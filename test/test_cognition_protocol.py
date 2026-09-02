from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

from self_cognition.application.process_event import ProcessEventService
from self_cognition.application.replay import ReplayService
from self_cognition.application.results import ProcessEventStatus
from self_cognition.blackboard.reducer import StateReducer
from self_cognition.blackboard.service import CognitiveSpaceService
from self_cognition.cognition.semantic.llm_extractor import LLMSemanticExtractor
from self_cognition.cognition.semantic.name_extractor import NameExtractor
from self_cognition.core.cognition import (
    CognitionFailureType,
    CognitionRequest,
    CognitionResultStatus,
)
from self_cognition.core.contributions import (
    CognitiveContribution,
    CognitionType,
    ContributionOperation,
)
from self_cognition.core.errors import MissingCognitionResultError
from self_cognition.core.events import (
    CognitionModuleResultPayload,
    EventEnvelope,
)
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.model_outputs import (
    ContributionCandidate,
    ModelExtractionResult,
)
from self_cognition.core.state import SubjectState
from self_cognition.core.workspace import RetrievalQuery
from self_cognition.infrastructure.persistence.in_memory_event_store import (
    InMemoryEventStore,
)
from self_cognition.infrastructure.persistence.file_event_store import FileEventStore
from self_cognition.infrastructure.persistence.in_memory_evidence_repository import (
    InMemoryEvidenceRepository,
)
from self_cognition.infrastructure.persistence.in_memory_state_repository import (
    InMemoryStateRepository,
)
from self_cognition.infrastructure.persistence.serialization import (
    event_from_json,
    event_to_json,
)
from self_cognition.runtime.engine import CognitionEngine
from self_cognition.runtime.run_context import RunContext


NOW = datetime(2026, 9, 3, 12, tzinfo=timezone.utc)


class FixedClock:
    def now(self) -> datetime:
        return NOW


def make_context(run_id: int) -> RunContext:
    return RunContext(
        run_id=UUID(int=run_id),
        correlation_id=UUID(int=100),
        deadline=NOW + timedelta(minutes=1),
        clock=FixedClock(),
    )


class RecordingModel:
    def __init__(self, value: str = "晚上") -> None:
        self.value = value
        self.calls = 0

    def extract(self, request: CognitionRequest) -> ModelExtractionResult:
        self.calls += 1
        context = request.run_context
        assert context is not None
        response = EventEnvelope.model_response(
            request.event,
            model="test-model",
            response_id=f"resp-{self.calls}",
            raw_output=f'{{"value":"{self.value}"}}',
            clock=context.clock,
            run_id=context.run_id,
            correlation_id=context.correlation_id,
        )
        context.emit_event(response)
        return ModelExtractionResult(
            response_id=response.payload.response_id,
            candidates=(
                ContributionCandidate(
                    target_field="preferences.study_time",
                    operation=ContributionOperation.SET,
                    cognition_type=CognitionType.PREFERENCE,
                    value=self.value,
                    confidence=1.0,
                    evidence_ids=(str(request.event.event_id),),
                ),
            ),
            response_evidence=EvidenceRef.for_event(response),
        )


class ContextReadingModule:
    subscriptions = frozenset({"user.message"})
    module_id = "test.context_reader"
    module_version = "1"
    deterministic = True

    def __init__(self, *, cross_subject: bool = False) -> None:
        self.cross_subject = cross_subject
        self.values: list[object] = []

    def run(
        self,
        request: CognitionRequest,
    ) -> tuple[CognitiveContribution, ...]:
        subject = (
            EventEnvelope.user_message("other-user", "test").subject
            if self.cross_subject
            else request.event.subject
        )
        packet = request.context.query(
            RetrievalQuery(
                subject=subject,
                task="read current name",
                purpose="cognition:test",
                field_patterns=("profile.name",),
            )
        )
        self.values.extend(item.content for item in packet.items)
        return ()


class FailOnceStateRepository(InMemoryStateRepository):
    def __init__(self) -> None:
        super().__init__()
        self.save_calls = 0

    def save(self, state: SubjectState, expected_version: int) -> None:
        self.save_calls += 1
        if self.save_calls == 1:
            raise OSError("simulated state write failure")
        super().save(state, expected_version)


def make_engine(model: RecordingModel) -> CognitionEngine:
    return CognitionEngine(
        (LLMSemanticExtractor(model),),
        CognitiveSpaceService(StateReducer()),
    )


def test_context_query_is_read_only_and_rejects_cross_subject_access():
    name_event = EventEnvelope.user_message("user-1", "我叫小明")
    state = CognitionEngine(
        (NameExtractor(),),
        CognitiveSpaceService(StateReducer()),
    ).process(name_event, SubjectState.empty("user-1"))
    event = EventEnvelope.user_message("user-1", "读取上下文")
    reader = ContextReadingModule()
    result = CognitionEngine(
        (reader,),
        CognitiveSpaceService(StateReducer()),
    ).analyze(event, state)[0]

    assert result.status is CognitionResultStatus.SUCCEEDED
    assert reader.values == ["小明"]

    denied = CognitionEngine(
        (ContextReadingModule(cross_subject=True),),
        CognitiveSpaceService(StateReducer()),
    ).analyze(event, state)[0]
    assert denied.status is CognitionResultStatus.FAILED
    assert denied.failure_type is CognitionFailureType.EXECUTION
    assert denied.error_type == "ScopeMismatchError"


def test_model_result_is_persisted_and_replay_does_not_call_model_again(
    tmp_path: Path,
):
    event_path = tmp_path / "events.jsonl"
    event_store = FileEventStore(event_path)
    state_repository = InMemoryStateRepository()
    model = RecordingModel()
    service = ProcessEventService(
        event_store,
        InMemoryEvidenceRepository(),
        state_repository,
        make_engine(model),
    )
    event = EventEnvelope.user_message("user-1", "我喜欢晚上学习")

    result = service.process(event, make_context(1))

    assert result.status is ProcessEventStatus.SUCCEEDED
    assert model.calls == 1
    stored = event_store.read_by_subject(event.subject)
    assert tuple(item.event_type for item in stored) == (
        "user.message",
        "model.response",
        "cognition.module_result",
        "state.reduced",
    )
    payload = stored[2].payload
    assert isinstance(payload, CognitionModuleResultPayload)
    assert payload.module_id == "semantic.llm_extractor"
    assert payload.module_version == "1"
    assert payload.deterministic is False
    assert payload.response_event_ids == (stored[1].event_id,)
    assert event_from_json(event_to_json(stored[2])) == stored[2]

    changed_model = RecordingModel("早上")
    replayed = ReplayService(
        FileEventStore(event_path),
        make_engine(changed_model),
    ).replay(event.subject)
    assert changed_model.calls == 0
    assert replayed == result.state
    assert replayed.get("preferences.study_time").value == "晚上"


def test_legacy_log_never_reinvokes_nondeterministic_module():
    event_store = InMemoryEventStore()
    event = EventEnvelope.user_message("user-1", "我喜欢晚上学习")
    event_store.append(event)
    model = RecordingModel()

    try:
        ReplayService(event_store, make_engine(model)).replay(event.subject)
    except MissingCognitionResultError as error:
        assert str(event.event_id) in str(error)
    else:
        raise AssertionError("missing model result must stop replay")
    assert model.calls == 0


def test_retry_reuses_persisted_model_result_after_state_write_failure():
    event_store = InMemoryEventStore()
    state_repository = FailOnceStateRepository()
    model = RecordingModel()
    service = ProcessEventService(
        event_store,
        InMemoryEvidenceRepository(),
        state_repository,
        make_engine(model),
    )
    event = EventEnvelope.user_message("user-1", "我喜欢晚上学习")

    first = service.process(event, make_context(1))
    second = service.process(event, make_context(2))

    assert first.status is ProcessEventStatus.FAILED
    assert first.error_type == "OSError"
    assert second.status is ProcessEventStatus.SUCCEEDED
    assert model.calls == 1
    assert second.state is not None
    assert second.state.get("preferences.study_time").value == "晚上"
