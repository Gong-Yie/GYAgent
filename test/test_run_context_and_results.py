from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid5, NAMESPACE_URL

from self_cognition.application.process_event import ProcessEventService
from self_cognition.application.results import ProcessEventStatus
from self_cognition.core.contributions import CognitiveContribution, CognitionType
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.events import Event
from self_cognition.infrastructure.persistence.in_memory_event_store import (
    InMemoryEventStore,
)
from self_cognition.infrastructure.persistence.in_memory_evidence_repository import (
    InMemoryEvidenceRepository,
)
from self_cognition.infrastructure.persistence.in_memory_state_repository import (
    InMemoryStateRepository,
)
from self_cognition.runtime.engine import CognitionEngine
from self_cognition.blackboard.reducer import StateReducer
from self_cognition.blackboard.service import CognitiveSpaceService
from self_cognition.runtime.run_context import RunContext


def make_context(run_id: int = 1) -> RunContext:
    return RunContext(
        run_id=UUID(int=run_id),
        correlation_id=UUID(int=100),
        deadline=datetime.now(timezone.utc) + timedelta(minutes=1),
    )


class RecordingReducer:
    def __init__(self) -> None:
        self.called = False

    def apply_many(self, state, contributions, *, decided_at):
        self.called = True
        return state


class RecordingModule:
    subscriptions = frozenset({"user.message"})

    def __init__(self) -> None:
        self.called = False

    def process(self, event: Event) -> tuple[CognitiveContribution, ...]:
        self.called = True
        return ()


class CancellingModule:
    subscriptions = frozenset({"user.message"})

    def __init__(self, context: RunContext) -> None:
        self._context = context

    def process(self, event: Event) -> tuple[CognitiveContribution, ...]:
        self._context.cancel()
        return (
            CognitiveContribution.set_from_event(
                event,
                contribution_id=uuid5(NAMESPACE_URL, str(event.event_id)),
                target_field="preferences.study_time",
                cognition_type=CognitionType.PREFERENCE,
                value="晚上",
                confidence=1.0,
                evidence_refs=(EvidenceRef.for_event(event),),
                source_module="test.cancelling_module",
                module_version="1",
            ),
        )


class PreferenceModule:
    subscriptions = frozenset({"user.message"})

    def process(self, event: Event) -> tuple[CognitiveContribution, ...]:
        return (
            CognitiveContribution.set_from_event(
                event,
                contribution_id=uuid5(NAMESPACE_URL, str(event.event_id)),
                target_field="preferences.study_time",
                cognition_type=CognitionType.PREFERENCE,
                value="晚上",
                confidence=1.0,
                evidence_refs=(EvidenceRef.for_event(event),),
                source_module="test.preference_module",
                module_version="1",
            ),
        )


class CancellingReducer:
    def __init__(self, context: RunContext) -> None:
        self._context = context

    def apply_many(self, state, contributions, *, decided_at):
        new_state = StateReducer().apply_many(
            state,
            contributions,
            decided_at=decided_at,
        )
        self._context.cancel()
        return new_state


class FailingModule:
    subscriptions = frozenset({"user.message"})

    def process(self, event: Event) -> tuple[CognitiveContribution, ...]:
        raise LookupError("test module failure")


def test_run_context_contains_ids_deadline_and_mutable_cancellation_state():
    context = make_context()

    assert context.run_id == UUID(int=1)
    assert context.correlation_id == UUID(int=100)
    assert context.deadline.tzinfo is not None
    assert context.is_cancelled is False

    context.cancel()

    assert context.is_cancelled is True
    assert not hasattr(context, "thread")
    assert not hasattr(context, "process")


def test_expired_deadline_is_treated_as_cancellation():
    context = RunContext(
        run_id=UUID(int=1),
        correlation_id=UUID(int=100),
        deadline=datetime.now(timezone.utc) - timedelta(seconds=1),
    )

    assert context.is_cancelled is True


def test_pre_cancelled_run_does_not_save_event_or_run_module():
    context = make_context()
    context.cancel()
    module = RecordingModule()
    event_store = InMemoryEventStore()
    state_repository = InMemoryStateRepository()
    service = ProcessEventService(
        event_store=event_store,
        evidence_repository=InMemoryEvidenceRepository(),
        state_repository=state_repository,
        engine=CognitionEngine(
            (module,),
            CognitiveSpaceService(RecordingReducer()),
        ),
    )
    event = Event.user_message("user-1", "我喜欢晚上学习")

    result = service.process(event, context)

    assert result.status is ProcessEventStatus.CANCELLED
    assert result.event_saved is False
    assert result.state_changed is False
    assert module.called is False
    assert event_store.read_by_subject(event.subject) == ()
    assert state_repository.load("user-1") is None


def test_cancellation_before_reducer_keeps_event_but_does_not_save_state():
    context = make_context()
    reducer = RecordingReducer()
    event_store = InMemoryEventStore()
    state_repository = InMemoryStateRepository()
    event = Event.user_message("user-1", "我喜欢晚上学习")
    service = ProcessEventService(
        event_store=event_store,
        evidence_repository=InMemoryEvidenceRepository(),
        state_repository=state_repository,
        engine=CognitionEngine(
            (CancellingModule(context),),
            CognitiveSpaceService(reducer),
        ),
    )

    result = service.process(event, context)

    assert result.status is ProcessEventStatus.CANCELLED
    assert result.old_version == 0
    assert result.new_version == 0
    assert result.event_saved is True
    assert reducer.called is False
    assert event_store.read_by_subject(event.subject)[0].event_id == event.event_id
    assert state_repository.load("user-1") is None


def test_cancellation_after_reducer_does_not_save_computed_state():
    context = make_context()
    event_store = InMemoryEventStore()
    state_repository = InMemoryStateRepository()
    event = Event.user_message("user-1", "我喜欢晚上学习")
    service = ProcessEventService(
        event_store=event_store,
        evidence_repository=InMemoryEvidenceRepository(),
        state_repository=state_repository,
        engine=CognitionEngine(
            (PreferenceModule(),),
            CognitiveSpaceService(CancellingReducer(context)),
        ),
    )

    result = service.process(event, context)

    assert result.status is ProcessEventStatus.CANCELLED
    assert result.old_version == 0
    assert result.new_version == 0
    assert result.event_saved is True
    assert event_store.read_by_subject(event.subject)[0].event_id == event.event_id
    assert state_repository.load("user-1") is None


def test_failure_result_preserves_error_type_and_correlation_id():
    context = make_context()
    event = Event.user_message("user-1", "触发测试失败")
    event_store = InMemoryEventStore()
    state_repository = InMemoryStateRepository()
    service = ProcessEventService(
        event_store=event_store,
        evidence_repository=InMemoryEvidenceRepository(),
        state_repository=state_repository,
        engine=CognitionEngine(
            (FailingModule(),),
            CognitiveSpaceService(RecordingReducer()),
        ),
    )

    result = service.process(event, context)

    assert result.status is ProcessEventStatus.FAILED
    assert result.error_type == "LookupError"
    assert result.correlation_id == context.correlation_id
    assert result.event_saved is True
    assert result.state is None
    assert tuple(
        stored_event.event_type
        for stored_event in event_store.read_by_subject(event.subject)
    ) == (
        "user.message",
        "processing.failed",
    )
    assert state_repository.load("user-1") is None
