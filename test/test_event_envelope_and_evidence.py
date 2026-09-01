from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from self_cognition.application.process_event import ProcessEventService
from self_cognition.application.results import ProcessEventStatus
from self_cognition.core.evidence import EvidenceRef, EvidenceSourceKind
from self_cognition.core.events import EventEnvelope, EventSource, UserMessagePayload
from self_cognition.core.events import StateReductionPayload
from self_cognition.core.scopes import (
    DataScope,
    DisclosureScope,
    MindScope,
    SubjectKind,
    SubjectRef,
    SubjectScope,
)
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


@dataclass(frozen=True, slots=True)
class FixedClock:
    value: datetime

    def now(self) -> datetime:
        return self.value


def make_subject(mind_id: str, subject_id: str) -> SubjectScope:
    return SubjectScope(
        MindScope(mind_id),
        SubjectRef(SubjectKind.USER, subject_id),
    )


def make_context(run_id: int = 10, correlation_id: int = 20) -> RunContext:
    return RunContext(
        run_id=UUID(int=run_id),
        correlation_id=UUID(int=correlation_id),
        deadline=datetime.now(timezone.utc) + timedelta(minutes=1),
    )


def test_user_message_envelope_has_typed_payload_times_and_private_scope():
    now = datetime(2026, 9, 1, 12, tzinfo=timezone.utc)
    subject = make_subject("mind-1", "user-1")

    event = EventEnvelope.user_message(
        subject,
        "测试消息",
        event_id=UUID(int=1),
        clock=FixedClock(now),
    )

    assert event.payload == UserMessagePayload("测试消息")
    assert event.actor == subject.subject
    assert event.subject == subject
    assert event.occurred_at == now
    assert event.recorded_at == now
    assert event.source is EventSource.USER
    assert event.scope == DataScope(subject, DisclosureScope.PRIVATE)


def test_process_service_binds_run_and_records_event_evidence():
    context = make_context()
    event = EventEnvelope.user_message(
        make_subject("mind-1", "user-1"),
        "测试消息",
    )
    event_store = InMemoryEventStore()
    evidence_repository = InMemoryEvidenceRepository()
    service = ProcessEventService(
        event_store=event_store,
        evidence_repository=evidence_repository,
        state_repository=InMemoryStateRepository(),
        engine=CognitionEngine((), CognitiveSpaceService(StateReducer())),
    )

    result = service.process(event, context)

    assert result.status is ProcessEventStatus.SUCCEEDED
    stored_events = event_store.read_by_subject(event.subject)
    recorded = stored_events[0]
    assert recorded.run_id == context.run_id
    assert recorded.correlation_id == context.correlation_id
    assert evidence_repository.get(event.subject, event.event_id) == (
        EvidenceRef.for_event(recorded)
    )
    reduction_event = stored_events[1]
    assert reduction_event.event_type == "state.reduced"
    assert reduction_event.causation_id == event.event_id
    assert reduction_event.payload == StateReductionPayload(0, 0, False, ())


def test_event_and_evidence_queries_isolate_same_subject_id_between_minds():
    first = EventEnvelope.user_message(
        make_subject("mind-1", "shared-user"),
        "第一心智",
        event_id=UUID(int=1),
    )
    second = EventEnvelope.user_message(
        make_subject("mind-2", "shared-user"),
        "第二心智",
        event_id=UUID(int=2),
    )
    event_store = InMemoryEventStore()
    evidence_repository = InMemoryEvidenceRepository()
    for event in (first, second):
        event_store.append(event)
        evidence_repository.append(EvidenceRef.for_event(event))

    assert event_store.read_by_subject(first.subject) == (first,)
    assert event_store.read_by_subject(second.subject) == (second,)
    assert evidence_repository.get(second.subject, first.event_id) is None
    with pytest.raises(TypeError):
        event_store.read_by_subject("shared-user")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "source_kind",
    [
        EvidenceSourceKind.FILE_FRAGMENT,
        EvidenceSourceKind.TOOL_RESULT,
        EvidenceSourceKind.MODEL_RESPONSE,
        EvidenceSourceKind.SYSTEM_PRIOR,
    ],
)
def test_evidence_ref_supports_non_event_source_kinds(
    source_kind: EvidenceSourceKind,
):
    subject = make_subject("mind-1", "user-1")

    evidence = EvidenceRef(
        evidence_id=UUID(int=10),
        source_kind=source_kind,
        source_ref="source-1",
        scope=DataScope(subject, DisclosureScope.PRIVATE),
    )

    assert evidence.source_kind is source_kind
