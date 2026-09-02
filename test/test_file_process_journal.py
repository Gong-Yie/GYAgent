from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

from self_cognition.application.process_event import ProcessEventService
from self_cognition.application.results import ProcessEventStatus
from self_cognition.blackboard.reducer import StateReducer
from self_cognition.blackboard.service import CognitiveSpaceService
from self_cognition.cognition.semantic.preference_extractor import (
    PreferenceExtractor,
)
from self_cognition.core.contributions import CognitiveContribution
from self_cognition.core.events import EventEnvelope, StateReductionPayload
from self_cognition.core.processing import (
    PROCESS_EVENT_FAILED,
    RUN_CANCELLED,
    ProcessingStatus,
)
from self_cognition.core.scopes import SubjectScope
from self_cognition.core.state import SubjectState
from self_cognition.infrastructure.persistence.file_event_store import FileEventStore
from self_cognition.infrastructure.persistence.file_process_journal import (
    FileProcessJournal,
)
from self_cognition.infrastructure.persistence.file_processing_recovery import (
    FileProcessingRecovery,
)
from self_cognition.infrastructure.persistence.file_state_repository import (
    FileStateRepository,
)
from self_cognition.infrastructure.persistence.in_memory_evidence_repository import (
    InMemoryEvidenceRepository,
)
from self_cognition.runtime.engine import CognitionEngine
from self_cognition.runtime.run_context import RunContext


NOW = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)


class FixedClock:
    def now(self) -> datetime:
        return NOW


class CountingModule:
    subscriptions = frozenset({"user.message"})

    def __init__(self) -> None:
        self.calls = 0

    def process(
        self,
        event: EventEnvelope,
    ) -> tuple[CognitiveContribution, ...]:
        self.calls += 1
        return ()


class CancellingModule:
    subscriptions = frozenset({"user.message"})

    def process(
        self,
        event: EventEnvelope,
        context: RunContext,
    ) -> tuple[CognitiveContribution, ...]:
        context.cancel()
        return ()


class FailingEventStore:
    def append(self, event: EventEnvelope) -> None:
        raise OSError("simulated event write failure")

    def read_by_subject(
        self,
        subject: SubjectScope,
    ) -> tuple[EventEnvelope, ...]:
        return ()


class FailingStateRepository:
    def load(self, subject: SubjectScope | str) -> SubjectState | None:
        return None

    def save(self, state: SubjectState, expected_version: int) -> None:
        raise OSError("simulated state write failure")


def make_context(run_id: int) -> RunContext:
    return RunContext(
        run_id=UUID(int=run_id),
        correlation_id=UUID(int=100),
        deadline=NOW + timedelta(minutes=1),
        clock=FixedClock(),
    )


def make_event(event_id: int, run_id: int) -> EventEnvelope:
    return EventEnvelope.user_message(
        "user-1",
        f"message-{event_id}",
        event_id=UUID(int=event_id),
        clock=FixedClock(),
        run_id=UUID(int=run_id),
        correlation_id=UUID(int=100),
    )


def make_service(
    event_store: FileEventStore | FailingEventStore,
    state_repository: FileStateRepository | FailingStateRepository,
    journal: FileProcessJournal,
    modules: tuple[object, ...],
) -> ProcessEventService:
    return ProcessEventService(
        event_store=event_store,
        evidence_repository=InMemoryEvidenceRepository(),
        state_repository=state_repository,
        engine=CognitionEngine(
            modules,
            CognitiveSpaceService(StateReducer()),
        ),
        process_journal=journal,
    )


def test_journal_records_transitions_and_acknowledges_once(tmp_path: Path) -> None:
    journal = FileProcessJournal(tmp_path / "processing")
    event = make_event(1, 1)

    journal.begin(event, UUID(int=1), NOW)
    pending_retry = journal.retry(
        event.event_id,
        UUID(int=1),
        NOW,
        available_at=NOW,
        error_code=PROCESS_EVENT_FAILED,
        error_type="LookupError",
    )
    journal.begin(event, UUID(int=2), NOW)
    completed = journal.complete(event.event_id, UUID(int=2), NOW)
    history_before_duplicate = journal.read_history(event.event_id)
    journal.complete(event.event_id, UUID(int=3), NOW)

    assert pending_retry.status is ProcessingStatus.PENDING
    assert pending_retry.error_code == PROCESS_EVENT_FAILED
    assert completed.status is ProcessingStatus.COMPLETED
    assert journal.pending_outbox() == ()
    assert tuple(record.status for record in history_before_duplicate) == (
        ProcessingStatus.PENDING,
        ProcessingStatus.PROCESSING,
        ProcessingStatus.PENDING,
        ProcessingStatus.PROCESSING,
        ProcessingStatus.COMPLETED,
    )
    assert journal.read_history(event.event_id) == history_before_duplicate


def test_recovery_rebuilds_pending_failed_and_completed_records(
    tmp_path: Path,
) -> None:
    event_log = tmp_path / "events.jsonl"
    store = FileEventStore(event_log)
    pending = make_event(1, 1)
    failed = make_event(2, 2)
    completed = make_event(3, 3)
    for event in (pending, failed, completed):
        store.append(event)
    store.append(
        EventEnvelope.processing_failed(
            failed,
            stage="process_event",
            error_type="LookupError",
            clock=FixedClock(),
            run_id=UUID(int=2),
            correlation_id=UUID(int=100),
        )
    )
    store.append(
        EventEnvelope.state_reduced(
            completed,
            StateReductionPayload(0, 0, False, ()),
            clock=FixedClock(),
            run_id=UUID(int=3),
            correlation_id=UUID(int=100),
        )
    )
    store.append(
        EventEnvelope.processing_failed(
            completed,
            stage="process_event",
            error_type="OSError",
            clock=FixedClock(),
            run_id=UUID(int=3),
            correlation_id=UUID(int=100),
        )
    )
    journal = FileProcessJournal(tmp_path / "processing")
    journal.begin(completed, UUID(int=3), NOW)
    recovery = FileProcessingRecovery(event_log, journal)

    records = recovery.reconcile()
    history_lengths = {
        event.event_id: len(journal.read_history(event.event_id))
        for event in (pending, failed, completed)
    }
    recovery.reconcile()

    assert {record.event_id: record.status for record in records} == {
        pending.event_id: ProcessingStatus.PENDING,
        failed.event_id: ProcessingStatus.FAILED,
        completed.event_id: ProcessingStatus.COMPLETED,
    }
    assert journal.get(failed.event_id).error_type == "LookupError"
    assert {entry.event_id for entry in journal.pending_outbox()} == {
        pending.event_id
    }
    assert tuple(record.event_id for record in journal.dead_letters()) == (
        failed.event_id,
    )
    assert {
        event.event_id: len(journal.read_history(event.event_id))
        for event in (pending, failed, completed)
    } == history_lengths


def test_completed_event_is_not_processed_twice(tmp_path: Path) -> None:
    event_store = FileEventStore(tmp_path / "events.jsonl")
    journal = FileProcessJournal(tmp_path / "processing")
    module = CountingModule()
    service = make_service(
        event_store,
        FileStateRepository(tmp_path / "states"),
        journal,
        (module,),
    )
    event = EventEnvelope.user_message(
        "user-1",
        "test",
        event_id=UUID(int=1),
        clock=FixedClock(),
    )

    first = service.process(event, make_context(1))
    second = service.process(event, make_context(2))

    assert first.status is ProcessEventStatus.SUCCEEDED
    assert second.status is ProcessEventStatus.SUCCEEDED
    assert second.old_version == second.new_version == 0
    assert second.state_changed is False
    assert module.calls == 1
    assert tuple(
        stored.event_type
        for stored in event_store.read_by_subject(event.subject)
    ) == ("user.message", "state.reduced")


def test_event_write_failure_does_not_create_processing_record(
    tmp_path: Path,
) -> None:
    journal = FileProcessJournal(tmp_path / "processing")
    event = EventEnvelope.user_message("user-1", "test", clock=FixedClock())
    service = make_service(
        FailingEventStore(),
        FileStateRepository(tmp_path / "states"),
        journal,
        (),
    )

    result = service.process(event, make_context(1))

    assert result.status is ProcessEventStatus.FAILED
    assert result.event_saved is False
    assert journal.get(event.event_id) is None
    assert journal.pending_outbox() == ()


def test_state_write_failure_is_recorded_as_dead_letter(
    tmp_path: Path,
) -> None:
    journal = FileProcessJournal(tmp_path / "processing")
    event_store = FileEventStore(tmp_path / "events.jsonl")
    service = make_service(
        event_store,
        FailingStateRepository(),
        journal,
        (PreferenceExtractor(),),
    )
    event = EventEnvelope.user_message("user-1", "我喜欢晚上学习", clock=FixedClock())

    result = service.process(event, make_context(1))
    record = journal.get(event.event_id)

    assert result.status is ProcessEventStatus.FAILED
    assert record is not None
    assert record.status is ProcessingStatus.FAILED
    assert record.error_code == PROCESS_EVENT_FAILED
    assert journal.pending_outbox() == ()
    assert tuple(record.event_id for record in journal.dead_letters()) == (
        event.event_id,
    )


def test_completed_record_survives_ack_failure_and_recovery(
    tmp_path: Path,
    monkeypatch,
) -> None:
    event_log = tmp_path / "events.jsonl"
    journal = FileProcessJournal(tmp_path / "processing")
    service = make_service(
        FileEventStore(event_log),
        FileStateRepository(tmp_path / "states"),
        journal,
        (PreferenceExtractor(),),
    )
    event = EventEnvelope.user_message("user-1", "我喜欢晚上学习", clock=FixedClock())

    def fail_ack(event_id: UUID, acknowledged_at: datetime) -> None:
        raise OSError("simulated acknowledgement failure")

    with monkeypatch.context() as patch:
        patch.setattr(journal, "_acknowledge_unlocked", fail_ack)
        result = service.process(event, make_context(1))

    record_before_recovery = journal.get(event.event_id)
    assert result.status is ProcessEventStatus.FAILED
    assert record_before_recovery is not None
    assert record_before_recovery.status is ProcessingStatus.COMPLETED
    assert {entry.event_id for entry in journal.pending_outbox()} == {
        event.event_id
    }

    FileProcessingRecovery(event_log, journal).reconcile()

    assert journal.get(event.event_id).status is ProcessingStatus.COMPLETED
    assert journal.pending_outbox() == ()


def test_cancelled_processing_uses_stable_error_code(tmp_path: Path) -> None:
    journal = FileProcessJournal(tmp_path / "processing")
    event = EventEnvelope.user_message("user-1", "test", clock=FixedClock())
    service = make_service(
        FileEventStore(tmp_path / "events.jsonl"),
        FileStateRepository(tmp_path / "states"),
        journal,
        (CancellingModule(),),
    )

    result = service.process(event, make_context(1))
    record = journal.get(event.event_id)

    assert result.status is ProcessEventStatus.CANCELLED
    assert record is not None
    assert record.status is ProcessingStatus.FAILED
    assert record.error_code == RUN_CANCELLED
