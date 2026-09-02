from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from time import sleep
from uuid import UUID

from self_cognition.application.process_event import ProcessEventService
from self_cognition.application.results import ProcessEventStatus
from self_cognition.blackboard.reducer import StateReducer
from self_cognition.blackboard.service import CognitiveSpaceService
from self_cognition.core.contributions import CognitiveContribution
from self_cognition.core.events import EventEnvelope
from self_cognition.core.errors import ContractValidationError
from self_cognition.core.processing import ProcessingStatus
from self_cognition.infrastructure.persistence.file_processing_recovery import (
    FileProcessingRecovery,
)
from self_cognition.infrastructure.persistence.file_event_store import FileEventStore
from self_cognition.infrastructure.persistence.file_process_journal import (
    FileProcessJournal,
)
from self_cognition.infrastructure.persistence.file_state_repository import (
    FileStateRepository,
)
from self_cognition.infrastructure.persistence.in_memory_evidence_repository import (
    InMemoryEvidenceRepository,
)
from self_cognition.runtime.engine import CognitionEngine
from self_cognition.runtime.event_bus import RetryPolicy, SingleMachineEventBus
from self_cognition.runtime.run_context import RunContext


class MutableClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)

    def now(self) -> datetime:
        return self.value

    def advance(self, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


class FlakyModule:
    subscriptions = frozenset({"user.message"})

    def __init__(self, failures: int) -> None:
        self.remaining = failures

    def process(self, event: EventEnvelope) -> tuple[CognitiveContribution, ...]:
        if self.remaining:
            self.remaining -= 1
            raise OSError("temporary storage failure")
        return ()


class ContractFailingModule:
    subscriptions = frozenset({"user.message"})

    def process(self, event: EventEnvelope) -> tuple[CognitiveContribution, ...]:
        raise ContractValidationError("invalid event contract")


class ConcurrencyProbeModule:
    subscriptions = frozenset({"user.message"})

    def __init__(self) -> None:
        self._lock = Lock()
        self._active_by_subject: dict[str, int] = {}
        self.calls: list[tuple[str, UUID]] = []
        self.max_total = 0
        self.max_per_subject = 0

    def process(self, event: EventEnvelope) -> tuple[CognitiveContribution, ...]:
        subject_id = event.subject.subject.subject_id
        with self._lock:
            self._active_by_subject[subject_id] = (
                self._active_by_subject.get(subject_id, 0) + 1
            )
            self.calls.append((subject_id, event.event_id))
            self.max_total = max(
                self.max_total,
                sum(self._active_by_subject.values()),
            )
            self.max_per_subject = max(
                self.max_per_subject,
                self._active_by_subject[subject_id],
            )
        sleep(0.02)
        with self._lock:
            self._active_by_subject[subject_id] -= 1
        return ()


def make_bus(tmp_path: Path, module: object, clock: MutableClock) -> tuple[
    SingleMachineEventBus, FileProcessJournal
]:
    event_store = FileEventStore(tmp_path / "events.jsonl")
    journal = FileProcessJournal(tmp_path / "processing")
    service = ProcessEventService(
        event_store,
        InMemoryEvidenceRepository(),
        FileStateRepository(tmp_path / "states"),
        CognitionEngine((module,), CognitiveSpaceService(StateReducer())),
        journal,
    )
    return (
        SingleMachineEventBus(
            event_store,
            journal,
            service,
            retry_policy=RetryPolicy(
                max_attempts=3,
                lease_timeout=timedelta(seconds=30),
                backoffs=(timedelta(seconds=1), timedelta(seconds=2)),
            ),
        ),
        journal,
    )


def test_retry_backoff_and_eventual_success(tmp_path: Path) -> None:
    clock = MutableClock()
    bus, journal = make_bus(tmp_path, FlakyModule(2), clock)
    event = EventEnvelope.user_message("user-1", "test", clock=clock)
    context = RunContext(
        UUID(int=1), UUID(int=2), clock.now() + timedelta(minutes=1), clock=clock
    )
    bus.publish(event, context)

    first = bus.drain(clock)
    assert first[0].status is ProcessEventStatus.FAILED
    assert tuple(entry.event_id for entry in bus.backlog()) == (event.event_id,)
    assert journal.get(event.event_id).status is ProcessingStatus.PENDING
    clock.advance(1)
    assert bus.drain(clock)[0].status is ProcessEventStatus.FAILED
    clock.advance(2)
    assert bus.drain(clock)[0].status is ProcessEventStatus.SUCCEEDED
    assert bus.dead_letters() == ()


def test_retry_exhaustion_is_queryable_dead_letter(tmp_path: Path) -> None:
    clock = MutableClock()
    bus, journal = make_bus(tmp_path, FlakyModule(5), clock)
    event = EventEnvelope.user_message("user-1", "test", clock=clock)
    context = RunContext(
        UUID(int=1), UUID(int=2), clock.now() + timedelta(minutes=1), clock=clock
    )
    bus.publish(event, context)

    bus.drain(clock)
    clock.advance(1)
    bus.drain(clock)
    clock.advance(2)
    result = bus.drain(clock)[0]

    assert result.status is ProcessEventStatus.FAILED
    assert journal.get(event.event_id).status is ProcessingStatus.FAILED
    assert tuple(record.event_id for record in bus.dead_letters()) == (
        event.event_id,
    )
    assert journal.pending_outbox() == ()


def test_contract_failure_is_not_retried(tmp_path: Path) -> None:
    clock = MutableClock()
    bus, journal = make_bus(tmp_path, ContractFailingModule(), clock)
    event = EventEnvelope.user_message("user-1", "test", clock=clock)
    context = RunContext(
        UUID(int=1), UUID(int=2), clock.now() + timedelta(minutes=1), clock=clock
    )
    bus.publish(event, context)

    result = bus.drain(clock)[0]

    assert result.retryable is False
    assert journal.get(event.event_id).attempt_count == 1
    assert tuple(record.event_id for record in bus.dead_letters()) == (
        event.event_id,
    )


def test_same_subject_is_ordered_and_different_subjects_run_in_parallel(
    tmp_path: Path,
) -> None:
    clock = MutableClock()
    module = ConcurrencyProbeModule()
    bus, _ = make_bus(tmp_path, module, clock)
    events = (
        EventEnvelope.user_message(
            "user-1", "one", event_id=UUID(int=1), clock=clock
        ),
        EventEnvelope.user_message(
            "user-1", "two", event_id=UUID(int=2), clock=clock
        ),
        EventEnvelope.user_message(
            "user-2", "one", event_id=UUID(int=3), clock=clock
        ),
        EventEnvelope.user_message(
            "user-2", "two", event_id=UUID(int=4), clock=clock
        ),
    )
    for index, event in enumerate(events, start=1):
        context = RunContext(
            UUID(int=index),
            UUID(int=100 + index),
            clock.now() + timedelta(minutes=1),
            clock=clock,
        )
        bus.publish(event, context)
    bus.publish(
        events[0],
        RunContext(
            UUID(int=1),
            UUID(int=101),
            clock.now() + timedelta(minutes=1),
            clock=clock,
        ),
    )

    results = bus.drain(clock)

    assert len(results) == 4
    assert module.max_total >= 2
    assert module.max_per_subject == 1
    assert [event_id for subject, event_id in module.calls if subject == "user-1"] == [
        UUID(int=1),
        UUID(int=2),
    ]
    assert [event_id for subject, event_id in module.calls if subject == "user-2"] == [
        UUID(int=3),
        UUID(int=4),
    ]


def test_stale_processing_lease_can_be_reclaimed(tmp_path: Path) -> None:
    clock = MutableClock()
    bus, journal = make_bus(tmp_path, FlakyModule(0), clock)
    event = EventEnvelope.user_message("user-1", "test", clock=clock)
    context = RunContext(
        UUID(int=1), UUID(int=2), clock.now() + timedelta(minutes=1), clock=clock
    )
    bus.publish(event, context)
    journal.claim(event.event_id, UUID(int=3), clock.now(), timedelta(seconds=1))

    clock.advance(2)
    restarted_bus, restarted_journal = make_bus(tmp_path, FlakyModule(0), clock)
    FileProcessingRecovery(
        tmp_path / "events.jsonl",
        restarted_journal,
    ).reconcile()
    results = restarted_bus.drain(clock)
    assert results[0].status is ProcessEventStatus.SUCCEEDED
