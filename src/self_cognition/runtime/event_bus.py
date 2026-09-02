from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import Lock
from uuid import UUID

from self_cognition.application.process_event import ProcessEventService
from self_cognition.application.results import (
    ProcessEventResult,
    ProcessEventStatus,
)
from self_cognition.core.events import EventEnvelope
from self_cognition.core.processing import (
    PROCESS_EVENT_FAILED,
    OutboxEntry,
    ProcessingRecord,
)
from self_cognition.core.protocols import EventStore, ProcessJournal
from self_cognition.core.scopes import SubjectScope
from self_cognition.core.time import Clock, SYSTEM_CLOCK
from self_cognition.runtime.run_context import RunContext


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    lease_timeout: timedelta = timedelta(seconds=30)
    backoffs: tuple[timedelta, ...] = (
        timedelta(seconds=1),
        timedelta(seconds=2),
    )

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if self.lease_timeout <= timedelta(0):
            raise ValueError("lease_timeout must be positive")
        if len(self.backoffs) < self.max_attempts - 1:
            raise ValueError("backoffs must cover every retry")
        if any(delay < timedelta(0) for delay in self.backoffs):
            raise ValueError("backoffs must not be negative")

    def delay_after(self, attempt_count: int) -> timedelta:
        return self.backoffs[attempt_count - 1]


class SingleMachineEventBus:
    """Explicit single-process dispatcher with per-subject ordering."""

    def __init__(
        self,
        event_store: EventStore,
        journal: ProcessJournal,
        process_event: ProcessEventService,
        *,
        max_workers: int = 4,
        retry_policy: RetryPolicy = RetryPolicy(),
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self._event_store = event_store
        self._journal = journal
        self._process_event = process_event
        self._max_workers = max_workers
        self._retry_policy = retry_policy
        self._subject_locks: dict[SubjectScope, Lock] = {}
        self._subject_locks_guard = Lock()
        self._accepting = True
        self._accepting_guard = Lock()

    def start(self) -> None:
        with self._accepting_guard:
            self._accepting = True

    def stop_accepting(self) -> None:
        with self._accepting_guard:
            self._accepting = False

    def close(self) -> None:
        self.stop_accepting()

    def publish(
        self,
        event: EventEnvelope,
        context: RunContext,
    ) -> EventEnvelope:
        with self._accepting_guard:
            if not self._accepting:
                raise RuntimeError("event bus is not accepting new events")
        return self._process_event.enqueue(event, context)

    def drain(
        self,
        clock: Clock = SYSTEM_CLOCK,
    ) -> tuple[ProcessEventResult, ...]:
        entries = self._journal.claimable_outbox(clock.now())
        if not entries:
            return ()
        grouped = defaultdict(list)
        for entry in entries:
            grouped[entry.subject].append(entry)
        groups = tuple(
            (
                subject,
                tuple(
                    sorted(
                        subject_entries,
                        key=lambda entry: (entry.enqueued_at, entry.event_id.int),
                    )
                ),
            )
            for subject, subject_entries in grouped.items()
        )
        with ThreadPoolExecutor(
            max_workers=min(self._max_workers, len(groups))
        ) as executor:
            futures = [
                executor.submit(self._drain_subject, subject, group, clock)
                for subject, group in groups
            ]
            return tuple(
                result
                for future in futures
                for result in future.result()
            )

    def dead_letters(self) -> tuple[ProcessingRecord, ...]:
        return self._journal.dead_letters()

    def backlog(self) -> tuple[OutboxEntry, ...]:
        return self._journal.pending_outbox()

    def _drain_subject(
        self,
        subject: SubjectScope,
        entries: tuple[OutboxEntry, ...],
        clock: Clock,
    ) -> tuple[ProcessEventResult, ...]:
        results = []
        with self._subject_lock(subject):
            for entry in entries:
                claimed_at = clock.now()
                claimed = self._journal.claim(
                    entry.event_id,
                    entry.run_id,
                    claimed_at,
                    self._retry_policy.lease_timeout,
                )
                if claimed is None:
                    continue
                event = self._event(entry.event_id, entry.subject)
                context = RunContext(
                    run_id=entry.run_id,
                    correlation_id=event.correlation_id or entry.run_id,
                    deadline=claimed_at + self._retry_policy.lease_timeout,
                    clock=clock,
                )
                result = self._process_event.process_claimed(event, context)
                self._settle(event, context, claimed, result)
                results.append(result)
        return tuple(results)

    def _settle(
        self,
        event: EventEnvelope,
        context: RunContext,
        claimed: ProcessingRecord,
        result: ProcessEventResult,
    ) -> None:
        if result.status is ProcessEventStatus.SUCCEEDED:
            return
        if result.status is ProcessEventStatus.CANCELLED:
            self._process_event.finalize_cancellation(event, context)
            return
        error_type = result.error_type or "UnknownError"
        if (
            result.retryable
            and claimed.attempt_count < self._retry_policy.max_attempts
        ):
            now = context.clock.now()
            self._journal.retry(
                event.event_id,
                context.run_id,
                now,
                available_at=(
                    now + self._retry_policy.delay_after(claimed.attempt_count)
                ),
                error_code=PROCESS_EVENT_FAILED,
                error_type=error_type,
            )
            return
        self._process_event.finalize_failure(
            event,
            context,
            error_type=error_type,
        )

    def _event(
        self,
        event_id: UUID,
        subject: SubjectScope,
    ) -> EventEnvelope:
        for event in self._event_store.read_by_subject(subject):
            if event.event_id == event_id:
                return event
        raise LookupError(f"outbox event is missing: {event_id}")

    def _subject_lock(self, subject: SubjectScope) -> Lock:
        with self._subject_locks_guard:
            return self._subject_locks.setdefault(subject, Lock())
