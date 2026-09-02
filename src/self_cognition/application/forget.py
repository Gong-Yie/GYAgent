from datetime import datetime
from uuid import UUID, uuid4

from self_cognition.application.replay import ReplayService
from self_cognition.core.deletions import (
    DeletionMode,
    DeletionPlan,
    DeletionSelector,
    DeletionStatus,
)
from self_cognition.core.evidence import EvidenceSourceKind
from self_cognition.core.events import EventEnvelope
from self_cognition.core.memories import MemoryRecord
from self_cognition.core.protocols import (
    DeletionRepository,
    EvidenceRepository,
    EventStore,
    MemoryRepository,
    ProcessJournal,
    StateRepository,
)


class ForgetService:
    def __init__(
        self,
        event_store: EventStore,
        evidence_repository: EvidenceRepository,
        state_repository: StateRepository,
        memory_repository: MemoryRepository,
        deletion_repository: DeletionRepository,
        replay: ReplayService,
        process_journal: ProcessJournal | None = None,
    ) -> None:
        self._event_store = event_store
        self._evidence_repository = evidence_repository
        self._state_repository = state_repository
        self._memory_repository = memory_repository
        self._deletion_repository = deletion_repository
        self._replay = replay
        self._process_journal = process_journal

    def dry_run(
        self,
        selector: DeletionSelector,
        *,
        now: datetime,
    ) -> DeletionPlan:
        records = self._memory_repository.read_by_subject(selector.subject)
        histories = {
            record.memory_id: self._memory_repository.read_history(
                selector.subject,
                record.memory_id,
            )
            for record in records
        }
        events = self._event_store.read_by_subject(selector.subject)
        selected = self._select_memories(selector, records)
        memory_ids = {record.memory_id for record in selected}
        event_ids = self._evidence_event_ids(
            tuple(
                version
                for record in selected
                for version in histories[record.memory_id]
            )
        )
        if selector.mode is DeletionMode.SUBJECT:
            memory_ids = {record.memory_id for record in records}
            event_ids = {event.event_id for event in events}
        else:
            changed = True
            while changed:
                event_ids = self._include_descendants(event_ids, events)
                cascading = {
                    record.memory_id
                    for record in records
                    if any(
                        evidence.evidence_id in event_ids
                        for version in histories[record.memory_id]
                        for evidence in version.evidence_refs
                    )
                }
                changed = not cascading.issubset(memory_ids)
                memory_ids.update(cascading)
                event_ids.update(
                    self._evidence_event_ids(
                        tuple(
                            version
                            for record in records
                            if record.memory_id in memory_ids
                            for version in histories[record.memory_id]
                        )
                    )
                )
            event_ids = self._include_descendants(event_ids, events)
        plan = DeletionPlan(
            plan_id=uuid4(),
            selector=selector,
            memory_ids=tuple(sorted(memory_ids, key=lambda value: value.int)),
            event_ids=tuple(sorted(event_ids, key=lambda value: value.int)),
            created_at=now,
        )
        self._deletion_repository.save(plan)
        return plan

    def execute(self, plan: DeletionPlan, *, now: datetime) -> DeletionPlan:
        stored = self._deletion_repository.get(plan.plan_id)
        if stored is None or stored.digest != plan.digest:
            raise ValueError("deletion plan is unknown or has changed")
        if stored.status is DeletionStatus.COMPLETED:
            return stored
        executing = stored.with_status(DeletionStatus.EXECUTING, now)
        self._deletion_repository.save(executing)
        try:
            subject = executing.selector.subject
            self._event_store.redact(subject, executing.event_ids, executing.plan_id)
            self._evidence_repository.delete(subject, executing.event_ids)
            self._memory_repository.delete(subject, executing.memory_ids)
            if self._process_journal is not None:
                self._process_journal.forget(executing.event_ids)
            remaining_events = self._event_store.read_by_subject(subject)
            if remaining_events:
                self._state_repository.replace(self._replay.replay(subject))
            else:
                self._state_repository.delete(subject)
            self._memory_repository.rebuild_index(subject)
            self._verify(executing)
        except Exception as error:
            failed = executing.with_status(
                DeletionStatus.FAILED,
                now,
                failure_type=type(error).__name__,
            )
            self._deletion_repository.save(failed)
            raise
        completed = executing.with_status(DeletionStatus.COMPLETED, now)
        self._deletion_repository.save(completed)
        return completed

    def recover(self, *, now: datetime) -> tuple[DeletionPlan, ...]:
        pending = self._deletion_repository.read_by_status(
            DeletionStatus.EXECUTING
        )
        return tuple(self.execute(plan, now=now) for plan in pending)

    @staticmethod
    def _select_memories(
        selector: DeletionSelector,
        records: tuple[MemoryRecord, ...],
    ) -> tuple[MemoryRecord, ...]:
        if selector.mode is DeletionMode.MEMORY:
            return tuple(
                record for record in records if record.memory_id == selector.memory_id
            )
        if selector.mode is DeletionMode.SUBJECT:
            return records
        return tuple(
            record
            for record in records
            if (
                not selector.memory_types
                or record.memory_type in selector.memory_types
            )
            and (
                selector.created_from is None
                or record.created_at >= selector.created_from
            )
            and (
                selector.created_to is None
                or record.created_at <= selector.created_to
            )
            and (
                selector.conversation_id is None
                or (
                    record.scope.conversation is not None
                    and record.scope.conversation.conversation_id
                    == selector.conversation_id
                )
            )
        )

    @staticmethod
    def _evidence_event_ids(records: tuple[MemoryRecord, ...]) -> set[UUID]:
        return {
            evidence.evidence_id
            for record in records
            for evidence in record.evidence_refs
            if evidence.source_kind
            in {EvidenceSourceKind.EVENT, EvidenceSourceKind.MODEL_RESPONSE}
        }

    @staticmethod
    def _include_descendants(
        event_ids: set[UUID],
        events: tuple[EventEnvelope, ...],
    ) -> set[UUID]:
        result = set(event_ids)
        changed = True
        while changed:
            descendants = {
                event.event_id
                for event in events
                if event.causation_id in result
            }
            changed = not descendants.issubset(result)
            result.update(descendants)
        return result

    def _verify(self, plan: DeletionPlan) -> None:
        subject = plan.selector.subject
        deleted_events = set(plan.event_ids)
        if any(
            event.event_id in deleted_events
            for event in self._event_store.read_by_subject(subject)
        ):
            raise RuntimeError("deleted event remains readable")
        remaining_memories = self._memory_repository.read_by_subject(subject)
        if any(record.memory_id in plan.memory_ids for record in remaining_memories):
            raise RuntimeError("deleted memory remains readable")
        if any(
            evidence.evidence_id in deleted_events
            for record in remaining_memories
            for version in self._memory_repository.read_history(
                subject,
                record.memory_id,
            )
            for evidence in version.evidence_refs
        ):
            raise RuntimeError("remaining memory references deleted evidence")
        state = self._state_repository.load(subject)
        if state is not None and any(
            evidence.evidence_id in deleted_events
            for atom in state.entries.values()
            for evidence in atom.evidence_refs
        ):
            raise RuntimeError("rebuilt state references deleted evidence")
