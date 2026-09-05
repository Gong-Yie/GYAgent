from datetime import datetime
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from self_cognition.application.replay import ReplayService
from self_cognition.core.deletions import (
    DeletionImpact,
    InvalidatedModuleResult,
    DeletionMode,
    DeletionPlan,
    DeletionSelector,
    DeletionStatus,
)
from self_cognition.core.evidence import EvidenceSourceKind
from self_cognition.core.events import (
    CognitionModuleResultPayload,
    EventEnvelope,
    EventSource,
    StateReductionPayload,
)
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.contributions import ContributionOperation
from self_cognition.core.scopes import SubjectScope
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
        event_ids.intersection_update(event.event_id for event in events)
        impacts = self._expand_impacts(selector.subject, event_ids, memory_ids)
        primary = next(item for item in impacts if item.subject == selector.subject)
        plan = DeletionPlan(
            plan_id=uuid4(),
            selector=selector,
            memory_ids=primary.memory_ids,
            event_ids=primary.event_ids,
            created_at=now,
            impacts=impacts,
        )
        self._deletion_repository.save(plan)
        return plan

    def execute(self, plan: DeletionPlan, *, now: datetime) -> DeletionPlan:
        stored = self._deletion_repository.get(plan.plan_id)
        if stored is None or stored.digest != plan.digest:
            raise ValueError("deletion plan is unknown or has changed")
        if stored.status is DeletionStatus.COMPLETED:
            return stored
        self._require_current_scope(stored)
        executing = stored.with_status(DeletionStatus.EXECUTING, now)
        self._deletion_repository.save(executing)
        try:
            for impact in executing.effective_impacts:
                self._event_store.redact(
                    impact.subject, impact.event_ids, executing.plan_id
                )
                self._evidence_repository.delete(impact.subject, impact.event_ids)
                self._memory_repository.delete(impact.subject, impact.memory_ids)
                if self._process_journal is not None:
                    self._process_journal.forget(impact.event_ids)
            self._record_invalidated_results(executing)
            for impact in executing.effective_impacts:
                subject = impact.subject
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

    def _expand_impacts(
        self,
        subject: SubjectScope,
        event_ids: set[UUID],
        memory_ids: set[UUID],
    ) -> tuple[DeletionImpact, ...]:
        events = self._event_store.read_by_mind(subject.mind)
        records = self._memory_repository.read_by_mind(subject.mind)
        histories = {
            record.memory_id: self._memory_repository.read_history(
                record.subject, record.memory_id
            )
            for record in records
        }
        deleted = set(event_ids)
        memories = set(memory_ids)
        changed = True
        while changed:
            before = (frozenset(deleted), frozenset(memories))
            removed_contributions = {
                contribution.contribution_id
                for event in events
                if event.event_id in deleted
                and isinstance(event.payload, CognitionModuleResultPayload)
                for contribution in event.payload.contributions
            }
            for event in events:
                payload = event.payload
                dependent = event.causation_id in deleted
                if isinstance(payload, CognitionModuleResultPayload):
                    dependent = (
                        dependent
                        or any(
                            ref.evidence_id in deleted
                            for contribution in payload.contributions
                            for ref in contribution.evidence_refs
                        )
                        or any(
                            str(item)
                            in contribution.value.get("candidate_contribution_ids", [])
                            for contribution in payload.contributions
                            if contribution.operation
                            is ContributionOperation.REVIEW_CONFLICT
                            and isinstance(contribution.value, dict)
                            for item in removed_contributions
                        )
                    )
                    if dependent or event.event_id in deleted:
                        deleted.update(payload.response_event_ids)
                if isinstance(payload, StateReductionPayload):
                    dependent = dependent or bool(
                        set(payload.applied_contribution_ids) & removed_contributions
                    )
                if dependent and event.source is not EventSource.USER:
                    deleted.add(event.event_id)
            for record in records:
                if any(
                    ref.evidence_id in deleted
                    for version in histories[record.memory_id]
                    for ref in version.evidence_refs
                ) or any(
                    source.contribution_id in removed_contributions
                    for version in histories[record.memory_id]
                    for source in version.sources
                ):
                    memories.add(record.memory_id)
            changed = before != (frozenset(deleted), frozenset(memories))
        subjects = {subject}
        subjects.update(event.subject for event in events if event.event_id in deleted)
        subjects.update(
            record.subject for record in records if record.memory_id in memories
        )
        return tuple(
            DeletionImpact(
                subject=owner,
                event_ids=tuple(
                    sorted(
                        event.event_id
                        for event in events
                        if event.subject == owner and event.event_id in deleted
                    )
                ),
                memory_ids=tuple(
                    sorted(
                        record.memory_id
                        for record in records
                        if record.subject == owner and record.memory_id in memories
                    )
                ),
                invalidated_results=tuple(
                    InvalidatedModuleResult(
                        event.event_id,
                        event.causation_id,
                        event.payload.module_id,
                        event.payload.module_version,
                        event.payload.deterministic,
                    )
                    for event in events
                    if event.subject == owner
                    and event.event_id in deleted
                    and isinstance(event.payload, CognitionModuleResultPayload)
                    and event.causation_id is not None
                    and event.causation_id not in deleted
                ),
            )
            for owner in sorted(
                subjects,
                key=lambda item: (item.subject.kind.value, item.subject.subject_id),
            )
        )

    def _require_current_scope(self, plan: DeletionPlan) -> None:
        event_ids = {
            item for impact in plan.effective_impacts for item in impact.event_ids
        }
        memory_ids = {
            item for impact in plan.effective_impacts for item in impact.memory_ids
        }
        current = self._expand_impacts(plan.selector.subject, event_ids, memory_ids)
        if any(
            not set(impact.event_ids).issubset(event_ids)
            or not set(impact.memory_ids).issubset(memory_ids)
            for impact in current
        ):
            raise ValueError(
                "deletion scope changed; create and confirm a new dry-run plan"
            )

    def _record_invalidated_results(self, plan: DeletionPlan) -> None:
        for impact in plan.effective_impacts:
            causes = {
                event.event_id: event
                for event in self._event_store.read_by_subject(impact.subject)
            }
            for invalidated in impact.invalidated_results:
                cause = causes.get(invalidated.cause_id)
                if cause is None:
                    continue
                event = EventEnvelope(
                    event_id=uuid5(
                        NAMESPACE_URL, f"deleted-result:{invalidated.event_id}"
                    ),
                    event_type="cognition.module_result",
                    actor=None,
                    subject=impact.subject,
                    payload=CognitionModuleResultPayload(
                        module_id=invalidated.module_id,
                        module_version=invalidated.module_version,
                        deterministic=invalidated.deterministic,
                        status="failed",
                        contributions=(),
                        response_event_ids=(),
                        failure_type="invalid_output",
                        error_type="EvidenceDeleted",
                    ),
                    occurred_at=plan.created_at,
                    recorded_at=plan.created_at,
                    source=EventSource.SYSTEM,
                    scope=cause.scope,
                    causation_id=cause.event_id,
                    correlation_id=cause.correlation_id,
                    run_id=cause.run_id,
                )
                self._event_store.append(event)
                self._evidence_repository.append(EvidenceRef.for_event(event))

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
            in {
                EvidenceSourceKind.EVENT,
                EvidenceSourceKind.MODEL_RESPONSE,
                EvidenceSourceKind.TOOL_RESULT,
            }
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
        deleted_events = {
            item for impact in plan.effective_impacts for item in impact.event_ids
        }
        for impact in plan.effective_impacts:
            self._verify_impact(impact, deleted_events)

    def _verify_impact(self, impact: DeletionImpact, deleted_events: set[UUID]) -> None:
        subject = impact.subject
        if any(
            event.event_id in deleted_events
            for event in self._event_store.read_by_subject(subject)
        ):
            raise RuntimeError("deleted event remains readable")
        remaining_memories = self._memory_repository.read_by_subject(subject)
        if any(record.memory_id in impact.memory_ids for record in remaining_memories):
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
