from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import NAMESPACE_URL, UUID, uuid5

from self_cognition.core.affect import Motive, compete_motives
from self_cognition.core.errors import ContractValidationError
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.events import (
    BehavioralDecisionPayload,
    EventEnvelope,
    MotiveFormedPayload,
    ProactiveIntentionPayload,
)
from self_cognition.core.proactivity import (
    BehavioralAction,
    BehavioralDecision,
    IntentionStatus,
    ProactiveIntention,
)
from self_cognition.core.protocols import EvidenceRepository, EventStore, GovernanceRepository
from self_cognition.core.scopes import SubjectScope
from self_cognition.core.time import Clock, SYSTEM_CLOCK
from self_cognition.runtime.run_context import RunContext


@dataclass(frozen=True, slots=True)
class ProactiveDecisionResult:
    intention: ProactiveIntention
    decision: BehavioralDecision
    event: EventEnvelope
    reused: bool = False


class ProactiveIntentionService:
    """Persist and fold proactive intentions without a second fact store."""

    def __init__(
        self,
        event_store: EventStore,
        evidence_repository: EvidenceRepository | None = None,
        governance: GovernanceRepository | None = None,
    ) -> None:
        self._events = event_store
        self._evidence = evidence_repository
        self._governance = governance

    def form_motive(
        self,
        subject: SubjectScope,
        motive: Motive,
        *,
        context: RunContext | None = None,
    ) -> EventEnvelope:
        if motive.subject_id != subject.mind.mind_id and motive.subject_id != subject.subject.subject_id:
            raise ContractValidationError("motive subject does not match target")
        event = EventEnvelope.motive_formed(
            subject,
            motive,
            clock=context.clock if context else SYSTEM_CLOCK,
            correlation_id=context.correlation_id if context else None,
            run_id=context.run_id if context else None,
        )
        self._append(event)
        return event

    def propose(
        self,
        intention: ProactiveIntention,
        *,
        context: RunContext | None = None,
    ) -> EventEnvelope:
        subject = SubjectScope.for_mind(intention.target.mind.mind_id)
        existing = next(
            (
                event
                for event in self._events.read_by_subject(subject)
                if isinstance(event.payload, ProactiveIntentionPayload)
                and event.payload.intention.intention_id == intention.intention_id
            ),
            None,
        )
        if existing is not None:
            if existing.payload.intention != intention:
                raise ContractValidationError("intention ID already has different content")
            return existing
        motive_event = next(
            (
                event
                for event in self._events.read_by_subject(subject)
                if isinstance(event.payload, MotiveFormedPayload)
                and event.payload.motive.motive_id == intention.motive_id
            ),
            None,
        )
        if motive_event is None:
            self.form_motive(subject, intention.motive, context=context)
        event = EventEnvelope.proactive_intention(
            intention,
            clock=context.clock if context else SYSTEM_CLOCK,
            correlation_id=context.correlation_id if context else None,
            run_id=context.run_id if context else None,
        )
        self._append(event)
        return event

    def decide(
        self,
        subject: SubjectScope,
        intention_id: UUID,
        action: BehavioralAction,
        reason: str,
        *,
        evidence_refs: tuple[EvidenceRef, ...] = (),
        state_version: int = 0,
        not_before: datetime | None = None,
        merged_into: UUID | None = None,
        response: str | None = None,
        idempotency_key: str | None = None,
        context: RunContext | None = None,
    ) -> ProactiveDecisionResult:
        intention = self._find_intention(subject, intention_id)
        key = idempotency_key or f"{intention.idempotency_key}:{action.value}"
        decision_id = uuid5(NAMESPACE_URL, f"behavior:{intention_id}:{key}")
        existing = next(
            (
                event
                for event in self._events.read_by_subject(subject)
                if isinstance(event.payload, BehavioralDecisionPayload)
                and event.payload.decision.decision_id == decision_id
            ),
            None,
        )
        if existing is not None:
            payload = existing.payload
            return ProactiveDecisionResult(intention, payload.decision, existing, True)
        decision = BehavioralDecision(
            decision_id,
            intention_id,
            action,
            reason,
            evidence_refs,
            state_version,
            context.clock.now() if context else SYSTEM_CLOCK.now(),
            key,
            not_before,
            merged_into,
            response,
        )
        event = EventEnvelope.behavior_decided(
            subject,
            decision,
            clock=context.clock if context else SYSTEM_CLOCK,
            correlation_id=context.correlation_id if context else None,
            run_id=context.run_id if context else None,
        )
        self._append(event)
        return ProactiveDecisionResult(intention, decision, event)

    def active(
        self,
        subject: SubjectScope,
        *,
        as_of: datetime,
    ) -> tuple[ProactiveIntention, ...]:
        intentions = {
            event.payload.intention.intention_id: event.payload.intention
            for event in self._events.read_by_subject(subject)
            if isinstance(event.payload, ProactiveIntentionPayload)
        }
        decisions = {
            event.payload.intention_id: event.payload.decision
            for event in self._events.read_by_subject(subject)
            if isinstance(event.payload, BehavioralDecisionPayload)
        }
        result = []
        for intention_id, intention in intentions.items():
            if self._governance is not None:
                controls = self._governance.load_controls(subject)
                if not controls.allows_proactive(intention.motive.kind):
                    continue
            if intention.is_expired(as_of):
                continue
            decision = decisions.get(intention_id)
            status = decision.status if decision is not None else intention.status
            if status in {
                IntentionStatus.CANCELLED,
                IntentionStatus.EXPIRED,
                IntentionStatus.EXECUTED,
                IntentionStatus.SILENT,
                IntentionStatus.MERGED,
            }:
                continue
            if decision is not None and decision.action is BehavioralAction.DELAY and decision.not_before and as_of < decision.not_before:
                result.append(intention)
                continue
            result.append(intention)
        return tuple(sorted(result, key=lambda item: (-item.priority, item.intention_id.int)))

    def compete(
        self,
        motives: tuple[Motive, ...],
        *,
        as_of: datetime,
    ) -> tuple[Motive, ...]:
        return compete_motives(motives, as_of)

    def cancel(
        self,
        subject: SubjectScope,
        intention_id: UUID,
        reason: str,
        *,
        context: RunContext | None = None,
    ) -> ProactiveDecisionResult:
        return self.decide(
            subject,
            intention_id,
            BehavioralAction.CANCEL,
            reason,
            context=context,
        )

    def expire(
        self,
        subject: SubjectScope,
        intention_id: UUID,
        *,
        as_of: datetime,
        context: RunContext | None = None,
    ) -> ProactiveDecisionResult:
        intention = self._find_intention(subject, intention_id)
        if as_of < intention.valid_until:
            raise ContractValidationError("intention is not yet expired")
        return self.decide(
            subject,
            intention_id,
            BehavioralAction.EXPIRE,
            "intention validity window elapsed",
            context=context,
        )

    def merge(
        self,
        subject: SubjectScope,
        intention_id: UUID,
        merged_into: UUID,
        reason: str,
        *,
        context: RunContext | None = None,
    ) -> ProactiveDecisionResult:
        if intention_id == merged_into:
            raise ContractValidationError("an intention cannot merge into itself")
        self._find_intention(subject, merged_into)
        return self.decide(
            subject,
            intention_id,
            BehavioralAction.MERGE,
            reason,
            merged_into=merged_into,
            context=context,
        )

    def reevaluate(
        self,
        subject: SubjectScope,
        intention_id: UUID,
        *,
        reason: str,
        action: BehavioralAction = BehavioralAction.ACCEPT,
        context: RunContext | None = None,
    ) -> ProactiveDecisionResult:
        return self.decide(subject, intention_id, action, reason, context=context)

    def _find_intention(self, subject: SubjectScope, intention_id: UUID) -> ProactiveIntention:
        for event in self._events.read_by_subject(subject):
            if isinstance(event.payload, ProactiveIntentionPayload) and event.payload.intention.intention_id == intention_id:
                return event.payload.intention
        raise ContractValidationError("unknown proactive intention")

    def _append(self, event: EventEnvelope) -> None:
        self._events.append(event)
        if self._evidence is not None:
            self._evidence.append(EvidenceRef.for_event(event))
