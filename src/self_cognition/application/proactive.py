from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import RLock
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from self_cognition.core.affect import Motive, compete_motives
from self_cognition.core.errors import ContractValidationError
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.events import (
    BehavioralDecisionPayload,
    EventEnvelope,
    MailboxAcknowledgedPayload,
    MailboxMessagePayload,
    MotiveFormedPayload,
    ProactiveIntentionPayload,
    UserMessagePayload,
)
from self_cognition.core.proactivity import (
    BehavioralAction,
    BehavioralDecision,
    IntentionStatus,
    MotiveProposal,
    ProactivityModel,
    ProactiveIntention,
)
from self_cognition.core.protocols import EvidenceRepository, EventStore, GovernanceRepository
from self_cognition.core.scopes import SubjectScope
from self_cognition.core.workspace import WorkspacePacket
from self_cognition.core.time import Clock, SYSTEM_CLOCK
from self_cognition.core.ids import new_event_id
from self_cognition.runtime.run_context import RunContext
from self_cognition.observability.metrics import MetricsRegistry


SOCIAL_CONNECTION_KIND = "social_connection"
PROACTIVE_KIND_COOLDOWN = timedelta(minutes=30)


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
        metrics: MetricsRegistry | None = None,
        model: ProactivityModel | None = None,
    ) -> None:
        self._events = event_store
        self._evidence = evidence_repository
        self._governance = governance
        self._metrics = metrics
        self._model = model
        self._proposal_lock = RLock()

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
        if self._metrics is not None:
            self._metrics.increment("proactive.motives.formed")
        return event

    def evaluate(
        self,
        event: EventEnvelope,
        workspace: WorkspacePacket,
        context: RunContext,
    ) -> EventEnvelope | None:
        if self._model is None:
            return None
        proposal = self._model.propose(event, workspace, context)
        if not proposal.should_form:
            return None
        available = {
            str(EvidenceRef.for_event(event).evidence_id),
            *(str(ref.evidence_id) for ref in workspace.evidence_refs),
        }
        if not set(proposal.evidence_ids).issubset(available):
            raise ContractValidationError("proactive motive cites unavailable evidence")
        now = context.clock.now()
        mind = SubjectScope.for_mind(event.subject.mind.mind_id)
        evidence = tuple(
            ref
            for ref in (EvidenceRef.for_event(event), *workspace.evidence_refs)
            if str(ref.evidence_id) in proposal.evidence_ids
        )
        motive = Motive(
            uuid4(),
            event.subject.mind.mind_id,
            proposal.kind,
            proposal.description,
            proposal.strength,
            proposal.priority,
            now,
            (event.event_id,),
        )
        intention = ProactiveIntention(
            uuid4(),
            motive,
            event.subject,
            proposal.expected_behavior,
            proposal.priority,
            0,
            evidence,
            now,
            now + timedelta(seconds=proposal.valid_for_seconds),
            idempotency_key=(
                f"model:{event.event_id}:{proposal.kind}:{proposal.expected_behavior}"
            ),
        )
        if any(
            isinstance(stored.payload, ProactiveIntentionPayload)
            and stored.payload.intention.idempotency_key == intention.idempotency_key
            for stored in self._read(mind)
        ):
            return None
        family = self._kind_family(proposal.kind)
        with self._proposal_lock:
            if self._recent_equivalent_intention(mind, family, now):
                return None
            return self.propose(intention, context=context)

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
                for event in self._read(subject)
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
                for event in self._read(subject)
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
        if self._metrics is not None:
            self._metrics.increment("proactive.intentions.proposed")
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
                for event in self._read(subject)
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
            self._scope(subject),
            decision,
            clock=context.clock if context else SYSTEM_CLOCK,
            correlation_id=context.correlation_id if context else None,
            run_id=context.run_id if context else None,
        )
        self._append(event)
        if self._metrics is not None:
            self._metrics.increment("proactive.decisions.recorded")
        return ProactiveDecisionResult(intention, decision, event)

    def active(
        self,
        subject: SubjectScope,
        *,
        as_of: datetime,
    ) -> tuple[ProactiveIntention, ...]:
        intentions = {
            event.payload.intention.intention_id: event.payload.intention
            for event in self._read(subject)
            if isinstance(event.payload, ProactiveIntentionPayload)
        }
        decisions = {
            event.payload.intention_id: event.payload.decision
            for event in self._read(subject)
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
        if self._metrics is not None:
            self._metrics.set_gauge("proactive.active", float(len(result)))
            self._metrics.increment("proactive.silenced", len(intentions) - len(result))
        return tuple(sorted(result, key=lambda item: (-item.priority, item.intention_id.int)))

    def consume_due(
        self,
        subject: SubjectScope,
        *,
        as_of: datetime,
        context: RunContext | None = None,
        workspace: WorkspacePacket | None = None,
    ) -> tuple[ProactiveDecisionResult, ...]:
        decisions = {
            event.payload.intention_id: event.payload.decision
            for event in self._read(subject)
            if isinstance(event.payload, BehavioralDecisionPayload)
        }
        results: list[ProactiveDecisionResult] = []
        for intention in self.active(subject, as_of=as_of):
            decision = decisions.get(intention.intention_id)
            if decision is not None and decision.action is not BehavioralAction.DELAY:
                continue
            if decision is not None and decision.not_before is not None and as_of < decision.not_before:
                continue
            result = self.decide(
                subject,
                intention.intention_id,
                BehavioralAction.ACCEPT,
                "scheduled proactive intention became due",
                state_version=intention.state_version,
                idempotency_key="scheduled-accept",
                response=self._expression_for(intention, workspace, context),
                context=context,
            )
            results.append(result)
            self._create_mailbox_message(subject, result, context=context)
        return tuple(results)

    def observe_event(
        self,
        event: EventEnvelope,
        *,
        context: RunContext | None = None,
    ) -> EventEnvelope | None:
        payload = event.payload
        if not isinstance(payload, UserMessagePayload):
            return None
        text = payload.text.strip()
        prefixes = ("提醒我", "记得提醒我", "请提醒我")
        prefix = next((item for item in prefixes if text.startswith(item)), None)
        if prefix is None:
            return None
        description = text[len(prefix):].strip(" ：:，,")
        if not description:
            return None
        now = context.clock.now() if context else event.recorded_at
        target = event.subject
        motive = Motive(
            uuid4(),
            target.mind.mind_id,
            "user_request",
            f"提醒：{description}",
            0.9,
            1,
            now,
            (event.event_id,),
        )
        intention = ProactiveIntention(
            uuid4(),
            motive,
            target,
            description,
            1,
            0,
            (EvidenceRef.for_event(event),),
            now,
            now + timedelta(days=1),
            idempotency_key=f"reminder:{event.event_id}",
        )
        return self.propose(intention, context=context)

    def form_boredom_social_motive(
        self,
        event: EventEnvelope,
        workspace: WorkspacePacket,
        context: RunContext | None = None,
    ) -> EventEnvelope | None:
        if not self._boredom_detected(workspace):
            return None
        now = context.clock.now() if context is not None else event.recorded_at
        mind = SubjectScope.for_mind(event.subject.mind.mind_id)
        with self._proposal_lock:
            if self._recent_equivalent_intention(
                mind,
                self._kind_family(SOCIAL_CONNECTION_KIND),
                now,
            ):
                return None
            motive = Motive(
                uuid4(),
                event.subject.mind.mind_id,
                SOCIAL_CONNECTION_KIND,
                "boredom and unmet social interaction need",
                0.8,
                1,
                now,
                (event.event_id,),
                expires_at=now + timedelta(hours=1),
            )
            intention = ProactiveIntention(
                uuid4(),
                motive,
                event.subject,
                "主动问候并邀请用户聊天",
                1,
                0,
                (EvidenceRef.for_event(event),),
                now,
                now + timedelta(minutes=10),
                idempotency_key=f"affect-social:{event.event_id}",
            )
            return self.propose(intention, context=context)

    @staticmethod
    def _boredom_detected(workspace: WorkspacePacket) -> bool:
        for item in workspace.fixed_context.emotion:
            content = item.get("content")
            if not isinstance(content, dict):
                continue
            emotion = str(content.get("emotion", ""))
            valence = str(content.get("valence", ""))
            try:
                arousal = float(content.get("arousal", 0.5))
            except (TypeError, ValueError):
                arousal = 0.5
            if emotion in {"boredom", "loneliness"} or (
                valence == "negative" and arousal <= 0.3
            ):
                return True
        return False

    def mailbox(
        self,
        subject: SubjectScope,
        *,
        as_of: datetime,
    ) -> tuple[dict[str, object], ...]:
        created = {
            event.payload.intention_id: event.payload
            for event in self._read(subject)
            if isinstance(event.payload, MailboxMessagePayload)
        }
        acknowledged = {
            event.payload.message_id
            for event in self._read(subject)
            if isinstance(event.payload, MailboxAcknowledgedPayload)
        }
        decisions = {
            event.payload.intention_id: event.payload.decision
            for event in self._read(subject)
            if isinstance(event.payload, BehavioralDecisionPayload)
        }
        items: list[dict[str, object]] = []
        for intention_id, message in created.items():
            decision = decisions.get(intention_id)
            if message.message_id in acknowledged:
                status = "acknowledged"
            elif decision is not None and decision.action is BehavioralAction.CANCEL:
                status = "cancelled"
            elif decision is not None and decision.action is BehavioralAction.EXPIRE:
                status = "expired"
            else:
                status = "pending"
            if message.valid_until <= as_of and status == "pending":
                status = "expired"
            items.append(
                {
                    "message_id": str(message.message_id),
                    "intention_id": str(intention_id),
                    "text": message.text,
                    "status": status,
                    "created_at": message.created_at.isoformat(),
                    "valid_until": message.valid_until.isoformat(),
                    "evidence_refs": [str(ref.evidence_id) for ref in message.evidence_refs],
                }
            )
        return tuple(sorted(items, key=lambda item: str(item["created_at"])))

    def acknowledge(
        self,
        subject: SubjectScope,
        intention_id: UUID,
        *,
        context: RunContext | None = None,
    ) -> ProactiveDecisionResult:
        intention = self._find_intention(subject, intention_id)
        message = next(
            (
                event.payload
                for event in self._read(subject)
                if isinstance(event.payload, MailboxMessagePayload)
                and event.payload.intention_id == intention_id
            ),
            None,
        )
        if message is None:
            raise ContractValidationError("proactive intention has no mailbox message")
        result = self.decide(
            subject,
            intention_id,
            BehavioralAction.EXECUTE,
            "user acknowledged proactive message",
            state_version=intention.state_version,
            response=intention.expected_behavior,
            idempotency_key="mailbox-acknowledged",
            context=context,
        )
        existing = next(
            (
                event
                for event in self._read(subject)
                if isinstance(event.payload, MailboxAcknowledgedPayload)
                and event.payload.message_id == message.message_id
            ),
            None,
        )
        if existing is None:
            self._append(
                EventEnvelope.mailbox_message_acknowledged(
                    self._scope(subject),
                    MailboxAcknowledgedPayload(message.message_id, intention_id),
                    clock=context.clock if context else SYSTEM_CLOCK,
                    causation_id=result.event.event_id,
                    correlation_id=context.correlation_id if context else None,
                    run_id=context.run_id if context else None,
                )
            )
        return result

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

    def _recent_equivalent_intention(
        self,
        mind: SubjectScope,
        family: str,
        as_of: datetime,
    ) -> bool:
        return any(
            isinstance(stored.payload, ProactiveIntentionPayload)
            and self._kind_family(
                stored.payload.intention.motive.kind
            )
            == family
            and stored.payload.intention.valid_until > as_of
            and stored.payload.intention.created_at
            > as_of - PROACTIVE_KIND_COOLDOWN
            for stored in self._read(mind)
        )

    @staticmethod
    def _kind_family(kind: str) -> str:
        normalized = kind.strip().lower().replace("-", "_")
        if "goal" in normalized:
            return "goal_followup"
        if "commit" in normalized or "remind" in normalized:
            return "commitment_reminder"
        if (
            "social" in normalized
            or "connection" in normalized
            or "check_in" in normalized
            or "chat" in normalized
        ):
            return "social_connection"
        return normalized.removeprefix("proactive_")

    def _find_intention(self, subject: SubjectScope, intention_id: UUID) -> ProactiveIntention:
        for event in self._read(subject):
            if isinstance(event.payload, ProactiveIntentionPayload) and event.payload.intention.intention_id == intention_id:
                return event.payload.intention
        raise ContractValidationError("unknown proactive intention")

    def _append(self, event: EventEnvelope) -> None:
        self._events.append(event)
        if self._evidence is not None:
            self._evidence.append(EvidenceRef.for_event(event))

    def _expression_for(
        self,
        intention: ProactiveIntention,
        workspace: WorkspacePacket | None,
        context: RunContext | None,
    ) -> str:
        if workspace is None or context is None or self._model is None:
            return intention.expected_behavior
        express = getattr(self._model, "express", None)
        if not callable(express):
            return intention.expected_behavior
        try:
            text = express(intention, workspace, context)
        except Exception:
            return intention.expected_behavior
        return text.strip() or intention.expected_behavior

    def _create_mailbox_message(
        self,
        subject: SubjectScope,
        result: ProactiveDecisionResult,
        *,
        context: RunContext | None,
    ) -> None:
        if result.decision.response is None:
            return
        if any(
            isinstance(event.payload, MailboxMessagePayload)
            and event.payload.intention_id == result.intention.intention_id
            for event in self._read(subject)
        ):
            return
        payload = MailboxMessagePayload(
            new_event_id(),
            result.intention.intention_id,
            result.decision.response,
            result.intention.evidence_refs,
            result.decision.decided_at,
            result.intention.valid_until,
        )
        self._append(
            EventEnvelope.mailbox_message_created(
                self._scope(subject),
                payload,
                clock=context.clock if context else SYSTEM_CLOCK,
                causation_id=result.event.event_id,
                correlation_id=context.correlation_id if context else None,
                run_id=context.run_id if context else None,
            )
        )

    @staticmethod
    def _scope(subject: SubjectScope) -> SubjectScope:
        return SubjectScope.for_mind(subject.mind.mind_id)

    def _read(self, subject: SubjectScope) -> tuple[EventEnvelope, ...]:
        return self._events.read_by_subject(self._scope(subject))
