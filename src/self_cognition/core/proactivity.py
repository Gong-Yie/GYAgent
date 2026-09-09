from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import UUID

from self_cognition.core.affect import Motive
from self_cognition.core.errors import ContractValidationError
from self_cognition.core.scopes import SubjectScope

if TYPE_CHECKING:
    from self_cognition.core.evidence import EvidenceRef


class IntentionStatus(str, Enum):
    ACTIVE = "active"
    ACCEPTED = "accepted"
    DELAYED = "delayed"
    MERGED = "merged"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    EXECUTED = "executed"
    SILENT = "silent"


class BehavioralAction(str, Enum):
    ACCEPT = "accept"
    DELAY = "delay"
    MERGE = "merge"
    CANCEL = "cancel"
    EXPIRE = "expire"
    EXECUTE = "execute"
    SILENCE = "silence"
    REFUSE = "refuse"
    NEGOTIATE = "negotiate"
    PERSIST = "persist"


@dataclass(frozen=True, slots=True)
class ProactiveIntention:
    intention_id: UUID
    motive: Motive
    target: SubjectScope
    expected_behavior: str
    priority: int
    state_version: int
    evidence_refs: tuple[EvidenceRef, ...]
    created_at: datetime
    valid_until: datetime
    budget: int = 1
    stop_conditions: tuple[str, ...] = ()
    idempotency_key: str = ""
    status: IntentionStatus = IntentionStatus.ACTIVE

    def __post_init__(self) -> None:
        from self_cognition.core.evidence import EvidenceRef

        if not isinstance(self.intention_id, UUID):
            raise ContractValidationError("intention ID must be a UUID")
        if not isinstance(self.motive, Motive):
            raise ContractValidationError("intention motive is invalid")
        if not isinstance(self.target, SubjectScope):
            raise ContractValidationError("intention target is invalid")
        if not isinstance(self.expected_behavior, str) or not self.expected_behavior.strip():
            raise ContractValidationError("expected behavior must not be blank")
        if type(self.priority) is not int or self.priority < 0:
            raise ContractValidationError("intention priority must be non-negative")
        if type(self.state_version) is not int or self.state_version < 0:
            raise ContractValidationError("intention state version must be non-negative")
        if any(not isinstance(ref, EvidenceRef) for ref in self.evidence_refs):
            raise ContractValidationError("intention evidence is invalid")
        if any(ref.scope.owner.mind != self.target.mind for ref in self.evidence_refs):
            raise ContractValidationError("intention evidence crosses minds")
        _aware(self.created_at, "intention created_at")
        _aware(self.valid_until, "intention valid_until")
        if self.valid_until <= self.created_at:
            raise ContractValidationError("intention validity must be positive")
        if type(self.budget) is not int or self.budget < 0:
            raise ContractValidationError("intention budget must be non-negative")
        if any(not isinstance(item, str) or not item.strip() for item in self.stop_conditions):
            raise ContractValidationError("intention stop conditions are invalid")
        if not isinstance(self.idempotency_key, str) or not self.idempotency_key.strip():
            raise ContractValidationError("intention idempotency key must not be blank")
        if not isinstance(self.status, IntentionStatus):
            raise ContractValidationError("intention status is invalid")

    @property
    def motive_id(self) -> UUID:
        return self.motive.motive_id

    def is_expired(self, as_of: datetime) -> bool:
        _aware(as_of, "intention as_of")
        return as_of >= self.valid_until

    def to_state_value(self) -> dict[str, Any]:
        return {
            "intention_id": str(self.intention_id),
            "motive_id": str(self.motive_id),
            "target": {
                "mind_id": self.target.mind.mind_id,
                "kind": self.target.subject.kind.value,
                "subject_id": self.target.subject.subject_id,
            },
            "expected_behavior": self.expected_behavior,
            "priority": self.priority,
            "state_version": self.state_version,
            "evidence_refs": [str(ref.evidence_id) for ref in self.evidence_refs],
            "created_at": self.created_at.isoformat(),
            "valid_until": self.valid_until.isoformat(),
            "budget": self.budget,
            "stop_conditions": list(self.stop_conditions),
            "idempotency_key": self.idempotency_key,
            "status": self.status.value,
        }


@dataclass(frozen=True, slots=True)
class BehavioralDecision:
    decision_id: UUID
    intention_id: UUID
    action: BehavioralAction
    reason: str
    evidence_refs: tuple[EvidenceRef, ...]
    state_version: int
    decided_at: datetime
    idempotency_key: str
    not_before: datetime | None = None
    merged_into: UUID | None = None
    response: str | None = None

    def __post_init__(self) -> None:
        from self_cognition.core.evidence import EvidenceRef

        if not isinstance(self.decision_id, UUID) or not isinstance(self.intention_id, UUID):
            raise ContractValidationError("decision IDs must be UUID values")
        if not isinstance(self.action, BehavioralAction):
            raise ContractValidationError("behavioral action is invalid")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ContractValidationError("decision reason must not be blank")
        if any(not isinstance(ref, EvidenceRef) for ref in self.evidence_refs):
            raise ContractValidationError("decision evidence is invalid")
        if type(self.state_version) is not int or self.state_version < 0:
            raise ContractValidationError("decision state version must be non-negative")
        _aware(self.decided_at, "decision decided_at")
        if not isinstance(self.idempotency_key, str) or not self.idempotency_key.strip():
            raise ContractValidationError("decision idempotency key must not be blank")
        if self.not_before is not None:
            _aware(self.not_before, "decision not_before")
        if self.merged_into is not None and not isinstance(self.merged_into, UUID):
            raise ContractValidationError("merged intention ID must be a UUID")
        if self.response is not None and (not isinstance(self.response, str) or not self.response.strip()):
            raise ContractValidationError("decision response must not be blank")
        if self.action is BehavioralAction.MERGE and self.merged_into is None:
            raise ContractValidationError("merge decision requires a target intention")
        if self.action is BehavioralAction.DELAY and self.not_before is None:
            raise ContractValidationError("delay decision requires not_before")

    @property
    def status(self) -> IntentionStatus:
        return {
            BehavioralAction.ACCEPT: IntentionStatus.ACCEPTED,
            BehavioralAction.DELAY: IntentionStatus.DELAYED,
            BehavioralAction.MERGE: IntentionStatus.MERGED,
            BehavioralAction.CANCEL: IntentionStatus.CANCELLED,
            BehavioralAction.EXPIRE: IntentionStatus.EXPIRED,
            BehavioralAction.EXECUTE: IntentionStatus.EXECUTED,
            BehavioralAction.SILENCE: IntentionStatus.SILENT,
            BehavioralAction.REFUSE: IntentionStatus.CANCELLED,
            BehavioralAction.NEGOTIATE: IntentionStatus.ACCEPTED,
            BehavioralAction.PERSIST: IntentionStatus.ACCEPTED,
        }[self.action]

    def to_state_value(self) -> dict[str, Any]:
        return {
            "decision_id": str(self.decision_id),
            "intention_id": str(self.intention_id),
            "action": self.action.value,
            "reason": self.reason,
            "evidence_refs": [str(ref.evidence_id) for ref in self.evidence_refs],
            "state_version": self.state_version,
            "decided_at": self.decided_at.isoformat(),
            "idempotency_key": self.idempotency_key,
            "not_before": self.not_before.isoformat() if self.not_before else None,
            "merged_into": str(self.merged_into) if self.merged_into else None,
            "response": self.response,
        }


def _aware(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ContractValidationError(f"{name} must include a timezone")


def proactive_dependency_ids(event: object) -> frozenset[UUID]:
    payload = getattr(event, "payload", None)
    ids: set[UUID] = set()
    causation_id = getattr(event, "causation_id", None)
    if isinstance(causation_id, UUID):
        ids.add(causation_id)
    for value in (
        getattr(payload, "intention", None),
        getattr(payload, "decision", None),
        getattr(payload, "motive", None),
    ):
        if isinstance(value, ProactiveIntention):
            ids.add(value.motive_id)
            ids.update(ref.evidence_id for ref in value.evidence_refs)
        elif isinstance(value, BehavioralDecision):
            ids.add(value.intention_id)
            ids.update(ref.evidence_id for ref in value.evidence_refs)
        elif isinstance(value, Motive):
            ids.update(value.source_event_ids)
    return frozenset(ids)


ProactiveIntentionStatus = IntentionStatus
BehaviorAction = BehavioralAction
