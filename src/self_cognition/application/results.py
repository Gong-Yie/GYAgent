from dataclasses import dataclass
from enum import Enum
from uuid import UUID

from self_cognition.core.state import SubjectState
from self_cognition.core.dialogue import DialogueDraft
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.identity import GoalRecord
from self_cognition.core.plans import Plan, PlanProgress

class ProcessEventStatus(str, Enum):
    SUCCEEDED = "succeeded"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ProcessEventResult:
    status: ProcessEventStatus
    run_id: UUID
    correlation_id: UUID
    old_version: int | None
    new_version: int | None
    state_changed: bool | None
    state: SubjectState | None
    event_saved: bool
    error_type: str | None = None
    retryable: bool | None = None


@dataclass(frozen=True, slots=True)
class ConverseResult:
    status: ProcessEventStatus
    run_id: UUID
    correlation_id: UUID
    old_version: int | None
    new_version: int | None
    state_changed: bool | None
    event_saved: bool
    response: DialogueDraft | None = None
    evidence_refs: tuple[EvidenceRef, ...] = ()
    response_event_id: UUID | None = None
    error_type: str | None = None
    reused: bool = False


@dataclass(frozen=True, slots=True)
class PursueGoalResult:
    status: ProcessEventStatus
    run_id: UUID
    correlation_id: UUID
    goal: GoalRecord | None = None
    plan: Plan | None = None
    progress: PlanProgress | None = None
    event_id: UUID | None = None
    error_type: str | None = None
    reused: bool = False
