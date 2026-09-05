from dataclasses import dataclass
from enum import Enum
from uuid import UUID

from self_cognition.core.errors import ContractValidationError


class KnowledgeStatus(str, Enum):
    KNOWN = "known"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"
    EXPIRED = "expired"


class EvidenceBasis(str, Enum):
    DIRECT = "direct"
    INFERENCE = "inference"
    HYPOTHESIS = "hypothesis"


class FailureCause(str, Enum):
    PERMISSION = "permission"
    ENVIRONMENT = "environment"
    INPUT = "input"
    MODEL = "model"
    STRATEGY = "strategy"
    UNKNOWN = "unknown"


class SuggestedAction(str, Enum):
    ASK = "ask"
    SEARCH = "search"
    RETRY = "retry"
    CHANGE_STRATEGY = "change_strategy"
    STOP = "stop"


class ConflictStatus(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"
    INVALIDATED = "invalidated"


@dataclass(frozen=True, slots=True)
class MetacognitiveAssessment:
    target: str
    status: KnowledgeStatus
    basis: EvidenceBasis
    explanation: str
    failure_cause: FailureCause | None = None
    suggestions: tuple[SuggestedAction, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.target, str) or not self.target.strip():
            raise ContractValidationError("assessment target must not be blank")
        if not isinstance(self.explanation, str) or not self.explanation.strip():
            raise ContractValidationError("assessment explanation is required")
        if not isinstance(self.status, KnowledgeStatus):
            raise ContractValidationError("assessment knowledge status is invalid")
        if not isinstance(self.basis, EvidenceBasis):
            raise ContractValidationError("assessment evidence basis is invalid")
        if self.failure_cause is not None and not isinstance(
            self.failure_cause, FailureCause
        ):
            raise ContractValidationError("assessment failure cause is invalid")
        if not isinstance(self.suggestions, tuple) or any(
            not isinstance(action, SuggestedAction) for action in self.suggestions
        ):
            raise ContractValidationError("assessment suggestions are invalid")

    def to_state_value(self) -> dict[str, object]:
        return {
            "target": self.target,
            "status": self.status.value,
            "basis": self.basis.value,
            "explanation": self.explanation,
            "failure_cause": (
                self.failure_cause.value if self.failure_cause is not None else None
            ),
            "suggestions": [action.value for action in self.suggestions],
        }

    @classmethod
    def from_state_value(cls, value: object) -> "MetacognitiveAssessment":
        fields = {
            "target",
            "status",
            "basis",
            "explanation",
            "failure_cause",
            "suggestions",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ContractValidationError("metacognition assessment fields are invalid")
        if not isinstance(value["suggestions"], list):
            raise ContractValidationError("assessment suggestions must be an array")
        try:
            return cls(
                target=value["target"],
                status=KnowledgeStatus(value["status"]),
                basis=EvidenceBasis(value["basis"]),
                explanation=value["explanation"],
                failure_cause=(
                    FailureCause(value["failure_cause"])
                    if value["failure_cause"] is not None
                    else None
                ),
                suggestions=tuple(
                    SuggestedAction(item) for item in value["suggestions"]
                ),
            )
        except (TypeError, ValueError) as error:
            raise ContractValidationError("invalid metacognition assessment") from error


@dataclass(frozen=True, slots=True)
class ConflictReview:
    candidate_contribution_ids: tuple[UUID, ...]
    status: ConflictStatus
    reason: str
    requires_confirmation: bool = False
    selected_contribution_id: UUID | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.candidate_contribution_ids, tuple)
            or len(set(self.candidate_contribution_ids)) < 2
            or any(
                not isinstance(item, UUID) for item in self.candidate_contribution_ids
            )
        ):
            raise ContractValidationError("a conflict requires distinct candidate IDs")
        if not isinstance(self.status, ConflictStatus):
            raise ContractValidationError("conflict status is invalid")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ContractValidationError("conflict review requires a reason")
        if not isinstance(self.requires_confirmation, bool):
            raise ContractValidationError("requires_confirmation must be boolean")
        if self.status is ConflictStatus.RESOLVED:
            if self.selected_contribution_id not in self.candidate_contribution_ids:
                raise ContractValidationError(
                    "resolution must select a conflict candidate"
                )
        elif self.selected_contribution_id is not None:
            raise ContractValidationError("only resolution may select a candidate")

    def to_state_value(self) -> dict[str, object]:
        return {
            "candidate_contribution_ids": [
                str(item) for item in self.candidate_contribution_ids
            ],
            "status": self.status.value,
            "reason": self.reason,
            "requires_confirmation": self.requires_confirmation,
            "selected_contribution_id": (
                str(self.selected_contribution_id)
                if self.selected_contribution_id is not None
                else None
            ),
        }

    @classmethod
    def from_state_value(cls, value: object) -> "ConflictReview":
        fields = {
            "candidate_contribution_ids",
            "status",
            "reason",
            "requires_confirmation",
            "selected_contribution_id",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ContractValidationError("conflict review fields are invalid")
        if not isinstance(value["candidate_contribution_ids"], list):
            raise ContractValidationError("conflict candidate IDs must be an array")
        try:
            return cls(
                candidate_contribution_ids=tuple(
                    UUID(item) for item in value["candidate_contribution_ids"]
                ),
                status=ConflictStatus(value["status"]),
                reason=value["reason"],
                requires_confirmation=value["requires_confirmation"],
                selected_contribution_id=(
                    UUID(value["selected_contribution_id"])
                    if value["selected_contribution_id"] is not None
                    else None
                ),
            )
        except (TypeError, ValueError, AttributeError) as error:
            raise ContractValidationError("invalid conflict review") from error
