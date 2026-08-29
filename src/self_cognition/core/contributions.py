from dataclasses import dataclass
from uuid import UUID

from self_cognition.core.errors import ContractValidationError


@dataclass(frozen=True, slots=True)
class Contribution:
    contribution_id: UUID
    target_subject_id: str
    target_field: str
    value: object
    confidence: float
    evidence_event_ids: tuple[UUID, ...]
    source_event_id: UUID
    source_module: str
    source_model_response_id: str | None = None
    explicitly_confirmed: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ContractValidationError("confidence must be between 0 and 1")
        if not self.target_subject_id.strip():
            raise ContractValidationError("target_subject_id must not be blank")
        if not self.target_field.strip():
            raise ContractValidationError("target_field must not be blank")
        if not self.evidence_event_ids:
            raise ContractValidationError("evidence_event_ids must not be empty")
        if not self.source_module.strip():
            raise ContractValidationError("source_module must not be blank")
        if not isinstance(self.explicitly_confirmed, bool):
            raise ContractValidationError(
                "explicitly_confirmed must be a boolean"
            )
        if (
            self.source_model_response_id is not None
            and not self.source_model_response_id.strip()
        ):
            raise ContractValidationError(
                "source_model_response_id must not be blank"
            )
        if self.source_event_id not in self.evidence_event_ids:
            raise ContractValidationError(
                "source_event_id must be included in evidence_event_ids"
            )
