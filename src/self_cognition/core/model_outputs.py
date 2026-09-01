from dataclasses import dataclass

from self_cognition.core.contributions import (
    CognitionType,
    ContributionOperation,
)
from self_cognition.core.evidence import EvidenceRef, EvidenceSourceKind
from self_cognition.core.errors import ModelOutputError


@dataclass(frozen=True, slots=True)
class ContributionCandidate:
    target_field: str
    operation: ContributionOperation
    cognition_type: CognitionType
    value: object
    confidence: float
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.target_field.strip():
            raise ModelOutputError("candidate target_field must not be blank")
        if not isinstance(self.operation, ContributionOperation):
            raise ModelOutputError("candidate operation is not supported")
        if not isinstance(self.cognition_type, CognitionType):
            raise ModelOutputError("candidate cognition_type is not supported")
        if not 0.0 <= self.confidence <= 1.0:
            raise ModelOutputError("candidate confidence must be between 0 and 1")
        if not self.evidence_ids:
            raise ModelOutputError("candidate evidence_ids must not be empty")
        if any(not evidence_id.strip() for evidence_id in self.evidence_ids):
            raise ModelOutputError("candidate evidence_ids must not be blank")


@dataclass(frozen=True, slots=True)
class ModelExtractionResult:
    response_id: str
    candidates: tuple[ContributionCandidate, ...]
    response_evidence: EvidenceRef

    def __post_init__(self) -> None:
        if not self.response_id.strip():
            raise ModelOutputError("model response_id must not be blank")
        if not isinstance(self.response_evidence, EvidenceRef):
            raise ModelOutputError("response_evidence must be an EvidenceRef")
        if (
            self.response_evidence.source_kind
            is not EvidenceSourceKind.MODEL_RESPONSE
        ):
            raise ModelOutputError(
                "response_evidence must reference a model response"
            )
