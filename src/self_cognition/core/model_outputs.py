from dataclasses import dataclass

from self_cognition.core.errors import ModelOutputError


ALLOWED_OPERATIONS = frozenset({"set"})


@dataclass(frozen=True, slots=True)
class ContributionCandidate:
    target_field: str
    operation: str
    value: object
    confidence: float
    evidence_event_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.target_field.strip():
            raise ModelOutputError("candidate target_field must not be blank")
        if self.operation not in ALLOWED_OPERATIONS:
            raise ModelOutputError("candidate operation is not supported")
        if not 0.0 <= self.confidence <= 1.0:
            raise ModelOutputError("candidate confidence must be between 0 and 1")
        if not self.evidence_event_ids:
            raise ModelOutputError("candidate evidence_event_ids must not be empty")
        if any(not event_id.strip() for event_id in self.evidence_event_ids):
            raise ModelOutputError("candidate evidence_event_ids must not be blank")


@dataclass(frozen=True, slots=True)
class ModelExtractionResult:
    response_id: str
    candidates: tuple[ContributionCandidate, ...]

    def __post_init__(self) -> None:
        if not self.response_id.strip():
            raise ModelOutputError("model response_id must not be blank")
