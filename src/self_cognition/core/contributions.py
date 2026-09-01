from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from uuid import UUID

from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.errors import ContractValidationError
from self_cognition.core.events import EventEnvelope
from self_cognition.core.scopes import DataScope, SubjectScope


class ContributionOperation(str, Enum):
    SET = "set"


class CognitionType(str, Enum):
    FACT = "fact"
    INFERENCE = "inference"
    PREFERENCE = "preference"
    GOAL = "goal"
    AFFECT = "affect"
    LIMITATION = "limitation"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class CognitiveContribution:
    contribution_id: UUID
    target: SubjectScope
    target_field: str
    operation: ContributionOperation
    cognition_type: CognitionType
    value: object
    confidence: float
    evidence_refs: tuple[EvidenceRef, ...]
    source_module: str
    module_version: str
    scope: DataScope
    created_at: datetime
    valid_from: datetime
    expires_at: datetime | None = None
    target_version: int | None = None
    explicitly_confirmed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.contribution_id, UUID):
            raise ContractValidationError("contribution_id must be a UUID")
        if not isinstance(self.target, SubjectScope):
            raise ContractValidationError("target must be a SubjectScope")
        if not 0.0 <= self.confidence <= 1.0:
            raise ContractValidationError("confidence must be between 0 and 1")
        if not self.target_field.strip():
            raise ContractValidationError("target_field must not be blank")
        if not isinstance(self.operation, ContributionOperation):
            raise ContractValidationError(
                "operation must be a ContributionOperation"
            )
        if not isinstance(self.cognition_type, CognitionType):
            raise ContractValidationError("cognition_type must be a CognitionType")
        if not self.evidence_refs:
            raise ContractValidationError("evidence_refs must not be empty")
        if any(not isinstance(ref, EvidenceRef) for ref in self.evidence_refs):
            raise ContractValidationError(
                "evidence_refs must contain EvidenceRef values"
            )
        if not self.source_module.strip():
            raise ContractValidationError("source_module must not be blank")
        if not self.module_version.strip():
            raise ContractValidationError("module_version must not be blank")
        if not isinstance(self.scope, DataScope):
            raise ContractValidationError("scope must be a DataScope")
        if self.scope.owner != self.target:
            raise ContractValidationError(
                "contribution scope owner must match its target"
            )
        if any(
            evidence.scope.owner.mind != self.target.mind
            for evidence in self.evidence_refs
        ):
            raise ContractValidationError(
                "contribution evidence must belong to the target mind"
            )
        _require_aware(self.created_at, "created_at")
        _require_aware(self.valid_from, "valid_from")
        if self.expires_at is not None:
            _require_aware(self.expires_at, "expires_at")
            if self.expires_at <= self.valid_from:
                raise ContractValidationError(
                    "expires_at must be later than valid_from"
                )
        if self.target_version is not None and self.target_version < 0:
            raise ContractValidationError("target_version must not be negative")
        if not isinstance(self.explicitly_confirmed, bool):
            raise ContractValidationError(
                "explicitly_confirmed must be a boolean"
            )

    @classmethod
    def set_from_event(
        cls,
        event: EventEnvelope,
        *,
        contribution_id: UUID,
        target_field: str,
        cognition_type: CognitionType,
        value: object,
        confidence: float,
        evidence_refs: tuple[EvidenceRef, ...],
        source_module: str,
        module_version: str,
        expires_at: datetime | None = None,
        target_version: int | None = None,
        explicitly_confirmed: bool = False,
    ) -> "CognitiveContribution":
        return cls(
            contribution_id=contribution_id,
            target=event.subject,
            target_field=target_field,
            operation=ContributionOperation.SET,
            cognition_type=cognition_type,
            value=value,
            confidence=confidence,
            evidence_refs=evidence_refs,
            source_module=source_module,
            module_version=module_version,
            scope=event.scope,
            created_at=event.recorded_at,
            valid_from=event.occurred_at,
            expires_at=expires_at,
            target_version=target_version,
            explicitly_confirmed=explicitly_confirmed,
        )


Contribution = CognitiveContribution


def _require_aware(value: datetime, name: str) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ContractValidationError(
            f"{name} must include timezone information"
        )
