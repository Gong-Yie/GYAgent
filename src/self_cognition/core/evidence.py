from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from uuid import UUID

from self_cognition.core.errors import ContractValidationError
from self_cognition.core.events import (
    EventEnvelope,
    ModelResponsePayload,
    UserMessagePayload,
)
from self_cognition.core.scopes import (
    DataScope,
    DisclosureScope,
    SubjectScope,
    normalize_subject_scope,
)


class EvidenceSourceKind(str, Enum):
    EVENT = "event"
    FILE_FRAGMENT = "file_fragment"
    TOOL_RESULT = "tool_result"
    MODEL_RESPONSE = "model_response"
    SYSTEM_PRIOR = "system_prior"


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    evidence_id: UUID
    source_kind: EvidenceSourceKind
    source_ref: str
    scope: DataScope
    locator: str | None = None
    excerpt: str | None = None
    observed_at: datetime | None = None
    reliability: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_id, UUID):
            raise ContractValidationError("evidence_id must be a UUID")
        if not isinstance(self.source_kind, EvidenceSourceKind):
            raise ContractValidationError(
                "source_kind must be an EvidenceSourceKind"
            )
        if not isinstance(self.source_ref, str) or not self.source_ref.strip():
            raise ContractValidationError("source_ref must not be blank")
        if not isinstance(self.scope, DataScope):
            raise ContractValidationError("scope must be a DataScope")
        _require_optional_text(self.locator, "locator")
        _require_optional_text(self.excerpt, "excerpt")
        if self.observed_at is not None and (
            not isinstance(self.observed_at, datetime)
            or self.observed_at.tzinfo is None
            or self.observed_at.utcoffset() is None
        ):
            raise ContractValidationError(
                "observed_at must include timezone information"
            )
        if self.reliability is not None and not 0.0 <= self.reliability <= 1.0:
            raise ContractValidationError(
                "reliability must be between 0 and 1"
            )

    @classmethod
    def for_event(cls, event: EventEnvelope) -> "EvidenceRef":
        if isinstance(event.payload, ModelResponsePayload):
            return cls(
                evidence_id=event.event_id,
                source_kind=EvidenceSourceKind.MODEL_RESPONSE,
                source_ref=event.payload.response_id,
                scope=event.scope,
                locator="payload.raw_output",
                observed_at=event.occurred_at,
                reliability=1.0,
            )
        return cls(
            evidence_id=event.event_id,
            source_kind=EvidenceSourceKind.EVENT,
            source_ref=str(event.event_id),
            scope=event.scope,
            locator=(
                "payload.text"
                if isinstance(event.payload, UserMessagePayload)
                else None
            ),
            excerpt=(
                event.payload.text
                if isinstance(event.payload, UserMessagePayload)
                else None
            ),
            observed_at=event.occurred_at,
            reliability=1.0,
        )

    @classmethod
    def for_event_id(
        cls,
        event_id: UUID,
        subject: SubjectScope | str,
    ) -> "EvidenceRef":
        subject_scope = normalize_subject_scope(subject)
        return cls(
            evidence_id=event_id,
            source_kind=EvidenceSourceKind.EVENT,
            source_ref=str(event_id),
            scope=DataScope(
                owner=subject_scope,
                disclosure=DisclosureScope.PRIVATE,
            ),
        )


def _require_optional_text(value: str | None, name: str) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ContractValidationError(f"{name} must not be blank")
