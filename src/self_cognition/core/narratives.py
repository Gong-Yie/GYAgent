from dataclasses import dataclass
from enum import Enum
from uuid import UUID

from self_cognition.core.errors import ContractValidationError
from self_cognition.core.scopes import MindScope, SubjectKind, SubjectRef, SubjectScope


class NarrativeLayer(str, Enum):
    TASK = "task"
    PERIOD = "period"
    IDENTITY = "identity"
    RELATIONSHIP = "relationship"


@dataclass(frozen=True, slots=True)
class NarrativeRecord:
    """An evidence-backed explanation that can be superseded without rewriting facts."""

    narrative_id: str
    layer: NarrativeLayer
    subject: SubjectScope
    theme: str
    stage: str
    summary: str
    occurred_at: str
    evidence_event_ids: tuple[UUID, ...]
    unknowns: tuple[str, ...] = ()
    revision_of: str | None = None
    version: int = 1

    def __post_init__(self) -> None:
        for name in ("narrative_id", "theme", "stage", "summary", "occurred_at"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ContractValidationError(f"narrative {name} must not be blank")
        if not isinstance(self.layer, NarrativeLayer):
            raise ContractValidationError("narrative layer is invalid")
        if not isinstance(self.subject, SubjectScope):
            raise ContractValidationError("narrative subject is invalid")
        if not self.evidence_event_ids or any(
            not isinstance(value, UUID) for value in self.evidence_event_ids
        ):
            raise ContractValidationError(
                "narrative evidence_event_ids must contain UUID values"
            )
        if any(not isinstance(value, str) or not value.strip() for value in self.unknowns):
            raise ContractValidationError("narrative unknowns must contain text values")
        if self.revision_of is not None and not self.revision_of.strip():
            raise ContractValidationError("narrative revision_of must not be blank")
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 1:
            raise ContractValidationError("narrative version must be positive")

    def to_state_value(self) -> dict[str, object]:
        return {
            "kind": "narrative",
            "narrative_id": self.narrative_id,
            "layer": self.layer.value,
            "subject": {
                "mind_id": self.subject.mind.mind_id,
                "kind": self.subject.subject.kind.value,
                "subject_id": self.subject.subject.subject_id,
            },
            "theme": self.theme,
            "stage": self.stage,
            "summary": self.summary,
            "occurred_at": self.occurred_at,
            "evidence_event_ids": [str(value) for value in self.evidence_event_ids],
            "unknowns": list(self.unknowns),
            "revision_of": self.revision_of,
            "version": self.version,
        }

    @classmethod
    def from_state_value(cls, value: object) -> "NarrativeRecord":
        if not isinstance(value, dict) or value.get("kind") != "narrative":
            raise ContractValidationError("narrative state value is invalid")
        subject_value = value.get("subject")
        if not isinstance(subject_value, dict):
            raise ContractValidationError("narrative subject value is invalid")
        try:
            subject = SubjectScope(
                MindScope(str(subject_value["mind_id"])),
                SubjectRef(
                    SubjectKind(str(subject_value["kind"])),
                    str(subject_value["subject_id"]),
                ),
            )
            layer = NarrativeLayer(str(value["layer"]))
            evidence_event_ids = tuple(
                UUID(str(item)) for item in value["evidence_event_ids"]
            )
        except (KeyError, ValueError) as error:
            raise ContractValidationError("narrative state value is invalid") from error
        return cls(
            narrative_id=str(value.get("narrative_id", "")),
            layer=layer,
            subject=subject,
            theme=str(value.get("theme", "")),
            stage=str(value.get("stage", "")),
            summary=str(value.get("summary", "")),
            occurred_at=str(value.get("occurred_at", "")),
            evidence_event_ids=evidence_event_ids,
            unknowns=tuple(str(item) for item in value.get("unknowns", ())),
            revision_of=(
                None
                if value.get("revision_of") is None
                else str(value["revision_of"])
            ),
            version=int(value.get("version", 1)),
        )
