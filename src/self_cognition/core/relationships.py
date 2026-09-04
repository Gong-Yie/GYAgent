from dataclasses import dataclass
from enum import Enum
from uuid import UUID

from self_cognition.core.errors import ContractValidationError
from self_cognition.core.scopes import (
    ConversationScope,
    DataScope,
    DisclosureScope,
    MindScope,
    SubjectKind,
    SubjectRef,
    SubjectScope,
)


class RelationshipDisclosureDecision(str, Enum):
    UNDECIDED = "undecided"
    HONOR = "honor"
    OVERRIDE = "override"


@dataclass(frozen=True, slots=True)
class RelationshipState:
    """A directed, scoped relationship view owned by one subject."""

    source: SubjectScope
    target: SubjectScope
    relation: str
    context: str
    scope: DataScope
    shared_experience_ids: tuple[UUID, ...] = ()
    boundaries: tuple[str, ...] = ()
    commitments: tuple[str, ...] = ()
    disclosure_decision: RelationshipDisclosureDecision = (
        RelationshipDisclosureDecision.UNDECIDED
    )
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.source, SubjectScope) or not isinstance(
            self.target, SubjectScope
        ):
            raise ContractValidationError("relationship subjects are invalid")
        if self.source.mind != self.target.mind:
            raise ContractValidationError(
                "relationship subjects must belong to the same mind"
            )
        if self.scope.owner != self.source:
            raise ContractValidationError(
                "relationship scope owner must match source subject"
            )
        for name in ("relation", "context"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ContractValidationError(
                    f"relationship {name} must not be blank"
                )
        for name in ("shared_experience_ids",):
            values = getattr(self, name)
            if any(not isinstance(value, UUID) for value in values):
                raise ContractValidationError(
                    f"relationship {name} must contain UUID values"
                )
        for name in ("boundaries", "commitments"):
            values = getattr(self, name)
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ContractValidationError(
                    f"relationship {name} must contain text values"
                )
        if not isinstance(
            self.disclosure_decision,
            RelationshipDisclosureDecision,
        ):
            raise ContractValidationError("relationship disclosure decision is invalid")
        if not 0.0 <= self.confidence <= 1.0:
            raise ContractValidationError(
                "relationship confidence must be between zero and one"
            )

    def to_state_value(self) -> dict[str, object]:
        conversation = self.scope.conversation
        return {
            "kind": "relationship",
            "source": _subject_value(self.source),
            "target": _subject_value(self.target),
            "relation": self.relation,
            "context": self.context,
            "shared_experience_ids": [str(value) for value in self.shared_experience_ids],
            "boundaries": list(self.boundaries),
            "commitments": list(self.commitments),
            "disclosure": self.scope.disclosure.value,
            "conversation": (
                {
                    "conversation_id": conversation.conversation_id,
                    "group_id": conversation.group_id,
                }
                if conversation is not None
                else None
            ),
            "disclosure_decision": self.disclosure_decision.value,
            "confidence": self.confidence,
        }

    @classmethod
    def from_state_value(cls, value: object) -> "RelationshipState":
        if not isinstance(value, dict) or value.get("kind") != "relationship":
            raise ContractValidationError("relationship state value is invalid")
        source = _subject_from_value(value.get("source"))
        target = _subject_from_value(value.get("target"))
        raw_conversation = value.get("conversation")
        conversation = None
        if raw_conversation is not None:
            if not isinstance(raw_conversation, dict):
                raise ContractValidationError("relationship conversation is invalid")
            conversation = ConversationScope(
                str(raw_conversation.get("conversation_id", "")),
                raw_conversation.get("group_id"),
            )
        try:
            disclosure = DisclosureScope(str(value["disclosure"]))
            decision = RelationshipDisclosureDecision(
                str(value["disclosure_decision"])
            )
        except (KeyError, ValueError) as error:
            raise ContractValidationError("relationship disclosure is invalid") from error
        return cls(
            source=source,
            target=target,
            relation=str(value.get("relation", "")),
            context=str(value.get("context", "")),
            scope=DataScope(source, disclosure, conversation),
            shared_experience_ids=tuple(
                UUID(str(item)) for item in value.get("shared_experience_ids", ())
            ),
            boundaries=tuple(str(item) for item in value.get("boundaries", ())),
            commitments=tuple(str(item) for item in value.get("commitments", ())),
            disclosure_decision=decision,
            confidence=float(value.get("confidence", 0.0)),
        )


def _subject_value(subject: SubjectScope) -> dict[str, str]:
    return {
        "mind_id": subject.mind.mind_id,
        "kind": subject.subject.kind.value,
        "subject_id": subject.subject.subject_id,
    }


def _subject_from_value(value: object) -> SubjectScope:
    if not isinstance(value, dict):
        raise ContractValidationError("relationship subject value is invalid")
    try:
        kind = SubjectKind(str(value["kind"]))
        mind_id = str(value["mind_id"])
        subject_id = str(value["subject_id"])
    except (KeyError, ValueError) as error:
        raise ContractValidationError("relationship subject value is invalid") from error
    return SubjectScope(MindScope(mind_id), SubjectRef(kind, subject_id))
