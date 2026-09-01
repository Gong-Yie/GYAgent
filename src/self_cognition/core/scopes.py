from dataclasses import dataclass
from enum import Enum

from self_cognition.core.errors import ContractValidationError, ScopeMismatchError


DEFAULT_MIND_ID = "default-mind"


class SubjectKind(str, Enum):
    MIND = "mind"
    USER = "user"
    GROUP = "group"


class DisclosureScope(str, Enum):
    PRIVATE = "private"
    CONVERSATION = "conversation"
    GROUP = "group"
    MIND = "mind"


@dataclass(frozen=True, slots=True)
class MindScope:
    mind_id: str

    def __post_init__(self) -> None:
        _require_non_blank(self.mind_id, "mind_id")


@dataclass(frozen=True, slots=True)
class SubjectRef:
    kind: SubjectKind
    subject_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SubjectKind):
            raise ContractValidationError("subject kind must be a SubjectKind")
        _require_non_blank(self.subject_id, "subject_id")


@dataclass(frozen=True, slots=True)
class SubjectScope:
    mind: MindScope
    subject: SubjectRef

    def __post_init__(self) -> None:
        if not isinstance(self.mind, MindScope):
            raise ContractValidationError("mind must be a MindScope")
        if not isinstance(self.subject, SubjectRef):
            raise ContractValidationError("subject must be a SubjectRef")
        if (
            self.subject.kind is SubjectKind.MIND
            and self.subject.subject_id != self.mind.mind_id
        ):
            raise ContractValidationError(
                "mind subject ID must match its mind scope"
            )

    @classmethod
    def legacy_user(cls, subject_id: str) -> "SubjectScope":
        return cls(
            mind=MindScope(DEFAULT_MIND_ID),
            subject=SubjectRef(SubjectKind.USER, subject_id),
        )


@dataclass(frozen=True, slots=True)
class ConversationScope:
    conversation_id: str
    group_id: str | None = None

    def __post_init__(self) -> None:
        _require_non_blank(self.conversation_id, "conversation_id")
        if self.group_id is not None:
            _require_non_blank(self.group_id, "group_id")


@dataclass(frozen=True, slots=True)
class DataScope:
    owner: SubjectScope
    disclosure: DisclosureScope
    conversation: ConversationScope | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.owner, SubjectScope):
            raise ContractValidationError("owner must be a SubjectScope")
        if not isinstance(self.disclosure, DisclosureScope):
            raise ContractValidationError(
                "disclosure must be a DisclosureScope"
            )
        if (
            self.disclosure is DisclosureScope.CONVERSATION
            and self.conversation is None
        ):
            raise ContractValidationError(
                "conversation disclosure requires a conversation scope"
            )
        if self.disclosure is DisclosureScope.GROUP and (
            self.conversation is None or self.conversation.group_id is None
        ):
            raise ContractValidationError(
                "group disclosure requires a group conversation scope"
            )

    def require_same_mind(self, requester: SubjectScope) -> None:
        if requester.mind != self.owner.mind:
            raise ScopeMismatchError("data belongs to a different mind")

    def matches_disclosure_intent(
        self,
        requester: SubjectScope,
        conversation: ConversationScope | None = None,
    ) -> bool:
        """Return whether the requester fits the stated disclosure intent.

        This is evidence for the mind's disclosure decision, not an access-control
        decision. A complete mind may deliberately override the recorded intent.
        """
        self.require_same_mind(requester)
        if self.disclosure is DisclosureScope.PRIVATE:
            return requester.subject == self.owner.subject
        if self.disclosure is DisclosureScope.CONVERSATION:
            return conversation == self.conversation
        if self.disclosure is DisclosureScope.GROUP:
            return (
                conversation is not None
                and self.conversation is not None
                and conversation.group_id == self.conversation.group_id
            )
        return requester.subject.kind is SubjectKind.MIND


def normalize_subject_scope(subject: SubjectScope | str) -> SubjectScope:
    if isinstance(subject, SubjectScope):
        return subject
    if isinstance(subject, str):
        return SubjectScope.legacy_user(subject)
    raise TypeError("subject must be a SubjectScope or legacy subject ID")


def _require_non_blank(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{name} must not be blank")
