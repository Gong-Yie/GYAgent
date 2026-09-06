from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol
from uuid import UUID

from self_cognition.core.errors import ContractValidationError, ModelOutputError
from self_cognition.core.scopes import DisclosureScope, SubjectScope

if TYPE_CHECKING:
    from self_cognition.core.events import EventEnvelope
    from self_cognition.core.evidence import EvidenceRef
    from self_cognition.core.workspace import WorkspacePacket
    from self_cognition.runtime.run_context import RunContext


class ClaimStance(str, Enum):
    SUPPORTED = "supported"
    UNCERTAIN = "uncertain"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class DialogueRequest:
    event: EventEnvelope
    max_tokens: int = 1024
    max_items: int = 16

    def __post_init__(self) -> None:
        from self_cognition.core.events import EventEnvelope

        if (
            not isinstance(self.event, EventEnvelope)
            or self.event.event_type != "user.message"
        ):
            raise ContractValidationError("dialogue requires a user message")
        for value in (self.max_tokens, self.max_items):
            if type(value) is not int or value < 1:
                raise ContractValidationError(
                    "dialogue budgets must be positive integers"
                )


@dataclass(frozen=True, slots=True)
class DialogueClaim:
    text: str
    evidence_ids: tuple[UUID, ...]
    stance: ClaimStance

    def __post_init__(self) -> None:
        _text(self.text, "claim text")
        _ids(self.evidence_ids)
        if not isinstance(self.stance, ClaimStance):
            raise ContractValidationError("claim stance is invalid")
        if self.stance is ClaimStance.SUPPORTED and not self.evidence_ids:
            raise ContractValidationError("supported claims require evidence")


@dataclass(frozen=True, slots=True)
class DisclosureDecision:
    action: str
    scope: DisclosureScope
    reason: str
    evidence_ids: tuple[UUID, ...]
    overrides_intent: bool = False

    def __post_init__(self) -> None:
        if self.action not in {"disclose", "withhold"}:
            raise ContractValidationError("disclosure action is invalid")
        if not isinstance(self.scope, DisclosureScope):
            raise ContractValidationError("disclosure scope is invalid")
        _text(self.reason, "disclosure reason")
        _ids(self.evidence_ids)
        if type(self.overrides_intent) is not bool:
            raise ContractValidationError("overrides_intent must be boolean")


@dataclass(frozen=True, slots=True)
class DialogueDraft:
    text: str
    claims: tuple[DialogueClaim, ...]
    disclosure: DisclosureDecision

    def __post_init__(self) -> None:
        _text(self.text, "answer text")
        if not isinstance(self.claims, tuple) or any(
            not isinstance(claim, DialogueClaim) for claim in self.claims
        ):
            raise ContractValidationError("answer claims are invalid")
        if not isinstance(self.disclosure, DisclosureDecision):
            raise ContractValidationError("answer disclosure decision is invalid")


@dataclass(frozen=True, slots=True)
class GroundingReview:
    supported: bool
    reason: str

    def __post_init__(self) -> None:
        if type(self.supported) is not bool:
            raise ContractValidationError("grounding supported must be boolean")
        _text(self.reason, "grounding reason")


@dataclass(frozen=True, slots=True)
class DialogueModelOutput:
    model: str
    response_id: str
    raw_output: str
    error_type: str | None = None

    def __post_init__(self) -> None:
        _text(self.model, "model")
        _text(self.response_id, "response ID")
        if not isinstance(self.raw_output, str):
            raise ContractValidationError("raw output must be text")


class DialogueModel(Protocol):
    def generate(
        self, workspace: WorkspacePacket, context: RunContext
    ) -> DialogueModelOutput: ...

    def review(
        self, workspace: WorkspacePacket, draft: DialogueDraft, context: RunContext
    ) -> DialogueModelOutput: ...


@dataclass(frozen=True, slots=True)
class DialogueContextPayload:
    request_event_id: UUID
    recipient: SubjectScope
    workspace_json: str
    evidence_refs: tuple[EvidenceRef, ...]
    old_version: int
    new_version: int


@dataclass(frozen=True, slots=True)
class AssistantMessagePayload:
    request_event_id: UUID
    recipient: SubjectScope
    answer: DialogueDraft
    evidence_refs: tuple[EvidenceRef, ...]
    review: GroundingReview | None
    old_version: int
    new_version: int


@dataclass(frozen=True, slots=True)
class DialogueFailurePayload:
    request_event_id: UUID
    recipient: SubjectScope
    stage: str
    error_type: str
    old_version: int | None
    new_version: int | None


def draft_to_dict(draft: DialogueDraft) -> dict[str, object]:
    return {
        "text": draft.text,
        "claims": [
            {
                "text": claim.text,
                "evidence_ids": [str(value) for value in claim.evidence_ids],
                "stance": claim.stance.value,
            }
            for claim in draft.claims
        ],
        "disclosure": {
            "action": draft.disclosure.action,
            "scope": draft.disclosure.scope.value,
            "reason": draft.disclosure.reason,
            "evidence_ids": [str(value) for value in draft.disclosure.evidence_ids],
            "overrides_intent": draft.disclosure.overrides_intent,
        },
    }


def draft_from_dict(value: object) -> DialogueDraft:
    try:
        values = _object(value, {"text", "claims", "disclosure"})
        if not isinstance(values["claims"], list):
            raise ModelOutputError("claims must be an array")
        claims = []
        for raw in values["claims"]:
            claim = _object(raw, {"text", "evidence_ids", "stance"})
            claims.append(
                DialogueClaim(
                    claim["text"],
                    _uuid_array(claim["evidence_ids"]),
                    ClaimStance(claim["stance"]),
                )
            )
        disclosure = _object(
            values["disclosure"],
            {"action", "scope", "reason", "evidence_ids", "overrides_intent"},
        )
        return DialogueDraft(
            values["text"],
            tuple(claims),
            DisclosureDecision(
                disclosure["action"],
                DisclosureScope(disclosure["scope"]),
                disclosure["reason"],
                _uuid_array(disclosure["evidence_ids"]),
                disclosure["overrides_intent"],
            ),
        )
    except (ValueError, TypeError, ContractValidationError) as error:
        raise ModelOutputError("invalid structured dialogue output") from error


def review_from_dict(value: object) -> GroundingReview:
    values = _object(value, {"supported", "reason"})
    try:
        return GroundingReview(values["supported"], values["reason"])
    except ContractValidationError as error:
        raise ModelOutputError("invalid grounding output") from error


def parse_model_json(raw_output: str) -> object:
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError as error:
        raise ModelOutputError("model output is not valid JSON") from error


def _object(value: object, keys: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ModelOutputError("structured dialogue fields are invalid")
    return value


def _uuid_array(value: object) -> tuple[UUID, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ModelOutputError("evidence IDs must be a string array")
    return tuple(UUID(item) for item in value)


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{name} must not be blank")


def _ids(values: tuple[UUID, ...]) -> None:
    if not isinstance(values, tuple) or any(
        not isinstance(value, UUID) for value in values
    ):
        raise ContractValidationError("evidence IDs must be UUID values")


def dialogue_dependency_ids(event: EventEnvelope) -> frozenset[UUID]:
    if event.event_type not in {
        "dialogue.started",
        "assistant.message",
        "dialogue.failed",
        "model.response",
    }:
        return frozenset()
    ids = {event.causation_id} if event.causation_id is not None else set()
    if isinstance(event.payload, (DialogueContextPayload, AssistantMessagePayload)):
        ids.update(ref.evidence_id for ref in event.payload.evidence_refs)
        ids.add(event.payload.request_event_id)
    return frozenset(ids)
