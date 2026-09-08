from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Mapping, Protocol
from uuid import UUID

from self_cognition.core.errors import ContractValidationError, ModelOutputError
from self_cognition.core.scopes import MindScope, SubjectKind, SubjectRef, SubjectScope

if TYPE_CHECKING:
    from self_cognition.core.events import EventEnvelope
    from self_cognition.core.evidence import EvidenceRef
    from self_cognition.core.plans import Plan, PlanStep
    from self_cognition.core.workspace import WorkspacePacket
    from self_cognition.runtime.run_context import RunContext


class ActionDecisionStatus(str, Enum):
    ALLOWED = "allowed"
    REJECTED = "rejected"
    DELAYED = "delayed"
    CONFIRMATION_REQUIRED = "confirmation_required"


class ActionResultStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


@dataclass(frozen=True, slots=True)
class ExpectedSideEffect:
    effect_type: str
    description: str
    reversible: bool

    def __post_init__(self) -> None:
        _text(self.effect_type, "side effect type")
        _text(self.description, "side effect description")
        if type(self.reversible) is not bool:
            raise ContractValidationError("side effect reversible must be boolean")


@dataclass(frozen=True, slots=True)
class ToolDescriptor:
    tool_id: str
    name: str
    description: str
    input_schema: Mapping[str, object]
    output_schema: Mapping[str, object]
    expected_side_effects: tuple[ExpectedSideEffect, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.tool_id, "tool_id")
        _text(self.name, "tool name")
        _text(self.description, "tool description")
        _json_object(self.input_schema, "tool input schema")
        _json_object(self.output_schema, "tool output schema")
        _side_effects(self.expected_side_effects)


@dataclass(frozen=True, slots=True)
class ActionProposalRequest:
    request_id: UUID
    owner: SubjectScope
    plan_id: UUID
    plan_version: int
    step_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, UUID) or not isinstance(self.plan_id, UUID):
            raise ContractValidationError("action proposal IDs must be UUID values")
        if (
            not isinstance(self.owner, SubjectScope)
            or self.owner.subject.kind is not SubjectKind.MIND
        ):
            raise ContractValidationError("action owner must be a MIND subject")
        if type(self.plan_version) is not int or self.plan_version < 1:
            raise ContractValidationError("action plan version must be positive")
        _identifier(self.step_id, "action step_id")


@dataclass(frozen=True, slots=True)
class ActionDraft:
    tool_id: str
    arguments: Mapping[str, object]
    expected_side_effects: tuple[ExpectedSideEffect, ...]

    def __post_init__(self) -> None:
        _identifier(self.tool_id, "action tool_id")
        _json_object(self.arguments, "action arguments")
        _side_effects(self.expected_side_effects)


@dataclass(frozen=True, slots=True)
class ActionRequest:
    action_id: UUID
    proposal_request_id: UUID
    owner: SubjectScope
    plan_id: UUID
    plan_version: int
    step_id: str
    tool_id: str
    arguments: Mapping[str, object]
    expected_side_effects: tuple[ExpectedSideEffect, ...]
    idempotency_key: str
    requested_at: datetime

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, UUID)
            for value in (self.action_id, self.proposal_request_id, self.plan_id)
        ):
            raise ContractValidationError("action request IDs must be UUID values")
        if (
            not isinstance(self.owner, SubjectScope)
            or self.owner.subject.kind is not SubjectKind.MIND
        ):
            raise ContractValidationError("action request owner must be a MIND subject")
        if type(self.plan_version) is not int or self.plan_version < 1:
            raise ContractValidationError(
                "action request plan version must be positive"
            )
        _identifier(self.step_id, "action request step_id")
        _identifier(self.tool_id, "action request tool_id")
        _json_object(self.arguments, "action request arguments")
        _side_effects(self.expected_side_effects)
        _text(self.idempotency_key, "action idempotency key")
        _aware(self.requested_at, "action requested_at")


@dataclass(frozen=True, slots=True)
class ActionDecisionDraft:
    status: ActionDecisionStatus
    reason: str
    value_basis: tuple[str, ...]
    relationship_context: str
    risks: tuple[str, ...]
    evidence_ids: tuple[UUID, ...]
    valid_until: datetime
    not_before: datetime | None = None
    confirmation_prompt: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, ActionDecisionStatus):
            raise ContractValidationError("action decision status is invalid")
        _text(self.reason, "action decision reason")
        _texts(self.value_basis, "action value basis", required=True)
        _text(self.relationship_context, "action relationship context")
        _texts(self.risks, "action risks", required=True)
        _uuids(self.evidence_ids, "action evidence IDs")
        _aware(self.valid_until, "action decision valid_until")
        if self.not_before is not None:
            _aware(self.not_before, "action decision not_before")
        if self.status is ActionDecisionStatus.DELAYED:
            if self.not_before is None or self.not_before >= self.valid_until:
                raise ContractValidationError(
                    "delayed action requires not_before before valid_until"
                )
        elif self.not_before is not None:
            raise ContractValidationError("only delayed actions may set not_before")
        if self.status is ActionDecisionStatus.CONFIRMATION_REQUIRED:
            _text(self.confirmation_prompt, "action confirmation prompt")
        elif self.confirmation_prompt is not None:
            raise ContractValidationError(
                "only confirmation decisions may set a prompt"
            )


@dataclass(frozen=True, slots=True)
class ActionDecision:
    decision_id: UUID
    action_id: UUID
    status: ActionDecisionStatus
    reason: str
    value_basis: tuple[str, ...]
    relationship_context: str
    risks: tuple[str, ...]
    evidence_ids: tuple[UUID, ...]
    decided_at: datetime
    valid_until: datetime
    one_time_scope: UUID
    not_before: datetime | None = None
    confirmation_prompt: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.decision_id, UUID) or not isinstance(
            self.action_id, UUID
        ):
            raise ContractValidationError("action decision IDs must be UUID values")
        if self.one_time_scope != self.action_id:
            raise ContractValidationError("action decision scope must be its action ID")
        ActionDecisionDraft(
            self.status,
            self.reason,
            self.value_basis,
            self.relationship_context,
            self.risks,
            self.evidence_ids,
            self.valid_until,
            self.not_before,
            self.confirmation_prompt,
        )
        _aware(self.decided_at, "action decided_at")
        if self.valid_until <= self.decided_at:
            raise ContractValidationError("action decision must expire in the future")


@dataclass(frozen=True, slots=True)
class ActionResult:
    result_id: UUID
    action_id: UUID
    owner: SubjectScope
    status: ActionResultStatus
    summary: str
    output: object | None
    actual_side_effects: tuple[ExpectedSideEffect, ...]
    recorded_at: datetime
    error_type: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.result_id, UUID) or not isinstance(
            self.action_id, UUID
        ):
            raise ContractValidationError("action result IDs must be UUID values")
        if (
            not isinstance(self.owner, SubjectScope)
            or self.owner.subject.kind is not SubjectKind.MIND
        ):
            raise ContractValidationError("action result owner must be a MIND subject")
        if not isinstance(self.status, ActionResultStatus):
            raise ContractValidationError("action result status is invalid")
        _text(self.summary, "action result summary")
        _json_value(self.output, "action result output")
        _side_effects(self.actual_side_effects)
        _aware(self.recorded_at, "action result recorded_at")
        if self.status in {ActionResultStatus.FAILED, ActionResultStatus.TIMED_OUT}:
            _text(self.error_type, "action result error_type")
        elif self.error_type is not None:
            raise ContractValidationError(
                "only failed or timed out actions may set error_type"
            )


@dataclass(frozen=True, slots=True)
class ActionModelOutput:
    model: str
    response_id: str
    raw_output: str
    error_type: str | None = None

    def __post_init__(self) -> None:
        _text(self.model, "action model")
        _text(self.response_id, "action response ID")
        if not isinstance(self.raw_output, str):
            raise ContractValidationError("action raw output must be text")


class ActionModel(Protocol):
    def propose(
        self,
        plan: Plan,
        step: PlanStep,
        workspace: WorkspacePacket,
        tools: tuple[ToolDescriptor, ...],
        context: RunContext,
    ) -> ActionModelOutput: ...

    def decide(
        self,
        request: ActionRequest,
        workspace: WorkspacePacket,
        context: RunContext,
    ) -> ActionModelOutput: ...


@dataclass(frozen=True, slots=True)
class ActionContextPayload:
    request: ActionProposalRequest
    workspace_json: str
    evidence_refs: tuple[EvidenceRef, ...]
    tools: tuple[ToolDescriptor, ...]


@dataclass(frozen=True, slots=True)
class ActionProposedPayload:
    request: ActionRequest


@dataclass(frozen=True, slots=True)
class ActionDecisionPayload:
    request: ActionRequest
    decision: ActionDecision


@dataclass(frozen=True, slots=True)
class ActionFailurePayload:
    proposal_request_id: UUID
    stage: str
    error_type: str

    def __post_init__(self) -> None:
        if not isinstance(self.proposal_request_id, UUID):
            raise ContractValidationError("action proposal request ID must be a UUID")
        _text(self.stage, "action failure stage")
        _text(self.error_type, "action failure error_type")


@dataclass(frozen=True, slots=True)
class ActionResultPayload:
    result: ActionResult


ActionEventPayload = (
    ActionContextPayload
    | ActionProposedPayload
    | ActionDecisionPayload
    | ActionFailurePayload
    | ActionResultPayload
)


def action_draft_from_json(raw_output: str) -> ActionDraft:
    try:
        values = _mapping(
            json.loads(raw_output),
            {"tool_id", "arguments", "expected_side_effects"},
            "action draft",
        )
        effects = _effects_from_value(values["expected_side_effects"])
        return ActionDraft(
            _string(values["tool_id"], "tool_id"),
            _json_object(values["arguments"], "arguments"),
            effects,
        )
    except (
        json.JSONDecodeError,
        TypeError,
        ValueError,
        ContractValidationError,
    ) as error:
        raise ModelOutputError("invalid structured action proposal") from error


def decision_draft_from_json(raw_output: str) -> ActionDecisionDraft:
    try:
        values = _mapping(
            json.loads(raw_output),
            {
                "status",
                "reason",
                "value_basis",
                "relationship_context",
                "risks",
                "evidence_ids",
                "valid_until",
                "not_before",
                "confirmation_prompt",
            },
            "action decision",
        )
        return ActionDecisionDraft(
            ActionDecisionStatus(_string(values["status"], "status")),
            _string(values["reason"], "reason"),
            _string_tuple(values["value_basis"], "value_basis"),
            _string(values["relationship_context"], "relationship_context"),
            _string_tuple(values["risks"], "risks"),
            _uuid_tuple(values["evidence_ids"], "evidence_ids"),
            datetime.fromisoformat(
                _string(values["valid_until"], "valid_until")
            ),
            _optional_datetime(values["not_before"], "not_before"),
            _optional_string(values["confirmation_prompt"], "confirmation_prompt"),
        )
    except (
        json.JSONDecodeError,
        TypeError,
        ValueError,
        ContractValidationError,
    ) as error:
        raise ModelOutputError("invalid structured action decision") from error


def action_draft_to_dict(draft: ActionDraft) -> dict[str, object]:
    return {
        "tool_id": draft.tool_id,
        "arguments": dict(draft.arguments),
        "expected_side_effects": [
            side_effect_to_dict(effect) for effect in draft.expected_side_effects
        ],
    }


def decision_draft_to_dict(draft: ActionDecisionDraft) -> dict[str, object]:
    return {
        "status": draft.status.value,
        "reason": draft.reason,
        "value_basis": list(draft.value_basis),
        "relationship_context": draft.relationship_context,
        "risks": list(draft.risks),
        "evidence_ids": [str(value) for value in draft.evidence_ids],
        "valid_until": draft.valid_until.isoformat(),
        "not_before": (
            draft.not_before.isoformat() if draft.not_before is not None else None
        ),
        "confirmation_prompt": draft.confirmation_prompt,
    }


def proposal_request_to_dict(request: ActionProposalRequest) -> dict[str, object]:
    return {
        "request_id": str(request.request_id),
        "owner": _subject_scope_to_dict(request.owner),
        "plan_id": str(request.plan_id),
        "plan_version": request.plan_version,
        "step_id": request.step_id,
    }


def proposal_request_from_dict(value: object) -> ActionProposalRequest:
    values = _mapping(
        value,
        {"request_id", "owner", "plan_id", "plan_version", "step_id"},
        "action proposal request",
    )
    return ActionProposalRequest(
        UUID(_string(values["request_id"], "request_id")),
        _subject_scope_from_dict(values["owner"]),
        UUID(_string(values["plan_id"], "plan_id")),
        _integer(values["plan_version"], "plan_version"),
        _string(values["step_id"], "step_id"),
    )


def action_request_to_dict(request: ActionRequest) -> dict[str, object]:
    return {
        "action_id": str(request.action_id),
        "proposal_request_id": str(request.proposal_request_id),
        "owner": _subject_scope_to_dict(request.owner),
        "plan_id": str(request.plan_id),
        "plan_version": request.plan_version,
        "step_id": request.step_id,
        "tool_id": request.tool_id,
        "arguments": dict(request.arguments),
        "expected_side_effects": [
            side_effect_to_dict(effect) for effect in request.expected_side_effects
        ],
        "idempotency_key": request.idempotency_key,
        "requested_at": request.requested_at.isoformat(),
    }


def action_request_from_dict(value: object) -> ActionRequest:
    values = _mapping(
        value,
        {
            "action_id",
            "proposal_request_id",
            "owner",
            "plan_id",
            "plan_version",
            "step_id",
            "tool_id",
            "arguments",
            "expected_side_effects",
            "idempotency_key",
            "requested_at",
        },
        "action request",
    )
    return ActionRequest(
        UUID(_string(values["action_id"], "action_id")),
        UUID(_string(values["proposal_request_id"], "proposal_request_id")),
        _subject_scope_from_dict(values["owner"]),
        UUID(_string(values["plan_id"], "plan_id")),
        _integer(values["plan_version"], "plan_version"),
        _string(values["step_id"], "step_id"),
        _string(values["tool_id"], "tool_id"),
        _json_object(values["arguments"], "arguments"),
        _effects_from_value(values["expected_side_effects"]),
        _string(values["idempotency_key"], "idempotency_key"),
        datetime.fromisoformat(_string(values["requested_at"], "requested_at")),
    )


def action_decision_to_dict(decision: ActionDecision) -> dict[str, object]:
    return {
        "decision_id": str(decision.decision_id),
        "action_id": str(decision.action_id),
        **decision_draft_to_dict(
            ActionDecisionDraft(
                decision.status,
                decision.reason,
                decision.value_basis,
                decision.relationship_context,
                decision.risks,
                decision.evidence_ids,
                decision.valid_until,
                decision.not_before,
                decision.confirmation_prompt,
            )
        ),
        "decided_at": decision.decided_at.isoformat(),
        "one_time_scope": str(decision.one_time_scope),
    }


def action_decision_from_dict(value: object) -> ActionDecision:
    values = _mapping(
        value,
        {
            "decision_id",
            "action_id",
            "status",
            "reason",
            "value_basis",
            "relationship_context",
            "risks",
            "evidence_ids",
            "valid_until",
            "not_before",
            "confirmation_prompt",
            "decided_at",
            "one_time_scope",
        },
        "action decision",
    )
    draft = decision_draft_from_json(
        json.dumps(
            {
                key: values[key]
                for key in (
                    "status",
                    "reason",
                    "value_basis",
                    "relationship_context",
                    "risks",
                    "evidence_ids",
                    "valid_until",
                    "not_before",
                    "confirmation_prompt",
                )
            }
        )
    )
    return ActionDecision(
        UUID(_string(values["decision_id"], "decision_id")),
        UUID(_string(values["action_id"], "action_id")),
        draft.status,
        draft.reason,
        draft.value_basis,
        draft.relationship_context,
        draft.risks,
        draft.evidence_ids,
        datetime.fromisoformat(_string(values["decided_at"], "decided_at")),
        draft.valid_until,
        UUID(_string(values["one_time_scope"], "one_time_scope")),
        draft.not_before,
        draft.confirmation_prompt,
    )


def action_result_to_dict(result: ActionResult) -> dict[str, object]:
    return {
        "result_id": str(result.result_id),
        "action_id": str(result.action_id),
        "owner": _subject_scope_to_dict(result.owner),
        "status": result.status.value,
        "summary": result.summary,
        "output": result.output,
        "actual_side_effects": [
            side_effect_to_dict(effect) for effect in result.actual_side_effects
        ],
        "recorded_at": result.recorded_at.isoformat(),
        "error_type": result.error_type,
    }


def action_result_from_dict(value: object) -> ActionResult:
    values = _mapping(
        value,
        {
            "result_id",
            "action_id",
            "owner",
            "status",
            "summary",
            "output",
            "actual_side_effects",
            "recorded_at",
            "error_type",
        },
        "action result",
    )
    return ActionResult(
        UUID(_string(values["result_id"], "result_id")),
        UUID(_string(values["action_id"], "action_id")),
        _subject_scope_from_dict(values["owner"]),
        ActionResultStatus(_string(values["status"], "status")),
        _string(values["summary"], "summary"),
        values["output"],
        _effects_from_value(values["actual_side_effects"]),
        datetime.fromisoformat(_string(values["recorded_at"], "recorded_at")),
        _optional_string(values["error_type"], "error_type"),
    )


def tool_descriptor_to_dict(descriptor: ToolDescriptor) -> dict[str, object]:
    return {
        "tool_id": descriptor.tool_id,
        "name": descriptor.name,
        "description": descriptor.description,
        "input_schema": dict(descriptor.input_schema),
        "output_schema": dict(descriptor.output_schema),
        "expected_side_effects": [
            side_effect_to_dict(effect)
            for effect in descriptor.expected_side_effects
        ],
    }


def tool_descriptor_from_dict(value: object) -> ToolDescriptor:
    values = _mapping(
        value,
        {
            "tool_id",
            "name",
            "description",
            "input_schema",
            "output_schema",
            "expected_side_effects",
        },
        "tool descriptor",
    )
    return ToolDescriptor(
        _string(values["tool_id"], "tool_id"),
        _string(values["name"], "name"),
        _string(values["description"], "description"),
        _json_object(values["input_schema"], "input_schema"),
        _json_object(values["output_schema"], "output_schema"),
        _effects_from_value(values["expected_side_effects"]),
    )


def side_effect_to_dict(effect: ExpectedSideEffect) -> dict[str, object]:
    return {
        "effect_type": effect.effect_type,
        "description": effect.description,
        "reversible": effect.reversible,
    }


def action_dependency_ids(event: EventEnvelope) -> frozenset[UUID]:
    if (
        not event.event_type.startswith("action.")
        and event.event_type != "model.response"
    ):
        return frozenset()
    ids = {event.causation_id} if event.causation_id is not None else set()
    if isinstance(event.payload, ActionContextPayload):
        ids.update(ref.evidence_id for ref in event.payload.evidence_refs)
    return frozenset(ids)


def _effects_from_value(value: object) -> tuple[ExpectedSideEffect, ...]:
    if not isinstance(value, list):
        raise ContractValidationError("side effects must be an array")
    effects = []
    for raw in value:
        values = _mapping(
            raw,
            {"effect_type", "description", "reversible"},
            "side effect",
        )
        reversible = values["reversible"]
        if type(reversible) is not bool:
            raise ContractValidationError("side effect reversible must be boolean")
        effects.append(
            ExpectedSideEffect(
                _string(values["effect_type"], "effect_type"),
                _string(values["description"], "description"),
                reversible,
            )
        )
    return tuple(effects)


def _mapping(value: object, keys: set[str], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ContractValidationError(f"{name} fields are invalid")
    return value


def _json_object(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ContractValidationError(f"{name} must be a JSON object")
    _json_value(dict(value), name)
    return value


def _json_value(value: object, name: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ContractValidationError(f"{name} must be valid JSON") from error


def _side_effects(values: tuple[ExpectedSideEffect, ...]) -> None:
    if not isinstance(values, tuple) or any(
        not isinstance(value, ExpectedSideEffect) for value in values
    ):
        raise ContractValidationError("side effects must be a tuple")
    if len({value.effect_type for value in values}) != len(values):
        raise ContractValidationError("side effect types must be unique")


def _identifier(value: str, name: str) -> None:
    _text(value, name)
    if any(character.isspace() for character in value):
        raise ContractValidationError(f"{name} must not contain whitespace")


def _text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{name} must not be blank")


def _texts(values: tuple[str, ...], name: str, *, required: bool = False) -> None:
    if not isinstance(values, tuple) or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise ContractValidationError(f"{name} must contain non-blank text")
    if required and not values:
        raise ContractValidationError(f"{name} must not be empty")


def _uuids(values: tuple[UUID, ...], name: str) -> None:
    if not isinstance(values, tuple) or any(
        not isinstance(value, UUID) for value in values
    ):
        raise ContractValidationError(f"{name} must contain UUID values")


def _aware(value: datetime, name: str) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ContractValidationError(f"{name} must include timezone information")


def _string(value: object, name: str) -> str:
    _text(value, name)
    assert isinstance(value, str)
    return value


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ContractValidationError(f"{name} must be a string array")
    return tuple(value)


def _uuid_tuple(value: object, name: str) -> tuple[UUID, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ContractValidationError(f"{name} must be a UUID string array")
    return tuple(UUID(item) for item in value)


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise ContractValidationError(f"{name} must be an integer")
    return value


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _string(value, name)


def _optional_datetime(value: object, name: str) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(_string(value, name))


def _subject_scope_to_dict(subject: SubjectScope) -> dict[str, object]:
    return {
        "mind_id": subject.mind.mind_id,
        "kind": subject.subject.kind.value,
        "subject_id": subject.subject.subject_id,
    }


def _subject_scope_from_dict(value: object) -> SubjectScope:
    values = _mapping(value, {"mind_id", "kind", "subject_id"}, "action owner")
    return SubjectScope(
        MindScope(_string(values["mind_id"], "mind_id")),
        SubjectRef(
            SubjectKind(_string(values["kind"], "kind")),
            _string(values["subject_id"], "subject_id"),
        ),
    )
