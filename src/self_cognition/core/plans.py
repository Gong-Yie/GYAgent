from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Mapping, Protocol
from uuid import UUID

from self_cognition.core.errors import ContractValidationError, ModelOutputError
from self_cognition.core.identity import GoalPriority, GoalRecord, GoalStatus
from self_cognition.core.scopes import SubjectKind, SubjectRef, SubjectScope

if TYPE_CHECKING:
    from self_cognition.core.events import EventEnvelope
    from self_cognition.core.evidence import EvidenceRef
    from self_cognition.core.identity import CapabilityRecord
    from self_cognition.core.workspace import WorkspacePacket
    from self_cognition.runtime.run_context import RunContext


class PlanStepResultStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


class PlanStepStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    BLOCKED = "blocked"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


class PlanProgressStatus(str, Enum):
    ACTIVE = "active"
    BLOCKED = "blocked"
    READY_TO_COMPLETE = "ready_to_complete"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class PlanBudget:
    max_steps: int = 16
    max_tool_steps: int = 8

    def __post_init__(self) -> None:
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ContractValidationError("max_steps must be a positive integer")
        if type(self.max_tool_steps) is not int or self.max_tool_steps < 0:
            raise ContractValidationError(
                "max_tool_steps must be a non-negative integer"
            )
        if self.max_tool_steps > self.max_steps:
            raise ContractValidationError("max_tool_steps cannot exceed max_steps")


@dataclass(frozen=True, slots=True)
class PlanStep:
    step_id: str
    description: str
    dependencies: tuple[str, ...] = ()
    required_tool_ids: tuple[str, ...] = ()
    completion_conditions: tuple[str, ...] = ()
    checkpoint: bool = True
    cancellable: bool = True

    def __post_init__(self) -> None:
        _identifier(self.step_id, "step_id")
        _text(self.description, "step description")
        for values, name in (
            (self.dependencies, "dependencies"),
            (self.required_tool_ids, "required_tool_ids"),
            (self.completion_conditions, "completion_conditions"),
        ):
            _text_tuple(values, name)
            if len(set(values)) != len(values):
                raise ContractValidationError(f"{name} must not contain duplicates")
        if self.step_id in self.dependencies:
            raise ContractValidationError("a plan step cannot depend on itself")
        if type(self.checkpoint) is not bool or type(self.cancellable) is not bool:
            raise ContractValidationError(
                "checkpoint and cancellable must be boolean"
            )


@dataclass(frozen=True, slots=True)
class PlanDraft:
    steps: tuple[PlanStep, ...]

    def __post_init__(self) -> None:
        if not self.steps or any(not isinstance(step, PlanStep) for step in self.steps):
            raise ContractValidationError("plan draft requires plan steps")


@dataclass(frozen=True, slots=True)
class Plan:
    plan_id: UUID
    goal_id: str
    version: int
    budget: PlanBudget
    steps: tuple[PlanStep, ...]
    created_at: datetime
    previous_version: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.plan_id, UUID):
            raise ContractValidationError("plan_id must be a UUID")
        _identifier(self.goal_id, "goal_id")
        if type(self.version) is not int or self.version < 1:
            raise ContractValidationError("plan version must be positive")
        if self.previous_version != (None if self.version == 1 else self.version - 1):
            raise ContractValidationError("previous plan version is invalid")
        if not isinstance(self.budget, PlanBudget):
            raise ContractValidationError("plan budget is invalid")
        if not self.steps or any(not isinstance(step, PlanStep) for step in self.steps):
            raise ContractValidationError("plan requires plan steps")
        if len(self.steps) > self.budget.max_steps:
            raise ContractValidationError("plan exceeds its step budget")
        tool_steps = sum(bool(step.required_tool_ids) for step in self.steps)
        if tool_steps > self.budget.max_tool_steps:
            raise ContractValidationError("plan exceeds its tool-step budget")
        if not any(step.checkpoint for step in self.steps):
            raise ContractValidationError("plan requires a checkpoint")
        if not any(step.cancellable for step in self.steps):
            raise ContractValidationError("plan requires a cancellation point")
        _validate_graph(self.steps)
        _aware(self.created_at, "plan created_at")


@dataclass(frozen=True, slots=True)
class GoalPlanningRequest:
    request_id: UUID
    owner: SubjectScope
    requested_by: SubjectScope | None
    goal_id: str
    description: str
    priority: GoalPriority
    completion_conditions: tuple[str, ...]
    budget: PlanBudget = PlanBudget()

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, UUID):
            raise ContractValidationError("request_id must be a UUID")
        if (
            not isinstance(self.owner, SubjectScope)
            or self.owner.subject.kind is not SubjectKind.MIND
        ):
            raise ContractValidationError("goal owner must be a MIND subject")
        if self.requested_by is not None and (
            not isinstance(self.requested_by, SubjectScope)
            or self.requested_by.subject.kind is not SubjectKind.USER
            or self.requested_by.mind != self.owner.mind
        ):
            raise ContractValidationError(
                "goal requester must be a user in the same mind"
            )
        _identifier(self.goal_id, "goal_id")
        _text(self.description, "goal description")
        if not isinstance(self.priority, GoalPriority):
            raise ContractValidationError("goal priority is invalid")
        _text_tuple(self.completion_conditions, "goal completion conditions")
        if not self.completion_conditions:
            raise ContractValidationError("goal requires completion conditions")
        if len(set(self.completion_conditions)) != len(self.completion_conditions):
            raise ContractValidationError(
                "goal completion conditions must not contain duplicates"
            )
        if not isinstance(self.budget, PlanBudget):
            raise ContractValidationError("goal plan budget is invalid")

    @property
    def goal(self) -> GoalRecord:
        source = (
            "system"
            if self.requested_by is None
            else f"user:{self.requested_by.subject.subject_id}"
        )
        return GoalRecord(
            self.goal_id,
            self.description,
            source,
            self.priority,
            GoalStatus.ACTIVE,
            self.completion_conditions,
        )


@dataclass(frozen=True, slots=True)
class PlanStepResult:
    result_id: UUID
    plan_id: UUID
    plan_version: int
    step_id: str
    status: PlanStepResultStatus
    summary: str
    recorded_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.result_id, UUID) or not isinstance(self.plan_id, UUID):
            raise ContractValidationError("plan result IDs must be UUID values")
        if type(self.plan_version) is not int or self.plan_version < 1:
            raise ContractValidationError("result plan version must be positive")
        _identifier(self.step_id, "result step_id")
        if not isinstance(self.status, PlanStepResultStatus):
            raise ContractValidationError("plan step result status is invalid")
        _text(self.summary, "plan step result summary")
        _aware(self.recorded_at, "plan result recorded_at")


@dataclass(frozen=True, slots=True)
class PlanStepProgress:
    step: PlanStep
    status: PlanStepStatus
    result: PlanStepResult | None = None


@dataclass(frozen=True, slots=True)
class PlanProgress:
    plan: Plan
    status: PlanProgressStatus
    steps: tuple[PlanStepProgress, ...]
    satisfied_conditions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PlanningModelOutput:
    model: str
    response_id: str
    raw_output: str
    error_type: str | None = None

    def __post_init__(self) -> None:
        _text(self.model, "planning model")
        _text(self.response_id, "planning response ID")
        if not isinstance(self.raw_output, str):
            raise ContractValidationError("planning raw output must be text")


class PlanningModel(Protocol):
    def create(
        self,
        goal: GoalRecord,
        budget: PlanBudget,
        workspace: WorkspacePacket,
        capabilities: tuple[CapabilityRecord, ...],
        context: RunContext,
    ) -> PlanningModelOutput: ...

    def replan(
        self,
        goal: GoalRecord,
        plan: Plan,
        progress: PlanProgress,
        workspace: WorkspacePacket,
        capabilities: tuple[CapabilityRecord, ...],
        context: RunContext,
    ) -> PlanningModelOutput: ...


@dataclass(frozen=True, slots=True)
class GoalRequestedPayload:
    goal: GoalRecord
    budget: PlanBudget
    requested_by: SubjectRef | None


@dataclass(frozen=True, slots=True)
class PlanningContextPayload:
    request_event_id: UUID
    goal: GoalRecord
    budget: PlanBudget
    workspace_json: str
    evidence_refs: tuple[EvidenceRef, ...]
    state_version: int


@dataclass(frozen=True, slots=True)
class GoalPlannedPayload:
    request_event_id: UUID
    goal: GoalRecord
    plan: Plan


@dataclass(frozen=True, slots=True)
class PlanRevisedPayload:
    request_event_id: UUID
    plan: Plan
    failed_result_id: UUID


@dataclass(frozen=True, slots=True)
class PlanStepResultPayload:
    result: PlanStepResult


@dataclass(frozen=True, slots=True)
class GoalStatusChangedPayload:
    plan_id: UUID
    previous_status: GoalStatus
    goal: GoalRecord
    changed_by: SubjectRef | None
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.plan_id, UUID):
            raise ContractValidationError("status change plan_id must be a UUID")
        if not isinstance(self.previous_status, GoalStatus):
            raise ContractValidationError("previous goal status is invalid")
        if not isinstance(self.goal, GoalRecord):
            raise ContractValidationError("changed goal is invalid")
        if self.goal.status is self.previous_status:
            raise ContractValidationError("goal status change must change status")
        if self.changed_by is not None and self.changed_by.kind is not SubjectKind.USER:
            raise ContractValidationError("goal status actor must be a user")
        _text(self.reason, "goal status reason")


@dataclass(frozen=True, slots=True)
class PlanningFailurePayload:
    request_event_id: UUID
    goal_id: str
    stage: str
    error_type: str

    def __post_init__(self) -> None:
        if not isinstance(self.request_event_id, UUID):
            raise ContractValidationError("planning request ID must be a UUID")
        _identifier(self.goal_id, "planning failure goal_id")
        _text(self.stage, "planning failure stage")
        _text(self.error_type, "planning failure error_type")


def plan_draft_from_json(raw_output: str) -> PlanDraft:
    try:
        value = json.loads(raw_output)
    except json.JSONDecodeError as error:
        raise ModelOutputError("planning output is not valid JSON") from error
    try:
        values = _mapping(value, {"steps"}, "plan draft")
        raw_steps = values["steps"]
        if not isinstance(raw_steps, list):
            raise ContractValidationError("plan steps must be an array")
        return PlanDraft(tuple(_step_from_value(step) for step in raw_steps))
    except (TypeError, ValueError, ContractValidationError) as error:
        raise ModelOutputError("invalid structured planning output") from error


def plan_draft_to_dict(draft: PlanDraft) -> dict[str, object]:
    return {"steps": [_step_to_value(step) for step in draft.steps]}


def plan_to_dict(plan: Plan) -> dict[str, object]:
    return {
        "plan_id": str(plan.plan_id),
        "goal_id": plan.goal_id,
        "version": plan.version,
        "budget": budget_to_dict(plan.budget),
        "steps": [_step_to_value(step) for step in plan.steps],
        "created_at": plan.created_at.isoformat(),
        "previous_version": plan.previous_version,
    }


def plan_from_dict(value: object) -> Plan:
    values = _mapping(
        value,
        {
            "plan_id",
            "goal_id",
            "version",
            "budget",
            "steps",
            "created_at",
            "previous_version",
        },
        "plan",
    )
    raw_steps = values["steps"]
    if not isinstance(raw_steps, list):
        raise ContractValidationError("plan steps must be an array")
    try:
        created_at = datetime.fromisoformat(_string(values["created_at"], "created_at"))
        return Plan(
            UUID(_string(values["plan_id"], "plan_id")),
            _string(values["goal_id"], "goal_id"),
            _integer(values["version"], "version"),
            budget_from_dict(values["budget"]),
            tuple(_step_from_value(step) for step in raw_steps),
            created_at,
            _optional_integer(values["previous_version"], "previous_version"),
        )
    except (ValueError, TypeError) as error:
        raise ContractValidationError("invalid plan value") from error


def step_result_to_dict(result: PlanStepResult) -> dict[str, object]:
    return {
        "result_id": str(result.result_id),
        "plan_id": str(result.plan_id),
        "plan_version": result.plan_version,
        "step_id": result.step_id,
        "status": result.status.value,
        "summary": result.summary,
        "recorded_at": result.recorded_at.isoformat(),
    }


def step_result_from_dict(value: object) -> PlanStepResult:
    values = _mapping(
        value,
        {
            "result_id",
            "plan_id",
            "plan_version",
            "step_id",
            "status",
            "summary",
            "recorded_at",
        },
        "plan step result",
    )
    try:
        return PlanStepResult(
            UUID(_string(values["result_id"], "result_id")),
            UUID(_string(values["plan_id"], "plan_id")),
            _integer(values["plan_version"], "plan_version"),
            _string(values["step_id"], "step_id"),
            PlanStepResultStatus(_string(values["status"], "status")),
            _string(values["summary"], "summary"),
            datetime.fromisoformat(_string(values["recorded_at"], "recorded_at")),
        )
    except (ValueError, TypeError) as error:
        raise ContractValidationError("invalid plan step result") from error


def budget_to_dict(budget: PlanBudget) -> dict[str, object]:
    return {
        "max_steps": budget.max_steps,
        "max_tool_steps": budget.max_tool_steps,
    }


def budget_from_dict(value: object) -> PlanBudget:
    values = _mapping(value, {"max_steps", "max_tool_steps"}, "plan budget")
    return PlanBudget(
        _integer(values["max_steps"], "max_steps"),
        _integer(values["max_tool_steps"], "max_tool_steps"),
    )


def planning_dependency_ids(event: EventEnvelope) -> frozenset[UUID]:
    planning_types = {
        "goal.requested",
        "planning.started",
        "goal.planned",
        "planning.failed",
        "plan.revised",
        "plan.step_result",
        "goal.status_changed",
    }
    if event.event_type not in planning_types and event.event_type != "model.response":
        return frozenset()
    ids = {event.causation_id} if event.causation_id is not None else set()
    payload = event.payload
    if isinstance(
        payload,
        (
            PlanningContextPayload,
            GoalPlannedPayload,
            PlanRevisedPayload,
            PlanningFailurePayload,
        ),
    ):
        ids.add(payload.request_event_id)
    if isinstance(payload, PlanningContextPayload):
        ids.update(ref.evidence_id for ref in payload.evidence_refs)
    if isinstance(payload, PlanRevisedPayload):
        ids.add(payload.failed_result_id)
    return frozenset(value for value in ids if value is not None)


def _step_to_value(step: PlanStep) -> dict[str, object]:
    return {
        "step_id": step.step_id,
        "description": step.description,
        "dependencies": list(step.dependencies),
        "required_tool_ids": list(step.required_tool_ids),
        "completion_conditions": list(step.completion_conditions),
        "checkpoint": step.checkpoint,
        "cancellable": step.cancellable,
    }


def _step_from_value(value: object) -> PlanStep:
    values = _mapping(
        value,
        {
            "step_id",
            "description",
            "dependencies",
            "required_tool_ids",
            "completion_conditions",
            "checkpoint",
            "cancellable",
        },
        "plan step",
    )
    return PlanStep(
        _string(values["step_id"], "step_id"),
        _string(values["description"], "description"),
        _string_tuple(values["dependencies"], "dependencies"),
        _string_tuple(values["required_tool_ids"], "required_tool_ids"),
        _string_tuple(values["completion_conditions"], "completion_conditions"),
        _boolean(values["checkpoint"], "checkpoint"),
        _boolean(values["cancellable"], "cancellable"),
    )


def _validate_graph(steps: tuple[PlanStep, ...]) -> None:
    by_id = {step.step_id: step for step in steps}
    if len(by_id) != len(steps):
        raise ContractValidationError("plan step IDs must be unique")
    if any(dependency not in by_id for step in steps for dependency in step.dependencies):
        raise ContractValidationError("plan contains an unknown dependency")
    remaining = set(by_id)
    resolved: set[str] = set()
    while remaining:
        ready = {
            step_id
            for step_id in remaining
            if set(by_id[step_id].dependencies).issubset(resolved)
        }
        if not ready:
            raise ContractValidationError("plan contains a dependency cycle")
        resolved.update(ready)
        remaining.difference_update(ready)


def _mapping(value: object, keys: set[str], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ContractValidationError(f"{name} fields are invalid")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{name} must be text")
    return value


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ContractValidationError(f"{name} must be an array of text")
    return tuple(value)


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise ContractValidationError(f"{name} must be an integer")
    return value


def _optional_integer(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _integer(value, name)


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ContractValidationError(f"{name} must be boolean")
    return value


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{name} must not be blank")


def _identifier(value: str, name: str) -> None:
    _text(value, name)
    if any(character.isspace() for character in value):
        raise ContractValidationError(f"{name} must not contain whitespace")


def _text_tuple(values: tuple[str, ...], name: str) -> None:
    if not isinstance(values, tuple) or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise ContractValidationError(f"{name} must contain text")


def _aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ContractValidationError(f"{name} must include timezone information")
