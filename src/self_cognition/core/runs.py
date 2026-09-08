from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from self_cognition.core.errors import ContractValidationError
from self_cognition.core.scopes import (
    MindScope,
    SubjectKind,
    SubjectRef,
    SubjectScope,
)


class RunKind(str, Enum):
    COGNITIVE_CYCLE = "cognitive_cycle"
    ACTION = "action"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    CHECKPOINTED = "checkpointed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    FAILED = "failed"
    INTERRUPTED = "interrupted"

    @property
    def is_terminal(self) -> bool:
        return self in {
            RunStatus.COMPLETED,
            RunStatus.CANCELLED,
            RunStatus.TIMED_OUT,
            RunStatus.FAILED,
            RunStatus.INTERRUPTED,
        }


@dataclass(frozen=True, slots=True)
class RunBudget:
    max_model_calls: int | None = None
    max_tool_calls: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("max_model_calls", self.max_model_calls),
            ("max_tool_calls", self.max_tool_calls),
        ):
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ContractValidationError(f"{name} must be non-negative")


@dataclass(frozen=True, slots=True)
class RunUsage:
    model_calls: int = 0
    tool_calls: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("model_calls", self.model_calls),
            ("tool_calls", self.tool_calls),
        ):
            if not isinstance(value, int) or value < 0:
                raise ContractValidationError(f"{name} must be non-negative")


@dataclass(frozen=True, slots=True)
class RunCheckpoint:
    checkpoint_id: UUID
    run_id: UUID
    position: str
    recorded_at: datetime
    action_id: UUID | None = None
    result_event_id: UUID | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint_id, UUID) or not isinstance(
            self.run_id, UUID
        ):
            raise ContractValidationError("checkpoint IDs must be UUID values")
        _require_non_blank(self.position, "checkpoint position")
        _require_aware(self.recorded_at, "checkpoint recorded_at")
        if self.result_event_id is not None and self.action_id is None:
            raise ContractValidationError(
                "result_event_id requires an action_id"
            )


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: UUID
    kind: RunKind
    subject: SubjectScope
    correlation_id: UUID
    started_at: datetime
    updated_at: datetime
    deadline: datetime
    status: RunStatus = RunStatus.PENDING
    parent_run_id: UUID | None = None
    budget: RunBudget = RunBudget()
    usage: RunUsage = RunUsage()
    checkpoint: RunCheckpoint | None = None
    input_event_ids: tuple[UUID, ...] = ()
    state_version: int | None = None
    motive_result: str | None = None
    termination_reason: str | None = None
    error_type: str | None = None
    cancel_requested: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("run_id", self.run_id),
            ("correlation_id", self.correlation_id),
        ):
            if not isinstance(value, UUID):
                raise ContractValidationError(f"{name} must be a UUID")
        if self.parent_run_id is not None and not isinstance(self.parent_run_id, UUID):
            raise ContractValidationError("parent_run_id must be a UUID")
        if not isinstance(self.kind, RunKind):
            raise ContractValidationError("kind must be a RunKind")
        if not isinstance(self.status, RunStatus):
            raise ContractValidationError("status must be a RunStatus")
        if not isinstance(self.subject, SubjectScope):
            raise ContractValidationError("subject must be a SubjectScope")
        for name, value in (
            ("started_at", self.started_at),
            ("updated_at", self.updated_at),
            ("deadline", self.deadline),
        ):
            _require_aware(value, name)
        if self.updated_at < self.started_at:
            raise ContractValidationError("updated_at cannot precede started_at")
        if self.deadline < self.started_at:
            raise ContractValidationError("deadline cannot precede started_at")
        if any(not isinstance(value, UUID) for value in self.input_event_ids):
            raise ContractValidationError("input_event_ids must contain UUID values")
        if self.state_version is not None and self.state_version < 0:
            raise ContractValidationError("state_version must be non-negative")
        if self.termination_reason is not None:
            _require_non_blank(self.termination_reason, "termination_reason")
        if self.error_type is not None:
            _require_non_blank(self.error_type, "error_type")
        if self.checkpoint is not None and self.checkpoint.run_id != self.run_id:
            raise ContractValidationError("checkpoint belongs to another run")


@dataclass(frozen=True, slots=True)
class CognitiveCycle:
    """Typed view of a slow cognition run recorded by ``RunRecord``."""

    record: RunRecord

    def __post_init__(self) -> None:
        if self.record.kind is not RunKind.COGNITIVE_CYCLE:
            raise ContractValidationError("record is not a cognitive cycle")

    @property
    def input_event_ids(self) -> tuple[UUID, ...]:
        return self.record.input_event_ids

    @property
    def state_version(self) -> int | None:
        return self.record.state_version

    @property
    def motive_result(self) -> str | None:
        return self.record.motive_result

    @property
    def termination_reason(self) -> str | None:
        return self.record.termination_reason


def run_to_dict(record: RunRecord) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": str(record.run_id),
        "kind": record.kind.value,
        "subject": {
            "mind_id": record.subject.mind.mind_id,
            "kind": record.subject.subject.kind.value,
            "subject_id": record.subject.subject.subject_id,
        },
        "correlation_id": str(record.correlation_id),
        "started_at": record.started_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
        "deadline": record.deadline.isoformat(),
        "status": record.status.value,
        "parent_run_id": (
            str(record.parent_run_id) if record.parent_run_id is not None else None
        ),
        "budget": {
            "max_model_calls": record.budget.max_model_calls,
            "max_tool_calls": record.budget.max_tool_calls,
        },
        "usage": {
            "model_calls": record.usage.model_calls,
            "tool_calls": record.usage.tool_calls,
        },
        "checkpoint": _checkpoint_to_dict(record.checkpoint),
        "input_event_ids": [str(value) for value in record.input_event_ids],
        "state_version": record.state_version,
        "motive_result": record.motive_result,
        "termination_reason": record.termination_reason,
        "error_type": record.error_type,
        "cancel_requested": record.cancel_requested,
    }


def run_from_dict(data: object) -> RunRecord:
    if not isinstance(data, dict):
        raise ContractValidationError("run record must be an object")
    try:
        if data.get("schema_version") != 1:
            raise ContractValidationError("unsupported run record schema")
        subject_data = data["subject"]
        budget_data = data["budget"]
        usage_data = data["usage"]
        if not isinstance(subject_data, dict):
            raise TypeError("subject must be an object")
        if not isinstance(budget_data, dict) or not isinstance(usage_data, dict):
            raise TypeError("budget and usage must be objects")
        subject = SubjectScope(
            MindScope(str(subject_data["mind_id"])),
            SubjectRef(
                SubjectKind(str(subject_data["kind"])),
                str(subject_data["subject_id"]),
            ),
        )
        parent = data.get("parent_run_id")
        return RunRecord(
            run_id=UUID(str(data["run_id"])),
            kind=RunKind(str(data["kind"])),
            subject=subject,
            correlation_id=UUID(str(data["correlation_id"])),
            started_at=datetime.fromisoformat(str(data["started_at"])),
            updated_at=datetime.fromisoformat(str(data["updated_at"])),
            deadline=datetime.fromisoformat(str(data["deadline"])),
            status=RunStatus(str(data["status"])),
            parent_run_id=UUID(str(parent)) if parent is not None else None,
            budget=RunBudget(
                max_model_calls=budget_data.get("max_model_calls"),
                max_tool_calls=budget_data.get("max_tool_calls"),
            ),
            usage=RunUsage(
                model_calls=int(usage_data.get("model_calls", 0)),
                tool_calls=int(usage_data.get("tool_calls", 0)),
            ),
            checkpoint=_checkpoint_from_dict(data.get("checkpoint")),
            input_event_ids=tuple(
                UUID(str(value)) for value in data.get("input_event_ids", ())
            ),
            state_version=data.get("state_version"),
            motive_result=data.get("motive_result"),
            termination_reason=data.get("termination_reason"),
            error_type=data.get("error_type"),
            cancel_requested=bool(data.get("cancel_requested", False)),
        )
    except (ContractValidationError, KeyError, TypeError, ValueError) as error:
        if isinstance(error, ContractValidationError):
            raise
        raise ContractValidationError("invalid run record") from error


def _checkpoint_to_dict(checkpoint: RunCheckpoint | None) -> dict[str, Any] | None:
    if checkpoint is None:
        return None
    return {
        "checkpoint_id": str(checkpoint.checkpoint_id),
        "run_id": str(checkpoint.run_id),
        "position": checkpoint.position,
        "recorded_at": checkpoint.recorded_at.isoformat(),
        "action_id": str(checkpoint.action_id) if checkpoint.action_id else None,
        "result_event_id": (
            str(checkpoint.result_event_id)
            if checkpoint.result_event_id
            else None
        ),
    }


def _checkpoint_from_dict(data: object) -> RunCheckpoint | None:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ContractValidationError("checkpoint must be an object")
    return RunCheckpoint(
        checkpoint_id=UUID(str(data["checkpoint_id"])),
        run_id=UUID(str(data["run_id"])),
        position=str(data["position"]),
        recorded_at=datetime.fromisoformat(str(data["recorded_at"])),
        action_id=UUID(str(data["action_id"])) if data.get("action_id") else None,
        result_event_id=(
            UUID(str(data["result_event_id"]))
            if data.get("result_event_id")
            else None
        ),
    )


def _require_aware(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ContractValidationError(f"{name} must include timezone information")


def _require_non_blank(value: str | None, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{name} must not be blank")
