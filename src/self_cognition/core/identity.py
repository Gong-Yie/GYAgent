from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

from self_cognition.core.errors import ContractValidationError
from self_cognition.core.scopes import SubjectKind, SubjectScope
from self_cognition.core.time import SYSTEM_CLOCK

if TYPE_CHECKING:
    from self_cognition.core.state import StateAtom, SubjectState


class SelfModelAspect(str, Enum):
    IDENTITY = "identity"
    VALUE = "values"
    LIMITATION = "limitations"
    GOAL = "goals"


class CapabilityKind(str, Enum):
    MODEL = "model"
    TOOL = "tool"
    KNOWLEDGE_SOURCE = "knowledge_source"


class CapabilityPermission(str, Enum):
    GRANTED = "granted"
    DENIED = "denied"


class CapabilityExecutionStatus(str, Enum):
    NOT_RUN = "not_run"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class LimitationStatus(str, Enum):
    ACTIVE = "active"
    RECOVERED = "recovered"


class GoalPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class GoalStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class CapabilityRecord:
    capability_id: str
    name: str
    kind: CapabilityKind
    registered: bool
    permission: CapabilityPermission
    execution_status: CapabilityExecutionStatus = (
        CapabilityExecutionStatus.NOT_RUN
    )
    reason: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.capability_id, "capability_id")
        _require_text(self.name, "capability name")
        if not isinstance(self.kind, CapabilityKind):
            raise ContractValidationError("capability kind is invalid")
        if not isinstance(self.registered, bool):
            raise ContractValidationError("registered must be a boolean")
        if not isinstance(self.permission, CapabilityPermission):
            raise ContractValidationError("capability permission is invalid")
        if not isinstance(self.execution_status, CapabilityExecutionStatus):
            raise ContractValidationError("capability execution status is invalid")
        _require_optional_text(self.reason, "capability reason")
        if (
            self.execution_status is CapabilityExecutionStatus.FAILED
            and self.reason is None
        ):
            raise ContractValidationError(
                "failed capability execution requires a reason"
            )
        if (not self.registered or self.permission is CapabilityPermission.DENIED) and (
            self.reason is None
        ):
            raise ContractValidationError(
                "unavailable capability requires a reason"
            )

    @property
    def available(self) -> bool:
        return self.registered and self.permission is CapabilityPermission.GRANTED

    @property
    def verified(self) -> bool:
        return self.execution_status is CapabilityExecutionStatus.SUCCEEDED

    def to_state_value(self) -> dict[str, object]:
        return {
            "capability_id": self.capability_id,
            "name": self.name,
            "kind": self.kind.value,
            "registered": self.registered,
            "permission": self.permission.value,
            "execution_status": self.execution_status.value,
            "reason": self.reason,
        }

    @classmethod
    def from_state_value(cls, value: object) -> CapabilityRecord:
        values = _require_mapping(value, "capability")
        _require_exact_keys(
            values,
            {
                "capability_id",
                "name",
                "kind",
                "registered",
                "permission",
                "execution_status",
                "reason",
            },
            "capability",
        )
        try:
            return cls(
                capability_id=_as_text(values["capability_id"], "capability_id"),
                name=_as_text(values["name"], "capability name"),
                kind=CapabilityKind(_as_text(values["kind"], "capability kind")),
                registered=_as_bool(values["registered"], "registered"),
                permission=CapabilityPermission(
                    _as_text(values["permission"], "capability permission")
                ),
                execution_status=CapabilityExecutionStatus(
                    _as_text(
                        values["execution_status"],
                        "capability execution status",
                    )
                ),
                reason=_as_optional_text(values["reason"], "capability reason"),
            )
        except ValueError as error:
            raise ContractValidationError("capability value is invalid") from error


@dataclass(frozen=True, slots=True)
class LimitationRecord:
    limitation_id: str
    description: str
    reason: str
    applies_to: str
    recovery_condition: str
    status: LimitationStatus = LimitationStatus.ACTIVE

    def __post_init__(self) -> None:
        _require_identifier(self.limitation_id, "limitation_id")
        for name in ("description", "reason", "applies_to", "recovery_condition"):
            _require_text(getattr(self, name), name)
        if not isinstance(self.status, LimitationStatus):
            raise ContractValidationError("limitation status is invalid")

    def to_state_value(self) -> dict[str, object]:
        return {
            "limitation_id": self.limitation_id,
            "description": self.description,
            "reason": self.reason,
            "applies_to": self.applies_to,
            "recovery_condition": self.recovery_condition,
            "status": self.status.value,
        }

    @classmethod
    def from_state_value(cls, value: object) -> LimitationRecord:
        values = _require_mapping(value, "limitation")
        _require_exact_keys(
            values,
            {
                "limitation_id",
                "description",
                "reason",
                "applies_to",
                "recovery_condition",
                "status",
            },
            "limitation",
        )
        try:
            return cls(
                limitation_id=_as_text(values["limitation_id"], "limitation_id"),
                description=_as_text(values["description"], "description"),
                reason=_as_text(values["reason"], "reason"),
                applies_to=_as_text(values["applies_to"], "applies_to"),
                recovery_condition=_as_text(
                    values["recovery_condition"],
                    "recovery_condition",
                ),
                status=LimitationStatus(
                    _as_text(values["status"], "limitation status")
                ),
            )
        except ValueError as error:
            raise ContractValidationError("limitation value is invalid") from error


@dataclass(frozen=True, slots=True)
class GoalRecord:
    goal_id: str
    description: str
    source: str
    priority: GoalPriority
    status: GoalStatus
    completion_conditions: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.goal_id, "goal_id")
        _require_text(self.description, "goal description")
        _require_text(self.source, "goal source")
        if not isinstance(self.priority, GoalPriority):
            raise ContractValidationError("goal priority is invalid")
        if not isinstance(self.status, GoalStatus):
            raise ContractValidationError("goal status is invalid")
        if not self.completion_conditions or any(
            not isinstance(condition, str) or not condition.strip()
            for condition in self.completion_conditions
        ):
            raise ContractValidationError(
                "goal completion conditions must contain text"
            )

    def to_state_value(self) -> dict[str, object]:
        return {
            "goal_id": self.goal_id,
            "description": self.description,
            "source": self.source,
            "priority": self.priority.value,
            "status": self.status.value,
            "completion_conditions": list(self.completion_conditions),
        }

    @classmethod
    def from_state_value(cls, value: object) -> GoalRecord:
        values = _require_mapping(value, "goal")
        _require_exact_keys(
            values,
            {
                "goal_id",
                "description",
                "source",
                "priority",
                "status",
                "completion_conditions",
            },
            "goal",
        )
        conditions = values["completion_conditions"]
        if not isinstance(conditions, list) or any(
            not isinstance(condition, str) for condition in conditions
        ):
            raise ContractValidationError(
                "goal completion conditions must be an array of text"
            )
        try:
            return cls(
                goal_id=_as_text(values["goal_id"], "goal_id"),
                description=_as_text(values["description"], "goal description"),
                source=_as_text(values["source"], "goal source"),
                priority=GoalPriority(_as_text(values["priority"], "goal priority")),
                status=GoalStatus(_as_text(values["status"], "goal status")),
                completion_conditions=tuple(conditions),
            )
        except ValueError as error:
            raise ContractValidationError("goal value is invalid") from error


SelfModelObservationValue = str | LimitationRecord | GoalRecord


@dataclass(frozen=True, slots=True)
class SelfModel:
    subject: SubjectScope
    version: int
    as_of: datetime
    entries: Mapping[str, StateAtom]
    identity: Mapping[str, StateAtom]
    values: Mapping[str, StateAtom]
    capabilities: tuple[CapabilityRecord, ...]
    limitations: tuple[LimitationRecord, ...]
    goals: tuple[GoalRecord, ...]

    @classmethod
    def from_state(
        cls,
        state: SubjectState,
        *,
        as_of: datetime | None = None,
    ) -> SelfModel:
        if state.subject_kind is not SubjectKind.MIND:
            raise ContractValidationError("self model requires a mind subject")
        evaluation_time = as_of or SYSTEM_CLOCK.now()
        identity: dict[str, StateAtom] = {}
        values: dict[str, StateAtom] = {}
        capabilities: list[CapabilityRecord] = []
        limitations: list[LimitationRecord] = []
        goals: list[GoalRecord] = []
        for target_field, atom in state.entries.items():
            if atom.valid_from > evaluation_time or (
                atom.expires_at is not None and atom.expires_at <= evaluation_time
            ):
                continue
            field_id = target_field.partition(".")[2]
            if target_field.startswith("identity."):
                identity[field_id] = atom
            elif target_field.startswith("values."):
                values[field_id] = atom
            elif target_field.startswith("capabilities."):
                capabilities.append(
                    CapabilityRecord.from_state_value(atom.value)
                )
            elif target_field.startswith("limitations."):
                limitations.append(LimitationRecord.from_state_value(atom.value))
            elif target_field.startswith("goals."):
                goals.append(GoalRecord.from_state_value(atom.value))
        return cls(
            subject=state.subject_scope,
            version=state.version,
            as_of=evaluation_time,
            entries=MappingProxyType(
                {
                    target_field: atom
                    for target_field, atom in state.entries.items()
                    if atom.valid_from <= evaluation_time
                    and (
                        atom.expires_at is None
                        or atom.expires_at > evaluation_time
                    )
                }
            ),
            identity=MappingProxyType(identity),
            values=MappingProxyType(values),
            capabilities=tuple(
                sorted(capabilities, key=lambda item: item.capability_id)
            ),
            limitations=tuple(
                sorted(limitations, key=lambda item: item.limitation_id)
            ),
            goals=tuple(sorted(goals, key=lambda item: item.goal_id)),
        )

    @property
    def available_capabilities(self) -> tuple[CapabilityRecord, ...]:
        return tuple(item for item in self.capabilities if item.available)

    @property
    def active_limitations(self) -> tuple[LimitationRecord, ...]:
        return tuple(
            item for item in self.limitations if item.status is LimitationStatus.ACTIVE
        )

    @property
    def active_goals(self) -> tuple[GoalRecord, ...]:
        return tuple(item for item in self.goals if item.status is GoalStatus.ACTIVE)


def _require_identifier(value: str, name: str) -> None:
    _require_text(value, name)
    if any(character.isspace() for character in value):
        raise ContractValidationError(f"{name} must not contain whitespace")


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{name} must not be blank")


def _require_optional_text(value: str | None, name: str) -> None:
    if value is not None:
        _require_text(value, name)


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{name} value must be a mapping")
    return value


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    name: str,
) -> None:
    if set(value) != expected:
        raise ContractValidationError(f"{name} value has invalid fields")


def _as_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ContractValidationError(f"{name} must be text")
    return value


def _as_optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _as_text(value, name)


def _as_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ContractValidationError(f"{name} must be a boolean")
    return value
