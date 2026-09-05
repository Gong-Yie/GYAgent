from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from hashlib import sha256
import json
from uuid import UUID

from self_cognition.core.errors import ContractValidationError
from self_cognition.core.memories import MemoryType
from self_cognition.core.scopes import SubjectScope


class DeletionMode(str, Enum):
    MEMORY = "memory"
    RANGE = "range"
    SUBJECT = "subject"


class DeletionStatus(str, Enum):
    PLANNED = "planned"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class DeletionSelector:
    subject: SubjectScope
    memory_id: UUID | None = None
    memory_types: tuple[MemoryType, ...] = ()
    created_from: datetime | None = None
    created_to: datetime | None = None
    conversation_id: str | None = None
    delete_subject: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.subject, SubjectScope):
            raise ContractValidationError("deletion subject must be a SubjectScope")
        if self.memory_id is not None and not isinstance(self.memory_id, UUID):
            raise ContractValidationError("deletion memory_id must be a UUID")
        if any(not isinstance(value, MemoryType) for value in self.memory_types):
            raise ContractValidationError("memory_types must contain MemoryType values")
        for name, value in (
            ("created_from", self.created_from),
            ("created_to", self.created_to),
        ):
            if value is not None and (
                value.tzinfo is None or value.utcoffset() is None
            ):
                raise ContractValidationError(
                    f"{name} must include timezone information"
                )
        if (
            self.created_from is not None
            and self.created_to is not None
            and self.created_from > self.created_to
        ):
            raise ContractValidationError("deletion time range is invalid")
        if self.conversation_id is not None and not self.conversation_id.strip():
            raise ContractValidationError("conversation_id must not be blank")
        selected_modes = sum(
            (
                self.memory_id is not None,
                self.delete_subject,
                bool(
                    self.memory_types
                    or self.created_from
                    or self.created_to
                    or self.conversation_id
                ),
            )
        )
        if selected_modes != 1:
            raise ContractValidationError(
                "deletion selector must choose one memory, one range, or the subject"
            )

    @property
    def mode(self) -> DeletionMode:
        if self.memory_id is not None:
            return DeletionMode.MEMORY
        if self.delete_subject:
            return DeletionMode.SUBJECT
        return DeletionMode.RANGE


@dataclass(frozen=True, slots=True)
class InvalidatedModuleResult:
    event_id: UUID
    cause_id: UUID
    module_id: str
    module_version: str
    deterministic: bool

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, UUID) or not isinstance(self.cause_id, UUID):
            raise ContractValidationError("invalidated result requires event IDs")
        if not self.module_id.strip() or not self.module_version.strip():
            raise ContractValidationError("invalidated result requires module metadata")
        if not isinstance(self.deterministic, bool):
            raise ContractValidationError("deterministic must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": str(self.event_id),
            "cause_id": str(self.cause_id),
            "module_id": self.module_id,
            "module_version": self.module_version,
            "deterministic": self.deterministic,
        }


@dataclass(frozen=True, slots=True)
class DeletionImpact:
    subject: SubjectScope
    event_ids: tuple[UUID, ...]
    memory_ids: tuple[UUID, ...]
    invalidated_results: tuple[InvalidatedModuleResult, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.subject, SubjectScope):
            raise ContractValidationError("deletion impact requires a subject")
        for values in (self.event_ids, self.memory_ids):
            if (
                any(not isinstance(item, UUID) for item in values)
                or tuple(sorted(set(values))) != values
            ):
                raise ContractValidationError(
                    "deletion impact IDs must be unique and sorted"
                )
        if any(
            item.event_id not in self.event_ids for item in self.invalidated_results
        ):
            raise ContractValidationError(
                "invalidated result must be selected for deletion"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "mind_id": self.subject.mind.mind_id,
            "subject_kind": self.subject.subject.kind.value,
            "subject_id": self.subject.subject.subject_id,
            "event_ids": [str(item) for item in self.event_ids],
            "memory_ids": [str(item) for item in self.memory_ids],
            "invalidated_results": [
                item.to_dict() for item in self.invalidated_results
            ],
        }


@dataclass(frozen=True, slots=True)
class DeletionPlan:
    plan_id: UUID
    selector: DeletionSelector
    memory_ids: tuple[UUID, ...]
    event_ids: tuple[UUID, ...]
    created_at: datetime
    reason: str = "user_request"
    status: DeletionStatus = DeletionStatus.PLANNED
    status_updated_at: datetime | None = None
    failure_type: str | None = None
    cache_result: str = "not_applicable"
    export_result: str = "not_applicable"
    impacts: tuple[DeletionImpact, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.plan_id, UUID):
            raise ContractValidationError("plan_id must be a UUID")
        if not isinstance(self.selector, DeletionSelector):
            raise ContractValidationError("selector must be a DeletionSelector")
        if not isinstance(self.status, DeletionStatus):
            raise ContractValidationError("status must be a DeletionStatus")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ContractValidationError("deletion reason must not be blank")
        for name, values in (
            ("memory_ids", self.memory_ids),
            ("event_ids", self.event_ids),
        ):
            if tuple(sorted(set(values), key=lambda value: value.int)) != values:
                raise ContractValidationError(f"{name} must be unique and sorted")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ContractValidationError(
                "created_at must include timezone information"
            )
        if self.status is DeletionStatus.PLANNED:
            if self.status_updated_at is not None or self.failure_type is not None:
                raise ContractValidationError(
                    "planned deletion has no execution result"
                )
        elif self.status_updated_at is None:
            raise ContractValidationError(
                "executed deletion requires status_updated_at"
            )
        if self.status_updated_at is not None and (
            self.status_updated_at.tzinfo is None
            or self.status_updated_at.utcoffset() is None
        ):
            raise ContractValidationError(
                "status_updated_at must include timezone information"
            )
        if self.status is DeletionStatus.FAILED:
            if not isinstance(self.failure_type, str) or not self.failure_type.strip():
                raise ContractValidationError("failed deletion requires failure_type")
        elif self.failure_type is not None:
            raise ContractValidationError("only failed deletion may have failure_type")
        if self.impacts:
            if len({item.subject for item in self.impacts}) != len(self.impacts):
                raise ContractValidationError("deletion impact subjects must be unique")
            if any(
                item.subject.mind != self.selector.subject.mind for item in self.impacts
            ):
                raise ContractValidationError("deletion impacts cannot cross minds")
            primary = next(
                (
                    item
                    for item in self.impacts
                    if item.subject == self.selector.subject
                ),
                None,
            )
            if (
                primary is None
                or primary.event_ids != self.event_ids
                or primary.memory_ids != self.memory_ids
            ):
                raise ContractValidationError(
                    "primary deletion impact must match its selector"
                )

    @property
    def effective_impacts(self) -> tuple[DeletionImpact, ...]:
        return self.impacts or (
            DeletionImpact(self.selector.subject, self.event_ids, self.memory_ids),
        )

    @property
    def digest(self) -> str:
        payload = {
            "plan_id": str(self.plan_id),
            "mode": self.selector.mode.value,
            "mind_id": self.selector.subject.mind.mind_id,
            "subject_kind": self.selector.subject.subject.kind.value,
            "subject_id": self.selector.subject.subject.subject_id,
            "selector_memory_id": (
                str(self.selector.memory_id)
                if self.selector.memory_id is not None
                else None
            ),
            "memory_types": [value.value for value in self.selector.memory_types],
            "created_from": (
                self.selector.created_from.isoformat()
                if self.selector.created_from is not None
                else None
            ),
            "created_to": (
                self.selector.created_to.isoformat()
                if self.selector.created_to is not None
                else None
            ),
            "conversation_id": self.selector.conversation_id,
            "memory_ids": [str(value) for value in self.memory_ids],
            "event_ids": [str(value) for value in self.event_ids],
            "created_at": self.created_at.isoformat(),
            "reason": self.reason,
        }
        if self.impacts:
            payload["impacts"] = [item.to_dict() for item in self.impacts]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return sha256(encoded.encode("utf-8")).hexdigest()

    def with_status(
        self,
        status: DeletionStatus,
        changed_at: datetime,
        *,
        failure_type: str | None = None,
    ) -> "DeletionPlan":
        return replace(
            self,
            status=status,
            status_updated_at=changed_at,
            failure_type=failure_type,
        )
