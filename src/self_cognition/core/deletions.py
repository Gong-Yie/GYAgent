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
