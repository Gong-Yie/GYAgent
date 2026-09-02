from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from uuid import UUID

from self_cognition.core.errors import ContractValidationError
from self_cognition.core.scopes import SubjectScope


PROCESS_EVENT_FAILED = "process_event_failed"
RUN_CANCELLED = "run_cancelled"


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ProcessingRecord:
    event_id: UUID
    subject: SubjectScope
    run_id: UUID
    status: ProcessingStatus
    updated_at: datetime
    error_code: str | None = None
    error_type: str | None = None
    attempt_count: int = 0
    available_at: datetime | None = None
    lease_expires_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, UUID) or not isinstance(
            self.run_id,
            UUID,
        ):
            raise ContractValidationError("processing IDs must be UUID values")
        if not isinstance(self.subject, SubjectScope):
            raise ContractValidationError("processing subject must be a SubjectScope")
        if not isinstance(self.status, ProcessingStatus):
            raise ContractValidationError("status must be a ProcessingStatus")
        _require_aware(self.updated_at, "updated_at")
        if not isinstance(self.attempt_count, int) or self.attempt_count < 0:
            raise ContractValidationError("attempt_count must be non-negative")
        if self.available_at is not None:
            _require_aware(self.available_at, "available_at")
        if self.lease_expires_at is not None:
            _require_aware(self.lease_expires_at, "lease_expires_at")
        if self.status is ProcessingStatus.FAILED:
            _require_non_blank(self.error_code, "error_code")
            _require_non_blank(self.error_type, "error_type")
        elif self.status is ProcessingStatus.PENDING:
            if (self.error_code is None) != (self.error_type is None):
                raise ContractValidationError(
                    "pending error_code and error_type must be set together"
                )
        elif self.error_code is not None or self.error_type is not None:
            raise ContractValidationError(
                "only pending or failed records may contain error details"
            )
        if (
            self.available_at is not None
            and self.status is not ProcessingStatus.PENDING
        ):
            raise ContractValidationError(
                "only pending records may contain available_at"
            )
        if (
            self.lease_expires_at is not None
            and self.status is not ProcessingStatus.PROCESSING
        ):
            raise ContractValidationError(
                "only processing records may contain lease_expires_at"
            )


@dataclass(frozen=True, slots=True)
class OutboxEntry:
    event_id: UUID
    subject: SubjectScope
    run_id: UUID
    enqueued_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, UUID) or not isinstance(
            self.run_id,
            UUID,
        ):
            raise ContractValidationError("outbox IDs must be UUID values")
        if not isinstance(self.subject, SubjectScope):
            raise ContractValidationError("outbox subject must be a SubjectScope")
        _require_aware(self.enqueued_at, "enqueued_at")


def _require_aware(value: datetime, name: str) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ContractValidationError(f"{name} must include timezone information")


def _require_non_blank(value: str | None, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{name} must not be blank")
