from dataclasses import dataclass
from enum import Enum
from uuid import UUID

from self_cognition.core.state import SubjectState


class ProcessEventStatus(str, Enum):
    SUCCEEDED = "succeeded"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ProcessEventResult:
    status: ProcessEventStatus
    run_id: UUID
    correlation_id: UUID
    old_version: int | None
    new_version: int | None
    state_changed: bool | None
    state: SubjectState | None
    event_saved: bool
    error_type: str | None = None
    retryable: bool | None = None
