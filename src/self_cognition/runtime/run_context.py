from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from self_cognition.core.errors import ContractValidationError
from self_cognition.core.time import Clock, SYSTEM_CLOCK


@dataclass(slots=True)
class RunContext:
    run_id: UUID
    correlation_id: UUID
    deadline: datetime
    cancelled: bool = False
    clock: Clock = SYSTEM_CLOCK

    def __post_init__(self) -> None:
        if self.deadline.tzinfo is None or self.deadline.utcoffset() is None:
            raise ContractValidationError("deadline must include timezone information")

    @property
    def is_cancelled(self) -> bool:
        return self.cancelled or self.clock.now() >= self.deadline

    def cancel(self) -> None:
        self.cancelled = True
