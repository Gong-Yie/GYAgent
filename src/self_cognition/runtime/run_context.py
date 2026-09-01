from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from self_cognition.core.errors import ContractValidationError
from self_cognition.core.events import EventEnvelope
from self_cognition.core.time import Clock, SYSTEM_CLOCK


@dataclass(slots=True)
class RunContext:
    run_id: UUID
    correlation_id: UUID
    deadline: datetime
    cancelled: bool = False
    clock: Clock = SYSTEM_CLOCK
    _emitted_events: list[EventEnvelope] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.deadline.tzinfo is None or self.deadline.utcoffset() is None:
            raise ContractValidationError("deadline must include timezone information")

    @property
    def is_cancelled(self) -> bool:
        return self.cancelled or self.clock.now() >= self.deadline

    def cancel(self) -> None:
        self.cancelled = True

    def emit_event(self, event: EventEnvelope) -> None:
        self._emitted_events.append(event)

    def drain_emitted_events(self) -> tuple[EventEnvelope, ...]:
        events = tuple(self._emitted_events)
        self._emitted_events.clear()
        return events
