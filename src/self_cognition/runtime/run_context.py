from dataclasses import dataclass, field, replace
from datetime import datetime
from uuid import UUID

from self_cognition.core.errors import ContractValidationError, RunBudgetExceededError
from self_cognition.core.events import EventEnvelope
from self_cognition.core.runs import RunBudget, RunUsage
from self_cognition.core.time import Clock, SYSTEM_CLOCK
from self_cognition.runtime.cancellation import CancellationToken


@dataclass(slots=True)
class RunContext:
    run_id: UUID
    correlation_id: UUID
    deadline: datetime
    cancelled: bool = False
    clock: Clock = SYSTEM_CLOCK
    parent_run_id: UUID | None = None
    budget: RunBudget = RunBudget()
    usage: RunUsage = RunUsage()
    cancellation_token: CancellationToken | None = None
    _emitted_events: list[EventEnvelope] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.deadline.tzinfo is None or self.deadline.utcoffset() is None:
            raise ContractValidationError("deadline must include timezone information")
        if self.cancellation_token is None:
            self.cancellation_token = CancellationToken()
        if self.cancelled:
            self.cancellation_token.cancel()

    @property
    def is_cancelled(self) -> bool:
        return (
            self.cancelled
            or self.cancellation_token.is_cancelled
            or self.clock.now() >= self.deadline
        )

    def cancel(self, reason: str = "cancelled") -> None:
        self.cancelled = True
        self.cancellation_token.cancel(reason)

    @property
    def cancellation_reason(self) -> str | None:
        return self.cancellation_token.reason

    def child(
        self,
        run_id: UUID,
        correlation_id: UUID | None = None,
        *,
        deadline: datetime | None = None,
        budget: RunBudget | None = None,
    ) -> "RunContext":
        return RunContext(
            run_id=run_id,
            correlation_id=correlation_id or self.correlation_id,
            deadline=deadline or self.deadline,
            clock=self.clock,
            parent_run_id=self.run_id,
            budget=budget or self.budget,
            cancellation_token=self.cancellation_token.child(),
        )

    def record_model_call(self) -> None:
        next_count = self.usage.model_calls + 1
        limit = self.budget.max_model_calls
        if limit is not None and next_count > limit:
            raise RunBudgetExceededError("model call budget exceeded")
        self.usage = replace(self.usage, model_calls=next_count)

    def record_tool_call(self) -> None:
        next_count = self.usage.tool_calls + 1
        limit = self.budget.max_tool_calls
        if limit is not None and next_count > limit:
            raise RunBudgetExceededError("tool call budget exceeded")
        self.usage = replace(self.usage, tool_calls=next_count)

    def emit_event(self, event: EventEnvelope) -> None:
        self._emitted_events.append(event)

    def drain_emitted_events(self) -> tuple[EventEnvelope, ...]:
        events = tuple(self._emitted_events)
        self._emitted_events.clear()
        return events
