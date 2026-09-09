from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from self_cognition.core.errors import ContractValidationError
from self_cognition.core.events import EventEnvelope
from self_cognition.core.proactivity import ProactiveIntention
from self_cognition.core.state import SubjectState
from self_cognition.runtime.run_context import RunContext


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class FastBehaviorResult(Generic[T]):
    event: EventEnvelope
    value: T
    intentions: tuple[ProactiveIntention, ...]


class FastBehaviorLoop:
    """Synchronous current-event path; it never waits for slow work."""

    def run(
        self,
        event: EventEnvelope,
        state: SubjectState,
        intentions: tuple[ProactiveIntention, ...],
        handler: Callable[[EventEnvelope, SubjectState, tuple[ProactiveIntention, ...]], T] | None = None,
        *,
        conversation_history: tuple[str, ...] = (),
    ) -> FastBehaviorResult[T | None]:
        if conversation_history:
            raise ContractValidationError(
                "fast behavior loop does not accept rolling conversation history"
            )
        value = handler(event, state, intentions) if handler is not None else None
        return FastBehaviorResult(event, value, intentions)


class SlowCognitionScheduler:
    """Explicit slow-loop executor; lifecycle owns its shutdown."""

    def __init__(
        self,
        handler: Callable[[EventEnvelope, RunContext], T],
        *,
        max_workers: int = 1,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self._handler = handler
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: set[Future[T]] = set()

    def submit(self, event: EventEnvelope, context: RunContext) -> Future[T]:
        future = self._executor.submit(self._handler, event, context)
        self._futures.add(future)
        future.add_done_callback(self._futures.discard)
        return future

    def pending(self) -> int:
        return sum(not future.done() for future in tuple(self._futures))

    def close(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=not wait)


class DualLoopScheduler:
    """Convenience facade combining the fast path and slow scheduler."""

    def __init__(
        self,
        slow_handler: Callable[[EventEnvelope, RunContext], T],
        *,
        max_workers: int = 1,
    ) -> None:
        self.fast = FastBehaviorLoop()
        self.slow = SlowCognitionScheduler(slow_handler, max_workers=max_workers)

    def submit_slow(self, event: EventEnvelope, context: RunContext) -> Future[T]:
        return self.slow.submit(event, context)

    def close(self, *, wait: bool = True) -> None:
        self.slow.close(wait=wait)


CognitiveScheduler = SlowCognitionScheduler
BehaviorScheduler = DualLoopScheduler
