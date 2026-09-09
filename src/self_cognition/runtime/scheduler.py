from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Generic, TypeVar

from self_cognition.core.errors import ContractValidationError
from self_cognition.core.events import EventEnvelope
from self_cognition.core.proactivity import ProactiveIntention
from self_cognition.core.state import SubjectState
from self_cognition.runtime.run_context import RunContext
from self_cognition.observability.metrics import MetricsRegistry
from self_cognition.observability.tracing import TraceRecorder


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class FastBehaviorResult(Generic[T]):
    event: EventEnvelope
    value: T
    intentions: tuple[ProactiveIntention, ...]


class FastBehaviorLoop:
    """Synchronous current-event path; it never waits for slow work."""

    def __init__(
        self,
        metrics: MetricsRegistry | None = None,
        traces: TraceRecorder | None = None,
    ) -> None:
        self._metrics = metrics
        self._traces = traces

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
        if self._metrics is not None:
            self._metrics.increment("behavior.fast.received")
        trace = (
            self._traces.start(
                "behavior.fast",
                event_id=event.event_id,
                state_version=state.version,
            )
            if self._traces is not None
            else None
        )
        value = handler(event, state, intentions) if handler is not None else None
        if self._metrics is not None and value is None:
            self._metrics.increment("behavior.fast.silenced")
        if trace is not None:
            self._traces.finish(trace)
        return FastBehaviorResult(event, value, intentions)


class SlowCognitionScheduler:
    """Explicit slow-loop executor; lifecycle owns its shutdown."""

    def __init__(
        self,
        handler: Callable[[EventEnvelope, RunContext], T],
        *,
        max_workers: int = 1,
        metrics: MetricsRegistry | None = None,
        traces: TraceRecorder | None = None,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self._handler = handler
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: set[Future[T]] = set()
        self._metrics = metrics
        self._traces = traces

    def submit(self, event: EventEnvelope, context: RunContext) -> Future[T]:
        started = perf_counter()
        if self._metrics is not None:
            self._metrics.increment("cognition.slow.submitted")
        future = self._executor.submit(self._handler, event, context)
        self._futures.add(future)
        trace = (
            self._traces.start(
                "cognition.slow",
                run_id=context.run_id,
                event_id=event.event_id,
            )
            if self._traces is not None
            else None
        )

        def complete(done: Future[T]) -> None:
            self._futures.discard(done)
            if self._metrics is not None:
                self._metrics.observe("cognition.slow.latency_seconds", perf_counter() - started)
                self._metrics.set_gauge("cognition.slow.pending", self.pending())
                failed = done.cancelled() or done.exception() is not None
                self._metrics.increment(
                    "cognition.slow.failed" if failed else "cognition.slow.completed"
                )
            if trace is not None:
                failed = done.cancelled() or done.exception() is not None
                self._traces.finish(trace, status="failed" if failed else "completed")

        future.add_done_callback(complete)
        if self._metrics is not None:
            self._metrics.set_gauge("cognition.slow.pending", self.pending())
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
        metrics: MetricsRegistry | None = None,
        traces: TraceRecorder | None = None,
    ) -> None:
        self.fast = FastBehaviorLoop(metrics, traces)
        self.slow = SlowCognitionScheduler(
            slow_handler,
            max_workers=max_workers,
            metrics=metrics,
            traces=traces,
        )

    def submit_slow(self, event: EventEnvelope, context: RunContext) -> Future[T]:
        return self.slow.submit(event, context)

    def close(self, *, wait: bool = True) -> None:
        self.slow.close(wait=wait)


CognitiveScheduler = SlowCognitionScheduler
BehaviorScheduler = DualLoopScheduler
