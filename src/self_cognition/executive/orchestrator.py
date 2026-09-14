from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from typing import Callable, Any

from self_cognition.core.events import EventEnvelope
from self_cognition.core.proactivity import ProactiveIntention
from self_cognition.core.protocols import StateRepository
from self_cognition.core.scopes import SubjectKind
from self_cognition.core.state import SubjectState
from self_cognition.application.proactive import ProactiveIntentionService
from self_cognition.runtime.run_context import RunContext
from self_cognition.runtime.scheduler import DualLoopScheduler, FastBehaviorResult


@dataclass(frozen=True, slots=True)
class OrchestrationResult:
    fast: FastBehaviorResult[Any]
    slow: Future[Any]


@dataclass(frozen=True, slots=True)
class FastBehaviorResponse:
    text: str
    intention_ids: tuple[str, ...] = ()


class ExecutiveOrchestrator:
    """Coordinates the two loops while keeping model and tool decisions separate."""

    def __init__(
        self,
        state_repository: StateRepository,
        intentions: ProactiveIntentionService,
        scheduler: DualLoopScheduler,
        *,
        fast_handler: Callable[[EventEnvelope, SubjectState, tuple[ProactiveIntention, ...]], Any] | None = None,
    ) -> None:
        self._states = state_repository
        self._intentions = intentions
        self._scheduler = scheduler
        self._fast_handler = fast_handler

    def handle(
        self,
        event: EventEnvelope,
        context: RunContext,
        *,
        conversation_history: tuple[str, ...] = (),
    ) -> OrchestrationResult:
        state = self._states.load(event.subject)
        if state is None:
            state = SubjectState.empty(
                event.subject.subject.subject_id,
                mind_id=event.subject.mind.mind_id,
                subject_kind=event.subject.subject.kind,
            )
        intentions = self._intentions.active(
            event.subject,
            as_of=context.clock.now(),
        )
        fast = self._scheduler.fast.run(
            event,
            state,
            intentions,
            self._fast_handler,
            conversation_history=conversation_history,
        )
        slow = self._scheduler.submit_slow(event, context)
        return OrchestrationResult(fast, slow)

    def fast_only(
        self,
        event: EventEnvelope,
        context: RunContext,
        *,
        conversation_history: tuple[str, ...] = (),
    ) -> FastBehaviorResult[Any]:
        state = self._states.load(event.subject)
        if state is None:
            state = SubjectState.empty(
                event.subject.subject.subject_id,
                mind_id=event.subject.mind.mind_id,
                subject_kind=event.subject.subject.kind,
            )
        intentions = self._intentions.active(
            event.subject,
            as_of=context.clock.now(),
        )
        return self._scheduler.fast.run(
            event,
            state,
            intentions,
            self._fast_handler,
            conversation_history=conversation_history,
        )

    def close(self) -> None:
        self._scheduler.close()


def default_fast_handler(
    event: EventEnvelope,
    state: SubjectState,
    intentions: tuple[ProactiveIntention, ...],
) -> FastBehaviorResponse:
    del state
    if intentions:
        intention = intentions[0]
        return FastBehaviorResponse(
            f"我先处理当前消息；还有一项待处理意图：{intention.expected_behavior}。",
            tuple(str(item.intention_id) for item in intentions),
        )
    text = getattr(event.payload, "text", "")
    return FastBehaviorResponse(
        "我已收到这条消息，正在并行更新相关认知。"
        if text
        else "我已收到当前事件，正在并行处理。"
    )
