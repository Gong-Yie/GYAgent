from dataclasses import dataclass
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Event
from time import monotonic
from uuid import UUID, uuid4

import pytest

from self_cognition.application.proactive import ProactiveIntentionService
from self_cognition.bootstrap import build_container
from self_cognition.core.affect import EmotionState, Motive, compete_motives, decay_emotion
from self_cognition.core.actions import ActionDecision, ActionDecisionStatus, ActionRequest
from self_cognition.core.errors import ContractValidationError
from self_cognition.core.events import ActionDecisionPayload, EventEnvelope, EventSource
from self_cognition.core.proactivity import BehavioralAction, ProactiveIntention
from self_cognition.core.scopes import DataScope, DisclosureScope, SubjectScope
from self_cognition.core.state import SubjectState
from self_cognition.infrastructure.persistence.in_memory_event_store import InMemoryEventStore
from self_cognition.infrastructure.persistence.serialization import event_from_json, event_to_json
from self_cognition.runtime.run_context import RunContext
from self_cognition.runtime.scheduler import DualLoopScheduler
from self_cognition.tools.executor import FileReadToolExecutor, ToolExecutionPolicy


NOW = datetime(2026, 9, 9, 10, tzinfo=timezone.utc)
MIND = SubjectScope.for_mind("mind-25")


@dataclass
class FixedClock:
    value: datetime = NOW

    def now(self) -> datetime:
        return self.value


def context() -> RunContext:
    return RunContext(uuid4(), uuid4(), NOW + timedelta(minutes=5), clock=FixedClock())


def test_emotion_decay_and_motive_competition_are_deterministic() -> None:
    emotion = EmotionState(uuid4(), "项目", "焦虑", "negative", "project", 0.8, NOW)
    decayed = decay_emotion(emotion, NOW + timedelta(hours=1))
    assert decayed is not None and decayed.intensity == 0.4
    assert decay_emotion(emotion, NOW + timedelta(hours=4)) is None

    motives = (
        Motive(uuid4(), MIND.mind.mind_id, "goal", "完成目标", 0.7, 1, NOW),
        Motive(uuid4(), MIND.mind.mind_id, "emotion", "缓解焦虑", 0.7, 2, NOW),
    )
    assert compete_motives(motives, NOW) == tuple(sorted(motives, key=lambda item: (-item.strength, -item.priority, item.motive_id.int)))


def test_intention_lifecycle_and_event_round_trip() -> None:
    store = InMemoryEventStore()
    service = ProactiveIntentionService(store)
    source = EventEnvelope.user_message("mind-25", "提醒我休息", clock=FixedClock())
    store.append(source)
    motive = Motive(uuid4(), MIND.mind.mind_id, "goal", "提醒休息", 0.8, 1, NOW, (source.event_id,))
    intention = ProactiveIntention(
        uuid4(),
        motive,
        MIND,
        "提醒用户休息",
        1,
        0,
        (),
        NOW,
        NOW + timedelta(hours=1),
        idempotency_key="rest-reminder",
    )
    first = service.propose(intention)
    second = service.propose(intention)
    assert second == first
    assert event_from_json(event_to_json(first)) == first

    decided = service.decide(MIND, intention.intention_id, BehavioralAction.DELAY, "当前仍在忙", not_before=NOW + timedelta(minutes=10))
    assert decided.decision.status.value == "delayed"
    assert service.decide(MIND, intention.intention_id, BehavioralAction.DELAY, "当前仍在忙", not_before=NOW + timedelta(minutes=10)).reused
    assert service.active(MIND, as_of=NOW) == (intention,)
    service.cancel(MIND, intention.intention_id, "用户取消")
    assert service.active(MIND, as_of=NOW) == ()


def test_fast_loop_rejects_history_and_slow_loop_does_not_block() -> None:
    started = Event()
    release = Event()

    def slow(event, run):
        del event, run
        started.set()
        release.wait(2)
        return "slow-done"

    scheduler = DualLoopScheduler(slow)
    try:
        event = EventEnvelope.user_message("mind-25", "当前事件", clock=FixedClock())
        future = scheduler.submit_slow(event, context())
        assert started.wait(1)
        begin = monotonic()
        state = SubjectState.empty("mind-25")
        result = scheduler.fast.run(event, state, ())
        assert monotonic() - begin < 0.2
        assert result.value is None
        with pytest.raises(ContractValidationError):
            scheduler.fast.run(event, state, (), conversation_history=("旧消息",))
        release.set()
        assert future.result(timeout=1) == "slow-done"
    finally:
        release.set()
        scheduler.close()


def test_container_exposes_orchestrator_without_rolling_context(tmp_path) -> None:
    app = build_container(tmp_path)
    try:
        event = EventEnvelope.user_message("mind-25", "hello", clock=FixedClock())
        with pytest.raises(ContractValidationError):
            app.orchestrator.handle(event, context(), conversation_history=("old",))
    finally:
        app.lifecycle.stop()


def test_action_result_reenters_cognition_before_return(tmp_path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "notes.txt").write_text("hello", encoding="utf-8")
    app = build_container(
        tmp_path / "data",
        tool_executor=FileReadToolExecutor(ToolExecutionPolicy((allowed,))),
    )
    action = ActionRequest(
        uuid4(), uuid4(), MIND, uuid4(), 1, "read", "file.read", {"path": "notes.txt"}, (), "stage25-action", NOW
    )
    decision = ActionDecision(
        uuid4(), action.action_id, ActionDecisionStatus.ALLOWED, "read", ("goal",), "user", ("none",), (), NOW, NOW + timedelta(hours=1), action.action_id
    )
    app.event_store.append(
        EventEnvelope(
            uuid4(), "action.decided", None, MIND, ActionDecisionPayload(action, decision), NOW, NOW,
            EventSource.SYSTEM, DataScope(MIND, DisclosureScope.MIND), causation_id=action.proposal_request_id,
        )
    )
    result = app.action.execute(action, decision, context())
    state = app.state_repository.load(MIND)
    assert result.result is not None and result.result.output["content"] == "hello"
    assert state is not None and any(field.startswith("procedural.action.") for field in state.entries)
    app.lifecycle.stop()
