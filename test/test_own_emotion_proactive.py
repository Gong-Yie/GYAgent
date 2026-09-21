from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from self_cognition.application.proactive import ProactiveIntentionService
from self_cognition.bootstrap import build_container
from self_cognition.core.events import EventEnvelope
from self_cognition.core.scopes import SubjectScope
from self_cognition.core.time import SYSTEM_CLOCK
from self_cognition.core.workspace import WorkspaceFixedContext, WorkspacePacket
from self_cognition.infrastructure.persistence.in_memory_event_store import (
    InMemoryEventStore,
)
from self_cognition.runtime.run_context import RunContext
from self_cognition.settings import ApplicationSettings


def _context() -> RunContext:
    now = datetime.now(timezone.utc)
    return RunContext(uuid4(), uuid4(), now + timedelta(minutes=5))


def _mind_workspace(emotion: str, arousal: float) -> WorkspacePacket:
    return WorkspacePacket(
        "mind",
        0,
        (),
        fixed_context=WorkspaceFixedContext(
            emotion=(
                {
                    "field": "affect.reaction.interaction",
                    "content": {
                        "emotion": emotion,
                        "valence": "negative",
                        "arousal": arousal,
                    },
                },
            )
        ),
    )


def _self_appraisal() -> EventEnvelope:
    source = EventEnvelope.user_message("user-1", "我最近有点忙")
    return EventEnvelope.assessment_requested(source, "自身情绪评估")


def test_own_emotion_forms_social_motive_targeting_user():
    store = InMemoryEventStore()
    service = ProactiveIntentionService(store)
    appraisal = _self_appraisal()
    recipient = SubjectScope.legacy_user("user-1")

    event = service.form_own_emotion_social_motive(
        appraisal,
        _mind_workspace("boredom", 0.1),
        _context(),
        target=recipient,
    )

    assert event is not None
    intention = event.payload.intention
    # 动机归属 Agent 自身（mind），投递目标为用户
    assert intention.motive.subject_id == appraisal.subject.mind.mind_id
    assert intention.target == recipient
    assert intention.motive.kind == "social_connection"
    # 动机必须带证据
    assert intention.evidence_refs


def test_own_emotion_motive_is_deduplicated_within_cooldown():
    store = InMemoryEventStore()
    service = ProactiveIntentionService(store)
    appraisal = _self_appraisal()
    recipient = SubjectScope.legacy_user("user-1")

    first = service.form_own_emotion_social_motive(
        appraisal, _mind_workspace("loneliness", 0.1), _context(), target=recipient
    )
    second = service.form_own_emotion_social_motive(
        appraisal, _mind_workspace("loneliness", 0.1), _context(), target=recipient
    )

    assert first is not None
    assert second is None


def test_own_emotion_without_social_need_forms_no_motive():
    store = InMemoryEventStore()
    service = ProactiveIntentionService(store)

    event = service.form_own_emotion_social_motive(
        _self_appraisal(),
        _mind_workspace("satisfaction", 0.6),
        _context(),
        target=SubjectScope.legacy_user("user-1"),
    )

    assert event is None


def _drain_until_idle(container, limit: int = 6) -> list:
    """像 worker 一样反复排空：用户消息之后还会有一次自我评估事件。"""
    drained = []
    for _ in range(limit):
        batch = container.event_bus.drain()
        drained.extend(batch)
        if not batch:
            break
    return drained


def _conversation(tmp_path: Path, text: str):
    data_dir = tmp_path / "data"
    container = build_container(
        data_dir,
        settings=ApplicationSettings(data_dir=data_dir, worker_enabled=False),
        dotenv_path=tmp_path / "missing.env",
    )
    user = SubjectScope.legacy_user("user-1")
    context = RunContext(
        uuid4(),
        uuid4(),
        SYSTEM_CLOCK.now() + timedelta(minutes=5),
    )
    event = EventEnvelope.user_message(
        user,
        text,
        run_id=context.run_id,
        correlation_id=context.correlation_id,
    )
    container.event_bus.publish(event, context)
    return container, user, event, context


def test_user_boredom_reaches_a_proactive_mailbox_addressed_to_that_user(
    tmp_path: Path,
) -> None:
    """完整链路：用户表达无聊 → Agent 自己产生无聊反应 → 主动联系该用户。"""
    container, user, event, context = _conversation(tmp_path, "好无聊，想找个人聊聊天")

    drained = _drain_until_idle(container)
    assert len(drained) == 2
    assert all(item.status.value == "succeeded" for item in drained)

    mind = SubjectScope.for_mind(user.mind.mind_id)
    mind_state = container.state_repository.load(mind)
    assert mind_state is not None
    reaction = mind_state.get("affect.reaction.interaction").value
    # 情绪属于 Agent 自己，对象是触发它的用户
    assert reaction["emotion"] == "boredom"
    assert reaction["target"] == user.subject.subject_id
    assert reaction["cause"] == "user.message"

    now = SYSTEM_CLOCK.now()
    container.proactive.consume_due(
        user,
        as_of=now,
        context=context,
        workspace=container.workspace_builder.build(event.payload.text, mind_state),
    )

    intentions = container.proactive.active(user, as_of=now)
    assert len(intentions) == 1
    # 动机由 mind 形成，主动接触的是用户而不是 Agent 自己
    assert intentions[0].motive.subject_id == mind.mind.mind_id
    assert intentions[0].target == user
    assert intentions[0].evidence_refs

    messages = container.proactive.mailbox(user, as_of=now)
    assert len(messages) == 1
    assert messages[0]["status"] == "pending"


def test_message_without_a_recognisable_emotion_stays_silent(
    tmp_path: Path,
) -> None:
    """沉默负例：没有情绪反应时既不形成动机，也不产生主动消息。"""
    container, user, event, context = _conversation(tmp_path, "我喜欢晚上学习")

    drained = _drain_until_idle(container)
    assert len(drained) >= 1

    mind = SubjectScope.for_mind(user.mind.mind_id)
    mind_state = container.state_repository.load(mind)
    assert mind_state is None or not any(
        field.startswith(("affect.", "mood.")) for field in mind_state.entries
    )

    now = SYSTEM_CLOCK.now()
    container.proactive.consume_due(
        user,
        as_of=now,
        context=context,
        workspace=None,
    )

    assert container.proactive.active(user, as_of=now) == ()
    assert container.proactive.mailbox(user, as_of=now) == ()
