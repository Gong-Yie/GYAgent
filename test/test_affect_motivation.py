from datetime import datetime, timedelta, timezone
from uuid import uuid4

from self_cognition.application.proactive import ProactiveIntentionService
from self_cognition.core.events import EventEnvelope
from self_cognition.core.scopes import SubjectScope
from self_cognition.core.workspace import WorkspaceFixedContext, WorkspacePacket
from self_cognition.infrastructure.persistence.in_memory_event_store import (
    InMemoryEventStore,
)
from self_cognition.runtime.run_context import RunContext

USER = SubjectScope.legacy_user("user-1")


def _context() -> RunContext:
    now = datetime.now(timezone.utc)
    return RunContext(uuid4(), uuid4(), now + timedelta(minutes=5))


def _self_appraisal() -> EventEnvelope:
    """Agent 评估自身反应：subject 是 MIND，不是用户。"""
    source = EventEnvelope.user_message("user-1", "我最近有点忙")
    return EventEnvelope.assessment_requested(source, "评估自身情绪")


def _mind_workspace(emotion: str, valence: str, arousal: float) -> WorkspacePacket:
    """从 Agent 自身状态投影出的 workspace。"""
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
                        "valence": valence,
                        "arousal": arousal,
                    },
                },
            )
        ),
    )


def test_own_boredom_forms_social_connection_motive_once():
    store = InMemoryEventStore()
    service = ProactiveIntentionService(store)
    appraisal = _self_appraisal()
    workspace = _mind_workspace("boredom", "negative", 0.1)

    first = service.form_own_emotion_social_motive(
        appraisal, workspace, _context(), target=USER
    )
    second = service.form_own_emotion_social_motive(
        appraisal, workspace, _context(), target=USER
    )

    assert first is not None
    assert second is None
    intention = first.payload.intention
    assert intention.motive.kind == "social_connection"
    # 情绪来自 Agent 自身：动机归属 mind
    assert intention.motive.subject_id == appraisal.subject.mind.mind_id
    # 但主动接触的对象是用户，不是 Agent 自己
    assert intention.target == USER
    assert intention.target != appraisal.subject
    # 动机必须带证据
    assert intention.evidence_refs


def test_non_boredom_does_not_form_social_motive():
    store = InMemoryEventStore()
    service = ProactiveIntentionService(store)

    assert (
        service.form_own_emotion_social_motive(
            _self_appraisal(),
            _mind_workspace("neutral", "neutral", 0.5),
            _context(),
            target=USER,
        )
        is None
    )


class _FakeExpressionModel:
    def propose(self, *args, **kwargs):
        raise AssertionError("motive formation is not used by this test")

    def express(self, intention, workspace, context):
        return "我有点无聊，想和你聊聊天。"


def test_due_social_intention_uses_dynamic_expression():
    store = InMemoryEventStore()
    service = ProactiveIntentionService(store, model=_FakeExpressionModel())
    appraisal = _self_appraisal()
    workspace = _mind_workspace("boredom", "negative", 0.1)
    context = _context()

    service.form_own_emotion_social_motive(
        appraisal, workspace, context, target=USER
    )
    service.consume_due(
        appraisal.subject,
        as_of=context.clock.now(),
        context=context,
        workspace=workspace,
    )

    messages = service.mailbox(appraisal.subject, as_of=context.clock.now())
    assert messages
    assert messages[0]["text"] == "我有点无聊，想和你聊聊天。"
