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


def _context() -> RunContext:
    now = datetime.now(timezone.utc)
    return RunContext(uuid4(), uuid4(), now + timedelta(minutes=5))


def test_boredom_forms_social_connection_motive_once():
    store = InMemoryEventStore()
    service = ProactiveIntentionService(store)
    event = EventEnvelope.user_message("user-1", "好无聊，想找人说说话")
    workspace = WorkspacePacket(
        "user-1",
        0,
        (),
        fixed_context=WorkspaceFixedContext(
            emotion=(
                {
                    "field": "affect.reaction.interaction",
                    "content": {
                        "emotion": "boredom",
                        "valence": "negative",
                        "arousal": 0.1,
                    },
                },
            )
        ),
    )

    first = service.form_boredom_social_motive(
        event,
        workspace,
        _context(),
    )
    second = service.form_boredom_social_motive(
        event,
        workspace,
        _context(),
    )

    assert first is not None
    assert second is None
    assert first.payload.intention.motive.kind == "social_connection"


def test_non_boredom_does_not_form_social_motive():
    store = InMemoryEventStore()
    service = ProactiveIntentionService(store)
    event = EventEnvelope.user_message("user-1", "今天天气不错")
    workspace = WorkspacePacket(
        "user-1",
        0,
        (),
        fixed_context=WorkspaceFixedContext(
            emotion=(
                {
                    "field": "mood.current",
                    "content": {
                        "emotion": "neutral",
                        "valence": "neutral",
                        "arousal": 0.5,
                    },
                },
            )
        ),
    )

    assert service.form_boredom_social_motive(event, workspace, _context()) is None
class _FakeExpressionModel:
    def propose(self, *args, **kwargs):
        raise AssertionError("motive formation is not used by this test")

    def express(self, intention, workspace, context):
        return "我有点无聊，想和你聊聊天。"


def test_due_social_intention_uses_dynamic_expression():
    store = InMemoryEventStore()
    service = ProactiveIntentionService(store, model=_FakeExpressionModel())
    event = EventEnvelope.user_message("user-1", "好无聊，想找人说说话")
    workspace = WorkspacePacket(
        "user-1",
        0,
        (),
        fixed_context=WorkspaceFixedContext(
            emotion=(
                {
                    "field": "mood.current",
                    "content": {
                        "emotion": "boredom",
                        "valence": "negative",
                        "arousal": 0.1,
                    },
                },
            )
        ),
    )
    context = _context()

    service.form_boredom_social_motive(event, workspace, context)
    service.consume_due(
        event.subject,
        as_of=context.clock.now(),
        context=context,
        workspace=workspace,
    )

    messages = service.mailbox(event.subject, as_of=context.clock.now())
    assert messages
    assert messages[0]["text"] == "我有点无聊，想和你聊聊天。"
