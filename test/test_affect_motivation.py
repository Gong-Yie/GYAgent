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