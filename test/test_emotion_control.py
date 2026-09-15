from datetime import datetime, timedelta, timezone
from uuid import uuid4

from self_cognition.bootstrap import build_container
from self_cognition.core.deletions import DeletionSelector, DeletionStatus
from self_cognition.core.events import EventEnvelope
from self_cognition.runtime.run_context import RunContext


def _context() -> RunContext:
    now = datetime.now(timezone.utc)
    return RunContext(uuid4(), uuid4(), now + timedelta(minutes=5))


def test_affect_module_can_be_disabled_per_subject(tmp_path):
    app = build_container(tmp_path, dotenv_path=tmp_path / "missing.env")
    event = EventEnvelope.user_message("user-1", "好无聊，想找个人聊聊天")
    try:
        app.user_control.disable_module(event.subject, "affect.fast_reaction")
        result = app.process_event.process(event, _context())
        state = app.state_repository.load(event.subject)
    finally:
        app.lifecycle.stop()

    assert result.status.value == "succeeded"
    assert state is None or "affect.reaction.interaction" not in state.entries


def test_export_includes_emotion_and_mood_projection(tmp_path):
    app = build_container(tmp_path, dotenv_path=tmp_path / "missing.env")
    event = EventEnvelope.user_message("user-1", "好无聊，想找个人聊聊天")
    try:
        app.process_event.process(event, _context())
        exported = app.user_control.export(event.subject)
        text = exported.path.read_text(encoding="utf-8")
    finally:
        app.lifecycle.stop()

    assert "affect.reaction.interaction" in text
    assert "mood.current" in text


def test_subject_deletion_removes_emotion_and_mood(tmp_path):
    app = build_container(tmp_path, dotenv_path=tmp_path / "missing.env")
    event = EventEnvelope.user_message("user-1", "好无聊，想找个人聊聊天")
    now = datetime.now(timezone.utc)
    try:
        app.process_event.process(event, _context())
        before = app.state_repository.load(event.subject)
        assert before is not None
        assert "mood.current" in before.entries

        plan = app.forget.dry_run(
            DeletionSelector(event.subject, delete_subject=True),
            now=now,
        )
        result = app.forget.execute(plan, now=now)
    finally:
        app.lifecycle.stop()

    assert result.status is DeletionStatus.COMPLETED
    assert app.state_repository.load(event.subject) is None