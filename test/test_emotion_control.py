from datetime import datetime, timedelta, timezone
from uuid import uuid4

from self_cognition.bootstrap import build_container
from self_cognition.core.affect import EmotionState
from self_cognition.core.deletions import DeletionSelector, DeletionStatus
from self_cognition.core.events import EventEnvelope
from self_cognition.core.scopes import SubjectScope
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

def test_affect_control_closes_and_reopens_emotion_modules(tmp_path):
    app = build_container(tmp_path, dotenv_path=tmp_path / "missing.env")
    user = "affect-control-user"
    subject = SubjectScope.legacy_user(user)
    try:
        closed = app.affect_control.close(subject)
        assert "affect.fast_reaction" in closed.disabled_modules
        assert "affect.affect_extractor" in closed.disabled_modules

        disabled_event = EventEnvelope.user_message(
            user,
            "好无聊，想找个人聊聊天",
        )
        result = app.process_event.process(disabled_event, _context())
        assert result.status.value == "succeeded"
        disabled_state = app.state_repository.load(subject)
        assert disabled_state is None or not any(
            field.startswith(("affect.", "mood."))
            for field in disabled_state.entries
        )

        reopened = app.affect_control.open(subject)
        assert "affect.fast_reaction" not in reopened.disabled_modules
        assert "affect.affect_extractor" not in reopened.disabled_modules

        enabled_event = EventEnvelope.user_message(
            user,
            "好无聊，想找个人聊聊天",
        )
        app.process_event.process(enabled_event, _context())
        enabled_state = app.state_repository.load(subject)
    finally:
        app.lifecycle.stop()

    assert enabled_state is not None
    assert "affect.reaction.interaction" in enabled_state.entries
    assert "mood.current" in enabled_state.entries


def test_affect_control_corrects_emotion_and_view_sees_it(tmp_path):
    app = build_container(tmp_path, dotenv_path=tmp_path / "missing.env")
    user = "affect-correct-user"
    event = EventEnvelope.user_message(user, "好无聊，想找个人聊聊天")
    corrected = EmotionState(
        emotion_id=uuid4(),
        target=user,
        emotion="calm",
        valence="positive",
        scope="interaction",
        intensity=0.9,
        assessed_at=datetime.now(timezone.utc),
        half_life_seconds=3600.0,
        arousal=0.2,
        control=0.8,
        certainty=0.7,
        cause="user correction",
    )
    try:
        app.process_event.process(event, _context())
        result = app.affect_control.correct(
            event.subject,
            field="affect.reaction.interaction",
            value=corrected.to_state_value(),
            context=_context(),
        )
        state = app.state_repository.load(event.subject)
        view = app.affect_view.view(
            event.subject,
            as_of=datetime.now(timezone.utc),
        )
    finally:
        app.lifecycle.stop()

    assert result.status.value == "succeeded"
    assert state is not None
    assert (
        state.entries["affect.reaction.interaction"].value["emotion"]
        == "calm"
    )
    assert any(
        item["field"] == "affect.reaction.interaction"
        and item["content"]["emotion"] == "calm"
        for item in view["emotions"]
    )
