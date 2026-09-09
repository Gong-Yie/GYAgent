from datetime import timedelta
from pathlib import Path
from uuid import UUID, uuid4

from self_cognition.bootstrap import build_container
from self_cognition.core.affect import Motive
from self_cognition.core.deletions import DeletionSelector
from self_cognition.core.events import EventEnvelope
from self_cognition.core.memories import MemoryType
from self_cognition.core.proactivity import ProactiveIntention
from self_cognition.core.scopes import SubjectScope
from self_cognition.core.time import SYSTEM_CLOCK
from self_cognition.runtime.run_context import RunContext


def _context() -> RunContext:
    now = SYSTEM_CLOCK.now()
    return RunContext(uuid4(), uuid4(), now + timedelta(minutes=5))


def test_explain_correction_export_and_audit(tmp_path: Path) -> None:
    app = build_container(tmp_path, dotenv_path=tmp_path / "missing.env")
    subject = SubjectScope.legacy_user("user-26")
    message = EventEnvelope.user_message(subject, "我喜欢晚上学习")
    app.process_event.process(message, _context())
    state = app.state_repository.load(subject)
    assert state is not None
    explanation = app.user_control.explain("state", "preferences.study_time", subject)
    assert explanation.evidence_ids
    corrected = app.user_control.correct(subject, target_field="preferences.study_time", cognition_type="preference", value="早上", context=_context())
    assert corrected.error_type is None
    exported = app.user_control.export(subject)
    payload = exported.path.read_text(encoding="utf-8")
    assert '"schema_version": 1' in payload
    assert exported.path.parent == tmp_path / "exports"
    assert app.user_control._governance.read_audit(subject)
    app.lifecycle.stop()


def test_retention_and_proactive_controls_propagate(tmp_path: Path) -> None:
    app = build_container(tmp_path, dotenv_path=tmp_path / "missing.env")
    subject = SubjectScope.legacy_user("user-26")
    now = SYSTEM_CLOCK.now()
    policy = app.user_control.set_retention(subject, retention_days=0, memory_types=(MemoryType.SEMANTIC,), now=now)
    app.user_control.apply_retention(policy, now=now + timedelta(days=1))
    proactive_subject = SubjectScope.for_mind(subject.mind.mind_id)
    source = EventEnvelope.user_message(subject, "触发提醒")
    app.event_store.append(source)
    motive = Motive(uuid4(), proactive_subject.subject.subject_id, "reminder", "提醒", 0.8, 1, now, (source.event_id,))
    intention = ProactiveIntention(uuid4(), motive, proactive_subject, "提醒", 1, 0, (), now, now + timedelta(hours=1), idempotency_key="stage26")
    app.proactive.propose(intention)
    app.user_control.disable_proactive_task(proactive_subject, "reminder")
    assert app.proactive.active(proactive_subject, as_of=now) == ()
    app.user_control.disable_proactive_channel(proactive_subject)
    app.user_control.set_proactive_frequency(proactive_subject, 0)
    plan = app.forget.dry_run(
        DeletionSelector(subject, delete_subject=True),
        now=now,
    )
    proactive_events = {
        event.event_id
        for event in app.event_store.read_by_subject(proactive_subject)
    }
    impacted_events = {
        event_id
        for impact in plan.effective_impacts
        for event_id in impact.event_ids
    }
    assert proactive_events.intersection(impacted_events)
    app.lifecycle.stop()
