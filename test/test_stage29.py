import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from self_cognition.bootstrap import build_container
from self_cognition.interfaces.cli import main
from self_cognition.interfaces.http.server import _handle
from self_cognition.settings import ApplicationSettings
from self_cognition.core.affect import Motive
from self_cognition.core.dialogue import DialogueRequest
from self_cognition.core.events import EventEnvelope
from self_cognition.core.proactivity import (
    BehavioralAction,
    MotiveProposal,
    ProactiveIntention,
)
from self_cognition.core.scopes import SubjectScope
from self_cognition.core.workspace import WorkspacePacket
from self_cognition.infrastructure.persistence.file_process_journal import FileProcessJournal
from self_cognition.infrastructure.persistence.serialization import (
    event_from_json,
    event_to_json,
)
from self_cognition.runtime.run_context import RunContext


NOW = datetime(2026, 9, 9, 16, tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class FixedClock:
    value: datetime = NOW

    def now(self) -> datetime:
        return self.value


def _context() -> RunContext:
    return RunContext(
        uuid4(),
        uuid4(),
        NOW + timedelta(minutes=5),
        clock=FixedClock(),
    )


def test_due_proactive_intention_is_consumed_once(tmp_path: Path) -> None:
    app = build_container(tmp_path / "data", dotenv_path=tmp_path / "missing.env")
    subject = SubjectScope.for_mind("mind-due")
    intention = ProactiveIntention(
        uuid4(),
        Motive(uuid4(), subject.mind.mind_id, "goal", "跟进目标", 0.8, 1, NOW),
        subject,
        "询问目标进展",
        1,
        0,
        (),
        NOW,
        NOW + timedelta(hours=1),
        idempotency_key="due-once",
    )
    app.proactive.propose(intention, context=_context())

    first = app.proactive.consume_due(
        subject,
        as_of=NOW + timedelta(minutes=1),
        context=_context(),
    )
    second = app.proactive.consume_due(
        subject,
        as_of=NOW + timedelta(minutes=1),
        context=_context(),
    )

    assert len(first) == 1
    assert first[0].decision.action is BehavioralAction.ACCEPT
    assert first[0].reused is False
    assert second == ()


def test_reminder_request_reaches_mailbox_and_acknowledges_once(tmp_path: Path) -> None:
    app = build_container(tmp_path / "data", dotenv_path=tmp_path / "missing.env")
    subject = SubjectScope.legacy_user("mailbox-user")
    event = EventEnvelope.user_message(subject, "提醒我提交论文", clock=FixedClock())

    proposed = app.proactive.observe_event(event, context=_context())
    assert proposed is not None
    intention = next(
        item
        for item in app.proactive.active(subject, as_of=NOW)
        if item.expected_behavior == "提交论文"
    )
    app.proactive.consume_due(subject, as_of=NOW, context=_context())

    pending = app.proactive.mailbox(subject, as_of=NOW)
    assert pending[0]["status"] == "pending"
    first = app.proactive.acknowledge(subject, intention.intention_id, context=_context())
    second = app.proactive.acknowledge(subject, intention.intention_id, context=_context())

    assert first.decision.action is BehavioralAction.EXECUTE
    assert second.reused is True
    assert app.proactive.mailbox(subject, as_of=NOW)[0]["status"] == "acknowledged"


def test_mailbox_event_stream_survives_restart_and_cancel(tmp_path: Path) -> None:
    data = tmp_path / "data"
    app = build_container(data, dotenv_path=tmp_path / "missing.env")
    subject = SubjectScope.legacy_user("mailbox-restart-user")
    source = EventEnvelope.user_message(subject, "提醒我检查收件箱", clock=FixedClock())
    app.proactive.observe_event(source, context=_context())
    intention = app.proactive.active(subject, as_of=NOW)[0]
    app.proactive.consume_due(subject, as_of=NOW, context=_context())
    created = next(
        event
        for event in app.event_store.read_by_mind(subject.mind)
        if event.event_type == "proactive.message.created"
    )
    assert event_from_json(event_to_json(created)) == created
    app.proactive.cancel(subject, intention.intention_id, "user cancelled", context=_context())
    assert app.proactive.mailbox(subject, as_of=NOW)[0]["status"] == "cancelled"
    app.lifecycle.stop()

    restarted = build_container(data, dotenv_path=tmp_path / "missing.env")
    try:
        assert restarted.proactive.mailbox(subject, as_of=NOW)[0]["status"] == "cancelled"
    finally:
        restarted.lifecycle.stop()


def test_mailbox_expiry_event_updates_status(tmp_path: Path) -> None:
    app = build_container(tmp_path / "data", dotenv_path=tmp_path / "missing.env")
    subject = SubjectScope.legacy_user("mailbox-expiry-user")
    source = EventEnvelope.user_message(subject, "提醒我检查截止日期", clock=FixedClock())
    app.proactive.observe_event(source, context=_context())
    intention = app.proactive.active(subject, as_of=NOW)[0]
    app.proactive.consume_due(subject, as_of=NOW, context=_context())
    app.proactive.expire(
        subject,
        intention.intention_id,
        as_of=NOW + timedelta(days=2),
        context=_context(),
    )
    try:
        assert app.proactive.mailbox(subject, as_of=NOW)[0]["status"] == "expired"
    finally:
        app.lifecycle.stop()


def test_cli_and_http_negative_proactivity_stay_silent(tmp_path, capsys) -> None:
    class SilentProactivityModel:
        def propose(self, event, workspace, context):
            del event, workspace, context
            return MotiveProposal(False, "", "", "", 0.0, 0, 1, ())

    settings = ApplicationSettings(
        data_dir=tmp_path / "data",
        worker_enabled=True,
        worker_poll_interval_seconds=0.01,
    )
    app = build_container(
        settings=settings,
        dotenv_path=tmp_path / "missing.env",
        proactive_model=SilentProactivityModel(),
    )
    subject = SubjectScope.legacy_user("silent-entry-user")
    try:
        assert main(["chat", subject.subject.subject_id, "项目一切正常"], container=app) == 0
        cli_output = capsys.readouterr().out
        app.event_bus.drain(clock=FixedClock())
        assert app.proactive.mailbox(subject, as_of=NOW) == ()

        http_output = _handle(
            app,
            "POST",
            "/chat",
            {},
            {"subject_id": subject.subject.subject_id, "message": "没有需要跟进的变化"},
        )
        assert http_output["status"] == "accepted"
        app.event_bus.drain(clock=FixedClock())
        assert app.proactive.mailbox(subject, as_of=NOW) == ()
        assert json.loads(cli_output)["slow_pending"] is True
    finally:
        app.lifecycle.stop()


def test_shutdown_propagates_to_cli_entry(tmp_path, capsys) -> None:
    app = build_container(tmp_path / "data", dotenv_path=tmp_path / "missing.env")
    app.lifecycle.start()
    app.lifecycle.stop(1)

    try:
        assert main(["chat", "closed-entry-user", "测试关闭传播"], container=app) == 1
        assert json.loads(capsys.readouterr().err)["error_type"] == "RuntimeError"
        try:
            _handle(
                app,
                "POST",
                "/chat",
                {},
                {"subject_id": "closed-entry-user", "message": "测试 HTTP 关闭传播"},
            )
        except RuntimeError as error:
            assert str(error) == "application lifecycle is closed"
        else:
            raise AssertionError("closed HTTP chat must be rejected")
    finally:
        app.lifecycle.stop()


def test_cli_commands_do_not_close_shared_container(tmp_path, capsys) -> None:
    app = build_container(tmp_path / "data", dotenv_path=tmp_path / "missing.env")
    subject = SubjectScope.legacy_user("shared-cli-user")
    try:
        assert main(["proactive", subject.subject.subject_id], container=app) == 0
        capsys.readouterr()
        assert app.lifecycle.is_closed is False
        assert main(["mailbox", subject.subject.subject_id], container=app) == 0
        capsys.readouterr()
        assert app.lifecycle.is_closed is False
        app.event_bus.publish(
            EventEnvelope.user_message(subject, "提醒我继续检查", clock=FixedClock()),
            _context(),
        )
    finally:
        app.lifecycle.stop()


def test_worker_reclaims_claimed_outbox_after_forced_stop(tmp_path) -> None:
    data = tmp_path / "data"
    app = build_container(data, dotenv_path=tmp_path / "missing.env")
    subject = SubjectScope.legacy_user("worker-recovery-user")
    context = _context()
    event = EventEnvelope.user_message(
        subject,
        "提醒我恢复任务",
        clock=FixedClock(),
        run_id=context.run_id,
        correlation_id=context.correlation_id,
    )
    app.event_bus.publish(event, context)
    journal = FileProcessJournal(data / "processing")
    claimed = journal.claim(event.event_id, context.run_id, NOW, timedelta(seconds=30))
    assert claimed is not None
    app.lifecycle.stop()

    restarted = build_container(data, dotenv_path=tmp_path / "missing.env")
    try:
        results = restarted.event_bus.drain(clock=FixedClock(NOW + timedelta(seconds=31)))
        assert len(results) == 1
        assert restarted.event_store.read_by_subject(subject)
    finally:
        restarted.lifecycle.stop()


def test_model_proposal_forms_evidence_bound_motive(tmp_path: Path) -> None:
    class FakeProactivityModel:
        def propose(self, event, workspace, context):
            del workspace, context
            return MotiveProposal(
                True,
                "goal",
                "跟进未完成目标",
                "询问目标进展",
                0.8,
                1,
                3600,
                (str(event.event_id),),
            )

    app = build_container(
        tmp_path / "data",
        dotenv_path=tmp_path / "missing.env",
        proactive_model=FakeProactivityModel(),
    )
    subject = SubjectScope.legacy_user("model-user")
    event = EventEnvelope.user_message(subject, "项目还没有完成", clock=FixedClock())
    created = app.proactive.evaluate(
        event,
        WorkspacePacket(subject.subject.subject_id, 0, ()),
        _context(),
    )

    assert created is not None
    intentions = app.proactive.active(subject, as_of=NOW)
    assert len(intentions) == 1
    assert intentions[0].motive.source_event_ids == (event.event_id,)


def test_restart_preserves_state_and_intention_without_rolling_context(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    app = build_container(data, dotenv_path=tmp_path / "missing.env")
    user = SubjectScope.legacy_user("stage29-user")
    mind = SubjectScope.for_mind(user.mind.mind_id)
    for text in (
        "我喜欢晚上学习",
        "小明是我的朋友",
        "这次考试通过了，我很开心",
    ):
        result = app.process_event.process(
            EventEnvelope.user_message(user, text, clock=FixedClock()),
            _context(),
        )
        assert result.error_type is None
    source = EventEnvelope.user_message(user, "记得跟进目标", clock=FixedClock())
    assert app.process_event.process(source, _context()).error_type is None
    motive = Motive(
        uuid4(),
        mind.subject.subject_id,
        "goal",
        "跟进未完成目标",
        0.8,
        1,
        NOW,
        (source.event_id,),
    )
    intention = ProactiveIntention(
        uuid4(),
        motive,
        mind,
        "询问目标进展",
        1,
        0,
        (),
        NOW,
        NOW + timedelta(hours=1),
        idempotency_key="stage29-follow-up",
    )
    app.proactive.propose(intention, context=_context())
    expected_user_state = app.state_repository.load(user)
    expected_intentions = app.proactive.active(mind, as_of=NOW)
    app.lifecycle.stop()

    restarted = build_container(data, dotenv_path=tmp_path / "missing.env")
    try:
        assert restarted.state_repository.load(user) == expected_user_state
        assert restarted.replay.replay(user) == expected_user_state
        assert restarted.proactive.active(mind, as_of=NOW) == expected_intentions
        current = EventEnvelope.user_message(user, "现在呢？", clock=FixedClock())
        fast = restarted.scheduler.fast.run(
            current,
            expected_user_state,
            expected_intentions,
        )
        assert fast.intentions == expected_intentions
    finally:
        restarted.lifecycle.stop()


def test_legacy_workspace_and_stage20_answer_event_remain_readable(
    tmp_path: Path,
) -> None:
    legacy_workspace = WorkspacePacket("stage20-user", 0, ())
    assert legacy_workspace.workspace_version == 1
    assert legacy_workspace.subject is None

    app = build_container(tmp_path, dotenv_path=tmp_path / "missing.env")
    try:
        user = SubjectScope.legacy_user("stage20-user")
        result = app.converse.converse(
            DialogueRequest(
                EventEnvelope.user_message(user, "你好！", clock=FixedClock())
            ),
            _context(),
        )
        answer = next(
            event
            for event in app.event_store.read_by_mind(user.mind)
            if event.event_id == result.response_event_id
        )
        assert event_from_json(event_to_json(answer)) == answer
    finally:
        app.lifecycle.stop()
