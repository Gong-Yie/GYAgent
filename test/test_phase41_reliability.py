from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from threading import Event
from uuid import UUID, uuid4

from self_cognition.application.results import ProcessEventStatus
from self_cognition.bootstrap import build_container
from self_cognition.cognition.registry import ModuleRegistration
from self_cognition.core.errors import RunCancelledError
from self_cognition.core.events import EventEnvelope
from self_cognition.core.runs import RunKind, RunStatus
from self_cognition.core.scopes import SubjectScope
from self_cognition.core.time import SYSTEM_CLOCK
from self_cognition.runtime.run_context import RunContext
from self_cognition.settings import ApplicationSettings


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "reliability_probe.py"


def _run_probe(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PROBE), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )


class BlockingCancellableModule:
    module_id = "test.blocking_cancel"
    module_version = "1"
    deterministic = True
    subscriptions = frozenset({"user.message"})

    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()

    def run(self, request):
        self.started.set()
        while not request.run_context.is_cancelled:
            self.release.wait(0.01)
        raise RunCancelledError("cancellation observed by test module")


def test_force_kill_pending_event_recovers_and_drains(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    dotenv = tmp_path / "missing.env"

    result = _run_probe(
        ["kill-pending", "--data-dir", str(data_dir), "--dotenv", str(dotenv)]
    )
    assert result.returncode == 99, result.stderr

    container = build_container(
        data_dir,
        settings=ApplicationSettings(data_dir=data_dir, worker_enabled=False),
        dotenv_path=dotenv,
    )
    subject = SubjectScope.legacy_user("kill-user")

    assert len(container.event_bus.backlog()) == 1
    drained = container.event_bus.drain()
    assert len(drained) == 1
    assert drained[0].status is ProcessEventStatus.SUCCEEDED

    state = container.state_repository.load(subject)
    assert state is not None
    assert state.version == 1
    assert container.event_bus.backlog() == ()


def test_force_kill_running_run_is_recovered_as_interrupted(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    dotenv = tmp_path / "missing.env"
    run_id = uuid4()

    result = _run_probe(
        [
            "kill-running",
            "--data-dir",
            str(data_dir),
            "--dotenv",
            str(dotenv),
            "--run-id",
            str(run_id),
        ]
    )
    assert result.returncode == 99, result.stderr

    container = build_container(
        data_dir,
        settings=ApplicationSettings(data_dir=data_dir, worker_enabled=False),
        dotenv_path=dotenv,
    )
    record = container.run_repository.get(run_id)

    assert record is not None
    assert record.status is RunStatus.INTERRUPTED


def test_running_worker_cancellation_propagates(tmp_path: Path) -> None:
    module = BlockingCancellableModule()
    settings = ApplicationSettings(
        data_dir=tmp_path,
        worker_enabled=True,
        worker_poll_interval_seconds=0.01,
        worker_lease_timeout_seconds=30.0,
    )
    container = build_container(
        tmp_path,
        settings=settings,
        dotenv_path=tmp_path / "missing.env",
        module_registrations=(
            ModuleRegistration(
                module.module_id,
                "test",
                module.module_version,
                module,
            ),
        ),
    )
    subject = SubjectScope.legacy_user("cancel-user")
    run_id = uuid4()
    context = RunContext(
        run_id,
        uuid4(),
        SYSTEM_CLOCK.now() + timedelta(minutes=5),
    )
    event = EventEnvelope.user_message(
        subject,
        "cancel me",
        run_id=run_id,
        correlation_id=context.correlation_id,
    )

    try:
        container.event_bus.publish(event, context)
        container.lifecycle.start()
        assert module.started.wait(5.0), "blocking module did not start"

        container.run_lifecycle.request_cancel(
            run_id,
            reason="test cancellation",
            subject=subject,
        )

        deadline = time.monotonic() + 5.0
        record = None
        while time.monotonic() < deadline:
            record = container.run_repository.get(run_id)
            if record is not None and record.status is RunStatus.CANCELLED:
                break
            time.sleep(0.05)

        assert record is not None
        assert record.status is RunStatus.CANCELLED
    finally:
        container.lifecycle.stop(5.0)

    assert container.event_bus.backlog() == ()


def test_affect_and_proactive_state_survive_restart(tmp_path: Path) -> None:
    dotenv = tmp_path / "missing.env"
    settings = ApplicationSettings(data_dir=tmp_path, worker_enabled=False)
    container = build_container(tmp_path, settings=settings, dotenv_path=dotenv)
    subject = SubjectScope.legacy_user("restart-user")
    context = RunContext(
        uuid4(),
        uuid4(),
        SYSTEM_CLOCK.now() + timedelta(minutes=5),
    )
    event = EventEnvelope.user_message(
        subject,
        "好无聊，想找个人聊聊天",
        run_id=context.run_id,
        correlation_id=context.correlation_id,
    )

    container.event_bus.publish(event, context)
    drained = container.event_bus.drain()
    assert len(drained) == 1
    assert drained[0].status is ProcessEventStatus.SUCCEEDED

    state = container.state_repository.load(subject)
    assert state is not None
    now = SYSTEM_CLOCK.now()
    workspace = container.workspace_builder.build(event.payload.text, state)
    container.proactive.consume_due(
        subject,
        as_of=now,
        context=context,
        workspace=workspace,
    )

    affect_before = container.affect_view.view(subject, as_of=now)
    active_before = [
        intention.intention_id
        for intention in container.proactive.active(subject, as_of=now)
    ]
    mailbox_before = [
        item["message_id"]
        for item in container.proactive.mailbox(subject, as_of=now)
    ]
    assert mailbox_before

    restarted = build_container(
        tmp_path,
        settings=ApplicationSettings(data_dir=tmp_path, worker_enabled=False),
        dotenv_path=dotenv,
    )
    affect_after = restarted.affect_view.view(subject, as_of=now)
    active_after = [
        intention.intention_id
        for intention in restarted.proactive.active(subject, as_of=now)
    ]
    mailbox_after = [
        item["message_id"]
        for item in restarted.proactive.mailbox(subject, as_of=now)
    ]

    assert affect_after == affect_before
    assert active_after == active_before
    assert mailbox_after == mailbox_before

def test_multi_process_workers_shard_subjects_without_duplicate_processing(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    dotenv = tmp_path / "missing.env"
    settings = ApplicationSettings(data_dir=data_dir, worker_enabled=False)
    producer = build_container(data_dir, settings=settings, dotenv_path=dotenv)
    subjects = [
        SubjectScope.legacy_user(f"mp-{index}")
        for index in range(3)
    ]

    for round_index in range(3):
        for subject in subjects:
            context = RunContext(
                uuid4(),
                uuid4(),
                SYSTEM_CLOCK.now() + timedelta(minutes=5),
            )
            text = "我喜欢晚上学习" if round_index != 1 else "我喜欢早上学习"
            event = EventEnvelope.user_message(
                subject,
                text,
                run_id=context.run_id,
                correlation_id=context.correlation_id,
            )
            producer.event_bus.publish(event, context)

    processes: list[subprocess.Popen[str]] = []
    try:
        for shard_index in range(3):
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        str(PROBE),
                        "drain",
                        "--data-dir",
                        str(data_dir),
                        "--dotenv",
                        str(dotenv),
                        "--shard-index",
                        str(shard_index),
                        "--shard-count",
                        "3",
                        "--idle-seconds",
                        "0.5",
                        "--max-seconds",
                        "20",
                    ],
                    cwd=ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env={**os.environ, "PYTHONPATH": "src"},
                )
            )

        for process in processes:
            _, stderr = process.communicate(timeout=25)
            assert process.returncode == 0, stderr
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)

    restarted = build_container(data_dir, settings=settings, dotenv_path=dotenv)
    for subject in subjects:
        state = restarted.state_repository.load(subject)
        assert state is not None
        assert state.version == 3

    assert restarted.event_bus.backlog() == ()
    assert restarted.event_bus.dead_letters() == ()

    reduced = [
        event
        for event in restarted.event_store.read_all()
        if event.event_type == "state.reduced" and event.subject in subjects
    ]
    assert len(reduced) == 9
