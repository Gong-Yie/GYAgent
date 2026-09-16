from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta
from pathlib import Path
from uuid import UUID, uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from self_cognition.bootstrap import build_container
from self_cognition.core.events import EventEnvelope
from self_cognition.core.runs import RunKind
from self_cognition.core.scopes import SubjectScope
from self_cognition.core.time import SYSTEM_CLOCK
from self_cognition.runtime.run_context import RunContext
from self_cognition.settings import ApplicationSettings


def _settings(data_dir: Path) -> ApplicationSettings:
    return ApplicationSettings(data_dir=data_dir, worker_enabled=False)


def _context(run_id: UUID) -> RunContext:
    return RunContext(
        run_id,
        uuid4(),
        SYSTEM_CLOCK.now() + timedelta(hours=1),
    )


def _container(data_dir: Path, dotenv_path: Path):
    return build_container(
        data_dir,
        settings=_settings(data_dir),
        dotenv_path=dotenv_path,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reliability probe for force-kill recovery")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    pending = subparsers.add_parser("kill-pending")
    pending.add_argument("--data-dir", type=Path, required=True)
    pending.add_argument("--dotenv", type=Path, default=PROJECT_ROOT / ".env")

    running = subparsers.add_parser("kill-running")
    running.add_argument("--data-dir", type=Path, required=True)
    running.add_argument("--dotenv", type=Path, default=PROJECT_ROOT / ".env")
    running.add_argument("--run-id", type=UUID, required=True)

    args = parser.parse_args(argv)

    if args.mode == "kill-pending":
        container = _container(args.data_dir, args.dotenv)
        subject = SubjectScope.legacy_user("kill-user")
        run_id = uuid4()
        context = _context(run_id)
        event = EventEnvelope.user_message(
            subject,
            "我喜欢晚上学习",
            run_id=run_id,
            correlation_id=context.correlation_id,
        )
        container.event_bus.publish(event, context)
        os._exit(99)

    if args.mode == "kill-running":
        container = _container(args.data_dir, args.dotenv)
        subject = SubjectScope.legacy_user("kill-user")
        context = _context(args.run_id)
        container.run_lifecycle.begin(
            context,
            RunKind.COGNITIVE_CYCLE,
            subject,
            input_event_ids=(),
            wake_reason="reliability-probe",
        )
        os._exit(99)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
