from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from self_cognition.bootstrap import build_container
from self_cognition.core.events import EventEnvelope
from self_cognition.core.scopes import SubjectScope
from self_cognition.core.time import SYSTEM_CLOCK
from self_cognition.runtime.run_context import RunContext
from self_cognition.settings import ApplicationSettings


def _context() -> RunContext:
    return RunContext(
        uuid4(),
        uuid4(),
        SYSTEM_CLOCK.now() + timedelta(minutes=5),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure local event/query/state latency")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dotenv", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args(argv)
    if args.samples < 1:
        raise ValueError("samples must be positive")

    settings = ApplicationSettings(data_dir=args.data_dir, worker_enabled=False)
    container = build_container(
        args.data_dir,
        settings=settings,
        dotenv_path=args.dotenv,
    )
    metrics = container.metrics
    subjects = [
        SubjectScope.legacy_user(f"latency-{index}")
        for index in range(args.samples)
    ]

    for subject in subjects:
        context = _context()
        event = EventEnvelope.user_message(
            subject,
            "我喜欢晚上学习",
            run_id=context.run_id,
            correlation_id=context.correlation_id,
        )
        with metrics.time("local.event_ingest.latency_seconds"):
            container.event_bus.publish(event, context)

    while container.event_bus.backlog():
        container.event_bus.drain()

    for subject in subjects:
        with metrics.time("local.query.latency_seconds"):
            state = container.state_repository.load(subject)
        if state is None:
            raise RuntimeError("latency sample state was not persisted")
        with metrics.time("local.state_write.latency_seconds"):
            container.state_repository.replace(state)

    metric_names = (
        "local.event_ingest.latency_seconds",
        "local.query.latency_seconds",
        "local.state_write.latency_seconds",
    )
    snapshot = metrics.snapshot()
    report: dict[str, object] = {
        "samples": args.samples,
        "latency_seconds": {
            name: {
                "p50": metrics.percentile(name, 0.50),
                "p95": metrics.percentile(name, 0.95),
                "p99": metrics.percentile(name, 0.99),
            }
            for name in metric_names
        },
        "timing_counts": {
            name: len(snapshot.timings.get(name, ()))
            for name in metric_names
        },
        "counters": dict(snapshot.counters),
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
