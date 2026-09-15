from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta, timezone
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
from self_cognition.settings import load_settings


NEUTRAL_MESSAGES = (
    "你好，我只是来看看",
    "今天过得还行",
    "好的，我知道了",
    "嗯，先这样",
)


def _parse_minutes(value: str) -> tuple[int, ...]:
    return tuple(
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    )


def _context() -> RunContext:
    return RunContext(
        uuid4(),
        uuid4(),
        SYSTEM_CLOCK.now() + timedelta(minutes=5),
    )


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _snapshot(container, subject: SubjectScope, sent: int, elapsed: float) -> dict[str, object]:
    now = SYSTEM_CLOCK.now()
    state = container.state_repository.load(subject)
    health = container.health.check().as_dict()
    mailbox = container.proactive.mailbox(subject, as_of=now)
    active = container.proactive.active(subject, as_of=now)
    affect = container.affect_view.view(subject, as_of=now)
    mailbox_status = Counter(str(item["status"]) for item in mailbox)
    return {
        "elapsed_seconds": round(elapsed, 2),
        "events_sent": sent,
        "health": health,
        "backlog": len(container.event_bus.backlog()),
        "dead_letters": len(container.event_bus.dead_letters()),
        "worker_error_type": getattr(container.lifecycle, "worker_error_type", None),
        "state_version": state.version if state is not None else 0,
        "mailbox_count": len(mailbox),
        "mailbox_status": dict(mailbox_status),
        "active_intention_count": len(active),
        "active_emotions": affect["emotions"],
        "mood": affect["mood"],
        "metrics": {
            "counters": dict(container.metrics.snapshot().counters),
            "gauges": dict(container.metrics.snapshot().gauges),
        },
    }


def _write_summary(path: Path, summary: dict[str, object]) -> None:
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Real dual-loop soak runner")
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--interval-seconds", type=float, default=180.0)
    parser.add_argument("--snapshot-seconds", type=float, default=60.0)
    parser.add_argument("--user", default="soak-user")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--dotenv", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--boredom-minutes", default="0,5,35")
    args = parser.parse_args(argv)

    if args.minutes < 1:
        raise ValueError("minutes must be positive")
    if args.interval_seconds <= 0 or args.snapshot_seconds <= 0:
        raise ValueError("interval and snapshot must be positive")

    if args.data_dir is None:
        run_root = Path(tempfile.mkdtemp(prefix="sc-soak-"))
    else:
        run_root = args.data_dir.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    data_dir = run_root / "data"
    snapshots_path = run_root / "snapshots.jsonl"
    summary_path = run_root / "summary.json"

    settings = load_settings(args.dotenv)
    settings = replace(
        settings,
        data_dir=data_dir,
        worker_enabled=True,
        worker_poll_interval_seconds=0.25,
        worker_max_workers=4,
        worker_lease_timeout_seconds=max(
            settings.worker_lease_timeout_seconds,
            60.0,
        ),
    )
    container = build_container(
        data_dir=data_dir,
        settings=settings,
        dotenv_path=args.dotenv,
    )
    subject = SubjectScope.legacy_user(args.user)
    boredom_minutes = set(_parse_minutes(args.boredom_minutes))
    duration_seconds = float(args.minutes * 60)
    start = time.monotonic()
    next_message_at = 0.0
    next_snapshot_at = 0.0
    sent = 0
    fast_success = 0
    fast_errors: list[str] = []
    snapshots: list[dict[str, object]] = []
    deleted_lifecycle = False
    try:
        container.lifecycle.start()
        if not container.lifecycle.wait_ready(10):
            raise RuntimeError("application lifecycle did not become ready")
        while True:
            elapsed = time.monotonic() - start
            if elapsed >= duration_seconds:
                break
            if elapsed >= next_message_at:
                minute = int(elapsed // 60)
                if minute in boredom_minutes:
                    text = "好无聊，想找个人聊聊天"
                else:
                    text = NEUTRAL_MESSAGES[sent % len(NEUTRAL_MESSAGES)]
                context = _context()
                event = EventEnvelope.user_message(
                    subject,
                    text,
                    run_id=context.run_id,
                    correlation_id=context.correlation_id,
                )
                try:
                    container.event_bus.publish(event, context)
                    container.orchestrator.fast_only(event, context)
                    fast_success += 1
                except Exception as error:
                    fast_errors.append(type(error).__name__)
                sent += 1
                next_message_at += args.interval_seconds
            if elapsed >= next_snapshot_at:
                snapshot = _snapshot(container, subject, sent, elapsed)
                snapshots.append(snapshot)
                _append_jsonl(snapshots_path, snapshot)
                next_snapshot_at += args.snapshot_seconds
            time.sleep(min(1.0, max(0.05, next_snapshot_at - (time.monotonic() - start))))

        drain_deadline = time.monotonic() + 180.0
        while time.monotonic() < drain_deadline:
            if not container.event_bus.backlog():
                break
            time.sleep(1.0)
        final_elapsed = time.monotonic() - start
        final_snapshot = _snapshot(container, subject, sent, final_elapsed)
        snapshots.append(final_snapshot)
        _append_jsonl(snapshots_path, final_snapshot)
    finally:
        container.lifecycle.stop(10)
        deleted_lifecycle = True

    ready_snapshots = sum(
        1 for item in snapshots if bool(item["health"]["ready"])
    )  # type: ignore[index]
    worker_errors = [
        str(item["worker_error_type"])
        for item in snapshots
        if item.get("worker_error_type")
    ]
    max_backlog = max(int(item["backlog"]) for item in snapshots)
    module_degradation_counts: Counter[str] = Counter()
    module_degradation_reasons: Counter[str] = Counter()
    module_recovery_counts: Counter[str] = Counter()
    model_unhealthy_counts: Counter[str] = Counter()
    model_degradation_reasons: Counter[str] = Counter()
    previous_degraded_modules: set[str] = set()
    for item in snapshots:
        health = item.get("health", {})
        if not isinstance(health, dict):
            continue
        components = health.get("components", ())
        if not isinstance(components, list):
            continue
        current_degraded_modules: set[str] = set()
        for component in components:
            if not isinstance(component, dict):
                continue
            name = component.get("name")
            details = component.get("details", ())
            if not isinstance(details, list):
                continue
            if name == "modules":
                for detail in details:
                    if not isinstance(detail, dict) or detail.get("health") != "degraded":
                        continue
                    module_id = str(detail.get("module_id", "unknown"))
                    reason = detail.get("degraded_reason")
                    module_degradation_counts[module_id] += 1
                    current_degraded_modules.add(module_id)
                    if reason:
                        module_degradation_reasons[
                            f"{module_id}:{reason}"
                        ] += 1
            elif name == "models":
                for detail in details:
                    if not isinstance(detail, dict) or detail.get("healthy"):
                        continue
                    task = str(detail.get("task", "unknown"))
                    reason = detail.get("degraded_reason")
                    model_unhealthy_counts[task] += 1
                    if reason:
                        model_degradation_reasons[
                            f"{task}:{reason}"
                        ] += 1
        for module_id in previous_degraded_modules - current_degraded_modules:
            module_recovery_counts[module_id] += 1
        previous_degraded_modules = current_degraded_modules

    final_mailbox = container.proactive.mailbox(subject, as_of=SYSTEM_CLOCK.now())
    final_active = container.proactive.active(subject, as_of=SYSTEM_CLOCK.now())
    final_state = container.state_repository.load(subject)
    summary = {
        "run_root": str(run_root),
        "data_dir": str(data_dir),
        "minutes": args.minutes,
        "interval_seconds": args.interval_seconds,
        "snapshot_seconds": args.snapshot_seconds,
        "events_sent": sent,
        "fast_success": fast_success,
        "fast_errors": fast_errors,
        "snapshots": len(snapshots),
        "ready_snapshots": ready_snapshots,
        "degraded_snapshots": len(snapshots) - ready_snapshots,
        "max_backlog": max_backlog,
        "final_backlog": len(container.event_bus.backlog()),
        "dead_letters": len(container.event_bus.dead_letters()),
        "worker_error_types": sorted(set(worker_errors)),
        "final_ready_before_stop": bool(snapshots[-1]["health"]["ready"]),
        "module_degradation_counts": dict(module_degradation_counts),
        "module_degradation_reasons": dict(module_degradation_reasons),
        "module_recovery_counts": dict(module_recovery_counts),
        "model_unhealthy_counts": dict(model_unhealthy_counts),
        "model_degradation_reasons": dict(model_degradation_reasons),
        "mailbox_count": len(final_mailbox),
        "mailbox": final_mailbox,
        "active_intention_count": len(final_active),
        "state_version": final_state.version if final_state is not None else 0,
        "final_health": container.health.check().as_dict(),
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "clean_lifecycle_shutdown": deleted_lifecycle,
    }
    _write_summary(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
