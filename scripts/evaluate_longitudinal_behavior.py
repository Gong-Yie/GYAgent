from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from self_cognition.bootstrap import build_container
from self_cognition.core.dialogue import (
    AssistantMessagePayload,
    DialogueContextPayload,
    DialogueFailurePayload,
)
from self_cognition.core.events import EventEnvelope
from self_cognition.core.scopes import SubjectScope
from self_cognition.core.time import SYSTEM_CLOCK
from self_cognition.observability.longitudinal import (
    PROBES,
    SETUP_MESSAGES,
    LongitudinalProbe,
    evaluate_probe,
    summarize_criteria,
)
from self_cognition.runtime.run_context import RunContext
from self_cognition.settings import load_settings


NEUTRAL_MESSAGES = (
    "你好，我只是来看看",
    "今天过得还行",
    "好的，我知道了",
    "嗯，先这样",
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


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _snapshot(container, subject: SubjectScope, sent: int, elapsed: float) -> dict[str, object]:
    state = container.state_repository.load(subject)
    return {
        "elapsed_seconds": round(elapsed, 2),
        "events_sent": sent,
        "health": container.health.check().as_dict(),
        "backlog": len(container.event_bus.backlog()),
        "dead_letters": len(container.event_bus.dead_letters()),
        "worker_error_type": getattr(container.lifecycle, "worker_error_type", None),
        "state_version": state.version if state is not None else 0,
        "metrics": {
            "counters": dict(container.metrics.snapshot().counters),
            "gauges": dict(container.metrics.snapshot().gauges),
        },
    }


def _workspace_summary(workspace_json: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    try:
        payload = json.loads(workspace_json)
    except json.JSONDecodeError:
        return (), ()
    if not isinstance(payload, dict):
        return (), ()
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        return (), ()
    sources: set[str] = set()
    targets: set[str] = set()
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        raw_source = item.get("source")
        if raw_source is not None:
            sources.add(str(raw_source))
        raw_target = item.get("target_field")
        if isinstance(raw_target, str) and raw_target:
            targets.add(raw_target)
    return tuple(sorted(sources)), tuple(sorted(targets))


def _wait_for_dialogue(
    container,
    subject: SubjectScope,
    request_event_id,
    timeout_seconds: float,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        events = container.event_store.read_by_mind(subject.mind)
        terminal = next(
            (
                event
                for event in events
                if isinstance(event.payload, AssistantMessagePayload)
                and event.payload.request_event_id == request_event_id
            ),
            None,
        )
        if terminal is not None:
            context_event = next(
                (
                    event
                    for event in events
                    if isinstance(event.payload, DialogueContextPayload)
                    and event.payload.request_event_id == request_event_id
                ),
                None,
            )
            sources, targets = (
                _workspace_summary(context_event.payload.workspace_json)
                if context_event is not None
                else ((), ())
            )
            return {
                "status": "succeeded",
                "error_type": None,
                "answer": terminal.payload.answer.text,
                "evidence_count": len(terminal.payload.evidence_refs),
                "sources": sources,
                "target_fields": targets,
                "event_id": str(terminal.event_id),
            }
        failure = next(
            (
                event
                for event in events
                if isinstance(event.payload, DialogueFailurePayload)
                and event.payload.request_event_id == request_event_id
            ),
            None,
        )
        if failure is not None:
            return {
                "status": "failed",
                "error_type": failure.payload.error_type,
                "answer": "",
                "evidence_count": 0,
                "sources": (),
                "target_fields": (),
                "event_id": str(failure.event_id),
            }
        time.sleep(0.2)
    return {
        "status": "timeout",
        "error_type": "ProbeTimeout",
        "answer": "",
        "evidence_count": 0,
        "sources": (),
        "target_fields": (),
        "event_id": None,
    }


def _publish_and_wait(
    container,
    subject: SubjectScope,
    text: str,
    timeout_seconds: float,
) -> dict[str, object]:
    context = _context()
    event = EventEnvelope.user_message(
        subject,
        text,
        run_id=context.run_id,
        correlation_id=context.correlation_id,
    )
    container.event_bus.publish(event, context)
    response = _wait_for_dialogue(
        container,
        subject,
        event.event_id,
        timeout_seconds,
    )
    response["request_event_id"] = str(event.event_id)
    response["question"] = text
    return response


def _run_probe_suite(
    container,
    subject: SubjectScope,
    probes: tuple[LongitudinalProbe, ...],
    phase: str,
    timeout_seconds: float,
) -> tuple[dict[str, object], ...]:
    results: list[dict[str, object]] = []
    for probe in probes:
        response = _publish_and_wait(
            container,
            subject,
            probe.question,
            timeout_seconds,
        )
        result = evaluate_probe(
            probe,
            status=str(response["status"]),
            error_type=(
                str(response["error_type"])
                if response["error_type"] is not None
                else None
            ),
            answer=str(response["answer"]),
            evidence_count=int(response["evidence_count"]),
            sources=tuple(str(value) for value in response["sources"]),
            target_fields=tuple(
                str(value) for value in response["target_fields"]
            ),
        )
        result["phase"] = phase
        result["request_event_id"] = response["request_event_id"]
        results.append(result)
    return tuple(results)


def _select_probes(raw_ids: str) -> tuple[LongitudinalProbe, ...]:
    normalized = raw_ids.strip()
    if not normalized or normalized.lower() == "all":
        return PROBES
    selected = {
        item.strip() for item in normalized.split(",") if item.strip()
    }
    return tuple(probe for probe in PROBES if probe.probe_id in selected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Longitudinal real-model behavior evaluator"
    )
    parser.add_argument("--minutes", type=int, default=60)
    parser.add_argument("--interval-seconds", type=float, default=180.0)
    parser.add_argument("--snapshot-seconds", type=float, default=60.0)
    parser.add_argument("--probe-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--user", default="longitudinal-user")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--dotenv", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--probe-ids", default="all")
    parser.add_argument("--skip-final-probes", action="store_true")
    parser.add_argument("--setup-limit", type=int, default=None)
    args = parser.parse_args(argv)

    if args.minutes < 0:
        raise ValueError("minutes must be non-negative")
    if args.interval_seconds <= 0 or args.snapshot_seconds <= 0:
        raise ValueError("interval and snapshot must be positive")
    if args.probe_timeout_seconds <= 0:
        raise ValueError("probe timeout must be positive")
    if args.setup_limit is not None and args.setup_limit < 1:
        raise ValueError("setup limit must be positive")
    selected_probes = _select_probes(args.probe_ids)
    if not selected_probes:
        raise ValueError("no probes selected")

    if args.data_dir is None:
        run_root = Path(tempfile.mkdtemp(prefix="sc-longitudinal-"))
    else:
        run_root = args.data_dir.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    data_dir = run_root / "data"
    raw_path = run_root / "raw.jsonl"
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
    duration_seconds = float(args.minutes * 60)
    start = time.monotonic()
    next_message_at = 0.0
    next_snapshot_at = 0.0
    sent = 0
    errors: list[str] = []
    snapshots: list[dict[str, object]] = []
    response_calls = 0
    response_failures = 0
    try:
        container.lifecycle.start()
        if not container.lifecycle.wait_ready(10):
            raise RuntimeError("application lifecycle did not become ready")

        setup_results = []
        for message in SETUP_MESSAGES[: args.setup_limit]:
            response = _publish_and_wait(
                container,
                subject,
                message,
                args.probe_timeout_seconds,
            )
            setup_results.append(response)
            _append_jsonl(
                raw_path,
                {"kind": "setup", **response},
            )

        probe_results = list(
            _run_probe_suite(
                container,
                subject,
                selected_probes,
                "initial",
                args.probe_timeout_seconds,
            )
        )
        for result in probe_results:
            _append_jsonl(raw_path, {"kind": "probe", **result})

        while time.monotonic() - start < duration_seconds:
            elapsed = time.monotonic() - start
            if elapsed >= next_message_at:
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
                    sent += 1
                except Exception as error:
                    errors.append(type(error).__name__)
                next_message_at += args.interval_seconds
            if elapsed >= next_snapshot_at:
                snapshot = _snapshot(container, subject, sent, elapsed)
                snapshots.append(snapshot)
                _append_jsonl(raw_path, {"kind": "snapshot", **snapshot})
                next_snapshot_at += args.snapshot_seconds
            time.sleep(
                min(
                    1.0,
                    max(0.05, next_snapshot_at - (time.monotonic() - start)),
                )
            )

        drain_deadline = time.monotonic() + 180.0
        while time.monotonic() < drain_deadline:
            if not container.event_bus.backlog():
                break
            time.sleep(1.0)

        final_probe_results: list[dict[str, object]] = []
        if not args.skip_final_probes:
            final_probe_results = list(
                _run_probe_suite(
                    container,
                    subject,
                    selected_probes,
                    "final",
                    args.probe_timeout_seconds,
                )
            )
            for result in final_probe_results:
                _append_jsonl(raw_path, {"kind": "probe", **result})
        probe_results.extend(final_probe_results)
        response_calls = len(setup_results) + len(probe_results)
        response_failures = sum(
            1
            for item in setup_results
            if item.get("status") != "succeeded"
        ) + sum(
            1 for item in probe_results if not item.get("passed")
        )

        final_elapsed = time.monotonic() - start
        final_snapshot = _snapshot(container, subject, sent, final_elapsed)
        snapshots.append(final_snapshot)
        _append_jsonl(raw_path, {"kind": "snapshot", **final_snapshot})
    finally:
        container.lifecycle.stop(10)

    criteria_by_phase = {
        phase: summarize_criteria(
            tuple(
                item for item in probe_results if item.get("phase") == phase
            )
        )
        for phase in ("initial", "final")
    }
    max_backlog = max(
        (int(item["backlog"]) for item in snapshots),
        default=0,
    )
    summary = {
        "run_root": str(run_root),
        "data_dir": str(data_dir),
        "minutes": args.minutes,
        "interval_seconds": args.interval_seconds,
        "snapshot_seconds": args.snapshot_seconds,
        "events_sent": sent,
        "probe_ids": [probe.probe_id for probe in selected_probes],
        "setup_limit": args.setup_limit,
        "skip_final_probes": args.skip_final_probes,
        "fast_publish_errors": errors,
        "response_calls": response_calls,
        "response_failures": response_failures,
        "setup_results": setup_results,
        "probe_results": probe_results,
        "criteria_by_phase": criteria_by_phase,
        "snapshots": snapshots,
        "max_backlog": max_backlog,
        "final_backlog": len(container.event_bus.backlog()),
        "dead_letters": len(container.event_bus.dead_letters()),
        "final_health": final_snapshot["health"],
        "final_ready_before_stop": bool(final_snapshot["health"]["ready"]),
        "ended_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
