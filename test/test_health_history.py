from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from self_cognition.bootstrap import build_container
from self_cognition.core.runs import RunKind, RunRecord, RunStatus
from self_cognition.core.scopes import SubjectScope
from self_cognition.interfaces.http.server import _handle, _query_int
from self_cognition.runtime.health import (
    ComponentHealth,
    HealthHistory,
    HealthReport,
)
from self_cognition.settings import ApplicationSettings


def _report() -> HealthReport:
    return HealthReport(
        ready=True,
        components=(
            ComponentHealth(
                "outbox",
                "healthy",
                "backlog=2;dead_letters=1",
            ),
            ComponentHealth("models", "healthy", "total=1;unavailable=0"),
            ComponentHealth("modules", "healthy", "healthy=1;degraded=0;disabled=0"),
        ),
    )


def test_health_history_is_bounded_and_serializable() -> None:
    history = HealthHistory(max_entries=2)
    first = datetime(2026, 9, 16, tzinfo=timezone.utc)
    history.record(_report(), checked_at=first)
    history.record(_report(), checked_at=first + timedelta(seconds=1))
    history.record(_report(), checked_at=first + timedelta(seconds=2))

    snapshots = history.snapshots(limit=10)
    assert len(snapshots) == 2
    assert snapshots[0].checked_at == first + timedelta(seconds=1)
    assert snapshots[-1].as_dict()["outbox"]["backlog"] == 2
    assert snapshots[-1].as_dict()["outbox"]["dead_letters"] == 1


def test_http_health_history_reports_backlog_and_degradation_fields(
    tmp_path: Path,
) -> None:
    container = build_container(
        tmp_path,
        settings=ApplicationSettings(data_dir=tmp_path, worker_enabled=False),
        dotenv_path=tmp_path / "missing.env",
    )
    _handle(container, "GET", "/health", {}, {})
    _handle(container, "GET", "/health", {}, {})

    payload = _handle(
        container,
        "GET",
        "/health/history",
        {"limit": ["10"]},
        {},
    )

    snapshots = payload["snapshots"]
    assert len(snapshots) >= 2
    latest = snapshots[-1]
    assert "checked_at" in latest
    assert latest["outbox"]["backlog"] == 0
    assert latest["outbox"]["dead_letters"] == 0
    assert latest["models"]["status"] == "healthy"
    assert latest["modules"]["status"] == "healthy"


def test_health_history_limit_validation() -> None:
    assert (
        _query_int(
            {"limit": ["10"]},
            "limit",
            default=50,
            minimum=1,
            maximum=500,
        )
        == 10
    )
    with pytest.raises(ValueError, match="must be an integer"):
        _query_int(
            {"limit": ["abc"]},
            "limit",
            default=50,
            minimum=1,
            maximum=500,
        )
    with pytest.raises(ValueError, match="between 1 and 500"):
        _query_int(
            {"limit": ["0"]},
            "limit",
            default=50,
            minimum=1,
            maximum=500,
        )

def _failed_run() -> RunRecord:
    now = datetime(2026, 9, 16, 12, tzinfo=timezone.utc)
    return RunRecord(
        run_id=uuid4(),
        kind=RunKind.COGNITIVE_CYCLE,
        subject=SubjectScope.for_mind("mind-health"),
        correlation_id=uuid4(),
        started_at=now,
        updated_at=now,
        deadline=now + timedelta(minutes=5),
        status=RunStatus.FAILED,
        error_type="ModelTimeoutError",
        termination_reason="provider timeout",
        input_event_ids=(uuid4(),),
    )


def test_http_health_failures_returns_recent_failure_chain(
    tmp_path: Path,
) -> None:
    container = build_container(
        tmp_path,
        settings=ApplicationSettings(data_dir=tmp_path, worker_enabled=False),
        dotenv_path=tmp_path / "missing.env",
    )
    record = _failed_run()
    container.run_repository.save(record)

    payload = _handle(
        container,
        "GET",
        "/health/failures",
        {"limit": ["10"]},
        {},
    )

    failures = payload["failures"]
    matched = next(
        item for item in failures if item["run_id"] == str(record.run_id)
    )
    assert matched["status"] == "failed"
    assert matched["kind"] == "cognitive_cycle"
    assert matched["error_type"] == "ModelTimeoutError"
    assert matched["termination_reason"] == "provider timeout"
    assert matched["input_event_ids"] == [str(record.input_event_ids[0])]

def test_health_history_restores_across_container_restart(
    tmp_path: Path,
) -> None:
    settings = ApplicationSettings(data_dir=tmp_path, worker_enabled=False)
    first = build_container(
        tmp_path,
        settings=settings,
        dotenv_path=tmp_path / "missing.env",
    )
    first.model_router.mark_degraded(
        "dialogue-default",
        "dialogue",
        "ModelTimeoutError",
    )
    first.health.check()
    before = first.health.history(limit=10)
    assert before
    assert before[-1]["models"]["status"] == "degraded"

    second = build_container(
        tmp_path,
        settings=ApplicationSettings(data_dir=tmp_path, worker_enabled=False),
        dotenv_path=tmp_path / "missing.env",
    )

    restored = second.health.history(limit=10)
    assert restored
    assert restored[-1]["checked_at"] == before[-1]["checked_at"]
    assert restored[-1]["models"]["status"] == "degraded"
