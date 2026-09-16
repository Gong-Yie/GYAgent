from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from self_cognition.bootstrap import build_container
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
