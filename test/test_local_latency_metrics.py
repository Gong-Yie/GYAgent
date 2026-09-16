from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from self_cognition.observability.metrics import MetricsRegistry


ROOT = Path(__file__).resolve().parents[1]


def test_metrics_percentile_interpolates_and_handles_missing() -> None:
    metrics = MetricsRegistry()
    for value in (1.0, 2.0, 3.0, 4.0, 5.0):
        metrics.observe("latency", value)

    assert metrics.percentile("latency", 0.50) == 3.0
    assert metrics.percentile("latency", 0.95) == 4.8
    assert metrics.percentile("latency", 1.0) == 5.0
    assert metrics.percentile("missing", 0.50) is None
    assert metrics.snapshot().percentile("latency", 0.50) == 3.0


def test_measure_local_latency_script_reports_percentiles(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "measure_local_latency.py"),
            "--data-dir",
            str(tmp_path / "data"),
            "--dotenv",
            str(tmp_path / "missing.env"),
            "--samples",
            "6",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["samples"] == 6

    for name in (
        "local.event_ingest.latency_seconds",
        "local.query.latency_seconds",
        "local.state_write.latency_seconds",
    ):
        values = report["latency_seconds"][name]
        assert values["p50"] is not None
        assert values["p50"] <= values["p95"] <= values["p99"]
        assert report["timing_counts"][name] == 6
