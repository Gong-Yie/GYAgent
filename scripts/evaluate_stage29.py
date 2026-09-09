"""Run the versioned Stage 29 deterministic evaluation suite."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import TypedDict


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "docs" / "evaluations" / "stage29_scenarios_v1.json"


class AutomatedCase(TypedDict):
    id: str
    capability: str
    node_id: str


class DeferredCase(TypedDict):
    id: str
    reason: str


class ScenarioSuite(TypedDict):
    schema_version: int
    suite_id: str
    suite_version: str
    automated_cases: list[AutomatedCase]
    deferred_cases: list[DeferredCase]


def load_suite(path: Path = SCENARIOS) -> ScenarioSuite:
    suite = json.loads(path.read_text(encoding="utf-8"))
    if suite.get("schema_version") != 1:
        raise ValueError("unsupported Stage 29 scenario schema")
    if not suite.get("automated_cases"):
        raise ValueError("Stage 29 suite must contain automated cases")
    node_ids = [case["node_id"] for case in suite["automated_cases"]]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("Stage 29 scenario node IDs must be unique")
    return suite


def evaluate(suite: ScenarioSuite) -> tuple[dict[str, object], int]:
    node_ids = [case["node_id"] for case in suite["automated_cases"]]
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *node_ids],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    passed = completed.returncode == 0
    total = len(node_ids)
    report: dict[str, object] = {
        "schema_version": 1,
        "suite_id": suite["suite_id"],
        "suite_version": suite["suite_version"],
        "status": "passed" if passed else "failed",
        "deterministic_fake": {
            "scenario_groups": total,
            "pass_rate": 1.0 if passed else None,
            "pytest_output": completed.stdout.strip(),
        },
        "covered_capabilities": sorted(
            {case["capability"] for case in suite["automated_cases"]}
        ),
        "deferred": suite["deferred_cases"],
        "subjective_metrics": {
            "presence": "not_run",
            "interaction_comfort": "not_run",
            "willingness_to_return": "not_run",
        },
        "live_model": "not_run",
    }
    if completed.stderr.strip():
        report["pytest_stderr"] = completed.stderr.strip()
    return report, completed.returncode


def main() -> None:
    report, return_code = evaluate(load_suite())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
