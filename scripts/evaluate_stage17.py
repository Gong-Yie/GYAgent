"""Run the small offline Stage 17 rule and fake-model comparison."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from self_cognition.cognition.episodic.memory_extractor import (  # noqa: E402
    EpisodicMemoryExtractor,
)
from self_cognition.cognition.semantic.name_extractor import NameExtractor  # noqa: E402
from self_cognition.cognition.semantic.preference_extractor import (  # noqa: E402
    PreferenceExtractor,
)
from self_cognition.core.events import Event  # noqa: E402


CASES = (
    ("我叫小明", "profile.name"),
    ("我喜欢晚上学习", "preferences.study_time"),
    ("今天我去了公园", "episodic.experience."),
)


def evaluate_rule_baseline() -> dict[str, float | int]:
    modules = (NameExtractor(), PreferenceExtractor(), EpisodicMemoryExtractor())
    correct = 0
    evidence_backed = 0
    for text, expected in CASES:
        event = Event.user_message("stage17-eval", text)
        predictions = tuple(
            contribution
            for module in modules
            for contribution in module.process(event)
        )
        matched = tuple(
            contribution
            for contribution in predictions
            if contribution.target_field == expected
            or contribution.target_field.startswith(expected)
        )
        correct += bool(matched)
        evidence_backed += bool(matched and matched[0].evidence_refs)
    return {
        "cases": len(CASES),
        "target_accuracy": correct / len(CASES),
        "evidence_coverage": evidence_backed / len(CASES),
    }


def evaluate_fake_model() -> dict[str, float | int]:
    """Evaluate scripted structured outputs, without claiming live model quality."""
    predictions = {
        "我叫小明": "profile.name",
        "我喜欢晚上学习": "preferences.study_time",
        "今天我去了公园": "episodic.experience.",
    }
    correct = sum(
        predictions[text] == expected or predictions[text].startswith(expected)
        for text, expected in CASES
    )
    return {"cases": len(CASES), "target_accuracy": correct / len(CASES)}


def main() -> None:
    report = {
        "stage": 17,
        "rule_baseline": evaluate_rule_baseline(),
        "fake_model": evaluate_fake_model(),
        "live_model": "not_run",
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
