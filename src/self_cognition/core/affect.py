from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import Any

from self_cognition.core.errors import ContractValidationError

ACTIVE_INTENSITY_THRESHOLD = 0.1


@dataclass(frozen=True, slots=True)
class AffectAssessment:
    target: str
    goal_ids: tuple[str, ...]
    emotion: str
    valence: str
    scope: str
    initial_intensity: float
    assessed_at: datetime
    half_life_seconds: float = 3600.0
    active_threshold: float = ACTIVE_INTENSITY_THRESHOLD

    def __post_init__(self) -> None:
        for name in ("target", "emotion", "scope"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ContractValidationError(f"affect {name} must not be blank")
        if self.valence not in {"positive", "negative", "neutral", "mixed"}:
            raise ContractValidationError("affect valence is invalid")
        if not isinstance(self.goal_ids, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.goal_ids
        ):
            raise ContractValidationError("affect goal IDs must be text values")
        for name in ("initial_intensity", "half_life_seconds", "active_threshold"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
            ):
                raise ContractValidationError(f"affect {name} must be a finite number")
        if not 0 <= self.initial_intensity <= 1 or not 0 < self.active_threshold <= 1:
            raise ContractValidationError(
                "affect intensity or threshold is out of range"
            )
        if self.half_life_seconds <= 0:
            raise ContractValidationError("affect half life must be positive")
        if (
            not isinstance(self.assessed_at, datetime)
            or self.assessed_at.utcoffset() is None
        ):
            raise ContractValidationError("affect timestamp must include a timezone")

    def to_state_value(self) -> dict[str, object]:
        return {
            "target": self.target,
            "goal_ids": list(self.goal_ids),
            "emotion": self.emotion,
            "valence": self.valence,
            "scope": self.scope,
            "initial_intensity": self.initial_intensity,
            "assessed_at": self.assessed_at.isoformat(),
            "half_life_seconds": self.half_life_seconds,
            "active_threshold": self.active_threshold,
        }

    @classmethod
    def from_state_value(cls, value: object) -> "AffectAssessment":
        fields = {
            "target",
            "goal_ids",
            "emotion",
            "valence",
            "scope",
            "initial_intensity",
            "assessed_at",
            "half_life_seconds",
            "active_threshold",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ContractValidationError("affect assessment fields are invalid")
        if not isinstance(value["goal_ids"], list):
            raise ContractValidationError("affect goal IDs must be an array")
        try:
            return cls(
                target=value["target"],
                goal_ids=tuple(value["goal_ids"]),
                emotion=value["emotion"],
                valence=value["valence"],
                scope=value["scope"],
                initial_intensity=value["initial_intensity"],
                assessed_at=datetime.fromisoformat(value["assessed_at"]),
                half_life_seconds=value["half_life_seconds"],
                active_threshold=value["active_threshold"],
            )
        except (TypeError, ValueError) as error:
            raise ContractValidationError("invalid affect assessment") from error


def decay_assessment(
    assessment: object,
    as_of: datetime,
) -> dict[str, Any] | None:
    """Return a read-only decayed affect view at a given time."""
    if not isinstance(assessment, dict):
        return None
    try:
        assessed_at = datetime.fromisoformat(str(assessment["assessed_at"]))
        initial_intensity = float(assessment["initial_intensity"])
        half_life_seconds = float(assessment["half_life_seconds"])
        threshold = float(
            assessment.get("active_threshold", ACTIVE_INTENSITY_THRESHOLD)
        )
    except (KeyError, TypeError, ValueError):
        return None
    if (
        assessed_at.tzinfo is None
        or assessed_at.utcoffset() is None
        or as_of.tzinfo is None
        or as_of.utcoffset() is None
        or not 0.0 <= initial_intensity <= 1.0
        or half_life_seconds <= 0.0
        or not isfinite(half_life_seconds)
        or not 0.0 < threshold <= 1.0
        or as_of < assessed_at
    ):
        return None

    elapsed_seconds = max(0.0, (as_of - assessed_at).total_seconds())
    current_intensity = initial_intensity * (
        0.5 ** (elapsed_seconds / half_life_seconds)
    )
    if current_intensity < threshold:
        return None

    decayed = dict(assessment)
    decayed["current_intensity"] = current_intensity
    return decayed
