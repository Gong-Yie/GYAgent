from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import Any
from uuid import UUID

from self_cognition.core.errors import ContractValidationError

ACTIVE_INTENSITY_THRESHOLD = 0.1


def _aware(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ContractValidationError(f"{name} must include a timezone")


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


@dataclass(frozen=True, slots=True)
class EmotionState:
    """A short-lived, computational affect projection."""

    emotion_id: UUID
    target: str
    emotion: str
    valence: str
    scope: str
    intensity: float
    assessed_at: datetime
    goal_ids: tuple[str, ...] = ()
    half_life_seconds: float = 3600.0
    active_threshold: float = ACTIVE_INTENSITY_THRESHOLD

    def __post_init__(self) -> None:
        if not isinstance(self.emotion_id, UUID):
            raise ContractValidationError("emotion ID must be a UUID")
        for name in ("target", "emotion", "scope"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ContractValidationError(f"emotion {name} must not be blank")
        if self.valence not in {"positive", "negative", "neutral", "mixed"}:
            raise ContractValidationError("emotion valence is invalid")
        if any(not isinstance(item, str) or not item.strip() for item in self.goal_ids):
            raise ContractValidationError("emotion goal IDs must be text values")
        for name in ("intensity", "half_life_seconds", "active_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                raise ContractValidationError(f"emotion {name} must be finite")
        if not 0.0 <= self.intensity <= 1.0:
            raise ContractValidationError("emotion intensity must be between 0 and 1")
        if self.half_life_seconds <= 0.0:
            raise ContractValidationError("emotion half life must be positive")
        if not 0.0 < self.active_threshold <= 1.0:
            raise ContractValidationError("emotion threshold must be between 0 and 1")
        _aware(self.assessed_at, "emotion assessed_at")

    @property
    def initial_intensity(self) -> float:
        return self.intensity

    def to_state_value(self) -> dict[str, object]:
        return {
            "emotion_id": str(self.emotion_id),
            "target": self.target,
            "emotion": self.emotion,
            "valence": self.valence,
            "scope": self.scope,
            "intensity": self.intensity,
            "assessed_at": self.assessed_at.isoformat(),
            "goal_ids": list(self.goal_ids),
            "half_life_seconds": self.half_life_seconds,
            "active_threshold": self.active_threshold,
        }

    @classmethod
    def from_state_value(cls, value: object) -> "EmotionState":
        if not isinstance(value, dict):
            raise ContractValidationError("emotion state must be an object")
        try:
            return cls(
                emotion_id=UUID(str(value["emotion_id"])),
                target=value["target"],
                emotion=value["emotion"],
                valence=value["valence"],
                scope=value["scope"],
                intensity=value.get("intensity", value["initial_intensity"]),
                assessed_at=datetime.fromisoformat(value["assessed_at"]),
                goal_ids=tuple(value.get("goal_ids", ())),
                half_life_seconds=value.get("half_life_seconds", 3600.0),
                active_threshold=value.get("active_threshold", ACTIVE_INTENSITY_THRESHOLD),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractValidationError("invalid emotion state") from error


def decay_emotion(state: EmotionState, as_of: datetime) -> EmotionState | None:
    """Return a derived decayed view without mutating the stored state."""
    if not isinstance(state, EmotionState):
        raise ContractValidationError("emotion state is invalid")
    _aware(as_of, "emotion as_of")
    if as_of < state.assessed_at:
        return None
    elapsed = (as_of - state.assessed_at).total_seconds()
    intensity = state.intensity * (0.5 ** (elapsed / state.half_life_seconds))
    if intensity < state.active_threshold:
        return None
    return EmotionState(
        state.emotion_id,
        state.target,
        state.emotion,
        state.valence,
        state.scope,
        intensity,
        state.assessed_at,
        state.goal_ids,
        state.half_life_seconds,
        state.active_threshold,
    )


@dataclass(frozen=True, slots=True)
class Motive:
    motive_id: UUID
    subject_id: str
    kind: str
    description: str
    strength: float
    priority: int
    created_at: datetime
    source_event_ids: tuple[UUID, ...] = ()
    goal_ids: tuple[str, ...] = ()
    emotion_ids: tuple[UUID, ...] = ()
    expires_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.motive_id, UUID):
            raise ContractValidationError("motive ID must be a UUID")
        for name in ("subject_id", "kind", "description"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ContractValidationError(f"motive {name} must not be blank")
        if isinstance(self.strength, bool) or not 0.0 <= self.strength <= 1.0:
            raise ContractValidationError("motive strength must be between 0 and 1")
        if type(self.priority) is not int or self.priority < 0:
            raise ContractValidationError("motive priority must be non-negative")
        if any(not isinstance(item, UUID) for item in self.source_event_ids + self.emotion_ids):
            raise ContractValidationError("motive references must contain UUIDs")
        if any(not isinstance(item, str) or not item.strip() for item in self.goal_ids):
            raise ContractValidationError("motive goal IDs must be text values")
        _aware(self.created_at, "motive created_at")
        if self.expires_at is not None:
            _aware(self.expires_at, "motive expires_at")
            if self.expires_at <= self.created_at:
                raise ContractValidationError("motive expiry must be after creation")

    def active(self, as_of: datetime) -> bool:
        _aware(as_of, "motive as_of")
        return self.expires_at is None or as_of < self.expires_at


def compete_motives(motives: tuple[Motive, ...], as_of: datetime) -> tuple[Motive, ...]:
    """Filter expired motives and order ties deterministically."""
    active = tuple(motive for motive in motives if motive.active(as_of))
    return tuple(sorted(active, key=lambda item: (-item.strength, -item.priority, item.motive_id.int)))


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
