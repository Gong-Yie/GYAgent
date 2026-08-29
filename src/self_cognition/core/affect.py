from datetime import datetime
from typing import Any


ACTIVE_INTENSITY_THRESHOLD = 0.1


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
    except (KeyError, TypeError, ValueError):
        return None
    if (
        assessed_at.tzinfo is None
        or assessed_at.utcoffset() is None
        or as_of.tzinfo is None
        or as_of.utcoffset() is None
        or not 0.0 <= initial_intensity <= 1.0
        or half_life_seconds <= 0.0
    ):
        return None

    elapsed_seconds = max(0.0, (as_of - assessed_at).total_seconds())
    current_intensity = initial_intensity * (
        0.5 ** (elapsed_seconds / half_life_seconds)
    )
    if current_intensity < ACTIVE_INTENSITY_THRESHOLD:
        return None

    decayed = dict(assessment)
    decayed["current_intensity"] = current_intensity
    return decayed
