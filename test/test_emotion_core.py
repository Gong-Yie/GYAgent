from datetime import datetime, timedelta, timezone
from uuid import uuid4

from self_cognition.core.affect import (
    AffectAssessment,
    EmotionState,
    MoodState,
    accumulate_mood,
    decay_emotion,
    decay_mood,
)


NOW = datetime(2026, 9, 15, 12, tzinfo=timezone.utc)


def test_affect_assessment_defaults_extended_dimensions():
    assessment = AffectAssessment.from_state_value(
        {
            "target": "alice",
            "goal_ids": [],
            "emotion": "relief",
            "valence": "positive",
            "scope": "project",
            "initial_intensity": 0.8,
            "assessed_at": NOW.isoformat(),
            "half_life_seconds": 3600.0,
            "active_threshold": 0.1,
        }
    )

    assert assessment.arousal == 0.5
    assert assessment.control == 0.5
    assert assessment.certainty == 0.5
    assert assessment.cause == ""
    assert AffectAssessment.from_state_value(assessment.to_state_value()) == assessment


def test_emotion_decay_preserves_extended_dimensions():
    emotion = EmotionState(
        uuid4(),
        "alice",
        "boredom",
        "negative",
        "conversation",
        0.8,
        NOW,
        arousal=0.1,
        control=0.2,
        certainty=0.9,
        cause="no recent interaction",
        half_life_seconds=3600.0,
    )

    decayed = decay_emotion(emotion, NOW + timedelta(hours=1))

    assert decayed is not None
    assert decayed.intensity == 0.4
    assert decayed.arousal == 0.1
    assert decayed.control == 0.2
    assert decayed.certainty == 0.9
    assert decayed.cause == "no recent interaction"


def test_mood_accumulates_and_decays():
    emotion = EmotionState(
        uuid4(),
        "alice",
        "relief",
        "positive",
        "project",
        0.8,
        NOW,
        arousal=0.7,
        control=0.8,
        certainty=0.9,
        half_life_seconds=3600.0,
    )

    mood = accumulate_mood(None, emotion, as_of=NOW)
    assert isinstance(mood, MoodState)
    assert mood.intensity == 0.8
    assert mood.source_emotion_ids == (emotion.emotion_id,)

    later_emotion = EmotionState(
        uuid4(),
        "alice",
        "boredom",
        "negative",
        "project",
        0.9,
        NOW + timedelta(minutes=30),
        arousal=0.1,
        control=0.2,
        certainty=0.8,
        half_life_seconds=3600.0,
    )
    updated = accumulate_mood(
        mood,
        later_emotion,
        as_of=later_emotion.assessed_at,
    )

    assert updated.intensity > mood.intensity
    assert updated.valence == "negative"
    assert updated.source_emotion_ids[-1] == later_emotion.emotion_id

    decayed = decay_mood(updated, later_emotion.assessed_at + timedelta(days=1))
    assert decayed is None or decayed.intensity < updated.intensity