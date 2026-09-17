from dataclasses import replace
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from self_cognition.application.affect import AffectViewService
from self_cognition.core.affect import (
    AffectAssessment,
    EmotionState,
    MoodState,
    accumulate_mood,
    decay_emotion,
    decay_mood,
    is_preference_shaped_affect,
)
from self_cognition.core.contributions import CognitionType
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.scopes import DataScope, DisclosureScope, SubjectScope
from self_cognition.core.state import StateAtom, SubjectState
from self_cognition.infrastructure.persistence.in_memory_state_repository import (
    InMemoryStateRepository,
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

def test_preference_shaped_affect_is_detected() -> None:
    assert is_preference_shaped_affect(
        "affect.current.fish_preference",
        {
            "emotion": "liking",
            "valence": "positive",
            "target": "鱼",
            "scope": "fish_preference",
            "cause": "鱼",
        },
    )
    assert not is_preference_shaped_affect(
        "affect.current.exam",
        {
            "emotion": "开心",
            "valence": "positive",
            "target": "考试",
            "scope": "exam",
            "cause": "考试通过",
        },
    )

def test_emotion_view_hides_preference_shaped_affect() -> None:
    subject = SubjectScope.legacy_user("fish-user")
    value = {
        "emotion": "liking",
        "valence": "positive",
        "target": "鱼",
        "scope": "fish_preference",
        "initial_intensity": 0.69,
        "assessed_at": NOW.isoformat(),
        "half_life_seconds": 3600.0,
        "active_threshold": 0.1,
        "goal_ids": [],
        "cause": "鱼",
    }
    atom = StateAtom(
        value=value,
        cognition_type=CognitionType.AFFECT,
        confidence=1.0,
        scope=DataScope(subject, DisclosureScope.PRIVATE),
        evidence_refs=(EvidenceRef.for_event_id(uuid4(), subject),),
        contribution_ids=(uuid4(),),
        created_at=NOW,
        valid_from=NOW,
    )
    state = replace(
        SubjectState.empty("fish-user"),
        version=1,
        entries={"affect.current.fish_preference": atom},
    )
    repository = InMemoryStateRepository()
    repository.replace(state)

    view = AffectViewService(repository).view(subject, as_of=NOW)

    assert view["emotions"] == []
