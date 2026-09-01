from uuid import NAMESPACE_URL, uuid5

import pytest

from self_cognition.core.events import Event
from self_cognition.cognition.semantic.preference_extractor import PreferenceExtractor
from self_cognition.core.contributions import CognitionType
from self_cognition.core.state import SubjectState
from self_cognition.blackboard.reducer import StateReducer
from self_cognition.blackboard.service import CognitiveSpaceService
from self_cognition.runtime.engine import CognitionEngine

@pytest.mark.parametrize(
    ("content", "expected_value"),
    [
        ("我喜欢晚上学习", "晚上"),
        ("我喜欢早上学习", "早上"),
    ],
)
def test_remembers_user_study_preference(content: str, expected_value: str):
    event = Event.user_message(
        actor="user-1",
        content=content,
    )
    extractor = PreferenceExtractor()
    contributions = extractor.process(event)

    assert len(contributions) == 1

    old_state = SubjectState.empty(subject_id="user-1")
    new_state = CognitionEngine(
        (extractor,),
        CognitiveSpaceService(StateReducer()),
    ).process(event, old_state)

    preference = new_state.get("preferences.study_time")

    assert contributions[0].target_field == "preferences.study_time"
    assert contributions[0].cognition_type is CognitionType.PREFERENCE
    assert contributions[0].value == expected_value
    assert preference.value == expected_value
    assert preference.evidence_refs[0].evidence_id == event.event_id
    assert new_state.version == 1

def test_ignores_unrelated_message():
    event = Event.user_message(
        actor="user-1",
        content="今天天气很好",
    )
    extractor = PreferenceExtractor()
    contributions = extractor.process(event)

    assert len(contributions) == 0


def test_extracting_same_event_twice_returns_same_contribution():
    event = Event.user_message(
        actor="user-1",
        content="我喜欢晚上学习",
    )
    extractor = PreferenceExtractor()

    first = extractor.process(event)
    second = extractor.process(event)

    expected_id = uuid5(
        NAMESPACE_URL,
        f"{event.event_id}:semantic.preference_extractor:preferences.study_time",
    )
    assert first[0].contribution_id == expected_id
    assert second[0].contribution_id == expected_id
    assert first == second


def test_new_study_preference_replaces_value_and_preserves_sources():
    evening_event = Event.user_message(
        actor="user-1",
        content="我喜欢晚上学习",
    )
    morning_event = Event.user_message(
        actor="user-1",
        content="我喜欢早上学习",
    )
    extractor = PreferenceExtractor()
    engine = CognitionEngine(
        (extractor,),
        CognitiveSpaceService(StateReducer()),
    )
    evening_contribution = extractor.process(evening_event)[0]
    morning_contribution = extractor.process(morning_event)[0]

    evening_state = engine.process(
        evening_event,
        SubjectState.empty(subject_id="user-1"),
    )
    morning_state = engine.process(morning_event, evening_state)

    assert evening_state.get("preferences.study_time").value == "晚上"
    current_preference = morning_state.get("preferences.study_time")
    assert current_preference.value == "早上"
    assert morning_state.version == 2
    assert current_preference.contribution_ids == (
        evening_contribution.contribution_id,
        morning_contribution.contribution_id,
    )
    assert tuple(ref.evidence_id for ref in current_preference.evidence_refs) == (
        evening_event.event_id,
        morning_event.event_id,
    )
