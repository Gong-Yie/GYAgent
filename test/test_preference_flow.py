from uuid import NAMESPACE_URL, uuid5

import pytest

from self_cognition.core.events import Event
from self_cognition.cognition.semantic.preference_extractor import PreferenceExtractor
from self_cognition.core.state import SubjectState
from self_cognition.runtime.reducer import StateReducer

@pytest.mark.parametrize(
    ("content", "expected_value"),
    [
        ("我喜欢晚上学习", "晚上"),
        ("我喜欢早上学习", "早上"),
    ],
)
def test_remembers_user_study_preference(content: str, expected_value: str):
    event = Event.user_message(
        actor_id="user-1",
        content=content,
    )
    extractor = PreferenceExtractor()
    contributions = extractor.process(event)

    assert len(contributions) == 1

    old_state = SubjectState.empty(subject_id="user-1")
    new_state = StateReducer().apply(
        old_state,
        contributions[0],
    )

    preference = new_state.get("preferences.study_time")

    assert contributions[0].target_field == "preferences.study_time"
    assert contributions[0].value == expected_value
    assert preference.value == expected_value
    assert preference.evidence_event_ids == (event.event_id,)
    assert new_state.version == 1

def test_ignores_unrelated_message():
    event = Event.user_message(
        actor_id="user-1",
        content="今天天气很好",
    )
    extractor = PreferenceExtractor()
    contributions = extractor.process(event)

    assert len(contributions) == 0


def test_extracting_same_event_twice_returns_same_contribution():
    event = Event.user_message(
        actor_id="user-1",
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
        actor_id="user-1",
        content="我喜欢晚上学习",
    )
    morning_event = Event.user_message(
        actor_id="user-1",
        content="我喜欢早上学习",
    )
    extractor = PreferenceExtractor()
    reducer = StateReducer()
    evening_contribution = extractor.process(evening_event)[0]
    morning_contribution = extractor.process(morning_event)[0]

    evening_state = reducer.apply(
        SubjectState.empty(subject_id="user-1"),
        evening_contribution,
    )
    morning_state = reducer.apply(evening_state, morning_contribution)

    assert evening_state.get("preferences.study_time").value == "晚上"
    current_preference = morning_state.get("preferences.study_time")
    assert current_preference.value == "早上"
    assert morning_state.version == 2
    assert current_preference.contribution_ids == (
        evening_contribution.contribution_id,
        morning_contribution.contribution_id,
    )
    assert current_preference.evidence_event_ids == (
        evening_event.event_id,
        morning_event.event_id,
    )
