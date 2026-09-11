from datetime import datetime, timedelta, timezone
from uuid import NAMESPACE_URL, uuid5

import pytest

from self_cognition.bootstrap import build_container
from self_cognition.application.results import ProcessEventStatus
from self_cognition.core.events import Event, EventEnvelope
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.dialogue import DialogueRequest
from self_cognition.cognition.semantic.preference_extractor import PreferenceExtractor
from self_cognition.core.contributions import CognitionType
from self_cognition.core.state import SubjectState
from self_cognition.blackboard.reducer import StateReducer
from self_cognition.blackboard.service import CognitiveSpaceService
from self_cognition.runtime.engine import CognitionEngine
from self_cognition.runtime.run_context import RunContext

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


@pytest.mark.parametrize(
    ("content", "expected_value"),
    [
        ("我偏好在早晨读书", "早上"),
        ("我更喜欢夜间学习", "晚上"),
    ],
)
def test_study_preference_supports_synonyms(content: str, expected_value: str):
    event = Event.user_message(actor="user-1", content=content)

    contributions = PreferenceExtractor().process(event)

    assert contributions[0].value == expected_value


@pytest.mark.parametrize("content", ["我不喜欢晚上学习", "我喜欢什么时候学习？"])
def test_non_preference_language_is_not_recorded(content: str):
    event = Event.user_message(actor="user-1", content=content)

    assert PreferenceExtractor().process(event) == ()


def test_preference_survives_restart_with_previous_value_as_history(tmp_path):
    first = build_container(tmp_path, dotenv_path=tmp_path / "missing.env")
    event_subject = Event.user_message("user-1", "x").subject
    subject = EventEnvelope.user_message(event_subject, "我喜欢晚上学习")
    updated = EventEnvelope.user_message(
        event_subject, "我以前喜欢晚上学习，但最近改成早上了"
    )
    context = RunContext(
        uuid5(NAMESPACE_URL, "stage33-run"),
        uuid5(NAMESPACE_URL, "stage33-correlation"),
        datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    for event in (subject, updated):
        assert (
            first.process_event.process(event, context).status
            is ProcessEventStatus.SUCCEEDED
        )
    expected = first.state_repository.load(event_subject)
    first.lifecycle.stop()

    restarted = build_container(tmp_path, dotenv_path=tmp_path / "missing.env")
    current = restarted.state_repository.load(event_subject).get("preferences.study_time")

    assert restarted.state_repository.load(event_subject) == expected
    assert current.value == "早上"
    assert tuple(ref.evidence_id for ref in current.evidence_refs) == (
        subject.event_id,
        updated.event_id,
    )
    answer = restarted.converse.converse(
        DialogueRequest(
            EventEnvelope.user_message(event_subject, "我喜欢什么时候学习？")
        ),
        context,
    )
    assert answer.response is not None
    assert answer.evidence_refs == (
        EvidenceRef.for_event(subject),
        EvidenceRef.for_event(updated),
    )
    restarted.lifecycle.stop()


def test_current_preference_replaces_previous_value_for_natural_language_update():
    evening_event = Event.user_message(
        actor="user-1",
        content="我以前喜欢晚上学习，但最近改成早上了",
    )
    extractor = PreferenceExtractor()

    contributions = extractor.process(evening_event)

    assert len(contributions) == 1
    assert contributions[0].value == "早上"
