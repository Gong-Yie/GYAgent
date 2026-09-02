from uuid import NAMESPACE_URL, uuid5

import pytest

from self_cognition.cognition.semantic.name_extractor import NameExtractor
from self_cognition.cognition.semantic.preference_extractor import (
    PreferenceExtractor,
)
from self_cognition.core.errors import SubjectMismatchError
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.events import Event
from self_cognition.core.contributions import CognitiveContribution, CognitionType
from self_cognition.core.state import SubjectState
from self_cognition.runtime.engine import CognitionEngine
from self_cognition.blackboard.reducer import StateReducer
from self_cognition.blackboard.service import CognitiveSpaceService


def make_engine() -> CognitionEngine:
    return CognitionEngine(
        modules=(PreferenceExtractor(), NameExtractor()),
        cognitive_space=CognitiveSpaceService(StateReducer()),
    )


class RecordingModule:
    def __init__(self, subscriptions: frozenset[str], target_field: str):
        self.subscriptions = subscriptions
        self.target_field = target_field
        self.processed_event_ids = []

    def process(self, event: Event) -> tuple[CognitiveContribution, ...]:
        self.processed_event_ids.append(event.event_id)
        contribution_id = uuid5(
            NAMESPACE_URL,
            f"{event.event_id}:test.recording_module:{self.target_field}",
        )
        return (
            CognitiveContribution.set_from_event(
                event,
                contribution_id=contribution_id,
                target_field=self.target_field,
                cognition_type=CognitionType.FACT,
                value=self.target_field,
                confidence=1.0,
                evidence_refs=(EvidenceRef.for_event(event),),
                source_module="test.recording_module",
                module_version="1",
            ),
        )


def test_processes_preference_event_end_to_end():
    event = Event.user_message("user-1", "我喜欢晚上学习")

    state = make_engine().process(event, SubjectState.empty("user-1"))

    preference = state.get("preferences.study_time")
    assert preference.value == "晚上"
    assert preference.evidence_refs[0].evidence_id == event.event_id
    assert state.changes[0].contribution.target_version == 0
    assert "profile.name" not in state.entries
    assert state.version == 1


def test_processes_name_event_without_creating_preference():
    event = Event.user_message("user-1", "我叫小明")

    state = make_engine().process(event, SubjectState.empty("user-1"))

    name = state.get("profile.name")
    assert name.value == "小明"
    assert name.evidence_refs[0].evidence_id == event.event_id
    assert "preferences.study_time" not in state.entries
    assert state.version == 1


def test_multiple_modules_feed_the_same_reducer_across_events():
    preference_event = Event.user_message("user-1", "我喜欢晚上学习")
    name_event = Event.user_message("user-1", "我叫小明")
    engine = make_engine()

    preference_state = engine.process(
        preference_event,
        SubjectState.empty("user-1"),
    )
    final_state = engine.process(name_event, preference_state)

    assert final_state.get("preferences.study_time").value == "晚上"
    assert final_state.get("profile.name").value == "小明"
    assert final_state.version == 2
    assert final_state.changes[-1].contribution.target_version == 1
    assert len(final_state.applied_contribution_ids) == 2


def test_unrelated_event_returns_unchanged_state():
    event = Event.user_message("user-1", "今天天气很好")
    old_state = SubjectState.empty("user-1")

    new_state = make_engine().process(event, old_state)

    assert new_state is old_state


def test_repeated_event_changes_state_only_once():
    event = Event.user_message("user-1", "我喜欢晚上学习")
    engine = make_engine()

    first_state = engine.process(event, SubjectState.empty("user-1"))
    second_state = engine.process(event, first_state)

    assert second_state is first_state
    assert second_state.version == 1
    assert len(second_state.applied_contribution_ids) == 1


def test_rejects_event_for_another_subject_without_changing_state():
    event = Event.user_message("user-2", "我喜欢晚上学习")
    old_state = SubjectState.empty("user-1")

    with pytest.raises(SubjectMismatchError):
        make_engine().process(event, old_state)

    assert old_state == SubjectState.empty("user-1")


def test_routes_subscribed_event_to_module():
    module = RecordingModule(
        subscriptions=frozenset({"user.message"}),
        target_field="test.routed",
    )
    event = Event.user_message("user-1", "测试路由")
    engine = CognitionEngine((module,), CognitiveSpaceService(StateReducer()))

    state = engine.process(event, SubjectState.empty("user-1"))

    assert module.processed_event_ids == [event.event_id]
    assert state.get("test.routed").value == "test.routed"


def test_does_not_call_module_for_unsubscribed_event():
    module = RecordingModule(
        subscriptions=frozenset({"tool.result"}),
        target_field="test.routed",
    )
    event = Event.user_message("user-1", "工具执行完成")
    old_state = SubjectState.empty("user-1")
    engine = CognitionEngine((module,), CognitiveSpaceService(StateReducer()))

    new_state = engine.process(event, old_state)

    assert module.processed_event_ids == []
    assert new_state is old_state


def test_module_registration_order_does_not_change_result():
    event = Event.user_message("user-1", "测试模块顺序")
    first = RecordingModule(frozenset({"user.message"}), "test.first")
    second = RecordingModule(frozenset({"user.message"}), "test.second")

    forward = CognitionEngine(
        (first, second),
        CognitiveSpaceService(StateReducer()),
    ).process(
        event,
        SubjectState.empty("user-1"),
    )
    reverse = CognitionEngine(
        (second, first),
        CognitiveSpaceService(StateReducer()),
    ).process(
        event,
        SubjectState.empty("user-1"),
    )

    assert forward == reverse
    assert forward.version == 2


def test_one_module_failure_does_not_block_independent_modules():
    class FailingModule:
        subscriptions = frozenset({"user.message"})

        def process(self, event: Event) -> tuple[CognitiveContribution, ...]:
            raise RuntimeError("module failure")

    event = Event.user_message("user-1", "我喜欢晚上学习")
    state = CognitionEngine(
        modules=(FailingModule(), PreferenceExtractor()),
        cognitive_space=CognitiveSpaceService(StateReducer()),
    ).process(event, SubjectState.empty("user-1"))

    assert state.get("preferences.study_time").value == "晚上"
    assert state.version == 1
