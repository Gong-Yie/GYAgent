from datetime import datetime, timedelta, timezone
from uuid import UUID

from self_cognition.cognition.episodic.memory_extractor import (
    EpisodicMemoryExtractor,
)
from self_cognition.core.contributions import CognitionType
from self_cognition.cognition.semantic.preference_extractor import (
    PreferenceExtractor,
)
from self_cognition.core.events import Event
from self_cognition.core.state import SubjectState
from self_cognition.core.workspace import WorkspaceBuilder
from self_cognition.executive.dialogue.rule_based import RuleBasedDialogueModel
from self_cognition.runtime.engine import CognitionEngine
from self_cognition.blackboard.reducer import StateReducer
from self_cognition.blackboard.service import CognitiveSpaceService
from self_cognition.runtime.run_context import RunContext


def make_context() -> RunContext:
    return RunContext(
        run_id=UUID(int=1),
        correlation_id=UUID(int=2),
        deadline=datetime.now(timezone.utc) + timedelta(minutes=1),
    )


def test_records_one_concrete_experience_with_event_evidence():
    event = Event.user_message("user-1", "今天我去了公园")

    first = EpisodicMemoryExtractor().process(event)
    second = EpisodicMemoryExtractor().process(event)

    assert first == second
    assert len(first) == 1
    contribution = first[0]
    assert contribution.target_field.startswith("episodic.experience.")
    assert contribution.cognition_type is CognitionType.FACT
    assert contribution.target_field.endswith(str(event.event_id))
    assert contribution.value == {
        "text": "今天我去了公园",
        "people": ("我",),
        "time": "今天",
        "environment": "公园",
        "action": "去了",
        "result": "",
        "salience": 0.5,
    }
    assert contribution.confidence == 1.0
    assert contribution.evidence_refs[0].evidence_id == event.event_id


def test_ignores_non_experiential_or_unsubscribed_text():
    extractor = EpisodicMemoryExtractor()

    assert extractor.process(Event.user_message("user-1", "我喜欢晚上学习")) == ()
    assert extractor.process(Event.user_message("user-1", "我叫什么名字？")) == ()


def test_multiple_experiences_reach_workspace_and_answer_with_sources():
    first_event = Event.user_message(
        "user-1",
        "昨天我读完了一本书",
        event_id=UUID(int=10),
        clock=FixedClock(datetime(2026, 8, 12, 12, tzinfo=timezone.utc)),
    )
    second_event = Event.user_message(
        "user-1",
        "今天我去了公园",
        event_id=UUID(int=11),
        clock=FixedClock(datetime(2026, 8, 13, 12, tzinfo=timezone.utc)),
    )
    engine = CognitionEngine(
        modules=(EpisodicMemoryExtractor(),),
        cognitive_space=CognitiveSpaceService(StateReducer()),
    )
    state = engine.process(second_event, SubjectState.empty("user-1"))
    state = engine.process(first_event, state)

    workspace = WorkspaceBuilder().build("我经历过什么？", state)
    response = RuleBasedDialogueModel().respond("我经历过什么？", workspace)

    assert state.version == 2
    assert len(workspace.items) == 2
    assert response.text == "我记得：昨天我读完了一本书；今天我去了公园"
    assert tuple(ref.evidence_id for ref in response.evidence_refs) == (
        first_event.event_id,
        second_event.event_id,
    )


def test_episodic_module_can_be_disabled_without_affecting_other_modules():
    event = Event.user_message("user-1", "我喜欢晚上学习")
    state = CognitionEngine(
        modules=(PreferenceExtractor(),),
        cognitive_space=CognitiveSpaceService(StateReducer()),
    ).process(event, SubjectState.empty("user-1"))

    assert state.get("preferences.study_time").value == "晚上"
    assert not any(
        field.startswith("episodic.experience.") for field in state.entries
    )


class FixedClock:
    def __init__(self, value: datetime) -> None:
        self._value = value

    def now(self) -> datetime:
        return self._value
