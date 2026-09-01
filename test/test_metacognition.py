from datetime import datetime, timezone
from uuid import UUID

from self_cognition.application.replay import ReplayService
from self_cognition.cognition.metacognition.conflict_extractor import (
    ConflictMetacognitionExtractor,
)
from self_cognition.cognition.semantic.preference_extractor import (
    PreferenceExtractor,
)
from self_cognition.core.contributions import CognitionType
from self_cognition.core.events import Event
from self_cognition.core.state import SubjectState
from self_cognition.core.workspace import WorkspaceBuilder
from self_cognition.executive.dialogue.rule_based import RuleBasedDialogueModel
from self_cognition.infrastructure.persistence.in_memory_event_store import (
    InMemoryEventStore,
)
from self_cognition.runtime.engine import CognitionEngine
from self_cognition.blackboard.reducer import StateReducer
from self_cognition.blackboard.service import CognitiveSpaceService


QUESTION = "我喜欢什么时候学习？"


def make_event(event_id: int, content: str) -> Event:
    return Event.user_message(
        "user-1",
        content,
        event_id=UUID(int=event_id),
        clock=FixedClock(datetime(2026, 8, 13, 12, tzinfo=timezone.utc)),
    )


class FixedClock:
    def __init__(self, value: datetime) -> None:
        self._value = value

    def now(self) -> datetime:
        return self._value


def make_engine() -> CognitionEngine:
    return CognitionEngine(
        modules=(PreferenceExtractor(), ConflictMetacognitionExtractor()),
        cognitive_space=CognitiveSpaceService(StateReducer()),
    )


def test_explicit_uncertainty_becomes_an_evidenced_contribution():
    event = make_event(1, "我不确定更喜欢早上还是晚上学习")
    extractor = ConflictMetacognitionExtractor()

    first = extractor.process(event)
    second = extractor.process(event)

    assert first == second
    assert len(first) == 1
    contribution = first[0]
    assert contribution.target_field == (
        "metacognition.uncertainties.preferences.study_time"
    )
    assert contribution.value == "早上或晚上"
    assert contribution.cognition_type is CognitionType.INFERENCE
    assert contribution.confidence == 1.0
    assert contribution.evidence_refs[0].evidence_id == event.event_id
    assert contribution.source_module == "metacognition.conflict_extractor"


def test_mutually_exclusive_values_create_conflict_and_metacognition():
    event = make_event(2, "我既喜欢早上学习，也喜欢晚上学习")

    state = make_engine().process(event, SubjectState.empty("user-1"))

    assert "preferences.study_time" not in state.entries
    assert PreferenceExtractor().process(event) == ()
    assert len(state.conflicts) == 1
    conflict = next(iter(state.conflicts))
    assert conflict.target_field == "preferences.study_time"
    assert len(conflict.candidate_contribution_ids) == 2
    meta_entry = state.get("metacognition.conflicts.preferences.study_time")
    assert meta_entry.cognition_type is CognitionType.INFERENCE
    assert meta_entry.value == "早上和晚上"
    assert meta_entry.evidence_refs[0].evidence_id == event.event_id


def test_uncertainty_reaches_workspace_and_observable_answer():
    event = make_event(3, "我不确定更喜欢早上还是晚上学习")
    state = make_engine().process(event, SubjectState.empty("user-1"))

    workspace = WorkspaceBuilder().build(QUESTION, state)
    response = RuleBasedDialogueModel().respond(QUESTION, workspace)

    assert [item.target_field for item in workspace.items] == [
        "metacognition.uncertainties.preferences.study_time"
    ]
    assert response.text == "你还不确定更喜欢早上还是晚上学习。"
    assert response.evidence_refs[0].evidence_id == event.event_id


def test_conflict_takes_priority_over_an_older_certain_preference():
    certain_event = make_event(4, "我喜欢晚上学习")
    conflict_event = make_event(5, "我既喜欢早上学习，也喜欢晚上学习")
    engine = make_engine()
    state = engine.process(certain_event, SubjectState.empty("user-1"))
    state = engine.process(conflict_event, state)

    workspace = WorkspaceBuilder().build(QUESTION, state)
    response = RuleBasedDialogueModel().respond(QUESTION, workspace)

    assert state.get("preferences.study_time").value == "晚上"
    assert response.text == "你的学习时间偏好存在冲突：同时提到了早上和晚上。"
    assert response.evidence_refs[0].evidence_id == conflict_event.event_id
    assert certain_event.event_id not in {
        ref.evidence_id for ref in response.evidence_refs
    }


def test_metacognition_module_can_be_disabled():
    event = make_event(6, "我不确定更喜欢早上还是晚上学习")
    old_state = SubjectState.empty("user-1")

    state = CognitionEngine(
        modules=(PreferenceExtractor(),),
        cognitive_space=CognitiveSpaceService(StateReducer()),
    ).process(event, old_state)

    assert state is old_state
    assert not any(field.startswith("metacognition.") for field in state.entries)


def test_conflict_and_metacognition_replay_is_deterministic():
    event = make_event(7, "我既喜欢早上学习，也喜欢晚上学习")
    store = InMemoryEventStore()
    store.append(event)

    replay = ReplayService(store, make_engine())
    first = replay.replay(event.subject)
    second = replay.replay(event.subject)

    assert second == first
    assert len(first.conflicts) == 1
    assert first.get(
        "metacognition.conflicts.preferences.study_time"
    ).evidence_refs[0].evidence_id == event.event_id
