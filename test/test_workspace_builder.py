from self_cognition.cognition.semantic.name_extractor import NameExtractor
from self_cognition.cognition.semantic.preference_extractor import (
    PreferenceExtractor,
)
from self_cognition.core.events import Event
from self_cognition.core.state import SubjectState
from self_cognition.core.workspace import Workspace, WorkspaceBuilder, WorkspaceItem
from self_cognition.runtime.engine import CognitionEngine
from self_cognition.runtime.reducer import StateReducer


def test_selects_only_the_study_time_preference_for_the_question():
    preference_event = Event.user_message("user-1", "我喜欢晚上学习")
    name_event = Event.user_message("user-1", "我叫小明")
    engine = CognitionEngine(
        modules=(PreferenceExtractor(), NameExtractor()),
        reducer=StateReducer(),
    )
    state = engine.process(preference_event, SubjectState.empty("user-1"))
    state = engine.process(name_event, state)

    workspace = WorkspaceBuilder().build("我喜欢什么时候学习？", state)

    assert workspace == Workspace(
        subject_id="user-1",
        state_version=2,
        items=(
            WorkspaceItem(
                target_field="preferences.study_time",
                content="晚上",
                evidence_event_ids=(preference_event.event_id,),
                confidence=1.0,
                selection_reason="question maps to this state field",
            ),
        ),
    )
    assert all(item.target_field != "profile.name" for item in workspace.items)


def test_empty_state_returns_an_empty_workspace():
    workspace = WorkspaceBuilder().build(
        "我喜欢什么时候学习？",
        SubjectState.empty("user-1"),
    )

    assert workspace == Workspace(
        subject_id="user-1",
        state_version=0,
        items=(),
    )


def test_unmapped_question_does_not_expose_unrelated_state():
    event = Event.user_message("user-1", "我喜欢晚上学习")
    state = CognitionEngine(
        modules=(PreferenceExtractor(),),
        reducer=StateReducer(),
    ).process(event, SubjectState.empty("user-1"))

    workspace = WorkspaceBuilder().build("我叫什么名字？", state)

    assert workspace.items == ()
    assert workspace.state_version == state.version
