from self_cognition.cognition.relationship.relationship_extractor import (
    RelationshipExtractor,
)
from self_cognition.cognition.semantic.preference_extractor import (
    PreferenceExtractor,
)
from self_cognition.core.events import Event
from self_cognition.core.state import SubjectState
from self_cognition.core.workspace import WorkspaceBuilder
from self_cognition.executive.dialogue.rule_based import RuleBasedDialogueModel
from self_cognition.runtime.engine import CognitionEngine
from self_cognition.runtime.reducer import StateReducer


def test_extracts_relationship_as_an_evidenced_contribution():
    event = Event.user_message("user-1", "小明是我的朋友")
    extractor = RelationshipExtractor()

    first = extractor.process(event)
    second = extractor.process(event)

    assert first == second
    assert len(first) == 1
    contribution = first[0]
    assert contribution.target_subject_id == "user-1"
    assert contribution.target_field == "relationships.小明.role"
    assert contribution.value == "朋友"
    assert contribution.confidence == 1.0
    assert contribution.evidence_event_ids == (event.event_id,)
    assert contribution.source_event_id == event.event_id


def test_distinguishes_two_relationship_objects_in_state():
    ming_event = Event.user_message("user-1", "小明是我的朋友")
    hong_event = Event.user_message("user-1", "小红是我的同事")
    engine = CognitionEngine((RelationshipExtractor(),), StateReducer())

    state = engine.process(ming_event, SubjectState.empty("user-1"))
    state = engine.process(hong_event, state)

    assert state.get("relationships.小明.role").value == "朋友"
    assert state.get("relationships.小红.role").value == "同事"
    assert state.version == 2


def test_relationship_workspace_and_answer_select_only_requested_object():
    ming_event = Event.user_message("user-1", "小明是我的朋友")
    hong_event = Event.user_message("user-1", "小红是我的同事")
    engine = CognitionEngine((RelationshipExtractor(),), StateReducer())
    state = engine.process(ming_event, SubjectState.empty("user-1"))
    state = engine.process(hong_event, state)

    workspace = WorkspaceBuilder().build("我和小明是什么关系？", state)
    response = RuleBasedDialogueModel().respond(
        "我和小明是什么关系？",
        workspace,
    )

    assert len(workspace.items) == 1
    assert workspace.items[0].target_field == "relationships.小明.role"
    assert response.text == "小明是你的朋友。"
    assert response.source_event_ids == (ming_event.event_id,)
    assert hong_event.event_id not in response.source_event_ids


def test_relationship_state_is_isolated_between_owners():
    engine = CognitionEngine((RelationshipExtractor(),), StateReducer())
    user_one_event = Event.user_message("user-1", "小明是我的朋友")
    user_two_event = Event.user_message("user-2", "小红是我的同事")

    user_one_state = engine.process(
        user_one_event,
        SubjectState.empty("user-1"),
    )
    user_two_state = engine.process(
        user_two_event,
        SubjectState.empty("user-2"),
    )

    assert "relationships.小明.role" in user_one_state.entries
    assert "relationships.小红.role" not in user_one_state.entries
    assert "relationships.小红.role" in user_two_state.entries
    assert "relationships.小明.role" not in user_two_state.entries


def test_unknown_relationship_is_not_invented():
    event = Event.user_message("user-1", "小红是我的同事")
    state = CognitionEngine(
        (RelationshipExtractor(),),
        StateReducer(),
    ).process(event, SubjectState.empty("user-1"))

    workspace = WorkspaceBuilder().build("我和小明是什么关系？", state)
    response = RuleBasedDialogueModel().respond(
        "我和小明是什么关系？",
        workspace,
    )

    assert workspace.items == ()
    assert response.text == "我还不知道你和小明是什么关系。"
    assert response.source_event_ids == ()


def test_relationship_module_can_be_disabled():
    event = Event.user_message("user-1", "我喜欢晚上学习")
    state = CognitionEngine(
        (PreferenceExtractor(),),
        StateReducer(),
    ).process(event, SubjectState.empty("user-1"))

    assert state.get("preferences.study_time").value == "晚上"
    assert not any(field.startswith("relationships.") for field in state.entries)
