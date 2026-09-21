from self_cognition.blackboard.reducer import StateReducer
from self_cognition.blackboard.service import CognitiveSpaceService
from self_cognition.core.events import Event
from self_cognition.core.state import SubjectState
from self_cognition.runtime.engine import CognitionEngine


def _engine():
    from self_cognition.cognition.affect.fast_reaction import FastAffectExtractor

    return CognitionEngine(
        (FastAffectExtractor(),),
        CognitiveSpaceService(StateReducer()),
    )


def test_fast_affect_reaction_and_mood_are_written():
    event = Event.user_message("user-1", "今天项目成功交付，我很开心")
    state = _engine().process(event, SubjectState.empty("user-1"))

    reaction_field = "affect.reaction.interaction"
    mood_field = "mood.current"
    assert reaction_field in state.entries
    assert mood_field in state.entries
    reaction = state.get(reaction_field).value
    mood = state.get(mood_field).value
    assert reaction["emotion"] == "positive"
    assert reaction["arousal"] == 0.6
    assert mood["intensity"] > 0


def test_boredom_reaction_is_low_arousal():
    event = Event.user_message("user-1", "好无聊，没什么事可做")
    state = _engine().process(event, SubjectState.empty("user-1"))

    reaction = state.get("affect.reaction.interaction").value
    assert reaction["emotion"] == "boredom"
    assert reaction["valence"] == "negative"
    assert reaction["arousal"] == 0.1

def test_workspace_projects_active_emotion_and_mood():
    from self_cognition.core.workspace import WorkspaceBuilder

    event = Event.user_message("user-1", "今天项目成功交付，我很开心")
    state = _engine().process(event, SubjectState.empty("user-1"))

    workspace = WorkspaceBuilder().build("聊一聊当前状态", state)

    fields = {item["field"] for item in workspace.fixed_context.emotion}
    assert "affect.reaction.interaction" in fields
    assert "mood.current" in fields


def test_self_appraisal_writes_the_agents_own_reaction_to_the_mind():
    """评估事件属于 mind：Agent 的反应写在 mind 上，对象仍指向被评估的主体。"""
    from self_cognition.core.scopes import SubjectKind
    from self_cognition.core.workspace import WorkspaceBuilder

    source = Event.user_message("user-1", "好无聊，没什么事可做")
    appraisal = Event.assessment_requested(source, "评估自身情绪")
    assert appraisal.subject.subject.kind is SubjectKind.MIND

    mind = SubjectState.empty(
        "default-mind",
        mind_id="default-mind",
        subject_kind=SubjectKind.MIND,
    )
    state = _engine().process(appraisal, mind)

    reaction = state.get("affect.reaction.interaction").value
    assert reaction["emotion"] == "boredom"
    assert reaction["arousal"] == 0.1
    # 情绪的对象是被评估的用户，而不是 Agent 自己
    assert reaction["target"] == "user-1"
    assert reaction["cause"] == "user.message"
    # 证据同时包含评估事件和被评估的源事件，反应可回溯
    assert len(state.get("affect.reaction.interaction").evidence_refs) == 2

    workspace = WorkspaceBuilder().build("自身情绪评估", state)
    fields = {item["field"] for item in workspace.fixed_context.emotion}
    assert "affect.reaction.interaction" in fields


def test_appraisal_without_a_recognisable_reaction_writes_nothing():
    from self_cognition.core.scopes import SubjectKind

    source = Event.user_message("user-1", "我喜欢晚上学习")
    appraisal = Event.assessment_requested(source, "评估自身情绪")
    mind = SubjectState.empty(
        "default-mind",
        mind_id="default-mind",
        subject_kind=SubjectKind.MIND,
    )

    state = _engine().process(appraisal, mind)

    assert not any(field.startswith("affect.") for field in state.entries)
