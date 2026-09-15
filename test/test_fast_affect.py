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
