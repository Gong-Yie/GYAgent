from datetime import datetime, timedelta, timezone
from uuid import UUID

from self_cognition.application.replay import ReplayService
from self_cognition.cognition.affect.affect_extractor import AffectExtractor
from self_cognition.core.events import Event
from self_cognition.core.state import SubjectState
from self_cognition.core.workspace import WorkspaceBuilder
from self_cognition.executive.dialogue.rule_based import RuleBasedDialogueModel
from self_cognition.infrastructure.persistence.in_memory_event_store import (
    InMemoryEventStore,
)
from self_cognition.infrastructure.persistence.serialization import (
    state_from_json,
    state_to_json,
)
from self_cognition.runtime.engine import CognitionEngine
from self_cognition.runtime.reducer import StateReducer


EXAM_QUESTION = "我对这次考试感觉怎么样？"
PROJECT_QUESTION = "我对这个项目感觉怎么样？"
HALF_LIFE = timedelta(hours=1)


def make_event(
    event_id: int,
    content: str,
    *,
    actor_id: str = "user-1",
    occurred_at: datetime | None = None,
) -> Event:
    return Event(
        event_id=UUID(int=event_id),
        event_type="user.message",
        actor_id=actor_id,
        content=content,
        occurred_at=occurred_at
        or datetime(2026, 8, 13, 12, tzinfo=timezone.utc),
    )


def make_engine() -> CognitionEngine:
    return CognitionEngine((AffectExtractor(),), StateReducer())


def test_extracts_structured_affect_with_target_scope_and_decay_parameters():
    event = make_event(1, "这次考试通过了，我很开心")
    extractor = AffectExtractor()

    first = extractor.process(event)
    second = extractor.process(event)

    assert first == second
    assert len(first) == 1
    contribution = first[0]
    assert contribution.target_field == "affect.current.exam"
    assert contribution.value == {
        "emotion": "开心",
        "valence": "positive",
        "target": "这次考试",
        "scope": "exam",
        "initial_intensity": 0.8,
        "assessed_at": event.occurred_at.isoformat(),
        "half_life_seconds": 3600.0,
    }
    assert contribution.confidence == 1.0
    assert contribution.evidence_event_ids == (event.event_id,)
    assert contribution.source_event_id == event.event_id
    assert contribution.source_module == "affect.affect_extractor"


def test_affect_scopes_are_selected_by_their_requested_target():
    exam_event = make_event(2, "这次考试通过了，我很开心")
    project_event = make_event(3, "这个项目失败了，我很失望")
    engine = make_engine()
    state = engine.process(exam_event, SubjectState.empty("user-1"))
    state = engine.process(project_event, state)

    exam_workspace = WorkspaceBuilder().build(
        EXAM_QUESTION,
        state,
        as_of=exam_event.occurred_at,
    )
    project_workspace = WorkspaceBuilder().build(
        PROJECT_QUESTION,
        state,
        as_of=project_event.occurred_at,
    )

    assert [item.target_field for item in exam_workspace.items] == [
        "affect.current.exam"
    ]
    assert exam_workspace.items[0].content["scope"] == "exam"
    assert [item.target_field for item in project_workspace.items] == [
        "affect.current.project"
    ]
    assert project_workspace.items[0].content["scope"] == "project"


def test_new_affect_for_the_same_target_replaces_current_view_and_keeps_sources():
    happy_event = make_event(20, "这次考试通过了，我很开心")
    disappointed_event = make_event(
        21,
        "这次考试没通过，我很失望",
        occurred_at=happy_event.occurred_at + timedelta(minutes=10),
    )
    engine = make_engine()
    state = engine.process(happy_event, SubjectState.empty("user-1"))
    state = engine.process(disappointed_event, state)

    entry = state.get("affect.current.exam")
    workspace = WorkspaceBuilder().build(
        EXAM_QUESTION,
        state,
        as_of=disappointed_event.occurred_at,
    )
    response = RuleBasedDialogueModel().respond(EXAM_QUESTION, workspace)

    assert entry.value["emotion"] == "失望"
    assert entry.value["valence"] == "negative"
    assert entry.evidence_event_ids == (
        happy_event.event_id,
        disappointed_event.event_id,
    )
    assert response.text == "你对这次考试感到失望，当前强度约为0.80。"
    assert response.source_event_ids == (
        happy_event.event_id,
        disappointed_event.event_id,
    )
    assert state.version == 2


def test_affect_intensity_halves_without_mutating_the_stored_state():
    event = make_event(4, "这次考试通过了，我很开心")
    state = make_engine().process(event, SubjectState.empty("user-1"))
    original_value = dict(state.get("affect.current.exam").value)

    workspace = WorkspaceBuilder().build(
        EXAM_QUESTION,
        state,
        as_of=event.occurred_at + HALF_LIFE,
    )
    response = RuleBasedDialogueModel().respond(EXAM_QUESTION, workspace)

    assert workspace.items[0].content["current_intensity"] == 0.4
    assert response.text == "你对这次考试感到开心，当前强度约为0.40。"
    assert response.source_event_ids == (event.event_id,)
    assert state.get("affect.current.exam").value == original_value
    assert "current_intensity" not in original_value


def test_affect_below_the_active_threshold_is_not_exposed_as_current():
    event = make_event(5, "这次考试通过了，我很开心")
    state = make_engine().process(event, SubjectState.empty("user-1"))

    workspace = WorkspaceBuilder().build(
        EXAM_QUESTION,
        state,
        as_of=event.occurred_at + 4 * HALF_LIFE,
    )
    response = RuleBasedDialogueModel().respond(EXAM_QUESTION, workspace)

    assert workspace.items == ()
    assert response.text == "我没有足够强的当前情感评估。"
    assert response.source_event_ids == ()


def test_affect_questions_do_not_create_affect_state():
    event = make_event(6, EXAM_QUESTION)
    old_state = SubjectState.empty("user-1")

    assert AffectExtractor().process(event) == ()
    state = make_engine().process(event, old_state)

    assert state is old_state


def test_affect_state_is_isolated_between_subjects():
    first_event = make_event(7, "这次考试通过了，我很开心")
    second_event = make_event(
        8,
        "这个项目失败了，我很失望",
        actor_id="user-2",
    )
    engine = make_engine()

    first_state = engine.process(first_event, SubjectState.empty("user-1"))
    second_state = engine.process(second_event, SubjectState.empty("user-2"))

    assert "affect.current.exam" in first_state.entries
    assert "affect.current.project" not in first_state.entries
    assert "affect.current.project" in second_state.entries
    assert "affect.current.exam" not in second_state.entries


def test_affect_module_can_be_disabled():
    event = make_event(9, "这次考试通过了，我很开心")
    old_state = SubjectState.empty("user-1")

    state = CognitionEngine((), StateReducer()).process(event, old_state)

    assert state is old_state
    assert not any(field.startswith("affect.") for field in state.entries)


def test_affect_state_roundtrips_and_replays_deterministically():
    events = (
        make_event(10, "这次考试通过了，我很开心"),
        make_event(11, "这个项目失败了，我很失望"),
    )
    store = InMemoryEventStore()
    for event in events:
        store.append(event)

    replay = ReplayService(store, make_engine())
    first = replay.replay("user-1")
    second = replay.replay("user-1")

    assert second == first
    assert state_from_json(state_to_json(first)) == first
    assert first.version == 2
