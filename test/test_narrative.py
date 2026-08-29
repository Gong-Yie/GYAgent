from datetime import datetime, timezone
from uuid import UUID

from self_cognition.application.replay import ReplayService
from self_cognition.cognition.episodic.memory_extractor import (
    EpisodicMemoryExtractor,
)
from self_cognition.cognition.narrative.narrative_extractor import (
    NarrativeExtractor,
)
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


QUESTION = "我的项目经历如何发展？"


def make_event(event_id: int, content: str) -> Event:
    return Event(
        event_id=UUID(int=event_id),
        event_type="user.message",
        actor_id="user-1",
        content=content,
        occurred_at=datetime(2026, 8, 13, event_id, tzinfo=timezone.utc),
    )


def make_engine() -> CognitionEngine:
    return CognitionEngine((NarrativeExtractor(),), StateReducer())


def test_extracts_a_deterministic_evidenced_narrative_chapter():
    event = make_event(1, "我开始准备研究项目")
    extractor = NarrativeExtractor()

    first = extractor.process(event)
    second = extractor.process(event)

    assert first == second
    assert len(first) == 1
    contribution = first[0]
    assert contribution.target_field.startswith(
        "narrative.chapter.2026-08-13T01:00:00+00:00."
    )
    assert contribution.value == {
        "theme": "研究项目",
        "stage": "启动",
        "summary": "开始准备研究项目",
        "occurred_at": event.occurred_at.isoformat(),
    }
    assert contribution.confidence == 1.0
    assert contribution.evidence_event_ids == (event.event_id,)
    assert contribution.source_event_id == event.event_id


def test_unrelated_or_narrative_question_events_are_ignored():
    extractor = NarrativeExtractor()

    assert extractor.process(make_event(2, "今天天气很好")) == ()
    assert extractor.process(make_event(3, QUESTION)) == ()


def test_existing_chapters_form_an_ordered_project_narrative_with_sources():
    start_event = make_event(4, "我开始准备研究项目")
    finish_event = make_event(5, "我完成了研究项目")
    engine = make_engine()
    state = engine.process(finish_event, SubjectState.empty("user-1"))
    state = engine.process(start_event, state)

    workspace = WorkspaceBuilder().build(QUESTION, state)
    response = RuleBasedDialogueModel().respond(QUESTION, workspace)

    assert state.version == 2
    assert len(workspace.items) == 2
    assert workspace.items[0].content["stage"] == "启动"
    assert workspace.items[1].content["stage"] == "完成"
    assert response.text == (
        "你的项目叙事是：先是开始准备研究项目，后来完成研究项目。"
    )
    assert response.source_event_ids == (
        start_event.event_id,
        finish_event.event_id,
    )


def test_narrative_is_an_explanation_layer_and_does_not_invent_unknown_history():
    workspace = WorkspaceBuilder().build(QUESTION, SubjectState.empty("user-1"))
    response = RuleBasedDialogueModel().respond(QUESTION, workspace)

    assert workspace.items == ()
    assert response.text == "我还没有形成项目叙事。"
    assert response.source_event_ids == ()


def test_narrative_module_can_be_disabled_without_affecting_state():
    event = make_event(6, "我开始准备研究项目")
    old_state = SubjectState.empty("user-1")

    state = CognitionEngine((), StateReducer()).process(event, old_state)

    assert state is old_state
    assert not any(field.startswith("narrative.") for field in state.entries)


def test_narrative_state_is_isolated_between_subjects():
    first_event = Event(
        event_id=UUID(int=20),
        event_type="user.message",
        actor_id="user-1",
        content="我开始准备研究项目",
        occurred_at=datetime(2026, 8, 13, 12, tzinfo=timezone.utc),
    )
    second_event = Event(
        event_id=UUID(int=21),
        event_type="user.message",
        actor_id="user-2",
        content="我完成了研究项目",
        occurred_at=datetime(2026, 8, 13, 13, tzinfo=timezone.utc),
    )
    engine = make_engine()

    first_state = engine.process(first_event, SubjectState.empty("user-1"))
    second_state = engine.process(second_event, SubjectState.empty("user-2"))

    assert next(iter(first_state.entries.values())).value["stage"] == "启动"
    assert next(iter(second_state.entries.values())).value["stage"] == "完成"
    assert first_state.subject_id == "user-1"
    assert second_state.subject_id == "user-2"


def test_narrative_and_episodic_modules_can_record_the_same_event():
    event = Event(
        event_id=UUID(int=22),
        event_type="user.message",
        actor_id="user-1",
        content="今天我开始准备研究项目",
        occurred_at=datetime(2026, 8, 13, 14, tzinfo=timezone.utc),
    )
    engine = CognitionEngine(
        (EpisodicMemoryExtractor(), NarrativeExtractor()),
        StateReducer(),
    )

    state = engine.process(event, SubjectState.empty("user-1"))

    assert any(field.startswith("episodic.experience.") for field in state.entries)
    assert any(field.startswith("narrative.chapter.") for field in state.entries)
    assert state.version == 2


def test_narrative_state_roundtrips_and_replays_deterministically():
    events = (
        make_event(7, "我开始准备研究项目"),
        make_event(8, "我完成了研究项目"),
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
