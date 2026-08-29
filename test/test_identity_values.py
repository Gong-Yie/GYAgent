from datetime import datetime, timezone
from uuid import UUID

from self_cognition.application.replay import ReplayService
from self_cognition.cognition.identity.identity_value_extractor import (
    IdentityValueExtractor,
)
from self_cognition.core.contributions import Contribution
from self_cognition.core.events import Event
from self_cognition.core.state import SubjectState
from self_cognition.core.workspace import WorkspaceBuilder
from self_cognition.executive.dialogue.rule_based import RuleBasedDialogueModel
from self_cognition.infrastructure.persistence.in_memory_event_store import (
    InMemoryEventStore,
)
from self_cognition.runtime.engine import CognitionEngine
from self_cognition.runtime.reducer import StateReducer


ROLE_QUESTION = "我的角色是什么？"
VALUE_QUESTION = "我最重视什么？"


def make_event(event_id: int, content: str, actor_id: str = "user-1") -> Event:
    return Event(
        event_id=UUID(int=event_id),
        event_type="user.message",
        actor_id=actor_id,
        content=content,
        occurred_at=datetime(2026, 8, 13, 12, tzinfo=timezone.utc),
    )


def make_contribution(
    event_id: int,
    *,
    target_field: str,
    value: str,
    confidence: float,
    confirmed: bool = False,
) -> Contribution:
    source_event_id = UUID(int=event_id)
    return Contribution(
        contribution_id=UUID(int=event_id + 100),
        target_subject_id="user-1",
        target_field=target_field,
        value=value,
        confidence=confidence,
        evidence_event_ids=(source_event_id,),
        source_event_id=source_event_id,
        source_module="test.identity_values",
        explicitly_confirmed=confirmed,
    )


def make_engine() -> CognitionEngine:
    return CognitionEngine((IdentityValueExtractor(),), StateReducer())


def test_identity_and_value_statements_become_evidenced_contributions():
    role_event = make_event(1, "我的角色是研究助手")
    value_event = make_event(2, "我最重视诚实")
    extractor = IdentityValueExtractor()

    role = extractor.process(role_event)
    value = extractor.process(value_event)

    assert role == extractor.process(role_event)
    assert role[0].target_field == "identity.role"
    assert role[0].value == "研究助手"
    assert role[0].confidence == 1.0
    assert role[0].explicitly_confirmed is False
    assert role[0].evidence_event_ids == (role_event.event_id,)
    assert value[0].target_field == "values.principle"
    assert value[0].value == "诚实"
    assert value[0].confidence == 1.0
    assert value[0].evidence_event_ids == (value_event.event_id,)


def test_identity_and_value_questions_are_not_recorded_as_answers():
    extractor = IdentityValueExtractor()

    assert extractor.process(make_event(29, ROLE_QUESTION)) == ()
    assert extractor.process(make_event(30, VALUE_QUESTION)) == ()

    role_workspace = WorkspaceBuilder().build(
        ROLE_QUESTION,
        SubjectState.empty("user-1"),
    )
    role_response = RuleBasedDialogueModel().respond(
        ROLE_QUESTION,
        role_workspace,
    )

    assert role_response.text == "我还不知道你的角色。"
    assert role_response.source_event_ids == ()


def test_protected_fields_require_high_confidence_for_first_write():
    contribution = make_contribution(
        3,
        target_field="identity.role",
        value="研究助手",
        confidence=0.89,
    )
    old_state = SubjectState.empty("user-1")

    new_state = StateReducer().apply(old_state, contribution)

    assert new_state is old_state
    assert "identity.role" not in new_state.entries
    assert contribution.contribution_id not in new_state.applied_contribution_ids


def test_low_confidence_rule_does_not_apply_to_ordinary_preferences():
    contribution = make_contribution(
        40,
        target_field="preferences.study_time",
        value="晚上",
        confidence=0.5,
    )

    state = StateReducer().apply(SubjectState.empty("user-1"), contribution)

    assert state.get("preferences.study_time").value == "晚上"
    assert state.version == 1


def test_low_confidence_protected_candidates_do_not_create_a_conflict():
    first = make_contribution(
        41,
        target_field="identity.role",
        value="研究助手",
        confidence=0.5,
    )
    second = make_contribution(
        42,
        target_field="identity.role",
        value="学习助手",
        confidence=0.5,
    )
    old_state = SubjectState.empty("user-1")

    state = StateReducer().apply_many(old_state, (first, second))

    assert state is old_state
    assert state.conflicts == frozenset()


def test_unconfirmed_identity_change_is_rejected_but_confirmation_is_accepted():
    reducer = StateReducer()
    initial = make_contribution(
        4,
        target_field="identity.role",
        value="研究助手",
        confidence=1.0,
    )
    unconfirmed = make_contribution(
        5,
        target_field="identity.role",
        value="学习助手",
        confidence=1.0,
    )
    confirmed = make_contribution(
        6,
        target_field="identity.role",
        value="学习助手",
        confidence=1.0,
        confirmed=True,
    )

    first_state = reducer.apply(SubjectState.empty("user-1"), initial)
    rejected_state = reducer.apply(first_state, unconfirmed)
    final_state = reducer.apply(rejected_state, confirmed)

    assert rejected_state is first_state
    assert rejected_state.get("identity.role").value == "研究助手"
    assert unconfirmed.contribution_id not in rejected_state.applied_contribution_ids
    entry = final_state.get("identity.role")
    assert entry.value == "学习助手"
    assert entry.evidence_event_ids == (
        initial.source_event_id,
        confirmed.source_event_id,
    )
    assert entry.contribution_ids == (
        initial.contribution_id,
        confirmed.contribution_id,
    )
    assert final_state.version == 2


def test_value_change_uses_the_same_confirmation_threshold():
    engine = make_engine()
    initial_event = make_event(7, "我最重视诚实")
    unconfirmed_event = make_event(8, "我最重视可靠")
    confirmed_event = make_event(9, "我确认将最重视的原则改为可靠")

    initial_state = engine.process(initial_event, SubjectState.empty("user-1"))
    rejected_state = engine.process(unconfirmed_event, initial_state)
    final_state = engine.process(confirmed_event, rejected_state)

    assert rejected_state is initial_state
    assert rejected_state.get("values.principle").value == "诚实"
    assert final_state.get("values.principle").value == "可靠"
    assert final_state.version == 2


def test_identity_and_values_reach_workspace_and_answers_with_sources():
    role_event = make_event(10, "我的角色是研究助手")
    value_event = make_event(11, "我最重视诚实")
    engine = make_engine()
    state = engine.process(role_event, SubjectState.empty("user-1"))
    state = engine.process(value_event, state)

    role_workspace = WorkspaceBuilder().build(ROLE_QUESTION, state)
    role_response = RuleBasedDialogueModel().respond(
        ROLE_QUESTION,
        role_workspace,
    )
    value_workspace = WorkspaceBuilder().build(VALUE_QUESTION, state)
    value_response = RuleBasedDialogueModel().respond(
        VALUE_QUESTION,
        value_workspace,
    )

    assert [item.target_field for item in role_workspace.items] == [
        "identity.role"
    ]
    assert role_response.text == "你的角色是研究助手。"
    assert role_response.source_event_ids == (role_event.event_id,)
    assert [item.target_field for item in value_workspace.items] == [
        "values.principle"
    ]
    assert value_response.text == "你最重视诚实。"
    assert value_response.source_event_ids == (value_event.event_id,)


def test_identity_state_is_isolated_between_subjects():
    engine = make_engine()
    first_event = make_event(12, "我的角色是研究助手", "user-1")
    second_event = make_event(13, "我的角色是学习助手", "user-2")

    first_state = engine.process(first_event, SubjectState.empty("user-1"))
    second_state = engine.process(second_event, SubjectState.empty("user-2"))

    assert first_state.get("identity.role").value == "研究助手"
    assert second_state.get("identity.role").value == "学习助手"


def test_identity_value_module_can_be_disabled():
    event = make_event(14, "我的角色是研究助手")
    old_state = SubjectState.empty("user-1")

    state = CognitionEngine((), StateReducer()).process(event, old_state)

    assert state is old_state
    assert "identity.role" not in state.entries


def test_identity_and_value_replay_is_deterministic():
    events = (
        make_event(15, "我的角色是研究助手"),
        make_event(16, "我的角色是学习助手"),
        make_event(17, "我确认将角色改为学习助手"),
        make_event(18, "我最重视诚实"),
    )
    store = InMemoryEventStore()
    for event in events:
        store.append(event)

    replay = ReplayService(store, make_engine())
    first = replay.replay("user-1")
    second = replay.replay("user-1")

    assert second == first
    assert first.get("identity.role").value == "学习助手"
    assert first.get("values.principle").value == "诚实"
    assert first.version == 3
