from uuid import UUID, uuid4

import pytest

from self_cognition.core.contributions import Contribution
from self_cognition.core.errors import SubjectMismatchError
from self_cognition.core.state import ConflictRecord, StateEntry, SubjectState
from self_cognition.runtime.reducer import StateReducer


def test_rejects_contribution_for_another_subject_without_changing_state():
    source_event_id = uuid4()
    existing_contribution_id = uuid4()
    existing_entry = StateEntry(
        value="晚上",
        confidence=1.0,
        evidence_event_ids=(uuid4(),),
        contribution_ids=(existing_contribution_id,),
    )
    old_state = SubjectState(
        subject_id="user-1",
        version=1,
        entries={"preferences.study_time": existing_entry},
        applied_contribution_ids=frozenset(),
        conflicts=frozenset(),
    )
    contribution = Contribution(
        contribution_id=uuid4(),
        target_subject_id="user-2",
        target_field="preferences.study_time",
        value="早上",
        confidence=1.0,
        evidence_event_ids=(source_event_id,),
        source_event_id=source_event_id,
        source_module="semantic.preference_extractor",
    )
    original_version = old_state.version
    original_entries = dict(old_state.entries)

    with pytest.raises(SubjectMismatchError):
        StateReducer().apply(old_state, contribution)

    assert old_state.version == original_version
    assert old_state.entries == original_entries


def test_empty_state_has_no_applied_contributions():
    state = SubjectState.empty(subject_id="user-1")

    assert state.applied_contribution_ids == frozenset()
    assert state.conflicts == frozenset()


def test_applying_same_contribution_twice_changes_state_only_once():
    source_event_id = uuid4()
    contribution = Contribution(
        contribution_id=uuid4(),
        target_subject_id="user-1",
        target_field="preferences.study_time",
        value="晚上",
        confidence=1.0,
        evidence_event_ids=(source_event_id,),
        source_event_id=source_event_id,
        source_module="semantic.preference_extractor",
    )
    reducer = StateReducer()

    first_state = reducer.apply(SubjectState.empty("user-1"), contribution)
    second_state = reducer.apply(first_state, contribution)

    assert first_state.version == 1
    assert second_state is first_state
    assert second_state.version == 1
    assert len(second_state.entries) == 1
    assert second_state.get("preferences.study_time").evidence_event_ids == (
        source_event_id,
    )
    assert second_state.applied_contribution_ids == frozenset(
        {contribution.contribution_id}
    )


def test_merges_sources_once_in_first_seen_order():
    first_event_id = uuid4()
    second_event_id = uuid4()
    first_contribution = Contribution(
        contribution_id=uuid4(),
        target_subject_id="user-1",
        target_field="preferences.study_time",
        value="晚上",
        confidence=1.0,
        evidence_event_ids=(first_event_id,),
        source_event_id=first_event_id,
        source_module="semantic.preference_extractor",
    )
    second_contribution = Contribution(
        contribution_id=uuid4(),
        target_subject_id="user-1",
        target_field="preferences.study_time",
        value="早上",
        confidence=0.9,
        evidence_event_ids=(first_event_id, second_event_id),
        source_event_id=second_event_id,
        source_module="semantic.preference_extractor",
    )
    reducer = StateReducer()

    first_state = reducer.apply(SubjectState.empty("user-1"), first_contribution)
    second_state = reducer.apply(first_state, second_contribution)
    entry = second_state.get("preferences.study_time")

    assert entry.value == "早上"
    assert entry.confidence == 0.9
    assert entry.evidence_event_ids == (first_event_id, second_event_id)
    assert entry.contribution_ids == (
        first_contribution.contribution_id,
        second_contribution.contribution_id,
    )


def test_apply_many_processes_duplicate_contribution_once():
    event_id = uuid4()
    contribution = Contribution(
        contribution_id=uuid4(),
        target_subject_id="user-1",
        target_field="preferences.study_time",
        value="晚上",
        confidence=1.0,
        evidence_event_ids=(event_id,),
        source_event_id=event_id,
        source_module="semantic.preference_extractor",
    )

    state = StateReducer().apply_many(
        SubjectState.empty("user-1"),
        (contribution, contribution),
    )

    assert state.version == 1
    assert state.get("preferences.study_time").contribution_ids == (
        contribution.contribution_id,
    )
    assert state.applied_contribution_ids == frozenset(
        {contribution.contribution_id}
    )
    assert state.conflicts == frozenset()


def test_apply_many_records_conflict_without_choosing_a_value():
    evening_event_id = UUID(int=101)
    morning_event_id = UUID(int=102)
    evening = Contribution(
        contribution_id=UUID(int=1),
        target_subject_id="user-1",
        target_field="preferences.study_time",
        value="晚上",
        confidence=1.0,
        evidence_event_ids=(evening_event_id,),
        source_event_id=evening_event_id,
        source_module="semantic.preference_extractor",
    )
    morning = Contribution(
        contribution_id=UUID(int=2),
        target_subject_id="user-1",
        target_field="preferences.study_time",
        value="早上",
        confidence=1.0,
        evidence_event_ids=(morning_event_id,),
        source_event_id=morning_event_id,
        source_module="semantic.preference_extractor",
    )

    state = StateReducer().apply_many(
        SubjectState.empty("user-1"),
        (morning, evening),
    )

    assert "preferences.study_time" not in state.entries
    assert state.version == 1
    assert state.applied_contribution_ids == frozenset()
    assert state.conflicts == frozenset(
        {
            ConflictRecord(
                target_field="preferences.study_time",
                candidate_contribution_ids=(UUID(int=1), UUID(int=2)),
                reason="different values for the same field in one batch",
            )
        }
    )


def test_apply_many_is_independent_of_input_order_and_safe_to_replay():
    evening_event_id = UUID(int=201)
    morning_event_id = UUID(int=202)
    name_event_id = UUID(int=203)
    evening = Contribution(
        contribution_id=UUID(int=11),
        target_subject_id="user-1",
        target_field="preferences.study_time",
        value="晚上",
        confidence=1.0,
        evidence_event_ids=(evening_event_id,),
        source_event_id=evening_event_id,
        source_module="semantic.preference_extractor",
    )
    morning = Contribution(
        contribution_id=UUID(int=12),
        target_subject_id="user-1",
        target_field="preferences.study_time",
        value="早上",
        confidence=1.0,
        evidence_event_ids=(morning_event_id,),
        source_event_id=morning_event_id,
        source_module="semantic.preference_extractor",
    )
    name = Contribution(
        contribution_id=UUID(int=13),
        target_subject_id="user-1",
        target_field="profile.name",
        value="小明",
        confidence=1.0,
        evidence_event_ids=(name_event_id,),
        source_event_id=name_event_id,
        source_module="semantic.name_extractor",
    )
    reducer = StateReducer()

    forward = reducer.apply_many(
        SubjectState.empty("user-1"),
        (evening, morning, name),
    )
    reverse = reducer.apply_many(
        SubjectState.empty("user-1"),
        (name, morning, evening),
    )
    replayed = reducer.apply_many(forward, (name, morning, evening))

    assert forward == reverse
    assert forward.version == 2
    assert forward.get("profile.name").value == "小明"
    assert "preferences.study_time" not in forward.entries
    assert replayed is forward


def test_apply_many_rejects_mixed_subjects_without_changing_state():
    event_id = uuid4()
    contribution = Contribution(
        contribution_id=uuid4(),
        target_subject_id="user-2",
        target_field="preferences.study_time",
        value="晚上",
        confidence=1.0,
        evidence_event_ids=(event_id,),
        source_event_id=event_id,
        source_module="semantic.preference_extractor",
    )
    old_state = SubjectState.empty("user-1")

    with pytest.raises(SubjectMismatchError):
        StateReducer().apply_many(old_state, (contribution,))

    assert old_state == SubjectState.empty("user-1")
