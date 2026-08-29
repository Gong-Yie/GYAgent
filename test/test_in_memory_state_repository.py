from uuid import uuid4

import pytest

from self_cognition.core.contributions import Contribution
from self_cognition.core.errors import VersionConflictError
from self_cognition.core.state import SubjectState
from self_cognition.infrastructure.persistence.in_memory_state_repository import (
    InMemoryStateRepository,
)
from self_cognition.runtime.reducer import StateReducer


def make_contribution(value: str) -> Contribution:
    event_id = uuid4()
    return Contribution(
        contribution_id=uuid4(),
        target_subject_id="user-1",
        target_field="preferences.study_time",
        value=value,
        confidence=1.0,
        evidence_event_ids=(event_id,),
        source_event_id=event_id,
        source_module="test.state_repository",
    )


def test_load_returns_none_for_unsaved_subject():
    repository = InMemoryStateRepository()

    assert repository.load("user-1") is None


def test_saves_and_loads_state_with_expected_old_version():
    repository = InMemoryStateRepository()
    reducer = StateReducer()
    version_one = reducer.apply(
        SubjectState.empty("user-1"),
        make_contribution("晚上"),
    )
    version_two = reducer.apply(version_one, make_contribution("早上"))

    repository.save(version_one, expected_version=0)
    assert repository.load("user-1") == version_one

    repository.save(version_two, expected_version=1)
    assert repository.load("user-1") == version_two


def test_save_requires_expected_version_argument():
    repository = InMemoryStateRepository()
    state = SubjectState.empty("user-1")

    with pytest.raises(TypeError):
        repository.save(state)  # type: ignore[call-arg]


def test_rejects_stale_save_without_changing_stored_state():
    repository = InMemoryStateRepository()
    reducer = StateReducer()
    version_one = reducer.apply(
        SubjectState.empty("user-1"),
        make_contribution("晚上"),
    )
    stored_version_two = reducer.apply(version_one, make_contribution("早上"))
    stale_version_two = reducer.apply(version_one, make_contribution("凌晨"))
    repository.save(version_one, expected_version=0)
    repository.save(stored_version_two, expected_version=1)

    with pytest.raises(VersionConflictError):
        repository.save(stale_version_two, expected_version=1)

    assert repository.load("user-1") is stored_version_two


def test_rejects_older_state_even_when_expected_version_is_current():
    repository = InMemoryStateRepository()
    reducer = StateReducer()
    version_one = reducer.apply(
        SubjectState.empty("user-1"),
        make_contribution("晚上"),
    )
    version_two = reducer.apply(version_one, make_contribution("早上"))
    repository.save(version_two, expected_version=0)

    with pytest.raises(VersionConflictError):
        repository.save(version_one, expected_version=2)

    assert repository.load("user-1") is version_two


def test_rejects_first_save_with_nonzero_expected_version():
    repository = InMemoryStateRepository()
    state = StateReducer().apply(
        SubjectState.empty("user-1"),
        make_contribution("晚上"),
    )

    with pytest.raises(VersionConflictError):
        repository.save(state, expected_version=1)

    assert repository.load("user-1") is None
