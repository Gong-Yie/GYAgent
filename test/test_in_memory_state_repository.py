from datetime import datetime, timezone
from uuid import uuid4

import pytest

from self_cognition.core.contributions import (
    CognitiveContribution,
    CognitionType,
    ContributionOperation,
)
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.errors import VersionConflictError
from self_cognition.core.scopes import DataScope, DisclosureScope, SubjectScope
from self_cognition.core.state import SubjectState
from self_cognition.infrastructure.persistence.in_memory_state_repository import (
    InMemoryStateRepository,
)
from self_cognition.blackboard.reducer import StateReducer


NOW = datetime(2026, 9, 1, tzinfo=timezone.utc)
SUBJECT = SubjectScope.legacy_user("user-1")


def make_contribution(
    value: str,
    target_version: int,
) -> CognitiveContribution:
    event_id = uuid4()
    return CognitiveContribution(
        contribution_id=uuid4(),
        target=SUBJECT,
        target_field="preferences.study_time",
        operation=ContributionOperation.SET,
        cognition_type=CognitionType.PREFERENCE,
        value=value,
        confidence=1.0,
        evidence_refs=(EvidenceRef.for_event_id(event_id, SUBJECT),),
        source_module="test.state_repository",
        module_version="1",
        scope=DataScope(SUBJECT, DisclosureScope.PRIVATE),
        created_at=NOW,
        valid_from=NOW,
        target_version=target_version,
    )


def test_load_returns_none_for_unsaved_subject():
    repository = InMemoryStateRepository()

    assert repository.load("user-1") is None


def test_saves_and_loads_state_with_expected_old_version():
    repository = InMemoryStateRepository()
    reducer = StateReducer()
    version_one = reducer.apply(
        SubjectState.empty("user-1"),
        make_contribution("晚上", 0),
        decided_at=NOW,
    )
    version_two = reducer.apply(
        version_one,
        make_contribution("早上", 1),
        decided_at=NOW,
    )

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
        make_contribution("晚上", 0),
        decided_at=NOW,
    )
    stored_version_two = reducer.apply(
        version_one,
        make_contribution("早上", 1),
        decided_at=NOW,
    )
    stale_version_two = reducer.apply(
        version_one,
        make_contribution("凌晨", 1),
        decided_at=NOW,
    )
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
        make_contribution("晚上", 0),
        decided_at=NOW,
    )
    version_two = reducer.apply(
        version_one,
        make_contribution("早上", 1),
        decided_at=NOW,
    )
    repository.save(version_two, expected_version=0)

    with pytest.raises(VersionConflictError):
        repository.save(version_one, expected_version=2)

    assert repository.load("user-1") is version_two


def test_rejects_first_save_with_nonzero_expected_version():
    repository = InMemoryStateRepository()
    state = StateReducer().apply(
        SubjectState.empty("user-1"),
        make_contribution("晚上", 0),
        decided_at=NOW,
    )

    with pytest.raises(VersionConflictError):
        repository.save(state, expected_version=1)

    assert repository.load("user-1") is None
