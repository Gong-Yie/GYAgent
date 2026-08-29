from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from self_cognition.core.contributions import Contribution
from self_cognition.core.errors import VersionConflictError
from self_cognition.core.events import Event
from self_cognition.core.state import SubjectState
from self_cognition.infrastructure.persistence.file_event_store import FileEventStore
from self_cognition.infrastructure.persistence.file_state_repository import (
    FileStateRepository,
)
from self_cognition.infrastructure.persistence.in_memory_event_store import (
    InMemoryEventStore,
)
from self_cognition.infrastructure.persistence.in_memory_state_repository import (
    InMemoryStateRepository,
)
from self_cognition.runtime.reducer import StateReducer


@pytest.fixture(params=("memory", "file"))
def event_store(request, tmp_path):
    if request.param == "memory":
        return InMemoryEventStore()
    return FileEventStore(tmp_path / "events.jsonl")


@pytest.fixture(params=("memory", "file"))
def state_repository(request, tmp_path):
    if request.param == "memory":
        return InMemoryStateRepository()
    return FileStateRepository(tmp_path / "states")


def make_event(event_id: int, actor_id: str) -> Event:
    return Event(
        event_id=UUID(int=event_id),
        event_type="user.message",
        actor_id=actor_id,
        content=f"消息 {event_id}",
        occurred_at=datetime(2026, 8, 13, event_id, tzinfo=timezone.utc),
    )


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
        source_module="test.persistence_contract",
    )


def test_event_store_contract_preserves_order_deduplication_and_subjects(
    event_store,
):
    first = make_event(1, "user-1")
    other = make_event(2, "user-2")
    second = make_event(3, "user-1")

    event_store.append(first)
    event_store.append(other)
    event_store.append(first)
    event_store.append(second)

    assert event_store.read_all() == (first, other, second)
    assert event_store.read_by_subject("user-1") == (first, second)
    assert event_store.contains(first.event_id) is True


def test_state_repository_contract_enforces_optimistic_versions(
    state_repository,
):
    reducer = StateReducer()
    version_one = reducer.apply(
        SubjectState.empty("user-1"),
        make_contribution("晚上"),
    )
    version_two = reducer.apply(version_one, make_contribution("早上"))

    assert state_repository.load("user-1") is None
    state_repository.save(version_one, expected_version=0)
    assert state_repository.load("user-1") == version_one

    with pytest.raises(VersionConflictError):
        state_repository.save(version_two, expected_version=0)

    assert state_repository.load("user-1") == version_one
    state_repository.save(version_two, expected_version=1)
    assert state_repository.load("user-1") == version_two
