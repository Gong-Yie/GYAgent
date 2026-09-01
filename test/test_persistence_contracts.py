from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from self_cognition.core.contributions import (
    CognitiveContribution,
    CognitionType,
    ContributionOperation,
)
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.errors import VersionConflictError
from self_cognition.core.events import Event
from self_cognition.core.scopes import DataScope, DisclosureScope, SubjectScope
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
from self_cognition.blackboard.reducer import StateReducer


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
    return Event.user_message(
        actor_id,
        f"消息 {event_id}",
        event_id=UUID(int=event_id),
        clock=FixedClock(datetime(2026, 8, 13, event_id, tzinfo=timezone.utc)),
    )


class FixedClock:
    def __init__(self, value: datetime) -> None:
        self._value = value

    def now(self) -> datetime:
        return self._value


SUBJECT = SubjectScope.legacy_user("user-1")
NOW = datetime(2026, 9, 1, tzinfo=timezone.utc)


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
        source_module="test.persistence_contract",
        module_version="1",
        scope=DataScope(SUBJECT, DisclosureScope.PRIVATE),
        created_at=NOW,
        valid_from=NOW,
        target_version=target_version,
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

    assert event_store.read_by_subject(first.subject) == (first, second)
    assert event_store.read_by_subject(other.subject) == (other,)


def test_state_repository_contract_enforces_optimistic_versions(
    state_repository,
):
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

    assert state_repository.load("user-1") is None
    state_repository.save(version_one, expected_version=0)
    assert state_repository.load("user-1") == version_one

    with pytest.raises(VersionConflictError):
        state_repository.save(version_two, expected_version=0)

    assert state_repository.load("user-1") == version_one
    state_repository.save(version_two, expected_version=1)
    assert state_repository.load("user-1") == version_two
