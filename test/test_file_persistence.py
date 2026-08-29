from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from self_cognition.application.process_event import ProcessEventService
from self_cognition.application.replay import ReplayService
from self_cognition.application.results import ProcessEventStatus
from self_cognition.cognition.semantic.preference_extractor import (
    PreferenceExtractor,
)
from self_cognition.core.errors import (
    MalformedSerializedDataError,
    VersionConflictError,
)
from self_cognition.core.events import Event
from self_cognition.core.state import SubjectState
from self_cognition.infrastructure.persistence.file_event_store import FileEventStore
from self_cognition.infrastructure.persistence.file_state_repository import (
    FileStateRepository,
)
from self_cognition.infrastructure.persistence.in_memory_event_store import (
    InMemoryEventStore,
)
from self_cognition.runtime.engine import CognitionEngine
from self_cognition.runtime.reducer import StateReducer
from self_cognition.runtime.run_context import RunContext


def make_context(run_id: int) -> RunContext:
    return RunContext(
        run_id=UUID(int=run_id),
        correlation_id=UUID(int=100),
        deadline=datetime.now(timezone.utc) + timedelta(minutes=1),
    )


def make_states() -> tuple[SubjectState, SubjectState]:
    extractor = PreferenceExtractor()
    reducer = StateReducer()
    evening_event = Event.user_message("user-1", "我喜欢晚上学习")
    morning_event = Event.user_message("user-1", "我喜欢早上学习")
    version_one = reducer.apply(
        SubjectState.empty("user-1"),
        extractor.process(evening_event)[0],
    )
    version_two = reducer.apply(
        version_one,
        extractor.process(morning_event)[0],
    )
    return version_one, version_two


def test_file_event_store_appends_deduplicates_and_reloads(tmp_path):
    path = tmp_path / "events.jsonl"
    first = Event.user_message("user-1", "第一条消息")
    second = Event.user_message("user-2", "第二条消息")
    store = FileEventStore(path)

    store.append(first)
    store.append(first)
    store.append(second)

    reloaded = FileEventStore(path)
    assert store.read_all() == (first, second)
    assert reloaded.read_all() == (first, second)
    assert reloaded.read_by_subject("user-1") == (first,)
    assert path.read_text(encoding="utf-8").count("\n") == 2
    assert len(
        [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    ) == 2


def test_file_event_store_rejects_corrupt_records_instead_of_ignoring_them(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text("{broken}\n", encoding="utf-8")

    with pytest.raises(MalformedSerializedDataError, match="line 1"):
        FileEventStore(path)


def test_event_write_failure_does_not_mark_event_as_appended(tmp_path, monkeypatch):
    import self_cognition.infrastructure.persistence.file_event_store as module

    path = tmp_path / "events.jsonl"
    store = FileEventStore(path)
    event = Event.user_message("user-1", "写入失败")

    def fail_write(self, value):
        raise OSError("simulated write failure")

    monkeypatch.setattr(module.Path, "open", lambda *args, **kwargs: _FailingHandle())

    with pytest.raises(OSError, match="write failure"):
        store.append(event)

    assert store.contains(event.event_id) is False
    assert store.read_all() == ()


class _FailingHandle:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def write(self, value):
        raise OSError("simulated write failure")


def test_file_state_repository_enforces_versions_and_reloads(tmp_path):
    repository = FileStateRepository(tmp_path / "states")
    version_one, version_two = make_states()

    repository.save(version_one, expected_version=0)
    assert repository.load("user-1") == version_one

    with pytest.raises(VersionConflictError):
        repository.save(version_two, expected_version=0)

    assert repository.load("user-1") == version_one

    repository.save(version_two, expected_version=1)
    assert FileStateRepository(tmp_path / "states").load("user-1") == version_two
    assert list((tmp_path / "states").glob("*.tmp")) == []


def test_application_service_works_with_file_adapters(tmp_path):
    event_store = FileEventStore(tmp_path / "events.jsonl")
    state_repository = FileStateRepository(tmp_path / "states")
    service = ProcessEventService(
        event_store=event_store,
        state_repository=state_repository,
        engine=CognitionEngine((PreferenceExtractor(),), StateReducer()),
    )
    event = Event.user_message("user-1", "我喜欢晚上学习")

    result = service.process(event, make_context(1))

    assert result.status is ProcessEventStatus.SUCCEEDED
    assert result.state is not None
    assert result.state.get("preferences.study_time").value == "晚上"
    assert FileEventStore(tmp_path / "events.jsonl").read_all() == (event,)
    assert (
        FileStateRepository(tmp_path / "states").load("user-1")
        == result.state
    )


def test_corrupt_snapshot_can_be_recovered_by_replaying_the_event_log(tmp_path):
    event_store = FileEventStore(tmp_path / "events.jsonl")
    state_directory = tmp_path / "states"
    state_repository = FileStateRepository(state_directory)
    engine = CognitionEngine((PreferenceExtractor(),), StateReducer())
    service = ProcessEventService(event_store, state_repository, engine)
    events = (
        Event.user_message("user-1", "我喜欢晚上学习"),
        Event.user_message("user-1", "我喜欢早上学习"),
    )
    for run_id, event in enumerate(events, start=1):
        result = service.process(event, make_context(run_id))
    assert result.state is not None
    expected_state = result.state

    snapshot_path = next(state_directory.glob("*.json"))
    snapshot_path.write_text("{broken}", encoding="utf-8")

    with pytest.raises(MalformedSerializedDataError):
        FileStateRepository(state_directory).load("user-1")

    recovered_state = ReplayService(
        FileEventStore(tmp_path / "events.jsonl"),
        engine,
    ).replay("user-1")
    assert recovered_state == expected_state


def test_failed_atomic_replace_preserves_previous_snapshot(tmp_path, monkeypatch):
    import self_cognition.infrastructure.persistence.file_state_repository as module

    repository = FileStateRepository(tmp_path / "states")
    version_one, version_two = make_states()
    repository.save(version_one, expected_version=0)

    def fail_replace(source, target):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failure"):
        repository.save(version_two, expected_version=1)

    assert repository.load("user-1") == version_one
    assert list((tmp_path / "states").glob("*.tmp")) == []
