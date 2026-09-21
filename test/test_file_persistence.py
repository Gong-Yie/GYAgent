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
from self_cognition.infrastructure.persistence.in_memory_evidence_repository import (
    InMemoryEvidenceRepository,
)
from self_cognition.runtime.engine import CognitionEngine
from self_cognition.blackboard.reducer import StateReducer
from self_cognition.blackboard.service import CognitiveSpaceService
from self_cognition.runtime.run_context import RunContext


def make_context(run_id: int) -> RunContext:
    return RunContext(
        run_id=UUID(int=run_id),
        correlation_id=UUID(int=100),
        deadline=datetime.now(timezone.utc) + timedelta(minutes=1),
    )


def make_states() -> tuple[SubjectState, SubjectState]:
    extractor = PreferenceExtractor()
    engine = CognitionEngine(
        (extractor,),
        CognitiveSpaceService(StateReducer()),
    )
    evening_event = Event.user_message("user-1", "我喜欢晚上学习")
    morning_event = Event.user_message("user-1", "我喜欢早上学习")
    version_one = engine.process(evening_event, SubjectState.empty("user-1"))
    version_two = engine.process(morning_event, version_one)
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
    assert store.read_by_subject(first.subject) == (first,)
    assert store.read_by_subject(second.subject) == (second,)
    assert reloaded.read_by_subject(first.subject) == (first,)
    assert reloaded.read_by_subject(second.subject) == (second,)
    assert path.read_text(encoding="utf-8").count("\n") == 2
    assert len(
        [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    ) == 2


def test_file_event_store_observes_records_a_sibling_appended(tmp_path):
    """一个实例必须看到另一个实例追加的事件，且偏移不落在记录中间。"""
    path = tmp_path / "events.jsonl"
    first = FileEventStore(path)
    second = FileEventStore(path)
    one = Event.user_message("user-1", "甲")
    two = Event.user_message("user-2", "乙")
    three = Event.user_message("user-1", "丙")

    first.append(one)
    second.append(two)
    first.append(three)

    # 顺序即物理追加顺序，尾部读取期间不会解析出半个记录
    assert second.read_all() == (one, two, three)
    assert first.read_all() == (one, two, three)
    assert second.read_by_subject(one.subject) == (one, three)


def test_file_event_store_propagates_a_sibling_deletion(tmp_path):
    """删除必须传播到其它实例，且被删除的事件不能通过追加复活。"""
    path = tmp_path / "events.jsonl"
    tombstones = tmp_path / "event_tombstones.jsonl"
    first = FileEventStore(path, tombstones)
    second = FileEventStore(path, tombstones)
    removed = Event.user_message("user-1", "要被遗忘的")
    kept = Event.user_message("user-2", "保留的")

    first.append(removed)
    first.append(kept)
    assert second.read_all() == (removed, kept)

    first.redact(removed.subject, (removed.event_id,), UUID(int=9))

    assert second.read_all() == (kept,)
    second.append(removed)
    assert second.read_all() == (kept,)
    assert FileEventStore(path, tombstones).read_all() == (kept,)


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

    assert store.read_by_subject(event.subject) == ()


class _FailingHandle:
    """A handle whose reads succeed but whose writes fail.

    Appending reads the log tail first, to keep its byte offset on a record
    boundary, so only ``write`` may raise for the assertion under test.
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def seek(self, offset):
        self.offset = offset

    def read(self, size=-1):
        return b""

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
        evidence_repository=InMemoryEvidenceRepository(),
        state_repository=state_repository,
        engine=CognitionEngine(
            (PreferenceExtractor(),),
            CognitiveSpaceService(StateReducer()),
        ),
    )
    event = Event.user_message("user-1", "我喜欢晚上学习")

    result = service.process(event, make_context(1))

    assert result.status is ProcessEventStatus.SUCCEEDED
    assert result.state is not None
    assert result.state.get("preferences.study_time").value == "晚上"
    stored_event = FileEventStore(tmp_path / "events.jsonl").read_by_subject(
        event.subject
    )[0]
    assert stored_event.event_id == event.event_id
    assert stored_event.run_id == UUID(int=1)
    assert (
        FileStateRepository(tmp_path / "states").load("user-1")
        == result.state
    )


def test_corrupt_snapshot_can_be_recovered_by_replaying_the_event_log(tmp_path):
    event_store = FileEventStore(tmp_path / "events.jsonl")
    state_directory = tmp_path / "states"
    state_repository = FileStateRepository(state_directory)
    engine = CognitionEngine(
        (PreferenceExtractor(),),
        CognitiveSpaceService(StateReducer()),
    )
    service = ProcessEventService(
        event_store,
        InMemoryEvidenceRepository(),
        state_repository,
        engine,
    )
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
    ).replay(events[0].subject)
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
