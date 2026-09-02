from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import pytest

from self_cognition.bootstrap import build_container
from self_cognition.core.errors import (
    FileLockUnavailableError,
    MalformedSerializedDataError,
)
from self_cognition.infrastructure.persistence.atomic_io import atomic_write_text
from self_cognition.infrastructure.persistence.file_layout import FileDataLayout
from self_cognition.infrastructure.persistence.file_lock import FileLock
from self_cognition.infrastructure.persistence.file_manifest import (
    build_manifest,
    write_manifest,
)
from self_cognition.infrastructure.persistence.file_recovery import (
    FileRecoveryLog,
    RecoveryRecord,
    RecoveryStatus,
)


def test_layout_creates_canonical_directories_and_uses_legacy_event_log(tmp_path):
    legacy = tmp_path / "events.jsonl"
    legacy.write_text("legacy\n", encoding="utf-8")

    layout = FileDataLayout(tmp_path).ensure()

    assert layout.event_log == legacy
    assert all(path.is_dir() for path in layout.directories)


def test_layout_prefers_canonical_event_log_for_new_data(tmp_path):
    layout = FileDataLayout(tmp_path).ensure()

    assert layout.event_log == tmp_path / "events" / "events.jsonl"


def test_atomic_write_replaces_target_and_cleans_temporary_files(tmp_path):
    target = tmp_path / "nested" / "record.json"

    atomic_write_text(target, "first")
    atomic_write_text(target, "second")

    assert target.read_text(encoding="utf-8") == "second"
    assert list(target.parent.glob("*.tmp")) == []


def test_file_lock_is_exclusive_and_released_after_context(tmp_path):
    path = tmp_path / "processing" / "write.lock"

    with FileLock(path):
        assert path.exists()
        with pytest.raises(FileLockUnavailableError):
            FileLock(path).acquire()

    with FileLock(path):
        assert path.exists()


def test_manifest_is_deterministic_and_excludes_its_own_output(tmp_path):
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    manifest_path = tmp_path / "snapshot.manifest.json"

    first = write_manifest(tmp_path, manifest_path)
    second = build_manifest(tmp_path)

    assert first == second
    assert [entry.path for entry in first.entries] == ["a.txt", "b.txt"]


def test_recovery_log_is_idempotent_and_reloads(tmp_path):
    path = tmp_path / "processing" / "recovery.jsonl"
    record = RecoveryRecord(
        operation="replay",
        target="mind/user-1",
        status=RecoveryStatus.COMPLETED,
        recorded_at=datetime(2026, 9, 2, tzinfo=timezone.utc),
        detail="state rebuilt",
        record_id=UUID(int=1),
    )
    log = FileRecoveryLog(path)

    log.append(record)
    log.append(record)

    assert log.read_all() == (record,)
    assert FileRecoveryLog(path).latest_for("mind/user-1") == record


def test_recovery_log_refreshes_records_written_by_another_instance(tmp_path):
    path = tmp_path / "processing" / "recovery.jsonl"
    first_log = FileRecoveryLog(path)
    second_log = FileRecoveryLog(path)
    first = RecoveryRecord(
        operation="replay",
        target="mind/user-1",
        status=RecoveryStatus.STARTED,
        recorded_at=datetime(2026, 9, 2, tzinfo=timezone.utc),
        detail="replay started",
        record_id=UUID(int=1),
    )
    second = RecoveryRecord(
        operation="replay",
        target="mind/user-1",
        status=RecoveryStatus.COMPLETED,
        recorded_at=datetime(2026, 9, 2, 0, 1, tzinfo=timezone.utc),
        detail="replay completed",
        record_id=UUID(int=2),
    )

    second_log.append(first)
    first_log.append(second)

    assert first_log.read_all() == (first, second)


def test_recovery_log_rejects_corrupt_records(tmp_path):
    path = tmp_path / "recovery.jsonl"
    path.write_text("{broken}\n", encoding="utf-8")

    with pytest.raises(MalformedSerializedDataError, match="line 1"):
        FileRecoveryLog(path)


def test_bootstrap_uses_file_layout_without_breaking_legacy_event_path(tmp_path):
    legacy = tmp_path / "events.jsonl"
    legacy.write_text("", encoding="utf-8")

    container = build_container(tmp_path)

    assert container.event_store is not None
    assert legacy.exists()
    assert (tmp_path / "memories").is_dir()
    assert (tmp_path / "processing").is_dir()
