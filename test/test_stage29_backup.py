import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from self_cognition.bootstrap import build_container
from self_cognition.core.deletions import DeletionSelector, DeletionStatus
from self_cognition.core.errors import MalformedSerializedDataError
from self_cognition.core.events import EventEnvelope
from self_cognition.core.scopes import SubjectScope
from self_cognition.core.state import SubjectState
from self_cognition.infrastructure.persistence.backup import (
    create_backup,
    migrate_data,
    restore_backup,
)
from self_cognition.runtime.run_context import RunContext


NOW = datetime(2026, 9, 9, 17, tzinfo=timezone.utc)


def _context() -> RunContext:
    return RunContext(uuid4(), uuid4(), NOW + timedelta(minutes=5))


def _seed(data: Path) -> tuple[SubjectScope, SubjectState]:
    app = build_container(data, dotenv_path=data / "missing.env")
    deleted = SubjectScope.legacy_user("deleted-user")
    retained = SubjectScope.legacy_user("retained-user")
    for subject in (deleted, retained):
        result = app.process_event.process(
            EventEnvelope.user_message(subject, "我喜欢晚上学习"),
            _context(),
        )
        assert result.error_type is None
    expected = app.state_repository.load(retained)
    assert expected is not None
    plan = app.forget.dry_run(
        DeletionSelector(deleted, delete_subject=True),
        now=NOW,
    )
    assert app.forget.execute(plan, now=NOW).status is DeletionStatus.COMPLETED
    app.lifecycle.stop()
    return retained, expected


def test_backup_restores_verified_data_and_rebuilds_indexes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    retained, expected = _seed(source)
    stale = source / "indexes" / "stale.json"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("stale", encoding="utf-8")
    config = tmp_path / ".env.example"
    config.write_text("SC_CONFIG_VERSION=1\n", encoding="utf-8")
    archive = tmp_path / "backup.zip"

    created = create_backup(source, archive, non_sensitive_config=(config,))
    restored = tmp_path / "restored"
    result = restore_backup(archive, restored)

    assert created.snapshot == result.snapshot
    assert created.snapshot.tombstone_count > 0
    assert not (restored / "indexes" / "stale.json").exists()
    assert (restored / "config" / ".env.example").is_file()
    indexes = tuple((restored / "indexes" / "memories").rglob("*.json"))
    assert indexes
    assert all(
        json.loads(path.read_text("utf-8"))["schema_version"] == 1
        for path in indexes
    )
    app = build_container(restored, dotenv_path=tmp_path / "missing.env")
    try:
        assert app.state_repository.load(retained) == expected
    finally:
        app.lifecycle.stop()


def test_restore_checksum_failure_leaves_target_absent(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _seed(source)
    archive = tmp_path / "backup.zip"
    create_backup(source, archive)
    corrupted = tmp_path / "corrupted.zip"
    with zipfile.ZipFile(archive) as original, zipfile.ZipFile(
        corrupted,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as output:
        for info in original.infolist():
            payload = original.read(info)
            if info.filename == "events/events.jsonl":
                payload += b"corrupt"
            output.writestr(info, payload)

    target = tmp_path / "must-not-exist"
    with pytest.raises(MalformedSerializedDataError, match="checksum"):
        restore_backup(corrupted, target)
    assert not target.exists()


def test_migration_writes_current_schemas_without_changing_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy"
    _seed(source)
    state_path = next((source / "states").glob("*.json"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["schema_version"] = 4
    state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    memory_path = next((source / "memories").rglob("*.json"))
    memory = json.loads(memory_path.read_text(encoding="utf-8"))
    memory["schema_version"] = 3
    for field in ("expires_at", "lifecycle_changed_at", "lifecycle_reason"):
        memory.pop(field)
    memory_path.write_text(json.dumps(memory, ensure_ascii=False), encoding="utf-8")

    target = tmp_path / "migrated"
    migrate_data(source, target)

    assert json.loads(state_path.read_text("utf-8"))["schema_version"] == 4
    assert json.loads(memory_path.read_text("utf-8"))["schema_version"] == 3
    assert all(
        json.loads(path.read_text("utf-8"))["schema_version"] == 5
        for path in (target / "states").glob("*.json")
    )
    assert all(
        json.loads(path.read_text("utf-8"))["schema_version"] == 4
        for path in (target / "memories").rglob("*.json")
    )
    assert (target / "migration.json").is_file()
