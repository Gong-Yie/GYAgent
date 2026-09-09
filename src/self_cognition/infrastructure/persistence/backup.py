from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from uuid import UUID

from self_cognition.core.errors import MalformedSerializedDataError
from self_cognition.core.scopes import SubjectScope
from self_cognition.infrastructure.persistence.atomic_io import atomic_write_text
from self_cognition.infrastructure.persistence.file_layout import FileDataLayout
from self_cognition.infrastructure.persistence.file_manifest import (
    FileManifestEntry,
    build_manifest,
    write_manifest,
)
from self_cognition.infrastructure.persistence.file_memory_repository import (
    FileMemoryRepository,
)
from self_cognition.infrastructure.persistence.serialization import (
    event_from_json,
    memory_from_json,
    memory_to_json,
    state_from_json,
    state_to_json,
)


BACKUP_SCHEMA_VERSION = 1
MEMORY_INDEX_SCHEMA_VERSION = 1
BACKUP_MANIFEST = "backup.manifest.json"
BACKUP_METADATA = "backup.metadata.json"
MIGRATION_RECORD = "migration.json"
INCLUDED_DIRECTORIES = (
    "events",
    "evidence",
    "states",
    "memories",
    "memory_access",
    "deletions",
    "processing",
    "runs",
    "blobs",
    "governance",
    "config",
)


@dataclass(frozen=True, slots=True)
class DataSnapshot:
    event_count: int
    event_schema_versions: tuple[int, ...]
    state_versions: tuple[tuple[str, int], ...]
    state_schema_versions: tuple[int, ...]
    memory_count: int
    memory_schema_versions: tuple[int, ...]
    tombstone_count: int
    memory_index_schema_version: int = MEMORY_INDEX_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "event_count": self.event_count,
            "event_schema_versions": list(self.event_schema_versions),
            "state_versions": [list(item) for item in self.state_versions],
            "state_schema_versions": list(self.state_schema_versions),
            "memory_count": self.memory_count,
            "memory_schema_versions": list(self.memory_schema_versions),
            "tombstone_count": self.tombstone_count,
            "memory_index_schema_version": self.memory_index_schema_version,
        }


@dataclass(frozen=True, slots=True)
class BackupResult:
    path: Path
    file_count: int
    snapshot: DataSnapshot


def create_backup(
    source: str | Path,
    archive: str | Path,
    *,
    non_sensitive_config: tuple[Path, ...] = (),
) -> BackupResult:
    source_path = Path(source).resolve()
    archive_path = Path(archive).resolve()
    if not source_path.is_dir():
        raise FileNotFoundError(f"data directory does not exist: {source_path}")
    if archive_path.exists():
        raise FileExistsError(f"backup already exists: {archive_path}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    snapshot = _snapshot(source_path)
    with tempfile.TemporaryDirectory(
        prefix=".self-cognition-backup-",
        dir=archive_path.parent,
    ) as temporary:
        staging = Path(temporary) / "payload"
        staging.mkdir()
        _copy_data(source_path, staging)
        if _selected_entries(source_path) != _selected_entries(staging):
            raise RuntimeError("data changed while the backup was being created")
        _require_same_snapshot(snapshot, _snapshot(staging))
        for config_path in non_sensitive_config:
            config = Path(config_path)
            if not config.is_file():
                raise FileNotFoundError(f"config file does not exist: {config}")
            target = staging / "config" / config.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(config, target)
        _write_metadata(staging, snapshot)
        manifest = write_manifest(staging, staging / BACKUP_MANIFEST)
        _write_zip(staging, archive_path)
    return BackupResult(archive_path, len(manifest.entries), snapshot)


def restore_backup(archive: str | Path, target: str | Path) -> BackupResult:
    archive_path = Path(archive).resolve()
    target_path = Path(target).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"backup does not exist: {archive_path}")
    if target_path.exists():
        raise FileExistsError(f"restore target already exists: {target_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{target_path.name}-restore-",
            dir=target_path.parent,
        )
    )
    try:
        _extract_zip(archive_path, temporary)
        entries = _verify_manifest(temporary)
        expected = _read_metadata(temporary)
        actual = _snapshot(temporary)
        _require_same_snapshot(expected, actual)
        _rebuild_memory_indexes(temporary)
        _validate_memory_indexes(temporary)
        os.replace(temporary, target_path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return BackupResult(target_path, len(entries), actual)


def migrate_data(
    source: str | Path,
    target: str | Path,
    *,
    non_sensitive_config: tuple[Path, ...] = (),
) -> BackupResult:
    source_path = Path(source).resolve()
    target_path = Path(target).resolve()
    if target_path.exists():
        raise FileExistsError(f"migration target already exists: {target_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".self-cognition-migration-",
        dir=target_path.parent,
    ) as temporary:
        temporary_path = Path(temporary)
        archive = temporary_path / "source.zip"
        restored = temporary_path / "restored"
        source_snapshot = create_backup(
            source_path,
            archive,
            non_sensitive_config=non_sensitive_config,
        ).snapshot
        restore_backup(archive, restored)
        _rewrite_supported_schemas(restored)
        _rebuild_memory_indexes(restored)
        migrated_snapshot = _snapshot(restored)
        migration = {
            "schema_version": 1,
            "migrated_at": datetime.now(timezone.utc).isoformat(),
            "source": source_snapshot.to_dict(),
            "result": migrated_snapshot.to_dict(),
        }
        atomic_write_text(
            restored / MIGRATION_RECORD,
            json.dumps(migration, ensure_ascii=False, sort_keys=True) + "\n",
        )
        _write_metadata(restored, migrated_snapshot)
        write_manifest(restored, restored / BACKUP_MANIFEST)
        _verify_manifest(restored)
        _validate_memory_indexes(restored)
        os.replace(restored, target_path)
    manifest = build_manifest(target_path)
    return BackupResult(target_path, len(manifest.entries), migrated_snapshot)


def _copy_data(source: Path, target: Path) -> None:
    for relative, _ in _selected_entries(source):
        source_file = source / Path(relative)
        target_file = target / Path(relative)
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target_file)


def _selected_entries(root: Path) -> tuple[tuple[str, str], ...]:
    paths = []
    for directory_name in INCLUDED_DIRECTORIES:
        directory = root / directory_name
        if directory.is_dir():
            paths.extend(path for path in directory.rglob("*") if path.is_file())
    legacy_event_log = root / "events.jsonl"
    if legacy_event_log.is_file():
        paths.append(legacy_event_log)
    return tuple(
        (path.relative_to(root).as_posix(), _sha256(path))
        for path in sorted(set(paths))
        if path.suffix not in {".lock", ".tmp"}
    )


def _snapshot(root: Path) -> DataSnapshot:
    layout = FileDataLayout(root)
    event_versions = []
    event_count = 0
    if layout.event_log.exists():
        for line in _records(layout.event_log, "event"):
            values = _json_object(line, "event")
            event_versions.append(_schema_version(values, "event"))
            event_from_json(line)
            event_count += 1

    state_versions = []
    state_schemas = []
    for path in sorted(layout.states.glob("*.json")):
        payload = path.read_text(encoding="utf-8")
        values = _json_object(payload, "state")
        state_schemas.append(_schema_version(values, "state"))
        state = state_from_json(payload)
        state_versions.append((path.name, state.version))

    memory_count = 0
    memory_schemas = []
    for path in _memory_paths(layout.memories):
        payload = path.read_text(encoding="utf-8")
        values = _json_object(payload, "memory")
        memory_schemas.append(_schema_version(values, "memory"))
        memory_from_json(payload)
        memory_count += 1

    tombstone_count = 0
    tombstones = layout.deletions / "event_tombstones.jsonl"
    if tombstones.exists():
        for line in _records(tombstones, "event tombstone"):
            values = _json_object(line, "event tombstone")
            if _schema_version(values, "event tombstone") != 1:
                raise MalformedSerializedDataError(
                    "unsupported event tombstone schema"
                )
            UUID(values["event_id"])
            tombstone_count += 1

    return DataSnapshot(
        event_count,
        tuple(sorted(set(event_versions))),
        tuple(state_versions),
        tuple(sorted(set(state_schemas))),
        memory_count,
        tuple(sorted(set(memory_schemas))),
        tombstone_count,
    )


def _rewrite_supported_schemas(root: Path) -> None:
    layout = FileDataLayout(root)
    for path in sorted(layout.states.glob("*.json")):
        atomic_write_text(path, state_to_json(state_from_json(path.read_text("utf-8"))))
    for path in _memory_paths(layout.memories):
        atomic_write_text(
            path,
            memory_to_json(memory_from_json(path.read_text("utf-8"))) + "\n",
        )


def _rebuild_memory_indexes(root: Path) -> None:
    layout = FileDataLayout(root)
    if layout.indexes.exists():
        shutil.rmtree(layout.indexes)
    subjects: set[SubjectScope] = set()
    for path in _memory_paths(layout.memories):
        subjects.add(memory_from_json(path.read_text(encoding="utf-8")).subject)
    repository = FileMemoryRepository(
        layout.memories,
        layout.indexes / "memories",
        layout.memory_access,
    )
    for subject in sorted(
        subjects,
        key=lambda item: (
            item.mind.mind_id,
            item.subject.kind.value,
            item.subject.subject_id,
        ),
    ):
        repository.rebuild_index(subject)


def _validate_memory_indexes(root: Path) -> None:
    index_root = FileDataLayout(root).indexes / "memories"
    for path in index_root.rglob("*.json"):
        values = _json_object(path.read_text(encoding="utf-8"), "memory index")
        if values.get("schema_version") != MEMORY_INDEX_SCHEMA_VERSION:
            raise MalformedSerializedDataError(
                "restored memory index schema is invalid"
            )


def _write_metadata(root: Path, snapshot: DataSnapshot) -> None:
    payload = {
        "schema_version": BACKUP_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "encrypted": False,
        "snapshot": snapshot.to_dict(),
    }
    atomic_write_text(
        root / BACKUP_METADATA,
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
    )


def _read_metadata(root: Path) -> DataSnapshot:
    values = _json_object(
        (root / BACKUP_METADATA).read_text(encoding="utf-8"),
        "backup metadata",
    )
    if values.get("schema_version") != BACKUP_SCHEMA_VERSION:
        raise MalformedSerializedDataError("unsupported backup schema")
    snapshot = values.get("snapshot")
    if not isinstance(snapshot, dict):
        raise MalformedSerializedDataError("backup snapshot is invalid")
    try:
        return DataSnapshot(
            event_count=int(snapshot["event_count"]),
            event_schema_versions=tuple(snapshot["event_schema_versions"]),
            state_versions=tuple(
                (str(path), int(version))
                for path, version in snapshot["state_versions"]
            ),
            state_schema_versions=tuple(snapshot["state_schema_versions"]),
            memory_count=int(snapshot["memory_count"]),
            memory_schema_versions=tuple(snapshot["memory_schema_versions"]),
            tombstone_count=int(snapshot["tombstone_count"]),
            memory_index_schema_version=int(
                snapshot["memory_index_schema_version"]
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise MalformedSerializedDataError("backup snapshot is invalid") from error


def _require_same_snapshot(expected: DataSnapshot, actual: DataSnapshot) -> None:
    if expected != actual:
        raise MalformedSerializedDataError(
            "restored event, state, memory or tombstone counts do not match"
        )


def _write_zip(root: Path, archive: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{archive.name}-",
        suffix=".tmp",
        dir=archive.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as output:
            for path in sorted(item for item in root.rglob("*") if item.is_file()):
                output.write(path, path.relative_to(root).as_posix())
        os.replace(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)


def _extract_zip(archive: Path, target: Path) -> None:
    with zipfile.ZipFile(archive) as source:
        names: set[str] = set()
        for info in source.infolist():
            relative = PurePosixPath(info.filename)
            if (
                info.is_dir()
                or info.filename in names
                or relative.is_absolute()
                or ".." in relative.parts
                or "\\" in info.filename
                or any(":" in part for part in relative.parts)
            ):
                raise MalformedSerializedDataError("backup contains an unsafe path")
            names.add(info.filename)
            destination = target.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open(info) as input_file, destination.open("wb") as output:
                shutil.copyfileobj(input_file, output)


def _verify_manifest(root: Path) -> tuple[FileManifestEntry, ...]:
    path = root / BACKUP_MANIFEST
    try:
        values = _json_object(path.read_text(encoding="utf-8"), "backup manifest")
        if values.get("schema_version") != 1 or not isinstance(
            values.get("entries"), list
        ):
            raise MalformedSerializedDataError("backup manifest is invalid")
        expected = tuple(
            FileManifestEntry(
                path=str(entry["path"]),
                size=int(entry["size"]),
                sha256=str(entry["sha256"]),
            )
            for entry in values["entries"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise MalformedSerializedDataError("backup manifest is invalid") from error
    if expected != build_manifest(root).entries:
        raise MalformedSerializedDataError("backup checksum verification failed")
    return expected


def _memory_paths(root: Path) -> tuple[Path, ...]:
    if not root.exists():
        return ()
    return tuple(
        path
        for path in sorted(root.rglob("*.json"))
        if ".locks" not in path.parts
    )


def _records(path: Path, label: str) -> tuple[str, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeError as error:
        raise MalformedSerializedDataError(f"{label} file is not UTF-8") from error
    if any(not line.strip() for line in lines):
        raise MalformedSerializedDataError(f"{label} file contains a blank record")
    return tuple(lines)


def _json_object(payload: str, label: str) -> dict[str, object]:
    try:
        values = json.loads(payload)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise MalformedSerializedDataError(f"{label} is not valid JSON") from error
    if not isinstance(values, dict):
        raise MalformedSerializedDataError(f"{label} must be a JSON object")
    return values


def _schema_version(values: dict[str, object], label: str) -> int:
    version = values.get("schema_version")
    if type(version) is not int:
        raise MalformedSerializedDataError(f"{label} schema version is invalid")
    return version


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
