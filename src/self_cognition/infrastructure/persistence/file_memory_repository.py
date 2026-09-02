import hashlib
import json
import os
import shutil
from pathlib import Path
from uuid import UUID

from self_cognition.core.errors import (
    ContractValidationError,
    MalformedSerializedDataError,
    VersionConflictError,
)
from self_cognition.core.memories import (
    MemoryAccessRecord,
    MemoryRecord,
    MemoryType,
)
from self_cognition.core.scopes import SubjectScope
from self_cognition.infrastructure.persistence.atomic_io import atomic_write_text
from self_cognition.infrastructure.persistence.file_lock import FileLock
from self_cognition.infrastructure.persistence.serialization import (
    memory_access_from_json,
    memory_access_to_json,
    memory_from_json,
    memory_to_json,
)


class FileMemoryRepository:
    def __init__(
        self,
        directory: str | Path,
        index_directory: str | Path,
        access_directory: str | Path | None = None,
    ) -> None:
        self._directory = Path(directory)
        self._index_directory = Path(index_directory)
        self._access_directory = Path(
            access_directory or self._directory.parent / "memory_access"
        )
        self._directory.mkdir(parents=True, exist_ok=True)
        self._index_directory.mkdir(parents=True, exist_ok=True)
        self._access_directory.mkdir(parents=True, exist_ok=True)

    def load(
        self,
        subject: SubjectScope,
        memory_id: UUID,
    ) -> MemoryRecord | None:
        history = self.read_history(subject, memory_id)
        return history[-1] if history else None

    def save(self, record: MemoryRecord, expected_version: int) -> None:
        if record.memory_type is MemoryType.WORKING:
            raise ContractValidationError(
                "working memory belongs to run checkpoints, not long-term memory"
            )
        with FileLock(self._lock_path(record.subject)):
            current = self.load(record.subject, record.memory_id)
            if current == record:
                self._write_index_unlocked(record.subject)
                return
            current_version = current.version if current is not None else 0
            if expected_version != current_version:
                raise VersionConflictError(
                    "expected version does not match stored memory version"
                )
            if record.version != current_version + 1:
                raise VersionConflictError(
                    "new memory version must follow the stored memory version"
                )
            if current is not None and record.memory_type is not current.memory_type:
                raise ContractValidationError(
                    "memory_type cannot change between versions"
                )
            atomic_write_text(
                self._version_path(
                    record.subject,
                    record.memory_id,
                    record.version,
                ),
                memory_to_json(record) + "\n",
            )
            self._write_index_unlocked(record.subject)

    def read_history(
        self,
        subject: SubjectScope,
        memory_id: UUID,
    ) -> tuple[MemoryRecord, ...]:
        memory_directory = self._memory_directory(subject, memory_id)
        if not memory_directory.exists():
            return ()
        records = tuple(
            self._read_record(path, subject, memory_id)
            for path in sorted(memory_directory.glob("*.json"))
        )
        expected_versions = tuple(range(1, len(records) + 1))
        if tuple(record.version for record in records) != expected_versions:
            raise MalformedSerializedDataError(
                "memory history versions must be contiguous"
            )
        return records

    def read_by_subject(
        self,
        subject: SubjectScope,
    ) -> tuple[MemoryRecord, ...]:
        subject_directory = self._subject_directory(subject)
        if not subject_directory.exists():
            return ()
        records = []
        for memory_directory in sorted(
            path for path in subject_directory.iterdir() if path.is_dir()
        ):
            try:
                memory_id = UUID(memory_directory.name)
            except ValueError as error:
                raise MalformedSerializedDataError(
                    "memory directory name must be a UUID"
                ) from error
            record = self.load(subject, memory_id)
            if record is not None:
                records.append(record)
        return tuple(records)

    def rebuild_index(self, subject: SubjectScope) -> None:
        with FileLock(self._lock_path(subject)):
            self._write_index_unlocked(subject)

    def delete(
        self,
        subject: SubjectScope,
        memory_ids: tuple[UUID, ...],
    ) -> None:
        with FileLock(self._lock_path(subject)):
            for memory_id in memory_ids:
                memory_directory = self._memory_directory(subject, memory_id)
                if memory_directory.exists():
                    shutil.rmtree(memory_directory)
                self._access_path(subject, memory_id).unlink(missing_ok=True)
                self._access_lock_path(subject, memory_id).unlink(missing_ok=True)
            self._write_index_unlocked(subject)

    def record_access(self, record: MemoryAccessRecord) -> None:
        memory = self.load(record.subject, record.memory_id)
        if memory is None:
            raise ContractValidationError("cannot audit access to an unknown memory")
        path = self._access_path(record.subject, record.memory_id)
        with FileLock(self._access_lock_path(record.subject, record.memory_id)):
            history = self.read_access_history(record.subject, record.memory_id)
            if any(existing.access_id == record.access_id for existing in history):
                if any(existing != record for existing in history):
                    raise ContractValidationError(
                        "access_id already has a different record"
                    )
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(memory_access_to_json(record) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def read_access_history(
        self,
        subject: SubjectScope,
        memory_id: UUID,
    ) -> tuple[MemoryAccessRecord, ...]:
        path = self._access_path(subject, memory_id)
        if not path.exists():
            return ()
        records: list[MemoryAccessRecord] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        raise MalformedSerializedDataError(
                            f"blank memory access record at line {line_number}"
                        )
                    try:
                        record = memory_access_from_json(line)
                    except MalformedSerializedDataError as error:
                        raise MalformedSerializedDataError(
                            f"invalid memory access record at line {line_number}"
                        ) from error
                    if (
                        record.subject != subject
                        or record.memory_id != memory_id
                    ):
                        raise MalformedSerializedDataError(
                            "memory access scope does not match requested memory"
                        )
                    records.append(record)
        except UnicodeError as error:
            raise MalformedSerializedDataError(
                "memory access log is not valid UTF-8"
            ) from error
        return tuple(records)

    def _write_index_unlocked(self, subject: SubjectScope) -> None:
        records = self.read_by_subject(subject)
        catalog = {
            "schema_version": 1,
            "subject": {
                "mind_id": subject.mind.mind_id,
                "kind": subject.subject.kind.value,
                "subject_id": subject.subject.subject_id,
            },
            "records": [
                {
                    "memory_id": str(record.memory_id),
                    "memory_type": record.memory_type.value,
                    "latest_version": record.version,
                    "lifecycle_status": record.lifecycle_status.value,
                }
                for record in records
            ],
        }
        atomic_write_text(
            self._index_path(subject),
            json.dumps(
                catalog,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
        )

    def _read_record(
        self,
        path: Path,
        subject: SubjectScope,
        memory_id: UUID,
    ) -> MemoryRecord:
        try:
            record = memory_from_json(path.read_text(encoding="utf-8"))
        except UnicodeError as error:
            raise MalformedSerializedDataError(
                "memory record is not valid UTF-8"
            ) from error
        if record.subject != subject:
            raise MalformedSerializedDataError(
                "memory record scope does not match requested scope"
            )
        if record.memory_id != memory_id:
            raise MalformedSerializedDataError(
                "memory record ID does not match its directory"
            )
        if path.stem != f"{record.version:08d}":
            raise MalformedSerializedDataError(
                "memory record version does not match its file name"
            )
        return record

    def _subject_directory(self, subject: SubjectScope) -> Path:
        return (
            self._directory
            / _digest(subject.mind.mind_id)
            / subject.subject.kind.value
            / _digest(subject.subject.subject_id)
        )

    def _memory_directory(
        self,
        subject: SubjectScope,
        memory_id: UUID,
    ) -> Path:
        return self._subject_directory(subject) / str(memory_id)

    def _version_path(
        self,
        subject: SubjectScope,
        memory_id: UUID,
        version: int,
    ) -> Path:
        return self._memory_directory(subject, memory_id) / f"{version:08d}.json"

    def _index_path(self, subject: SubjectScope) -> Path:
        return (
            self._index_directory
            / _digest(subject.mind.mind_id)
            / subject.subject.kind.value
            / f"{_digest(subject.subject.subject_id)}.json"
        )

    def _lock_path(self, subject: SubjectScope) -> Path:
        return (
            self._directory
            / ".locks"
            / _digest(subject.mind.mind_id)
            / subject.subject.kind.value
            / f"{_digest(subject.subject.subject_id)}.lock"
        )

    def _access_path(self, subject: SubjectScope, memory_id: UUID) -> Path:
        return (
            self._access_directory
            / _digest(subject.mind.mind_id)
            / subject.subject.kind.value
            / _digest(subject.subject.subject_id)
            / f"{memory_id}.jsonl"
        )

    def _access_lock_path(self, subject: SubjectScope, memory_id: UUID) -> Path:
        return self._access_path(subject, memory_id).with_suffix(".lock")


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
