from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from uuid import UUID

from self_cognition.core.errors import (
    ContractValidationError,
    MalformedSerializedDataError,
)
from self_cognition.core.runs import RunRecord, run_from_dict, run_to_dict
from self_cognition.infrastructure.persistence.atomic_io import atomic_write_text
from self_cognition.infrastructure.persistence.file_lock import FileLock


class FileRunRepository:
    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._locks = self._directory / ".locks"
        self._locks.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def get(self, run_id: UUID) -> RunRecord | None:
        path = self._path(run_id)
        if not path.exists():
            return None
        try:
            return run_from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            ContractValidationError,
        ) as error:
            raise MalformedSerializedDataError("invalid run record") from error

    def save(self, record: RunRecord) -> RunRecord:
        with self._lock, FileLock(self._lock_path(record.run_id)):
            current = self.get(record.run_id)
            if current is not None and current != record:
                if record.updated_at < current.updated_at:
                    raise ContractValidationError("run record update is stale")
            atomic_write_text(
                self._path(record.run_id),
                json.dumps(run_to_dict(record), ensure_ascii=False, sort_keys=True)
                + "\n",
            )
            return record

    def read_by_parent(self, parent_run_id: UUID) -> tuple[RunRecord, ...]:
        return tuple(
            record
            for record in self._read_all()
            if record.parent_run_id == parent_run_id
        )

    def read_incomplete(self) -> tuple[RunRecord, ...]:
        return tuple(
            record for record in self._read_all() if not record.status.is_terminal
        )

    def _read_all(self) -> tuple[RunRecord, ...]:
        records = []
        for path in sorted(self._directory.glob("*.json")):
            try:
                records.append(
                    run_from_dict(json.loads(path.read_text(encoding="utf-8")))
                )
            except (
                OSError,
                UnicodeError,
                json.JSONDecodeError,
                ContractValidationError,
            ) as error:
                raise MalformedSerializedDataError(
                    f"invalid run record: {path.name}"
                ) from error
        return tuple(records)

    def _path(self, run_id: UUID) -> Path:
        return self._directory / f"{run_id}.json"

    def _lock_path(self, run_id: UUID) -> Path:
        return self._locks / f"{run_id}.lock"
