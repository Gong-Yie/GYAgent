from __future__ import annotations

import hashlib
import json
from pathlib import Path

from self_cognition.core.scopes import MindScope
from self_cognition.core.vector_index import VectorIndex
from self_cognition.infrastructure.persistence.atomic_io import atomic_write_text
from self_cognition.infrastructure.persistence.file_lock import BlockingFileLock


MANIFEST_SCHEMA_VERSION = 1


class FileVectorIndexStore:
    """File-backed, rebuildable local vector index."""

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)

    def write(self, index: VectorIndex) -> None:
        if not isinstance(index, VectorIndex):
            raise TypeError("index must be a VectorIndex")
        payload = json.dumps(
            index.to_state_value(),
            ensure_ascii=False,
            sort_keys=True,
        )
        with BlockingFileLock(self._lock_path(index.mind_id)):
            self._directory.mkdir(parents=True, exist_ok=True)
            atomic_write_text(self._path_for(index.mind_id), payload)
        mind_ids = set(self.known_mind_ids())
        mind_ids.add(index.mind_id)
        self._write_manifest(mind_ids)

    def read(self, mind: MindScope) -> VectorIndex | None:
        if not isinstance(mind, MindScope):
            raise TypeError("mind must be a MindScope")
        path = self._path_for(mind.mind_id)
        if not path.exists():
            return None
        with BlockingFileLock(self._lock_path(mind.mind_id)):
            if not path.exists():
                return None
            return VectorIndex.from_state_value(
                json.loads(path.read_text(encoding="utf-8"))
            )

    def delete(self, mind: MindScope) -> None:
        if not isinstance(mind, MindScope):
            raise TypeError("mind must be a MindScope")
        with BlockingFileLock(self._lock_path(mind.mind_id)):
            self._path_for(mind.mind_id).unlink(missing_ok=True)
        mind_ids = set(self.known_mind_ids())
        mind_ids.discard(mind.mind_id)
        self._write_manifest(mind_ids)

    def known_mind_ids(self) -> tuple[str, ...]:
        manifest = self._read_manifest()
        if manifest is None:
            return ()
        raw = manifest.get("mind_ids")
        if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
            raise ValueError("vector index manifest mind_ids must be a string array")
        return tuple(sorted(set(raw)))

    def delete_all(self) -> None:
        if not self._directory.exists():
            return
        for path in self._directory.glob("*.json"):
            path.unlink(missing_ok=True)

    def _path_for(self, mind_id: str) -> Path:
        return self._directory / f"{_digest(mind_id)}.json"

    def _lock_path(self, mind_id: str) -> Path:
        return self._directory / f"{_digest(mind_id)}.lock"

    def _manifest_path(self) -> Path:
        return self._directory / "manifest.json"

    def _manifest_lock_path(self) -> Path:
        return self._directory / "manifest.lock"

    def _read_manifest(self) -> dict[str, object] | None:
        path = self._manifest_path()
        if not path.exists():
            return None
        with BlockingFileLock(self._manifest_lock_path()):
            if not path.exists():
                return None
            value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("vector index manifest must be an object")
        if value.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ValueError("vector index manifest schema version is invalid")
        return value

    def _write_manifest(self, mind_ids: set[str]) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "mind_ids": sorted(mind_ids),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        with BlockingFileLock(self._manifest_lock_path()):
            atomic_write_text(self._manifest_path(), payload)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
