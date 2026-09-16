from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from self_cognition.infrastructure.persistence.atomic_io import atomic_write_text
from self_cognition.infrastructure.persistence.file_lock import BlockingFileLock


MODEL_HEALTH_SNAPSHOT_SCHEMA_VERSION = 1


class FileModelHealthStore:
    """Bounded, best-effort diagnostic snapshot of model provider health.

    The snapshot is deliberately not an authoritative state store.  It is a
    cache-like aid so a process restart does not immediately forget a provider
    that is in a bounded failure cooldown.  Corrupt or stale data is ignored
    and can be cleared without affecting events, state or governance controls.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = BlockingFileLock(
            self._path.with_name(self._path.name + ".lock")
        )

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> tuple[dict[str, object], ...]:
        if not self._path.exists():
            return ()
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return ()
        if not isinstance(payload, dict):
            return ()
        if payload.get("schema_version") != MODEL_HEALTH_SNAPSHOT_SCHEMA_VERSION:
            return ()
        entries = payload.get("entries")
        if not isinstance(entries, list):
            return ()
        return tuple(
            dict(entry) for entry in entries if isinstance(entry, dict)
        )

    def save(self, entries: Iterable[Mapping[str, object]]) -> None:
        materialized = tuple(dict(entry) for entry in entries)
        payload = {
            "schema_version": MODEL_HEALTH_SNAPSHOT_SCHEMA_VERSION,
            "entries": materialized,
        }
        with self._lock:
            atomic_write_text(
                self._path,
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
            )

    def clear(self) -> None:
        with self._lock:
            self._path.unlink(missing_ok=True)
