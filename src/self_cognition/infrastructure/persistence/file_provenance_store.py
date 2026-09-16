from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import UUID

from self_cognition.core.provenance import ProvenanceGraph
from self_cognition.core.scopes import MindScope
from self_cognition.infrastructure.persistence.atomic_io import atomic_write_text
from self_cognition.infrastructure.persistence.file_lock import BlockingFileLock


class FileProvenanceStore:
    """File-backed, rebuildable provenance graph index."""

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)

    def write(self, graph: ProvenanceGraph) -> None:
        if not isinstance(graph, ProvenanceGraph):
            raise TypeError("graph must be a ProvenanceGraph")
        path = self._path_for(graph.mind_id)
        payload = json.dumps(
            graph.to_state_value(),
            ensure_ascii=False,
            sort_keys=True,
        )
        with BlockingFileLock(self._lock_path(graph.mind_id)):
            self._directory.mkdir(parents=True, exist_ok=True)
            atomic_write_text(path, payload)

    def read(self, mind: MindScope) -> ProvenanceGraph | None:
        if not isinstance(mind, MindScope):
            raise TypeError("mind must be a MindScope")
        path = self._path_for(mind.mind_id)
        if not path.exists():
            return None
        with BlockingFileLock(self._lock_path(mind.mind_id)):
            if not path.exists():
                return None
            return ProvenanceGraph.from_state_value(
                json.loads(path.read_text(encoding="utf-8"))
            )

    def delete(self, mind: MindScope) -> None:
        if not isinstance(mind, MindScope):
            raise TypeError("mind must be a MindScope")
        path = self._path_for(mind.mind_id)
        with BlockingFileLock(self._lock_path(mind.mind_id)):
            path.unlink(missing_ok=True)

    def _path_for(self, mind_id: str) -> Path:
        return self._directory / f"{_digest(mind_id)}.json"

    def _lock_path(self, mind_id: str) -> Path:
        return self._directory / f"{_digest(mind_id)}.lock"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
