from __future__ import annotations

from self_cognition.core.protocols import EventStore, MemoryRepository
from self_cognition.core.scopes import MindScope
from self_cognition.core.vector_index import VectorIndex, VectorMatch, build_vector_index, embed_text
from self_cognition.infrastructure.persistence.file_vector_index_store import (
    FileVectorIndexStore,
)


class VectorIndexService:
    """Rebuildable local vector index over memory records.

    Vector similarity only narrows retrieval candidates. It is not evidence of
    facts, relationships, identity, causality or capability.
    """

    def __init__(
        self,
        memory_repository: MemoryRepository,
        event_store: EventStore,
        store: FileVectorIndexStore,
    ) -> None:
        self._memory_repository = memory_repository
        self._event_store = event_store
        self._store = store

    def rebuild(self, mind: MindScope) -> VectorIndex:
        if not isinstance(mind, MindScope):
            raise TypeError("mind must be a MindScope")
        index = build_vector_index(
            mind.mind_id,
            self._memory_repository.read_by_mind(mind),
        )
        self._store.write(index)
        return index

    def rebuild_all(self) -> tuple[VectorIndex, ...]:
        known = set(self._known_mind_ids())
        for stale_mind_id in set(self._store.known_mind_ids()) - known:
            self._store.delete(MindScope(stale_mind_id))
        return tuple(
            self.rebuild(MindScope(mind_id))
            for mind_id in sorted(known)
        )

    def load(self, mind: MindScope) -> VectorIndex | None:
        if not isinstance(mind, MindScope):
            raise TypeError("mind must be a MindScope")
        return self._store.read(mind)

    def delete(self, mind: MindScope) -> None:
        if not isinstance(mind, MindScope):
            raise TypeError("mind must be a MindScope")
        self._store.delete(mind)

    def delete_all(self) -> None:
        self._store.delete_all()

    def verify(self) -> bool:
        known = set(self._known_mind_ids())
        stored = set(self._store.known_mind_ids())
        if known != stored:
            return False
        for mind_id in known:
            mind = MindScope(mind_id)
            index = self._store.read(mind)
            if index is None or index.mind_id != mind_id:
                return False
            if not index.is_compatible(self._memory_repository.read_by_mind(mind)):
                return False
        return True

    def search(
        self,
        mind: MindScope,
        query: str,
        *,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> tuple[VectorMatch, ...]:
        index = self.load(mind)
        records = self._memory_repository.read_by_mind(mind)
        if index is None:
            if mind.mind_id not in set(self._known_mind_ids()):
                return ()
            index = self.rebuild(mind)
        elif not index.is_compatible(records):
            index = self.rebuild(mind)
        return index.search(query, limit=limit, min_score=min_score)

    def _known_mind_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    event.subject.mind.mind_id
                    for event in self._event_store.read_all()
                }
            )
        )

    @staticmethod
    def embed(text: str) -> tuple[float, ...]:
        return embed_text(text)
