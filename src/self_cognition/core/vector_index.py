from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import isfinite, sqrt
from uuid import UUID

from self_cognition.core.errors import ContractValidationError
from self_cognition.core.indexes import text_terms
from self_cognition.core.memories import MemoryRecord


VECTOR_INDEX_SCHEMA_VERSION = 1
VECTOR_DIMENSION = 64


def embed_terms(terms: frozenset[str], dimension: int = VECTOR_DIMENSION) -> tuple[float, ...]:
    if dimension < 1:
        raise ValueError("vector dimension must be positive")
    vector = [0.0] * dimension
    for term in sorted(terms):
        digest = hashlib.sha256(term.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[bucket] += sign
    norm = sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return tuple(vector)
    return tuple(round(value / norm, 12) for value in vector)


def embed_text(text: str, dimension: int = VECTOR_DIMENSION) -> tuple[float, ...]:
    return embed_terms(text_terms(text), dimension)


def cosine_similarity(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimension")
    return sum(a * b for a, b in zip(left, right))


@dataclass(frozen=True, slots=True)
class VectorEntry:
    memory_id: UUID
    memory_type: str
    target_fields: tuple[str, ...]
    content_hash: str
    vector: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.memory_id, UUID):
            raise ContractValidationError("vector entry memory_id must be a UUID")
        if not isinstance(self.memory_type, str) or not self.memory_type.strip():
            raise ContractValidationError("vector entry memory_type must not be blank")
        if not isinstance(self.target_fields, tuple) or any(
            not isinstance(value, str) or not value.strip()
            for value in self.target_fields
        ):
            raise ContractValidationError("vector entry target_fields are invalid")
        if not isinstance(self.content_hash, str) or not self.content_hash.strip():
            raise ContractValidationError("vector entry content_hash must not be blank")
        if not isinstance(self.vector, tuple) or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not isfinite(value)
            for value in self.vector
        ):
            raise ContractValidationError("vector entry vector is invalid")

    def to_state_value(self) -> dict[str, object]:
        return {
            "memory_id": str(self.memory_id),
            "memory_type": self.memory_type,
            "target_fields": list(self.target_fields),
            "content_hash": self.content_hash,
            "vector": list(self.vector),
        }

    @classmethod
    def from_state_value(cls, value: object) -> "VectorEntry":
        data = _require_mapping(value, "vector entry")
        raw_vector = data.get("vector")
        if not isinstance(raw_vector, list):
            raise ContractValidationError("vector entry vector must be an array")
        raw_fields = data.get("target_fields")
        if not isinstance(raw_fields, list):
            raise ContractValidationError("vector entry target_fields must be an array")
        return cls(
            memory_id=_require_uuid(data.get("memory_id"), "vector entry.memory_id"),
            memory_type=_require_text(
                data.get("memory_type"),
                "vector entry.memory_type",
            ),
            target_fields=tuple(
                _require_text(item, "vector entry.target_fields[]")
                for item in raw_fields
            ),
            content_hash=_require_text(
                data.get("content_hash"),
                "vector entry.content_hash",
            ),
            vector=tuple(
                _require_float(item, "vector entry.vector[]")
                for item in raw_vector
            ),
        )


@dataclass(frozen=True, slots=True)
class VectorMatch:
    memory_id: UUID
    memory_version: int
    score: float
    source_ref: str


@dataclass(frozen=True, slots=True)
class VectorIndex:
    mind_id: str
    dimension: int
    entries: tuple[VectorEntry, ...]
    source_memory_versions: tuple[tuple[str, int], ...]
    schema_version: int = VECTOR_INDEX_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.mind_id, str) or not self.mind_id.strip():
            raise ContractValidationError("vector index mind_id must not be blank")
        if type(self.dimension) is not int or self.dimension < 1:
            raise ContractValidationError("vector index dimension must be positive")
        if self.schema_version != VECTOR_INDEX_SCHEMA_VERSION:
            raise ContractValidationError("vector index schema version is invalid")
        memory_ids = {entry.memory_id for entry in self.entries}
        if len(memory_ids) != len(self.entries):
            raise ContractValidationError("vector index memory IDs must be unique")
        for entry in self.entries:
            if len(entry.vector) != self.dimension:
                raise ContractValidationError(
                    "vector entry dimension does not match index dimension"
                )
        version_ids = {memory_id for memory_id, _ in self.source_memory_versions}
        if len(version_ids) != len(self.source_memory_versions):
            raise ContractValidationError(
                "vector index source memory versions must be unique"
            )
        for memory_id, version in self.source_memory_versions:
            try:
                UUID(memory_id)
            except ValueError as error:
                raise ContractValidationError(
                    "vector index source memory ID is invalid"
                ) from error
            if type(version) is not int or version < 1:
                raise ContractValidationError(
                    "vector index source memory version is invalid"
                )

    def is_compatible(self, memories: tuple[MemoryRecord, ...]) -> bool:
        current = tuple(
            sorted(
                (str(record.memory_id), record.version)
                for record in memories
            )
        )
        return (
            self.schema_version == VECTOR_INDEX_SCHEMA_VERSION
            and self.source_memory_versions == current
        )

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> tuple[VectorMatch, ...]:
        if limit < 1:
            raise ValueError("limit must be positive")
        if not isinstance(query, str):
            raise TypeError("query must be text")
        if not self.entries:
            return ()
        query_vector = embed_text(query, self.dimension)
        versions = dict(self.source_memory_versions)
        scored: list[tuple[float, VectorEntry]] = []
        for entry in self.entries:
            score = cosine_similarity(query_vector, entry.vector)
            if score >= min_score:
                scored.append((score, entry))
        scored.sort(key=lambda item: (-item[0], item[1].memory_id.int))
        return tuple(
            VectorMatch(
                memory_id=entry.memory_id,
                memory_version=versions[str(entry.memory_id)],
                score=score,
                source_ref=f"memory:{entry.memory_id}",
            )
            for score, entry in scored[:limit]
        )

    def to_state_value(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mind_id": self.mind_id,
            "dimension": self.dimension,
            "entries": [entry.to_state_value() for entry in self.entries],
            "source_memory_versions": [
                [memory_id, version]
                for memory_id, version in self.source_memory_versions
            ],
        }

    @classmethod
    def from_state_value(cls, value: object) -> "VectorIndex":
        data = _require_mapping(value, "vector index")
        entries = data.get("entries")
        source_versions = data.get("source_memory_versions")
        if not isinstance(entries, list):
            raise ContractValidationError("vector index entries must be an array")
        if not isinstance(source_versions, list):
            raise ContractValidationError(
                "vector index source_memory_versions must be an array"
            )
        parsed_versions: list[tuple[str, int]] = []
        for item in source_versions:
            if not isinstance(item, list) or len(item) != 2:
                raise ContractValidationError(
                    "vector index source memory version is invalid"
                )
            parsed_versions.append(
                (
                    _require_text(item[0], "vector index.source_memory_versions[]"),
                    _require_int(item[1], "vector index.source_memory_versions[]"),
                )
            )
        return cls(
            mind_id=_require_text(data.get("mind_id"), "vector index.mind_id"),
            dimension=_require_int(
                data.get("dimension"),
                "vector index.dimension",
            ),
            entries=tuple(
                VectorEntry.from_state_value(item) for item in entries
            ),
            source_memory_versions=tuple(parsed_versions),
            schema_version=_require_int(
                data.get("schema_version"),
                "vector index.schema_version",
            ),
        )


def build_vector_index(
    mind_id: str,
    memories: tuple[MemoryRecord, ...],
) -> VectorIndex:
    entries: list[VectorEntry] = []
    for record in sorted(memories, key=lambda item: item.memory_id.int):
        target_fields = tuple(
            sorted(
                {
                    source.target_field
                    for source in record.sources
                    if source.target_field
                }
            )
        ) or (f"memory.{record.memory_type.value}",)
        entries.append(
            VectorEntry(
                memory_id=record.memory_id,
                memory_type=record.memory_type.value,
                target_fields=target_fields,
                content_hash=hashlib.sha256(
                    _canonical_record_text(record).encode("utf-8")
                ).hexdigest(),
                vector=embed_terms(
                    text_terms(record.content, record.memory_type.value, record.cues),
                    VECTOR_DIMENSION,
                ),
            )
        )
    return VectorIndex(
        mind_id=mind_id,
        dimension=VECTOR_DIMENSION,
        entries=tuple(entries),
        source_memory_versions=tuple(
            sorted((str(record.memory_id), record.version) for record in memories)
        ),
    )


def _canonical_record_text(record: MemoryRecord) -> str:
    return json.dumps(
        {
            "content": record.content,
            "memory_type": record.memory_type.value,
            "cues": {
                "people": list(record.cues.people),
                "topics": list(record.cues.topics),
                "time_keys": list(record.cues.time_keys),
                "relationships": list(record.cues.relationships),
                "tasks": list(record.cues.tasks),
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _require_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ContractValidationError(f"{name} must be an object")
    return value


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{name} must be non-blank text")
    return value


def _require_uuid(value: object, name: str) -> UUID:
    try:
        return UUID(str(value))
    except (TypeError, ValueError) as error:
        raise ContractValidationError(f"{name} must be a UUID") from error


def _require_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise ContractValidationError(f"{name} must be an integer")
    return value


def _require_float(value: object, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(value)
    ):
        raise ContractValidationError(f"{name} must be a finite number")
    return float(value)
