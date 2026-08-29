from dataclasses import dataclass
from typing import Mapping
from uuid import UUID


@dataclass(frozen=True, slots=True)
class ConflictRecord:
    target_field: str
    candidate_contribution_ids: tuple[UUID, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class StateEntry:
    value: object
    confidence: float
    evidence_event_ids: tuple[UUID, ...]
    contribution_ids: tuple[UUID, ...]

@dataclass(frozen=True, slots=True)
class SubjectState:
    subject_id: str
    version: int
    entries: Mapping[str, StateEntry]
    applied_contribution_ids: frozenset[UUID]
    conflicts: frozenset[ConflictRecord]

    @classmethod
    def empty(cls, subject_id: str) -> "SubjectState":
        return cls(
            subject_id=subject_id,
            version=0,
            entries={},
            applied_contribution_ids=frozenset(),
            conflicts=frozenset(),
        )

    def get(self, field: str) -> StateEntry:
        return self.entries[field]
