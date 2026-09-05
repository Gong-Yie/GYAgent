from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Mapping
from uuid import NAMESPACE_URL, UUID, uuid5

from self_cognition.core.contributions import CognitiveContribution, CognitionType
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.errors import ContractValidationError
from self_cognition.core.metacognition import ConflictStatus
from self_cognition.core.scopes import (
    DataScope,
    DEFAULT_MIND_ID,
    MindScope,
    SubjectKind,
    SubjectRef,
    SubjectScope,
)


@dataclass(frozen=True, slots=True)
class ConflictRecord:
    target_field: str
    candidate_contribution_ids: tuple[UUID, ...]
    reason: str
    status: ConflictStatus = ConflictStatus.OPEN
    evidence_refs: tuple[EvidenceRef, ...] = ()
    requires_confirmation: bool = False
    confirmed: bool = False
    resolution_reason: str | None = None
    reviewed_by: UUID | None = None
    selected_contribution_id: UUID | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, ConflictStatus):
            raise ContractValidationError("conflict status is invalid")
        if not isinstance(self.requires_confirmation, bool) or not isinstance(
            self.confirmed, bool
        ):
            raise ContractValidationError("conflict confirmation flags must be boolean")
        if self.status is not ConflictStatus.OPEN and not self.resolution_reason:
            raise ContractValidationError(
                "closed conflict requires a resolution reason"
            )
        if self.status is ConflictStatus.RESOLVED and (
            self.selected_contribution_id not in self.candidate_contribution_ids
        ):
            raise ContractValidationError("resolved conflict must select a candidate")

    @property
    def conflict_id(self) -> UUID:
        candidates = ":".join(
            sorted(str(item) for item in self.candidate_contribution_ids)
        )
        return uuid5(NAMESPACE_URL, f"conflict:{self.target_field}:{candidates}")


@dataclass(frozen=True, slots=True)
class StateAtom:
    value: object
    cognition_type: CognitionType
    confidence: float
    scope: DataScope
    evidence_refs: tuple[EvidenceRef, ...]
    contribution_ids: tuple[UUID, ...]
    created_at: datetime
    valid_from: datetime
    expires_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cognition_type, CognitionType):
            raise ContractValidationError("cognition_type must be a CognitionType")
        if not 0.0 <= self.confidence <= 1.0:
            raise ContractValidationError("confidence must be between 0 and 1")
        if not isinstance(self.scope, DataScope):
            raise ContractValidationError("scope must be a DataScope")
        if not self.evidence_refs:
            raise ContractValidationError("evidence_refs must not be empty")
        if not self.contribution_ids:
            raise ContractValidationError("contribution_ids must not be empty")
        _require_aware(self.created_at, "created_at")
        _require_aware(self.valid_from, "valid_from")
        if self.expires_at is not None:
            _require_aware(self.expires_at, "expires_at")
            if self.expires_at <= self.valid_from:
                raise ContractValidationError(
                    "expires_at must be later than valid_from"
                )


StateEntry = StateAtom


class StateDecisionStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    PENDING = "pending"


@dataclass(frozen=True, slots=True)
class StateChangeRecord:
    contribution: CognitiveContribution
    status: StateDecisionStatus
    reason: str
    old_version: int
    new_version: int
    decided_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.contribution, CognitiveContribution):
            raise ContractValidationError(
                "change contribution must be a CognitiveContribution"
            )
        if not isinstance(self.status, StateDecisionStatus):
            raise ContractValidationError(
                "change status must be a StateDecisionStatus"
            )
        if not self.reason.strip():
            raise ContractValidationError("change reason must not be blank")
        if self.old_version < 0 or self.new_version <= self.old_version:
            raise ContractValidationError("change versions are invalid")
        _require_aware(self.decided_at, "decided_at")

@dataclass(frozen=True, slots=True)
class SubjectState:
    subject_id: str
    version: int
    entries: Mapping[str, StateAtom]
    applied_contribution_ids: frozenset[UUID]
    conflicts: frozenset[ConflictRecord]
    changes: tuple[StateChangeRecord, ...] = ()
    mind_id: str = DEFAULT_MIND_ID
    subject_kind: SubjectKind = SubjectKind.USER

    def __post_init__(self) -> None:
        if not isinstance(self.subject_id, str) or not self.subject_id.strip():
            raise ContractValidationError("subject_id must not be blank")
        if not isinstance(self.mind_id, str) or not self.mind_id.strip():
            raise ContractValidationError("mind_id must not be blank")
        if not isinstance(self.subject_kind, SubjectKind):
            raise ContractValidationError("subject_kind must be a SubjectKind")
        if (
            self.subject_kind is SubjectKind.MIND
            and self.subject_id != self.mind_id
        ):
            raise ContractValidationError(
                "mind subject ID must match its mind scope"
            )
        if any(not isinstance(atom, StateAtom) for atom in self.entries.values()):
            raise ContractValidationError("entries must contain StateAtom values")
        if any(
            atom.scope.owner != self.subject_scope
            for atom in self.entries.values()
        ):
            raise ContractValidationError(
                "state atom scope owner must match the state subject"
            )
        if any(
            change.contribution.target != self.subject_scope
            for change in self.changes
        ):
            raise ContractValidationError(
                "state changes must target the state subject"
            )
        if any(change.new_version > self.version for change in self.changes):
            raise ContractValidationError(
                "state change version must not exceed state version"
            )

    @property
    def subject_scope(self) -> SubjectScope:
        return SubjectScope(
            mind=MindScope(self.mind_id),
            subject=SubjectRef(self.subject_kind, self.subject_id),
        )

    @classmethod
    def empty(
        cls,
        subject_id: str,
        *,
        mind_id: str = DEFAULT_MIND_ID,
        subject_kind: SubjectKind = SubjectKind.USER,
    ) -> "SubjectState":
        return cls(
            subject_id=subject_id,
            version=0,
            entries={},
            applied_contribution_ids=frozenset(),
            conflicts=frozenset(),
            changes=(),
            mind_id=mind_id,
            subject_kind=subject_kind,
        )

    def get(self, field: str) -> StateAtom:
        return self.entries[field]

    @property
    def decided_contribution_ids(self) -> frozenset[UUID]:
        return self.applied_contribution_ids | frozenset(
            change.contribution.contribution_id for change in self.changes
        )


def _require_aware(value: datetime, name: str) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ContractValidationError(
            f"{name} must include timezone information"
        )
