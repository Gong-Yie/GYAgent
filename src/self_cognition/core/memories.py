from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from uuid import UUID

from self_cognition.core.errors import ContractValidationError
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.scopes import DataScope, SubjectScope


class MemoryType(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    RELATIONSHIP = "relationship"
    PROCEDURAL = "procedural"
    NARRATIVE = "narrative"


class MemoryLifecycleStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"
    INVALIDATED = "invalidated"


class MemoryConsolidationStatus(str, Enum):
    RAW = "raw"
    CONSOLIDATED = "consolidated"


@dataclass(frozen=True, slots=True)
class MemoryCues:
    people: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    time_keys: tuple[str, ...] = ()
    relationships: tuple[str, ...] = ()
    tasks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("people", "topics", "time_keys", "relationships", "tasks"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ContractValidationError(f"{name} must contain text values")


@dataclass(frozen=True, slots=True)
class MemorySourceRef:
    contribution_id: UUID
    old_state_version: int
    new_state_version: int
    target_field: str

    def __post_init__(self) -> None:
        if not isinstance(self.contribution_id, UUID):
            raise ContractValidationError("source contribution_id must be a UUID")
        if (
            not isinstance(self.old_state_version, int)
            or isinstance(self.old_state_version, bool)
            or self.old_state_version < 0
        ):
            raise ContractValidationError(
                "source old state version must not be negative"
            )
        if (
            not isinstance(self.new_state_version, int)
            or isinstance(self.new_state_version, bool)
            or self.new_state_version <= self.old_state_version
        ):
            raise ContractValidationError("source state versions are invalid")
        if not isinstance(self.target_field, str) or not self.target_field.strip():
            raise ContractValidationError("source target_field must not be blank")


@dataclass(frozen=True, slots=True)
class MemoryAccessRecord:
    access_id: UUID
    memory_id: UUID
    subject: SubjectScope
    accessed_at: datetime
    purpose: str
    context: str

    def __post_init__(self) -> None:
        if not isinstance(self.access_id, UUID):
            raise ContractValidationError("access_id must be a UUID")
        if not isinstance(self.memory_id, UUID):
            raise ContractValidationError("access memory_id must be a UUID")
        if not isinstance(self.subject, SubjectScope):
            raise ContractValidationError("access subject must be a SubjectScope")
        if (
            not isinstance(self.accessed_at, datetime)
            or self.accessed_at.tzinfo is None
            or self.accessed_at.utcoffset() is None
        ):
            raise ContractValidationError(
                "accessed_at must include timezone information"
            )
        if not isinstance(self.purpose, str) or not self.purpose.strip():
            raise ContractValidationError("access purpose must not be blank")
        if not isinstance(self.context, str) or not self.context.strip():
            raise ContractValidationError("access context must not be blank")


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    memory_id: UUID
    memory_type: MemoryType
    subject: SubjectScope
    scope: DataScope
    content: object
    evidence_refs: tuple[EvidenceRef, ...]
    confidence: float
    salience: float
    stability: float
    retrievability: float
    version: int
    lifecycle_status: MemoryLifecycleStatus
    created_at: datetime
    source_module: str
    source_module_version: str
    sources: tuple[MemorySourceRef, ...] = ()
    cues: MemoryCues = MemoryCues()
    consolidation_status: MemoryConsolidationStatus = MemoryConsolidationStatus.RAW
    expires_at: datetime | None = None
    lifecycle_changed_at: datetime | None = None
    lifecycle_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.memory_id, UUID):
            raise ContractValidationError("memory_id must be a UUID")
        if not isinstance(self.memory_type, MemoryType):
            raise ContractValidationError("memory_type must be a MemoryType")
        if not isinstance(self.subject, SubjectScope):
            raise ContractValidationError("subject must be a SubjectScope")
        if not isinstance(self.scope, DataScope):
            raise ContractValidationError("scope must be a DataScope")
        if self.scope.owner != self.subject:
            raise ContractValidationError(
                "memory scope owner must match the memory subject"
            )
        if self.content is None or (
            isinstance(self.content, str) and not self.content.strip()
        ):
            raise ContractValidationError("memory content must not be empty")
        if not self.evidence_refs:
            raise ContractValidationError("evidence_refs must not be empty")
        if any(
            not isinstance(evidence, EvidenceRef)
            for evidence in self.evidence_refs
        ):
            raise ContractValidationError(
                "evidence_refs must contain EvidenceRef values"
            )
        if any(
            evidence.scope.owner.mind != self.subject.mind
            for evidence in self.evidence_refs
        ):
            raise ContractValidationError(
                "memory evidence must belong to the same mind"
            )
        if any(
            not isinstance(source, MemorySourceRef) for source in self.sources
        ):
            raise ContractValidationError(
                "memory sources must contain MemorySourceRef values"
            )
        if not isinstance(self.cues, MemoryCues):
            raise ContractValidationError("cues must be a MemoryCues")
        if not isinstance(
            self.consolidation_status,
            MemoryConsolidationStatus,
        ):
            raise ContractValidationError(
                "consolidation_status must be a MemoryConsolidationStatus"
            )
        for name, value in (
            ("confidence", self.confidence),
            ("salience", self.salience),
            ("stability", self.stability),
            ("retrievability", self.retrievability),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ContractValidationError(f"{name} must be a number")
            if not 0.0 <= value <= 1.0:
                raise ContractValidationError(
                    f"{name} must be between 0 and 1"
                )
        if (
            not isinstance(self.version, int)
            or isinstance(self.version, bool)
            or self.version < 1
        ):
            raise ContractValidationError("memory version must be positive")
        if not isinstance(self.lifecycle_status, MemoryLifecycleStatus):
            raise ContractValidationError(
                "lifecycle_status must be a MemoryLifecycleStatus"
            )
        if (
            not isinstance(self.created_at, datetime)
            or self.created_at.tzinfo is None
            or self.created_at.utcoffset() is None
        ):
            raise ContractValidationError(
                "created_at must include timezone information"
            )
        for name, value in (
            ("expires_at", self.expires_at),
            ("lifecycle_changed_at", self.lifecycle_changed_at),
        ):
            if value is not None and (
                not isinstance(value, datetime)
                or value.tzinfo is None
                or value.utcoffset() is None
            ):
                raise ContractValidationError(
                    f"{name} must include timezone information"
                )
        if self.expires_at is not None and self.expires_at <= self.created_at:
            raise ContractValidationError("expires_at must be later than created_at")
        if self.lifecycle_status is MemoryLifecycleStatus.ACTIVE:
            if (
                self.lifecycle_changed_at is not None
                or self.lifecycle_reason is not None
            ):
                raise ContractValidationError(
                    "active memory must not have lifecycle transition metadata"
                )
        elif (self.lifecycle_changed_at is None) != (self.lifecycle_reason is None):
            raise ContractValidationError(
                "lifecycle transition time and reason must be set together"
            )
        elif self.lifecycle_reason is not None and not self.lifecycle_reason.strip():
            raise ContractValidationError("lifecycle_reason must not be blank")
        if (
            not isinstance(self.source_module, str)
            or not self.source_module.strip()
        ):
            raise ContractValidationError("source_module must not be blank")
        if (
            not isinstance(self.source_module_version, str)
            or not self.source_module_version.strip()
        ):
            raise ContractValidationError(
                "source_module_version must not be blank"
            )
