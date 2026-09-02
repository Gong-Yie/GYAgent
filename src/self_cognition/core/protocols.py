from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from typing import Protocol, runtime_checkable
from uuid import UUID

from self_cognition.core.contributions import CognitiveContribution
from self_cognition.core.deletions import DeletionPlan, DeletionStatus
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.events import EventEnvelope
from self_cognition.core.memories import MemoryAccessRecord, MemoryRecord
from self_cognition.core.model_outputs import ModelExtractionResult
from self_cognition.core.processing import (
    OutboxEntry,
    ProcessingRecord,
    ProcessingStatus,
)
from self_cognition.core.scopes import SubjectScope
from self_cognition.core.state import SubjectState

if TYPE_CHECKING:
    from self_cognition.runtime.run_context import RunContext


@runtime_checkable
class CognitiveModule(Protocol):
    subscriptions: frozenset[str]

    def process(
        self,
        event: EventEnvelope,
    ) -> tuple[CognitiveContribution, ...]: ...


@runtime_checkable
class ContextualCognitiveModule(Protocol):
    subscriptions: frozenset[str]

    def process(
        self,
        event: EventEnvelope,
        context: RunContext,
    ) -> tuple[CognitiveContribution, ...]: ...


@runtime_checkable
class CognitionModel(Protocol):
    def extract(
        self,
        event: EventEnvelope,
        context: RunContext,
    ) -> ModelExtractionResult: ...


@runtime_checkable
class EventStore(Protocol):
    def append(self, event: EventEnvelope) -> None: ...

    def read_by_subject(
        self,
        subject: SubjectScope,
    ) -> tuple[EventEnvelope, ...]: ...

    def redact(
        self,
        subject: SubjectScope,
        event_ids: tuple[UUID, ...],
        plan_id: UUID,
    ) -> None: ...


@runtime_checkable
class EvidenceRepository(Protocol):
    def append(self, evidence: EvidenceRef) -> None: ...

    def get(
        self,
        subject: SubjectScope,
        evidence_id: UUID,
    ) -> EvidenceRef | None: ...

    def read_by_subject(
        self,
        subject: SubjectScope,
    ) -> tuple[EvidenceRef, ...]: ...

    def delete(
        self,
        subject: SubjectScope,
        evidence_ids: tuple[UUID, ...],
    ) -> None: ...


@runtime_checkable
class StateRepository(Protocol):
    def load(self, subject: SubjectScope | str) -> SubjectState | None: ...

    def save(self, state: SubjectState, expected_version: int) -> None: ...

    def replace(self, state: SubjectState) -> None: ...

    def delete(self, subject: SubjectScope) -> None: ...


@runtime_checkable
class MemoryRepository(Protocol):
    def load(
        self,
        subject: SubjectScope,
        memory_id: UUID,
    ) -> MemoryRecord | None: ...

    def save(self, record: MemoryRecord, expected_version: int) -> None: ...

    def read_history(
        self,
        subject: SubjectScope,
        memory_id: UUID,
    ) -> tuple[MemoryRecord, ...]: ...

    def read_by_subject(
        self,
        subject: SubjectScope,
    ) -> tuple[MemoryRecord, ...]: ...

    def record_access(self, record: MemoryAccessRecord) -> None: ...

    def read_access_history(
        self,
        subject: SubjectScope,
        memory_id: UUID,
    ) -> tuple[MemoryAccessRecord, ...]: ...

    def delete(
        self,
        subject: SubjectScope,
        memory_ids: tuple[UUID, ...],
    ) -> None: ...

    def rebuild_index(self, subject: SubjectScope) -> None: ...


@runtime_checkable
class DeletionRepository(Protocol):
    def save(self, plan: DeletionPlan) -> None: ...

    def get(self, plan_id: UUID) -> DeletionPlan | None: ...

    def read_by_status(
        self,
        status: DeletionStatus,
    ) -> tuple[DeletionPlan, ...]: ...


@runtime_checkable
class ProcessJournal(Protocol):
    def get(self, event_id: UUID) -> ProcessingRecord | None: ...

    def begin(
        self,
        event: EventEnvelope,
        run_id: UUID,
        updated_at: datetime,
    ) -> ProcessingRecord: ...

    def enqueue(
        self,
        event: EventEnvelope,
        run_id: UUID,
        enqueued_at: datetime,
    ) -> ProcessingRecord: ...

    def claim(
        self,
        event_id: UUID,
        run_id: UUID,
        claimed_at: datetime,
        lease_timeout: timedelta,
    ) -> ProcessingRecord | None: ...

    def retry(
        self,
        event_id: UUID,
        run_id: UUID,
        updated_at: datetime,
        *,
        available_at: datetime,
        error_code: str,
        error_type: str,
    ) -> ProcessingRecord: ...

    def complete(
        self,
        event_id: UUID,
        run_id: UUID,
        updated_at: datetime,
    ) -> ProcessingRecord: ...

    def fail(
        self,
        event_id: UUID,
        run_id: UUID,
        updated_at: datetime,
        *,
        error_code: str,
        error_type: str,
        dead_letter: bool = True,
    ) -> ProcessingRecord: ...

    def pending_outbox(self) -> tuple[OutboxEntry, ...]: ...

    def claimable_outbox(self, now: datetime) -> tuple[OutboxEntry, ...]: ...

    def dead_letters(self) -> tuple[ProcessingRecord, ...]: ...

    def forget(self, event_ids: tuple[UUID, ...]) -> None: ...
