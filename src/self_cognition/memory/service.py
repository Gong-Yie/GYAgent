from dataclasses import replace
from datetime import datetime
from uuid import UUID

from self_cognition.core.ids import memory_access_id
from self_cognition.core.memories import (
    MemoryAccessRecord,
    MemoryLifecycleStatus,
    MemoryRecord,
)
from self_cognition.core.protocols import MemoryRepository
from self_cognition.core.scopes import SubjectScope
from self_cognition.core.state import StateChangeRecord
from self_cognition.memory.encoder import StateChangeMemoryEncoder


class MemoryEncodingService:
    def __init__(
        self,
        repository: MemoryRepository,
        encoder: StateChangeMemoryEncoder,
    ) -> None:
        self._repository = repository
        self._encoder = encoder

    def validate_correction(
        self,
        subject: SubjectScope,
        target_field: str,
        corrected_memory_id: UUID | None,
    ) -> None:
        if not self._encoder.supports(target_field):
            raise ValueError("correction target field is not memory-backed")
        candidates = self._repository.read_by_subject(subject)
        if corrected_memory_id is not None:
            candidates = tuple(
                record
                for record in candidates
                if record.memory_id == corrected_memory_id
            )
        if not candidates:
            raise LookupError("correction requires an existing memory")
        if not any(
            record.lifecycle_status is MemoryLifecycleStatus.ACTIVE
            and any(
                source.target_field == target_field for source in record.sources
            )
            for record in candidates
        ):
            raise ValueError("correction target does not match an active memory")

    def encode_changes(
        self,
        changes: tuple[StateChangeRecord, ...],
    ) -> tuple[MemoryRecord, ...]:
        saved: list[MemoryRecord] = []
        for change in changes:
            record = self._encoder.encode(change)
            if record is None:
                continue
            self._repository.save(record, expected_version=0)
            saved.append(record)
        return tuple(saved)

    def supersede_for_correction(
        self,
        subject: SubjectScope,
        target_field: str,
        replacement_id: UUID,
        *,
        corrected_memory_id: UUID | None,
        changed_at: datetime,
        correction_event_id: UUID,
    ) -> tuple[MemoryRecord, ...]:
        candidates = self._repository.read_by_subject(subject)
        if corrected_memory_id is not None:
            candidates = tuple(
                record
                for record in candidates
                if record.memory_id == corrected_memory_id
            )
            if not candidates:
                raise LookupError(
                    f"corrected memory does not exist: {corrected_memory_id}"
                )
        superseded = []
        for current in candidates:
            if current.memory_id == replacement_id:
                continue
            if current.lifecycle_status is not MemoryLifecycleStatus.ACTIVE:
                continue
            if not any(
                source.target_field == target_field
                for source in current.sources
            ):
                if corrected_memory_id is not None:
                    raise ValueError(
                        "corrected memory does not match the correction field"
                    )
                continue
            updated = replace(
                current,
                version=current.version + 1,
                lifecycle_status=MemoryLifecycleStatus.SUPERSEDED,
                lifecycle_changed_at=changed_at,
                lifecycle_reason=f"corrected_by:{correction_event_id}",
            )
            self._repository.save(updated, expected_version=current.version)
            superseded.append(updated)
        return tuple(superseded)


class MemoryLifecycleService:
    def __init__(self, repository: MemoryRepository) -> None:
        self._repository = repository

    def archive(
        self,
        subject: SubjectScope,
        memory_id: UUID,
        *,
        changed_at: datetime,
    ) -> MemoryRecord:
        record = self._require_active(subject, memory_id)
        return self._transition(
            record,
            MemoryLifecycleStatus.ARCHIVED,
            "user_archive",
            changed_at,
        )

    def expire_due(
        self,
        subject: SubjectScope,
        *,
        now: datetime,
    ) -> tuple[MemoryRecord, ...]:
        expired = []
        for record in self._repository.read_by_subject(subject):
            if (
                record.lifecycle_status is MemoryLifecycleStatus.ACTIVE
                and record.expires_at is not None
                and record.expires_at <= now
            ):
                expired.append(
                    self._transition(
                        record,
                        MemoryLifecycleStatus.EXPIRED,
                        "retention_expired",
                        now,
                    )
                )
        return tuple(expired)

    def _require_active(
        self,
        subject: SubjectScope,
        memory_id: UUID,
    ) -> MemoryRecord:
        record = self._repository.load(subject, memory_id)
        if record is None:
            raise LookupError(f"memory does not exist: {memory_id}")
        if record.lifecycle_status is not MemoryLifecycleStatus.ACTIVE:
            raise ValueError("only active memory can change lifecycle status")
        return record

    def _transition(
        self,
        record: MemoryRecord,
        status: MemoryLifecycleStatus,
        reason: str,
        changed_at: datetime,
    ) -> MemoryRecord:
        updated = replace(
            record,
            version=record.version + 1,
            lifecycle_status=status,
            lifecycle_changed_at=changed_at,
            lifecycle_reason=reason,
        )
        self._repository.save(updated, expected_version=record.version)
        return updated


class MemoryAccessService:
    def __init__(self, repository: MemoryRepository) -> None:
        self._repository = repository

    def record_access(
        self,
        subject: SubjectScope,
        memory_id: UUID,
        *,
        accessed_at: datetime,
        purpose: str,
        context: str,
    ) -> MemoryAccessRecord:
        record = MemoryAccessRecord(
            access_id=memory_access_id(),
            memory_id=memory_id,
            subject=subject,
            accessed_at=accessed_at,
            purpose=purpose,
            context=context,
        )
        self._repository.record_access(record)
        return record

    def history(
        self,
        subject: SubjectScope,
        memory_id: UUID,
    ) -> tuple[MemoryAccessRecord, ...]:
        return self._repository.read_access_history(subject, memory_id)
