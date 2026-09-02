import json
from datetime import datetime, timedelta
from pathlib import Path
from uuid import UUID

from self_cognition.core.errors import MalformedSerializedDataError
from self_cognition.core.events import EventEnvelope
from self_cognition.core.processing import (
    OutboxEntry,
    PROCESS_EVENT_FAILED,
    ProcessingRecord,
    ProcessingStatus,
)
from self_cognition.core.scopes import (
    MindScope,
    SubjectKind,
    SubjectRef,
    SubjectScope,
)
from self_cognition.infrastructure.persistence.atomic_io import atomic_write_text
from self_cognition.infrastructure.persistence.file_lock import FileLock


class FileProcessJournal:
    def __init__(self, directory: str | Path) -> None:
        root = Path(directory)
        self._records = root / "records"
        self._history = root / "history"
        self._pending = root / "outbox" / "pending"
        self._acknowledged = root / "outbox" / "acknowledged"
        self._dead_letters = root / "dead_letters"
        self._locks = root / "locks"
        for path in (
            self._records,
            self._history,
            self._pending,
            self._acknowledged,
            self._dead_letters,
            self._locks,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def get(self, event_id: UUID) -> ProcessingRecord | None:
        path = self._record_path(event_id)
        if not path.exists():
            return None
        return _record_from_json(path.read_text(encoding="utf-8"))

    def begin(
        self,
        event: EventEnvelope,
        run_id: UUID,
        updated_at: datetime,
    ) -> ProcessingRecord:
        self.enqueue(event, run_id, updated_at)
        claimed = self.claim(
            event.event_id,
            run_id,
            updated_at,
            timedelta(seconds=30),
        )
        return claimed or self._require_record(event.event_id)

    def enqueue(
        self,
        event: EventEnvelope,
        run_id: UUID,
        enqueued_at: datetime,
    ) -> ProcessingRecord:
        with FileLock(self._lock_path(event.event_id)):
            current = self.get(event.event_id)
            if current is not None and current.status in (
                ProcessingStatus.COMPLETED,
                ProcessingStatus.FAILED,
            ):
                return current
            if current is None:
                current = ProcessingRecord(
                    event_id=event.event_id,
                    subject=event.subject,
                    run_id=run_id,
                    status=ProcessingStatus.PENDING,
                    updated_at=enqueued_at,
                    available_at=enqueued_at,
                )
                self._write_record_unlocked(current)
            self._enqueue_unlocked(event, run_id, enqueued_at)
            return current

    def claim(
        self,
        event_id: UUID,
        run_id: UUID,
        claimed_at: datetime,
        lease_timeout: timedelta,
    ) -> ProcessingRecord | None:
        if lease_timeout <= timedelta(0):
            raise ValueError("lease_timeout must be positive")
        with FileLock(self._lock_path(event_id)):
            current = self._require_record(event_id)
            if current.status in (
                ProcessingStatus.COMPLETED,
                ProcessingStatus.FAILED,
            ):
                return None
            if (
                current.status is ProcessingStatus.PENDING
                and current.available_at is not None
                and current.available_at > claimed_at
            ):
                return None
            if (
                current.status is ProcessingStatus.PROCESSING
                and current.lease_expires_at is not None
                and current.lease_expires_at > claimed_at
            ):
                return None
            processing = ProcessingRecord(
                event_id=event_id,
                subject=current.subject,
                run_id=run_id,
                status=ProcessingStatus.PROCESSING,
                updated_at=claimed_at,
                attempt_count=current.attempt_count + 1,
                lease_expires_at=claimed_at + lease_timeout,
            )
            self._write_record_unlocked(processing)
            return processing

    def retry(
        self,
        event_id: UUID,
        run_id: UUID,
        updated_at: datetime,
        *,
        available_at: datetime,
        error_code: str,
        error_type: str,
    ) -> ProcessingRecord:
        with FileLock(self._lock_path(event_id)):
            current = self._require_record(event_id)
            if current.status is not ProcessingStatus.PROCESSING:
                return current
            pending = ProcessingRecord(
                event_id=event_id,
                subject=current.subject,
                run_id=run_id,
                status=ProcessingStatus.PENDING,
                updated_at=updated_at,
                error_code=error_code,
                error_type=error_type,
                attempt_count=current.attempt_count,
                available_at=available_at,
            )
            self._write_record_unlocked(pending)
            return pending

    def complete(
        self,
        event_id: UUID,
        run_id: UUID,
        updated_at: datetime,
    ) -> ProcessingRecord:
        with FileLock(self._lock_path(event_id)):
            current = self._require_record(event_id)
            if current.status is ProcessingStatus.COMPLETED:
                self._acknowledge_unlocked(event_id, updated_at)
                return current
            if current.status is ProcessingStatus.FAILED:
                return current
            completed = ProcessingRecord(
                event_id=event_id,
                subject=current.subject,
                run_id=run_id,
                status=ProcessingStatus.COMPLETED,
                updated_at=updated_at,
                attempt_count=current.attempt_count,
            )
            self._write_record_unlocked(completed)
            self._acknowledge_unlocked(event_id, updated_at)
            return completed

    def fail(
        self,
        event_id: UUID,
        run_id: UUID,
        updated_at: datetime,
        *,
        error_code: str,
        error_type: str,
        dead_letter: bool = True,
    ) -> ProcessingRecord:
        with FileLock(self._lock_path(event_id)):
            current = self._require_record(event_id)
            if current.status is ProcessingStatus.COMPLETED:
                return current
            failed = ProcessingRecord(
                event_id=event_id,
                subject=current.subject,
                run_id=run_id,
                status=ProcessingStatus.FAILED,
                updated_at=updated_at,
                error_code=error_code,
                error_type=error_type,
                attempt_count=current.attempt_count,
            )
            self._write_record_unlocked(failed)
            if dead_letter:
                atomic_write_text(
                    self._dead_letter_path(event_id),
                    _record_to_json(failed),
                )
            self._acknowledge_unlocked(event_id, updated_at)
            return failed

    def pending_outbox(self) -> tuple[OutboxEntry, ...]:
        entries = []
        for path in sorted(self._pending.glob("*.json")):
            event_id = UUID(path.stem)
            if self._ack_path(event_id).exists():
                continue
            entries.append(_outbox_from_json(path.read_text(encoding="utf-8")))
        return tuple(entries)

    def claimable_outbox(self, now: datetime) -> tuple[OutboxEntry, ...]:
        entries = []
        for entry in self.pending_outbox():
            record = self.get(entry.event_id)
            if record is None:
                continue
            if record.status is ProcessingStatus.PENDING and (
                record.available_at is None or record.available_at <= now
            ):
                entries.append(entry)
            elif record.status is ProcessingStatus.PROCESSING and (
                record.lease_expires_at is None
                or record.lease_expires_at <= now
            ):
                entries.append(entry)
        return tuple(entries)

    def dead_letters(self) -> tuple[ProcessingRecord, ...]:
        return tuple(
            _record_from_json(path.read_text(encoding="utf-8"))
            for path in sorted(self._dead_letters.glob("*.json"))
        )

    def recover(
        self,
        event: EventEnvelope,
        status: ProcessingStatus,
        updated_at: datetime,
        *,
        error_type: str | None = None,
    ) -> ProcessingRecord:
        if event.run_id is None:
            raise MalformedSerializedDataError(
                "recoverable event must contain a run_id"
            )
        with FileLock(self._lock_path(event.event_id)):
            current = self.get(event.event_id)
            if status is ProcessingStatus.COMPLETED:
                record = ProcessingRecord(
                    event_id=event.event_id,
                    subject=event.subject,
                    run_id=event.run_id,
                    status=status,
                    updated_at=updated_at,
                    attempt_count=(current.attempt_count if current else 0),
                )
                if current != record:
                    self._write_record_unlocked(record)
                self._acknowledge_unlocked(event.event_id, updated_at)
                return record
            if status is ProcessingStatus.FAILED:
                record = ProcessingRecord(
                    event_id=event.event_id,
                    subject=event.subject,
                    run_id=event.run_id,
                    status=status,
                    updated_at=updated_at,
                    error_code=PROCESS_EVENT_FAILED,
                    error_type=error_type or "UnknownError",
                    attempt_count=(current.attempt_count if current else 0),
                )
                if current != record:
                    self._write_record_unlocked(record)
                atomic_write_text(
                    self._dead_letter_path(event.event_id),
                    _record_to_json(record),
                )
                self._acknowledge_unlocked(event.event_id, updated_at)
                return record
            if status is not ProcessingStatus.PENDING:
                raise ValueError(
                    "recovery status must be pending, completed, or failed"
                )
            if current is not None:
                self._enqueue_unlocked(event, event.run_id, updated_at)
                return current
            record = ProcessingRecord(
                event_id=event.event_id,
                subject=event.subject,
                run_id=event.run_id,
                status=ProcessingStatus.PENDING,
                updated_at=updated_at,
                available_at=updated_at,
            )
            if current != record:
                self._write_record_unlocked(record)
            self._enqueue_unlocked(event, event.run_id, updated_at)
            return record

    def read_history(self, event_id: UUID) -> tuple[ProcessingRecord, ...]:
        path = self._history_path(event_id)
        if not path.exists():
            return ()
        records = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            try:
                records.append(_record_from_json(line))
            except MalformedSerializedDataError as error:
                raise MalformedSerializedDataError(
                    f"invalid processing history at line {line_number}"
                ) from error
        return tuple(records)

    def _write_record_unlocked(self, record: ProcessingRecord) -> None:
        atomic_write_text(
            self._record_path(record.event_id),
            _record_to_json(record),
        )
        history_path = self._history_path(record.event_id)
        history = (
            history_path.read_text(encoding="utf-8")
            if history_path.exists()
            else ""
        )
        atomic_write_text(history_path, history + _record_to_json(record) + "\n")

    def _enqueue_unlocked(
        self,
        event: EventEnvelope,
        run_id: UUID,
        enqueued_at: datetime,
    ) -> None:
        if self._ack_path(event.event_id).exists():
            return
        path = self._pending_path(event.event_id)
        if path.exists():
            return
        entry = OutboxEntry(event.event_id, event.subject, run_id, enqueued_at)
        atomic_write_text(path, _outbox_to_json(entry))

    def _acknowledge_unlocked(
        self,
        event_id: UUID,
        acknowledged_at: datetime,
    ) -> None:
        if self._ack_path(event_id).exists():
            self._pending_path(event_id).unlink(missing_ok=True)
            return
        receipt = {
            "schema_version": 1,
            "event_id": str(event_id),
            "acknowledged_at": acknowledged_at.isoformat(),
        }
        atomic_write_text(
            self._ack_path(event_id),
            json.dumps(receipt, ensure_ascii=False, separators=(",", ":")),
        )
        self._pending_path(event_id).unlink(missing_ok=True)

    def _require_record(self, event_id: UUID) -> ProcessingRecord:
        record = self.get(event_id)
        if record is None:
            raise MalformedSerializedDataError(
                f"processing record does not exist: {event_id}"
            )
        return record

    def _record_path(self, event_id: UUID) -> Path:
        return self._records / f"{event_id}.json"

    def _history_path(self, event_id: UUID) -> Path:
        return self._history / f"{event_id}.jsonl"

    def _pending_path(self, event_id: UUID) -> Path:
        return self._pending / f"{event_id}.json"

    def _ack_path(self, event_id: UUID) -> Path:
        return self._acknowledged / f"{event_id}.json"

    def _dead_letter_path(self, event_id: UUID) -> Path:
        return self._dead_letters / f"{event_id}.json"

    def _lock_path(self, event_id: UUID) -> Path:
        return self._locks / f"{event_id}.lock"


def _record_to_json(record: ProcessingRecord) -> str:
    payload = {
        "schema_version": 2,
        "event_id": str(record.event_id),
        "mind_id": record.subject.mind.mind_id,
        "subject_kind": record.subject.subject.kind.value,
        "subject_id": record.subject.subject.subject_id,
        "run_id": str(record.run_id),
        "status": record.status.value,
        "updated_at": record.updated_at.isoformat(),
        "error_code": record.error_code,
        "error_type": record.error_type,
        "attempt_count": record.attempt_count,
        "available_at": (
            record.available_at.isoformat()
            if record.available_at is not None
            else None
        ),
        "lease_expires_at": (
            record.lease_expires_at.isoformat()
            if record.lease_expires_at is not None
            else None
        ),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _record_from_json(payload: str) -> ProcessingRecord:
    try:
        values = json.loads(payload)
        schema_version = values.get("schema_version")
        if schema_version not in (1, 2):
            raise ValueError("unsupported processing schema")
        subject = SubjectScope(
            MindScope(values["mind_id"]),
            SubjectRef(SubjectKind(values["subject_kind"]), values["subject_id"]),
        )
        return ProcessingRecord(
            event_id=UUID(values["event_id"]),
            subject=subject,
            run_id=UUID(values["run_id"]),
            status=ProcessingStatus(values["status"]),
            updated_at=datetime.fromisoformat(values["updated_at"]),
            error_code=values["error_code"],
            error_type=values["error_type"],
            attempt_count=values.get("attempt_count", 0),
            available_at=(
                datetime.fromisoformat(values["available_at"])
                if values.get("available_at") is not None
                else None
            ),
            lease_expires_at=(
                datetime.fromisoformat(values["lease_expires_at"])
                if values.get("lease_expires_at") is not None
                else None
            ),
        )
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise MalformedSerializedDataError("invalid processing record") from error


def _outbox_to_json(entry: OutboxEntry) -> str:
    payload = {
        "schema_version": 1,
        "event_id": str(entry.event_id),
        "mind_id": entry.subject.mind.mind_id,
        "subject_kind": entry.subject.subject.kind.value,
        "subject_id": entry.subject.subject.subject_id,
        "run_id": str(entry.run_id),
        "enqueued_at": entry.enqueued_at.isoformat(),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _outbox_from_json(payload: str) -> OutboxEntry:
    try:
        values = json.loads(payload)
        if values.get("schema_version") != 1:
            raise ValueError("unsupported outbox schema")
        return OutboxEntry(
            event_id=UUID(values["event_id"]),
            subject=SubjectScope(
                MindScope(values["mind_id"]),
                SubjectRef(
                    SubjectKind(values["subject_kind"]),
                    values["subject_id"],
                ),
            ),
            run_id=UUID(values["run_id"]),
            enqueued_at=datetime.fromisoformat(values["enqueued_at"]),
        )
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise MalformedSerializedDataError("invalid outbox entry") from error
