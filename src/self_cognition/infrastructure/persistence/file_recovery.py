import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from uuid import UUID, uuid4

from self_cognition.core.errors import (
    ContractValidationError,
    MalformedSerializedDataError,
)
from self_cognition.infrastructure.persistence.atomic_io import atomic_write_text
from self_cognition.infrastructure.persistence.file_lock import FileLock


class RecoveryStatus(str, Enum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class RecoveryRecord:
    operation: str
    target: str
    status: RecoveryStatus
    recorded_at: datetime
    detail: str
    record_id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        if not self.operation.strip() or not self.target.strip():
            raise ContractValidationError("recovery operation and target are required")
        if not self.detail.strip():
            raise ContractValidationError("recovery detail must not be blank")
        if self.recorded_at.tzinfo is None or self.recorded_at.utcoffset() is None:
            raise ContractValidationError("recorded_at must include timezone information")

    def to_dict(self) -> dict[str, object]:
        return {
            "record_id": str(self.record_id),
            "operation": self.operation,
            "target": self.target,
            "status": self.status.value,
            "recorded_at": self.recorded_at.isoformat(),
            "detail": self.detail,
        }


class FileRecoveryLog:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._records = list(self._read())
        self._record_ids = {record.record_id for record in self._records}

    def append(self, record: RecoveryRecord) -> None:
        if record.record_id in self._record_ids:
            return
        lock_path = self._path.with_suffix(f"{self._path.suffix}.lock")
        with FileLock(lock_path):
            current_records = self._read()
            current_ids = {item.record_id for item in current_records}
            if record.record_id in current_ids:
                self._records = list(current_records)
                self._record_ids = current_ids
                return
            existing = (
                self._path.read_text(encoding="utf-8")
                if self._path.exists()
                else ""
            )
            payload = json.dumps(
                record.to_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            atomic_write_text(self._path, existing + payload + "\n")
            self._records = [*current_records, record]
            self._record_ids = current_ids | {record.record_id}

    def read_all(self) -> tuple[RecoveryRecord, ...]:
        return tuple(self._records)

    def latest_for(self, target: str) -> RecoveryRecord | None:
        for record in reversed(self._records):
            if record.target == target:
                return record
        return None

    def _read(self) -> tuple[RecoveryRecord, ...]:
        if not self._path.exists():
            return ()
        records: list[RecoveryRecord] = []
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except UnicodeError as error:
            raise MalformedSerializedDataError(
                "recovery log is not valid UTF-8"
            ) from error
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                raise MalformedSerializedDataError(
                    f"blank recovery record at line {line_number}"
                )
            try:
                values = json.loads(line)
                records.append(
                    RecoveryRecord(
                        operation=values["operation"],
                        target=values["target"],
                        status=RecoveryStatus(values["status"]),
                        recorded_at=datetime.fromisoformat(values["recorded_at"]),
                        detail=values["detail"],
                        record_id=UUID(values["record_id"]),
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise MalformedSerializedDataError(
                    f"invalid recovery record at line {line_number}"
                ) from error
        return tuple(records)
