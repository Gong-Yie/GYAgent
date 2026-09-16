from threading import RLock
from uuid import UUID

from self_cognition.core.errors import ContractValidationError
from self_cognition.core.runs import RunRecord, RunStatus
from self_cognition.core.scopes import SubjectScope


class InMemoryRunRepository:
    def __init__(self) -> None:
        self._records: dict[UUID, RunRecord] = {}
        self._lock = RLock()

    def get(self, run_id: UUID) -> RunRecord | None:
        with self._lock:
            return self._records.get(run_id)

    def save(self, record: RunRecord) -> RunRecord:
        with self._lock:
            current = self._records.get(record.run_id)
            if current is not None and current != record:
                if record.updated_at < current.updated_at:
                    raise ContractValidationError("run record update is stale")
            self._records[record.run_id] = record
            return record

    def read_by_parent(self, parent_run_id: UUID) -> tuple[RunRecord, ...]:
        with self._lock:
            return tuple(
                record
                for record in self._records.values()
                if record.parent_run_id == parent_run_id
            )

    def read_incomplete(self) -> tuple[RunRecord, ...]:
        with self._lock:
            return tuple(
                record
                for record in self._records.values()
                if not record.status.is_terminal
            )

    def read_by_subject(self, subject: SubjectScope) -> tuple[RunRecord, ...]:
        with self._lock:
            return tuple(record for record in self._records.values() if record.subject == subject)

    def read_recent_failures(self, limit: int = 50) -> tuple[RunRecord, ...]:
        if limit < 1:
            raise ValueError("limit must be positive")
        with self._lock:
            failures = [
                record
                for record in self._records.values()
                if record.status.is_terminal
                and record.status is not RunStatus.COMPLETED
            ]
        failures.sort(
            key=lambda record: (record.updated_at, str(record.run_id)),
            reverse=True,
        )
        return tuple(failures[:limit])

    def forget(self, event_ids: tuple[UUID, ...]) -> None:
        targets = set(event_ids)
        with self._lock:
            for run_id, record in tuple(self._records.items()):
                if targets.intersection(record.input_event_ids):
                    del self._records[run_id]
