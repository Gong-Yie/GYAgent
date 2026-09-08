from threading import RLock
from uuid import UUID

from self_cognition.core.errors import ContractValidationError
from self_cognition.core.runs import RunRecord


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
