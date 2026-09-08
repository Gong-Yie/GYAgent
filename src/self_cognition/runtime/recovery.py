from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from uuid import UUID

from self_cognition.core.actions import ActionResultPayload
from self_cognition.core.events import EventEnvelope
from self_cognition.core.runs import RunRecord
from self_cognition.core.protocols import EventStore, RunRepository
from self_cognition.core.runs import RunKind, RunStatus


class RunRecoveryService:
    def __init__(self, repository: RunRepository, event_store: EventStore) -> None:
        self._repository = repository
        self._events = event_store

    def recover(self, now: datetime) -> tuple[RunRecord, ...]:
        recovered: list[RunRecord] = []
        for record in self._repository.read_incomplete():
            if record.cancel_requested:
                recovered.append(
                    self._save(record, RunStatus.CANCELLED, "persisted cancellation request")
                )
                continue
            if now >= record.deadline:
                recovered.append(
                    self._save(record, RunStatus.TIMED_OUT, "deadline expired")
                )
                continue
            checkpoint = record.checkpoint
            if record.kind is RunKind.ACTION:
                result = self._action_result(record.subject, checkpoint.action_id if checkpoint else None)
                if result is not None:
                    recovered.append(
                        self._save(
                            record,
                            RunStatus.COMPLETED,
                            "recovered recorded action result",
                            checkpoint=replace(
                                checkpoint,
                                position="action.result_recorded",
                                result_event_id=result.event_id,
                            ),
                        )
                    )
                else:
                    recovered.append(
                        self._save(
                            record,
                            RunStatus.INTERRUPTED,
                            "action result missing after possible side effect",
                        )
                    )
                continue
            if checkpoint is not None and checkpoint.position == "state_reduced":
                recovered.append(
                    self._save(
                        record,
                        RunStatus.COMPLETED,
                        "recovered completed state reduction",
                    )
                )
                continue
            recovered.append(
                self._save(
                    record,
                    RunStatus.INTERRUPTED,
                    "cognitive cycle interrupted before terminal checkpoint",
                )
            )
        return tuple(recovered)

    def _action_result(
        self, subject, action_id: UUID | None
    ) -> EventEnvelope | None:
        if action_id is None:
            return None
        return next(
            (
                event
                for event in self._events.read_by_subject(subject)
                if isinstance(event.payload, ActionResultPayload)
                and event.payload.result.action_id == action_id
            ),
            None,
        )

    def _save(
        self,
        record: RunRecord,
        status: RunStatus,
        reason: str,
        *,
        checkpoint=None,
    ) -> RunRecord:
        return self._repository.save(
            replace(
                record,
                status=status,
                updated_at=max(record.updated_at, record.started_at),
                termination_reason=reason,
                checkpoint=checkpoint if checkpoint is not None else record.checkpoint,
            )
        )
