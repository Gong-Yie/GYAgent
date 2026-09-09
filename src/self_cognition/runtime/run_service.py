from __future__ import annotations

from dataclasses import replace
from threading import RLock
from uuid import UUID, uuid5

from self_cognition.core.errors import ContractValidationError
from self_cognition.core.runs import (
    RunCheckpoint,
    RunKind,
    RunRecord,
    RunStatus,
)
from self_cognition.core.protocols import RunRepository
from self_cognition.core.scopes import SubjectScope
from self_cognition.runtime.run_context import RunContext


class RunLifecycle:
    def __init__(self, repository: RunRepository) -> None:
        self._repository = repository
        self._contexts: dict[UUID, RunContext] = {}
        self._lock = RLock()

    def begin(
        self,
        context: RunContext,
        kind: RunKind,
        subject: SubjectScope,
        *,
        input_event_ids: tuple[UUID, ...] = (),
        state_version: int | None = None,
        wake_reason: str | None = None,
    ) -> RunRecord:
        with self._lock:
            record = self._repository.get(context.run_id)
            if record is not None:
                self._validate_context(record, context, subject, kind)
                self._contexts[context.run_id] = context
                if record.cancel_requested:
                    context.cancel("persisted cancellation request")
                return record
            record = RunRecord(
                run_id=context.run_id,
                kind=kind,
                subject=subject,
                correlation_id=context.correlation_id,
                started_at=context.clock.now(),
                updated_at=context.clock.now(),
                deadline=context.deadline,
                status=RunStatus.RUNNING,
                parent_run_id=context.parent_run_id,
                budget=context.budget,
                usage=context.usage,
                input_event_ids=input_event_ids,
                state_version=state_version,
                wake_reason=wake_reason,
            )
            self._contexts[context.run_id] = context
            return self._repository.save(record)

    def checkpoint(
        self,
        context: RunContext,
        record: RunRecord,
        position: str,
        *,
        action_id: UUID | None = None,
        result_event_id: UUID | None = None,
        state_version: int | None = None,
    ) -> RunRecord:
        checkpoint = RunCheckpoint(
            checkpoint_id=uuid5(record.run_id, f"checkpoint:{position}"),
            run_id=record.run_id,
            position=position,
            recorded_at=context.clock.now(),
            action_id=action_id,
            result_event_id=result_event_id,
        )
        return self._repository.save(
            replace(
                record,
                updated_at=context.clock.now(),
                status=RunStatus.CHECKPOINTED,
                usage=context.usage,
                checkpoint=checkpoint,
                state_version=(
                    state_version if state_version is not None else record.state_version
                ),
            )
        )

    def finish(
        self,
        context: RunContext,
        record: RunRecord,
        status: RunStatus,
        *,
        reason: str | None = None,
        error_type: str | None = None,
        state_version: int | None = None,
    ) -> RunRecord:
        if not status.is_terminal:
            raise ContractValidationError("run finish requires a terminal status")
        return self._repository.save(
            replace(
                record,
                updated_at=context.clock.now(),
                status=status,
                usage=context.usage,
                latency_seconds=(context.clock.now() - record.started_at).total_seconds(),
                result=status.value,
                state_version=(
                    state_version if state_version is not None else record.state_version
                ),
                termination_reason=reason,
                error_type=error_type,
            )
        )

    def request_cancel(
        self,
        run_id: UUID,
        reason: str = "cancelled",
        *,
        subject: SubjectScope | None = None,
    ) -> RunRecord:
        with self._lock:
            record = self._repository.get(run_id)
            if record is None:
                raise ContractValidationError("cannot cancel an unknown run")
            if subject is not None and record.subject != subject:
                raise ContractValidationError("run does not belong to subject")
            context = self._contexts.get(run_id)
            if context is not None:
                context.cancel(reason)
            updated = replace(
                record,
                updated_at=(context.clock.now() if context is not None else record.updated_at),
                cancel_requested=True,
                termination_reason=reason,
            )
            return self._repository.save(updated)

    @staticmethod
    def _validate_context(
        record: RunRecord,
        context: RunContext,
        subject: SubjectScope,
        kind: RunKind,
    ) -> None:
        if record.subject != subject or record.kind is not kind:
            raise ContractValidationError("run context does not match stored run")
        if record.correlation_id != context.correlation_id:
            raise ContractValidationError("run correlation ID does not match")
