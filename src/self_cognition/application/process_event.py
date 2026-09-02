import logging
from dataclasses import replace
from datetime import datetime

from self_cognition.application.results import (
    ProcessEventResult,
    ProcessEventStatus,
)
from self_cognition.core.errors import (
    ContractValidationError,
    FileLockUnavailableError,
    ModelTimeoutError,
    RunCancelledError,
)
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.events import EventEnvelope, StateReductionPayload
from self_cognition.core.processing import (
    PROCESS_EVENT_FAILED,
    RUN_CANCELLED,
    ProcessingStatus,
)
from self_cognition.core.protocols import (
    EvidenceRepository,
    EventStore,
    ProcessJournal,
    StateRepository,
)
from self_cognition.core.state import SubjectState
from self_cognition.runtime.engine import CognitionEngine
from self_cognition.runtime.run_context import RunContext


logger = logging.getLogger(__name__)


class ProcessEventService:
    def __init__(
        self,
        event_store: EventStore,
        evidence_repository: EvidenceRepository,
        state_repository: StateRepository,
        engine: CognitionEngine,
        process_journal: ProcessJournal | None = None,
    ) -> None:
        self._event_store = event_store
        self._evidence_repository = evidence_repository
        self._state_repository = state_repository
        self._engine = engine
        self._process_journal = process_journal

    def process(
        self,
        event: EventEnvelope,
        context: RunContext,
    ) -> ProcessEventResult:
        return self._process(event, context, event_is_claimed=False)

    def enqueue(
        self,
        event: EventEnvelope,
        context: RunContext,
    ) -> EventEnvelope:
        recorded_event = self._bind_run(event, context)
        self._append_event(recorded_event)
        if self._process_journal is not None:
            self._process_journal.enqueue(
                recorded_event,
                context.run_id,
                context.clock.now(),
            )
        return recorded_event

    def process_claimed(
        self,
        event: EventEnvelope,
        context: RunContext,
    ) -> ProcessEventResult:
        return self._process(event, context, event_is_claimed=True)

    def finalize_failure(
        self,
        event: EventEnvelope,
        context: RunContext,
        *,
        error_type: str,
    ) -> None:
        failure_event = EventEnvelope.processing_failed(
            event,
            stage="process_event",
            error_type=error_type,
            clock=context.clock,
            run_id=context.run_id,
            correlation_id=context.correlation_id,
        )
        self._append_event(failure_event)
        self._fail_journal(
            event,
            context,
            error_code=PROCESS_EVENT_FAILED,
            error_type=error_type,
            updated_at=failure_event.recorded_at,
        )

    def finalize_cancellation(
        self,
        event: EventEnvelope,
        context: RunContext,
    ) -> None:
        self._fail_journal(
            event,
            context,
            error_code=RUN_CANCELLED,
            error_type=RunCancelledError.__name__,
            dead_letter=False,
        )

    def _process(
        self,
        event: EventEnvelope,
        context: RunContext,
        *,
        event_is_claimed: bool,
    ) -> ProcessEventResult:
        if context.is_cancelled:
            return self._cancelled_result(context, event_saved=False, state=None)

        event_saved = False
        old_state: SubjectState | None = None
        try:
            processing_record = None
            if event_is_claimed:
                recorded_event = self._bind_run(event, context)
                event_saved = True
            else:
                recorded_event = self.enqueue(event, context)
                event_saved = True
            if self._process_journal is not None and not event_is_claimed:
                processing_record = self._process_journal.begin(
                    recorded_event,
                    context.run_id,
                    context.clock.now(),
                )

            old_state = self._state_repository.load(recorded_event.subject)
            if old_state is None:
                subject = recorded_event.subject
                old_state = SubjectState.empty(
                    subject.subject.subject_id,
                    mind_id=subject.mind.mind_id,
                    subject_kind=subject.subject.kind,
                )

            if (
                processing_record is not None
                and processing_record.status is ProcessingStatus.COMPLETED
            ):
                self._process_journal.complete(
                    recorded_event.event_id,
                    context.run_id,
                    context.clock.now(),
                )
                return ProcessEventResult(
                    status=ProcessEventStatus.SUCCEEDED,
                    run_id=context.run_id,
                    correlation_id=context.correlation_id,
                    old_version=old_state.version,
                    new_version=old_state.version,
                    state_changed=False,
                    state=old_state,
                    event_saved=event_saved,
                )
            if (
                processing_record is not None
                and processing_record.status is ProcessingStatus.FAILED
            ):
                return ProcessEventResult(
                    status=ProcessEventStatus.FAILED,
                    run_id=context.run_id,
                    correlation_id=context.correlation_id,
                    old_version=old_state.version,
                    new_version=None,
                    state_changed=None,
                    state=None,
                    event_saved=event_saved,
                    error_type=processing_record.error_type,
                    retryable=False,
                )

            if context.is_cancelled:
                if not event_is_claimed:
                    self.finalize_cancellation(recorded_event, context)
                return self._cancelled_result(context, event_saved, old_state)

            new_state = self._engine.process(recorded_event, old_state, context)
            self._append_emitted_events(context)
            if context.is_cancelled:
                if not event_is_claimed:
                    self.finalize_cancellation(recorded_event, context)
                return self._cancelled_result(context, event_saved, old_state)

            state_changed = new_state.version != old_state.version
            if state_changed:
                self._state_repository.save(
                    new_state,
                    expected_version=old_state.version,
                )

            reduction_event = EventEnvelope.state_reduced(
                recorded_event,
                StateReductionPayload(
                    old_version=old_state.version,
                    new_version=new_state.version,
                    state_changed=state_changed,
                    applied_contribution_ids=tuple(
                        sorted(
                            new_state.applied_contribution_ids
                            - old_state.applied_contribution_ids,
                            key=lambda value: value.int,
                        )
                    ),
                ),
                clock=context.clock,
                run_id=context.run_id,
                correlation_id=context.correlation_id,
            )
            self._append_event(reduction_event)
            if self._process_journal is not None:
                self._process_journal.complete(
                    recorded_event.event_id,
                    run_id=context.run_id,
                    updated_at=reduction_event.recorded_at,
                )

            return ProcessEventResult(
                status=ProcessEventStatus.SUCCEEDED,
                run_id=context.run_id,
                correlation_id=context.correlation_id,
                old_version=old_state.version,
                new_version=new_state.version,
                state_changed=state_changed,
                state=new_state,
                event_saved=event_saved,
            )

        except RunCancelledError:
            self._append_emitted_events(context)
            if event_saved and not event_is_claimed:
                self.finalize_cancellation(recorded_event, context)
            return self._cancelled_result(context, event_saved, old_state)
        except Exception as error:
            self._append_emitted_events(context)
            if event_saved and not event_is_claimed:
                try:
                    self.finalize_failure(
                        recorded_event,
                        context,
                        error_type=type(error).__name__,
                    )
                except Exception:
                    logger.exception("failed to persist processing failure event")
            return ProcessEventResult(
                status=ProcessEventStatus.FAILED,
                run_id=context.run_id,
                correlation_id=context.correlation_id,
                old_version=(old_state.version if old_state is not None else None),
                new_version=None,
                state_changed=None,
                state=None,
                event_saved=event_saved,
                error_type=type(error).__name__,
                retryable=self._is_retryable(error),
            )

    @staticmethod
    def _bind_run(
        event: EventEnvelope,
        context: RunContext,
    ) -> EventEnvelope:
        if event.run_id is not None and event.run_id != context.run_id:
            raise ContractValidationError(
                "event run_id does not match RunContext"
            )
        if (
            event.correlation_id is not None
            and event.correlation_id != context.correlation_id
        ):
            raise ContractValidationError(
                "event correlation_id does not match RunContext"
            )
        return replace(
            event,
            run_id=event.run_id or context.run_id,
            correlation_id=event.correlation_id or context.correlation_id,
        )

    def _append_event(self, event: EventEnvelope) -> None:
        self._event_store.append(event)
        self._evidence_repository.append(EvidenceRef.for_event(event))

    def _append_emitted_events(self, context: RunContext) -> None:
        for event in context.drain_emitted_events():
            self._append_event(event)

    def _fail_journal(
        self,
        event: EventEnvelope,
        context: RunContext,
        *,
        error_code: str,
        error_type: str,
        updated_at: datetime | None = None,
        dead_letter: bool = True,
    ) -> None:
        if self._process_journal is None:
            return
        try:
            self._process_journal.fail(
                event.event_id,
                context.run_id,
                updated_at or context.clock.now(),
                error_code=error_code,
                error_type=error_type,
                dead_letter=dead_letter,
            )
        except Exception:
            logger.exception("failed to persist processing journal failure")

    @staticmethod
    def _cancelled_result(
        context: RunContext,
        event_saved: bool,
        state: SubjectState | None,
    ) -> ProcessEventResult:
        version = state.version if state is not None else None
        return ProcessEventResult(
            status=ProcessEventStatus.CANCELLED,
            run_id=context.run_id,
            correlation_id=context.correlation_id,
            old_version=version,
            new_version=version,
            state_changed=False,
            state=state,
            event_saved=event_saved,
        )

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        return isinstance(
            error,
            (OSError, FileLockUnavailableError, ModelTimeoutError),
        )
