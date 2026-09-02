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
from self_cognition.core.cognition import (
    CognitionFailureType,
    CognitionModuleResult,
    CognitionResultStatus,
)
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.events import (
    CognitionModuleResultPayload,
    CognitionCorrectionPayload,
    EventEnvelope,
    StateReductionPayload,
)
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
from self_cognition.runtime.engine import CognitionEngine, raise_terminal_failure
from self_cognition.runtime.run_context import RunContext
from self_cognition.memory.service import MemoryEncodingService


logger = logging.getLogger(__name__)


class ProcessEventService:
    def __init__(
        self,
        event_store: EventStore,
        evidence_repository: EvidenceRepository,
        state_repository: StateRepository,
        engine: CognitionEngine,
        process_journal: ProcessJournal | None = None,
        memory_encoding: MemoryEncodingService | None = None,
    ) -> None:
        self._event_store = event_store
        self._evidence_repository = evidence_repository
        self._state_repository = state_repository
        self._engine = engine
        self._process_journal = process_journal
        self._memory_encoding = memory_encoding

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
                self._encode_memories(old_state, recorded_event)
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

            payload = recorded_event.payload
            if (
                self._memory_encoding is not None
                and isinstance(payload, CognitionCorrectionPayload)
            ):
                self._memory_encoding.validate_correction(
                    recorded_event.subject,
                    payload.target_field,
                    payload.corrected_memory_id,
                )
            stored_results = self._stored_cognition_results(recorded_event)
            reusable_results = self._reusable_results(stored_results)
            cognition_results = self._engine.analyze(
                recorded_event,
                old_state,
                context,
                existing_results=reusable_results,
            )
            new_results = tuple(
                result
                for result in cognition_results
                if result.module_id
                not in {stored.module_id for stored in reusable_results}
            )
            if new_results:
                self._append_cognition_results(
                    recorded_event,
                    context,
                    new_results,
                )
            raise_terminal_failure(cognition_results)
            if context.is_cancelled:
                if not event_is_claimed:
                    self.finalize_cancellation(recorded_event, context)
                return self._cancelled_result(context, event_saved, old_state)

            new_state = self._engine.reduce(
                old_state,
                cognition_results,
                decided_at=recorded_event.recorded_at,
            )
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
            self._encode_memories(new_state, recorded_event)

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
                        error_type=self._error_type(error),
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
                error_type=self._error_type(error),
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

    def _append_events(self, events: tuple[EventEnvelope, ...]) -> None:
        self._event_store.append_many(events)
        for event in events:
            self._evidence_repository.append(EvidenceRef.for_event(event))

    def _append_cognition_results(
        self,
        cause: EventEnvelope,
        context: RunContext,
        results: tuple[CognitionModuleResult, ...],
    ) -> None:
        events: list[EventEnvelope] = []
        for result in results:
            events.extend(result.emitted_events)
            response_event_ids = tuple(
                event.event_id
                for event in result.emitted_events
                if event.event_type == "model.response"
            )
            events.append(
                EventEnvelope.cognition_module_result(
                    cause,
                    CognitionModuleResultPayload(
                        module_id=result.module_id,
                        module_version=result.module_version,
                        deterministic=result.deterministic,
                        status=result.status.value,
                        contributions=result.contributions,
                        response_event_ids=response_event_ids,
                        failure_type=(
                            result.failure_type.value
                            if result.failure_type is not None
                            else None
                        ),
                        error_type=result.error_type,
                    ),
                    clock=context.clock,
                    run_id=context.run_id,
                    correlation_id=context.correlation_id,
                )
            )
        if events:
            self._append_events(tuple(events))

    def _stored_cognition_results(
        self,
        cause: EventEnvelope,
    ) -> tuple[CognitionModuleResult, ...]:
        stored: list[CognitionModuleResult] = []
        for event in self._event_store.read_by_subject(cause.subject):
            if (
                event.event_type != "cognition.module_result"
                or event.causation_id != cause.event_id
                or not isinstance(event.payload, CognitionModuleResultPayload)
            ):
                continue
            payload = event.payload
            stored.append(
                CognitionModuleResult(
                    module_id=payload.module_id,
                    module_version=payload.module_version,
                    deterministic=payload.deterministic,
                    status=CognitionResultStatus(payload.status),
                    contributions=payload.contributions,
                    failure_type=(
                        CognitionFailureType(payload.failure_type)
                        if payload.failure_type is not None
                        else None
                    ),
                    error_type=payload.error_type,
                )
            )
        return tuple(stored)

    @staticmethod
    def _reusable_results(
        results: tuple[CognitionModuleResult, ...],
    ) -> tuple[CognitionModuleResult, ...]:
        reusable: dict[str, CognitionModuleResult] = {}
        for result in results:
            if result.status is CognitionResultStatus.SUCCEEDED:
                reusable[result.module_id] = result
                continue
            if result.status is CognitionResultStatus.CANCELLED:
                continue
            retryable = (
                result.failure_type is CognitionFailureType.TIMEOUT
                or result.error_type
                in {"OSError", "FileLockUnavailableError", "ModelTimeoutError"}
            )
            if not retryable:
                reusable[result.module_id] = result
        return tuple(reusable.values())

    def _append_emitted_events(self, context: RunContext) -> None:
        for event in context.drain_emitted_events():
            self._append_event(event)

    def _encode_memories(
        self,
        state: SubjectState,
        event: EventEnvelope,
    ) -> None:
        if self._memory_encoding is None:
            return
        encoded = self._memory_encoding.encode_changes(state.changes)
        payload = event.payload
        if not isinstance(payload, CognitionCorrectionPayload):
            return
        replacement = next(
            (
                record
                for record in encoded
                if any(
                    source.target_field == payload.target_field
                    for source in record.sources
                )
                and any(
                    evidence.evidence_id == event.event_id
                    for evidence in record.evidence_refs
                )
            ),
            None,
        )
        if replacement is None:
            raise ContractValidationError(
                "correction did not produce a replacement memory"
            )
        self._memory_encoding.supersede_for_correction(
            event.subject,
            payload.target_field,
            replacement.memory_id,
            corrected_memory_id=payload.corrected_memory_id,
            changed_at=event.recorded_at,
            correction_event_id=event.event_id,
        )

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

    @staticmethod
    def _error_type(error: Exception) -> str:
        recorded = getattr(error, "recorded_error_type", None)
        return recorded if isinstance(recorded, str) else type(error).__name__
