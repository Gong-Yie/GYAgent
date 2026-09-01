import logging
from dataclasses import replace

from self_cognition.application.results import (
    ProcessEventResult,
    ProcessEventStatus,
)
from self_cognition.core.errors import ContractValidationError, RunCancelledError
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.events import EventEnvelope, StateReductionPayload
from self_cognition.core.protocols import (
    EvidenceRepository,
    EventStore,
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
    ) -> None:
        self._event_store = event_store
        self._evidence_repository = evidence_repository
        self._state_repository = state_repository
        self._engine = engine

    def process(
        self,
        event: EventEnvelope,
        context: RunContext,
    ) -> ProcessEventResult:
        if context.is_cancelled:
            return self._cancelled_result(context, event_saved=False, state=None)

        event_saved = False
        old_state: SubjectState | None = None
        try:
            recorded_event = self._bind_run(event, context)
            self._append_event(recorded_event)
            event_saved = True

            old_state = self._state_repository.load(recorded_event.subject)
            if old_state is None:
                subject = recorded_event.subject
                old_state = SubjectState.empty(
                    subject.subject.subject_id,
                    mind_id=subject.mind.mind_id,
                    subject_kind=subject.subject.kind,
                )

            if context.is_cancelled:
                return self._cancelled_result(context, event_saved, old_state)

            new_state = self._engine.process(recorded_event, old_state, context)
            self._append_emitted_events(context)
            if context.is_cancelled:
                return self._cancelled_result(context, event_saved, old_state)

            state_changed = new_state.version != old_state.version
            if state_changed:
                self._state_repository.save(
                    new_state,
                    expected_version=old_state.version,
                )

            self._append_event(
                EventEnvelope.state_reduced(
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
            return self._cancelled_result(context, event_saved, old_state)
        except Exception as error:
            self._append_emitted_events(context)
            if event_saved:
                try:
                    self._append_event(
                        EventEnvelope.processing_failed(
                            recorded_event,
                            stage="process_event",
                            error_type=type(error).__name__,
                            clock=context.clock,
                            run_id=context.run_id,
                            correlation_id=context.correlation_id,
                        )
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
