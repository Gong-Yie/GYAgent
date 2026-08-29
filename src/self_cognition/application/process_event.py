from self_cognition.application.results import (
    ProcessEventResult,
    ProcessEventStatus,
)
from self_cognition.core.errors import RunCancelledError
from self_cognition.core.events import Event
from self_cognition.core.protocols import EventStore, StateRepository
from self_cognition.core.state import SubjectState
from self_cognition.runtime.engine import CognitionEngine
from self_cognition.runtime.run_context import RunContext


class ProcessEventService:
    def __init__(
        self,
        event_store: EventStore,
        state_repository: StateRepository,
        engine: CognitionEngine,
    ) -> None:
        self._event_store = event_store
        self._state_repository = state_repository
        self._engine = engine

    def process(self, event: Event, context: RunContext) -> ProcessEventResult:
        if context.is_cancelled:
            return self._cancelled_result(context, event_saved=False, state=None)

        event_saved = False
        old_state: SubjectState | None = None
        try:
            self._event_store.append(event)
            event_saved = True

            old_state = self._state_repository.load(event.actor_id)
            if old_state is None:
                old_state = SubjectState.empty(event.actor_id)

            if context.is_cancelled:
                return self._cancelled_result(context, event_saved, old_state)

            new_state = self._engine.process(event, old_state, context)
            if context.is_cancelled:
                return self._cancelled_result(context, event_saved, old_state)

            state_changed = new_state.version != old_state.version
            if state_changed:
                self._state_repository.save(
                    new_state,
                    expected_version=old_state.version,
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
            return self._cancelled_result(context, event_saved, old_state)
        except Exception as error:
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
