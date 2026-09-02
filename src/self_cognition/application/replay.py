from collections import defaultdict
from uuid import UUID

from self_cognition.core.cognition import (
    CognitionFailureType,
    CognitionModuleResult,
    CognitionResultStatus,
)
from self_cognition.core.errors import MissingCognitionResultError
from self_cognition.core.events import (
    CognitionModuleResultPayload,
    EventEnvelope,
)
from self_cognition.core.protocols import EventStore
from self_cognition.core.scopes import SubjectScope
from self_cognition.core.state import SubjectState
from self_cognition.runtime.engine import CognitionEngine


DERIVED_EVENT_TYPES = frozenset(
    {
        "model.response",
        "cognition.module_result",
        "state.reduced",
        "processing.failed",
    }
)


class ReplayService:
    def __init__(
        self,
        event_store: EventStore,
        engine: CognitionEngine,
    ) -> None:
        self._event_store = event_store
        self._engine = engine

    def replay(self, subject: SubjectScope) -> SubjectState:
        if not isinstance(subject, SubjectScope):
            raise TypeError("subject must be a SubjectScope")
        state = SubjectState.empty(
            subject.subject.subject_id,
            mind_id=subject.mind.mind_id,
            subject_kind=subject.subject.kind,
        )
        events = self._event_store.read_by_subject(subject)
        stored_results = _results_by_cause(events)
        for event in events:
            if event.event_type in DERIVED_EVENT_TYPES:
                continue
            results = stored_results.get(event.event_id)
            if results is not None:
                state = self._engine.reduce(
                    state,
                    results,
                    decided_at=event.recorded_at,
                    rebind_target_version=True,
                )
                continue
            if self._engine.requires_persisted_result(event.event_type):
                raise MissingCognitionResultError(
                    "replay cannot invoke a nondeterministic cognition module "
                    f"for event {event.event_id}"
                )
            state = self._engine.process(event, state)
        return state


def _results_by_cause(
    events: tuple[EventEnvelope, ...],
) -> dict[UUID, tuple[CognitionModuleResult, ...]]:
    grouped: defaultdict[UUID, list[CognitionModuleResult]] = defaultdict(list)
    for event in events:
        payload = event.payload
        if (
            event.event_type != "cognition.module_result"
            or event.causation_id is None
            or not isinstance(payload, CognitionModuleResultPayload)
        ):
            continue
        grouped[event.causation_id].append(
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
    return {cause_id: tuple(results) for cause_id, results in grouped.items()}
