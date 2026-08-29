from self_cognition.core.protocols import EventStore
from self_cognition.core.state import SubjectState
from self_cognition.runtime.engine import CognitionEngine


class ReplayService:
    def __init__(
        self,
        event_store: EventStore,
        engine: CognitionEngine,
    ) -> None:
        self._event_store = event_store
        self._engine = engine

    def replay(self, subject_id: str) -> SubjectState:
        state = SubjectState.empty(subject_id)
        for event in self._event_store.read_by_subject(subject_id):
            state = self._engine.process(event, state)

        return state
