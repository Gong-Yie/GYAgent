from self_cognition.core.protocols import EventStore
from self_cognition.core.scopes import SubjectScope
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

    def replay(self, subject: SubjectScope) -> SubjectState:
        if not isinstance(subject, SubjectScope):
            raise TypeError("subject must be a SubjectScope")
        state = SubjectState.empty(
            subject.subject.subject_id,
            mind_id=subject.mind.mind_id,
            subject_kind=subject.subject.kind,
        )
        for event in self._event_store.read_by_subject(subject):
            state = self._engine.process(event, state)

        return state
