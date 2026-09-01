from datetime import datetime

from self_cognition.blackboard.reducer import StateReducer
from self_cognition.core.contributions import CognitiveContribution
from self_cognition.core.state import SubjectState


class CognitiveSpaceService:
    def __init__(self, reducer: StateReducer) -> None:
        self._reducer = reducer

    def submit(
        self,
        state: SubjectState,
        contributions: tuple[CognitiveContribution, ...],
        *,
        decided_at: datetime,
    ) -> SubjectState:
        return self._reducer.apply_many(
            state,
            contributions,
            decided_at=decided_at,
        )
