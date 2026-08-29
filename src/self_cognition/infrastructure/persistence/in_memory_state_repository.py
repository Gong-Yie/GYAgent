from self_cognition.core.errors import VersionConflictError
from self_cognition.core.state import SubjectState


class InMemoryStateRepository:
    def __init__(self) -> None:
        self._states: dict[str, SubjectState] = {}

    def load(self, subject_id: str) -> SubjectState | None:
        return self._states.get(subject_id)

    def save(self, state: SubjectState, expected_version: int) -> None:
        current_state = self._states.get(state.subject_id)
        current_version = current_state.version if current_state is not None else 0
        if expected_version != current_version:
            raise VersionConflictError(
                "expected version does not match stored state version"
            )
        if state.version <= current_version:
            raise VersionConflictError(
                "new state version must be greater than stored state version"
            )

        self._states[state.subject_id] = state
