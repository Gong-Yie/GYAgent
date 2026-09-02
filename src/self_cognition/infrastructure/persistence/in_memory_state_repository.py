from self_cognition.core.errors import VersionConflictError
from self_cognition.core.scopes import SubjectScope, normalize_subject_scope
from self_cognition.core.state import SubjectState


class InMemoryStateRepository:
    def __init__(self) -> None:
        self._states: dict[SubjectScope, SubjectState] = {}

    def load(self, subject: SubjectScope | str) -> SubjectState | None:
        return self._states.get(normalize_subject_scope(subject))

    def save(self, state: SubjectState, expected_version: int) -> None:
        subject_scope = state.subject_scope
        current_state = self._states.get(subject_scope)
        current_version = current_state.version if current_state is not None else 0
        if expected_version != current_version:
            raise VersionConflictError(
                "expected version does not match stored state version"
            )
        if state.version <= current_version:
            raise VersionConflictError(
                "new state version must be greater than stored state version"
            )

        self._states[subject_scope] = state

    def replace(self, state: SubjectState) -> None:
        self._states[state.subject_scope] = state

    def delete(self, subject: SubjectScope) -> None:
        self._states.pop(subject, None)
