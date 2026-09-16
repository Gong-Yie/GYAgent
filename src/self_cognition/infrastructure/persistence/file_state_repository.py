import hashlib
import json
import os
import tempfile
from pathlib import Path

from self_cognition.core.errors import (
    MalformedSerializedDataError,
    VersionConflictError,
)
from self_cognition.core.scopes import (
    DEFAULT_MIND_ID,
    SubjectKind,
    SubjectScope,
    normalize_subject_scope,
)
from self_cognition.core.state import SubjectState
from self_cognition.infrastructure.persistence.file_lock import BlockingFileLock
from self_cognition.infrastructure.persistence.serialization import (
    state_from_json,
    state_to_json,
)


class FileStateRepository:
    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)

    def load(self, subject: SubjectScope | str) -> SubjectState | None:
        subject_scope = normalize_subject_scope(subject)
        with BlockingFileLock(self._lock_path_for(subject_scope)):
            return self._load_unlocked(subject_scope)

    def _load_unlocked(self, subject_scope: SubjectScope) -> SubjectState | None:
        path = self._path_for(subject_scope)
        if not path.exists():
            return None

        try:
            payload = path.read_text(encoding="utf-8")
        except UnicodeError as error:
            raise MalformedSerializedDataError(
                "state snapshot is not valid UTF-8"
            ) from error
        state = state_from_json(payload)
        if state.subject_scope != subject_scope:
            raise MalformedSerializedDataError(
                "state snapshot scope does not match requested scope"
            )
        return state

    def save(self, state: SubjectState, expected_version: int) -> None:
        subject_scope = state.subject_scope
        with BlockingFileLock(self._lock_path_for(subject_scope)):
            current_state = self._load_unlocked(subject_scope)
            current_version = (
                current_state.version if current_state is not None else 0
            )
            if expected_version != current_version:
                raise VersionConflictError(
                    "expected version does not match stored state version"
                )
            if state.version <= current_version:
                raise VersionConflictError(
                    "new state version must be greater than stored state version"
                )
            self._write_unlocked(state)

    def replace(self, state: SubjectState) -> None:
        with BlockingFileLock(self._lock_path_for(state.subject_scope)):
            self._write_unlocked(state)

    def delete(self, subject: SubjectScope) -> None:
        subject_scope = normalize_subject_scope(subject)
        with BlockingFileLock(self._lock_path_for(subject_scope)):
            self._path_for(subject_scope).unlink(missing_ok=True)

    def _write_unlocked(self, state: SubjectState) -> None:
        target = self._path_for(state.subject_scope)
        payload = state_to_json(state)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=self._directory,
                prefix=".state-",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, target)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _lock_path_for(self, subject: SubjectScope | str) -> Path:
        return self._path_for(subject).with_suffix(".lock")

    def _path_for(self, subject: SubjectScope | str) -> Path:
        subject_scope = normalize_subject_scope(subject)
        if (
            subject_scope.mind.mind_id == DEFAULT_MIND_ID
            and subject_scope.subject.kind is SubjectKind.USER
        ):
            key = subject_scope.subject.subject_id
        else:
            key = json.dumps(
                [
                    subject_scope.mind.mind_id,
                    subject_scope.subject.kind.value,
                    subject_scope.subject.subject_id,
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self._directory / f"{digest}.json"
