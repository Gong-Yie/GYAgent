import os
from pathlib import Path
from uuid import uuid4

from self_cognition.core.errors import FileLockUnavailableError


class FileLock:
    """A cross-platform, non-blocking lock backed by exclusive file creation."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._token: str | None = None

    def acquire(self) -> None:
        if self._token is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        token = str(uuid4())
        try:
            descriptor = os.open(
                self._path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError as error:
            raise FileLockUnavailableError(
                f"file lock is already held: {self._path}"
            ) from error
        succeeded = False
        try:
            os.write(descriptor, token.encode("ascii"))
            os.fsync(descriptor)
            succeeded = True
        finally:
            os.close(descriptor)
            if not succeeded:
                self._path.unlink(missing_ok=True)
        self._token = token

    def release(self) -> None:
        if self._token is None:
            return
        try:
            owner = self._path.read_text(encoding="ascii")
        except FileNotFoundError:
            owner = None
        if owner == self._token:
            self._path.unlink(missing_ok=True)
        self._token = None

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()
