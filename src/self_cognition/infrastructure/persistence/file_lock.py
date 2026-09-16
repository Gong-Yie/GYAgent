import os
import time
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
        owner: str | None = None
        descriptor: int | None = None
        try:
            descriptor = os.open(self._path, os.O_RDONLY)
        except OSError:
            descriptor = None
        if descriptor is not None:
            try:
                owner = os.read(descriptor, 4096).decode("ascii")
            except OSError:
                owner = None
            finally:
                os.close(descriptor)
        if owner == self._token:
            self._path.unlink(missing_ok=True)
        self._token = None

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


class BlockingFileLock(FileLock):
    """FileLock with bounded waiting for short cross-process critical sections."""

    def __init__(
        self,
        path: str | Path,
        *,
        timeout_seconds: float = 10.0,
        poll_interval_seconds: float = 0.01,
    ) -> None:
        super().__init__(path)
        if timeout_seconds <= 0:
            raise ValueError("lock timeout must be positive")
        if poll_interval_seconds <= 0:
            raise ValueError("lock poll interval must be positive")
        self._timeout_seconds = timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds

    def acquire(self) -> None:
        if self._token is not None:
            return
        deadline = time.monotonic() + self._timeout_seconds
        while True:
            try:
                super().acquire()
                return
            except FileLockUnavailableError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(self._poll_interval_seconds)
