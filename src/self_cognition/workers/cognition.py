from __future__ import annotations

from threading import Event, Thread
from typing import Callable


class CognitionWorker:
    """Drain the persisted outbox until stopped."""

    def __init__(self, drain: Callable[[], object], *, poll_interval_seconds: float = 0.1) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll interval must be positive")
        self._drain = drain
        self._interval = poll_interval_seconds
        self._stop = Event()
        self._thread: Thread | None = None
        self.error_type: str | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self.error_type = None
        self._thread = Thread(target=self._run, name="self-cognition-cognition", daemon=False)
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise TimeoutError("cognition worker did not stop")
            self._thread = None

    def close(self) -> None:
        self.stop()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self._drain()
                self._stop.wait(self._interval)
        except Exception as error:
            self.error_type = type(error).__name__
            self._stop.set()
