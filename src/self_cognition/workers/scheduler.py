from __future__ import annotations

from threading import Event, Thread
from typing import Callable


class SchedulerWorker:
    """Run explicitly supplied maintenance tasks on a bounded interval."""

    def __init__(self, tasks: tuple[Callable[[], object], ...] = (), *, interval_seconds: float = 60.0) -> None:
        if interval_seconds <= 0:
            raise ValueError("scheduler interval must be positive")
        self._tasks = tasks
        self._interval = interval_seconds
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
        self._thread = Thread(target=self._run, name="self-cognition-scheduler", daemon=False)
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise TimeoutError("scheduler worker did not stop")
            self._thread = None

    def close(self) -> None:
        self.stop()

    def run_once(self) -> None:
        for task in self._tasks:
            task()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self.run_once()
                self._stop.wait(self._interval)
        except Exception as error:
            self.error_type = type(error).__name__
            self._stop.set()
