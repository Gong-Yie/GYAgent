from __future__ import annotations

from threading import Event, Lock, Thread
from typing import Protocol


class EventBus(Protocol):
    def drain(self) -> tuple[object, ...]: ...


class ApplicationLifecycle:
    def __init__(
        self,
        event_bus: EventBus,
        *,
        worker_enabled: bool = False,
        worker_poll_interval_seconds: float = 0.1,
        resources: tuple[object, ...] = (),
    ) -> None:
        if worker_poll_interval_seconds <= 0:
            raise ValueError("worker poll interval must be positive")
        self._event_bus = event_bus
        self._worker_enabled = worker_enabled
        self._poll_interval = worker_poll_interval_seconds
        self._resources = resources
        self._stop_requested = Event()
        self._ready = Event()
        self._lock = Lock()
        self._worker: Thread | None = None
        self._worker_error_type: str | None = None

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set() and self._worker_error_type is None

    @property
    def worker_error_type(self) -> str | None:
        return self._worker_error_type

    def start(self) -> None:
        with self._lock:
            if self._ready.is_set():
                return
            self._stop_requested.clear()
            self._worker_error_type = None
            started_resources: list[object] = []
            try:
                for resource in self._resources:
                    start = getattr(resource, "start", None)
                    if callable(start):
                        start()
                    started_resources.append(resource)
                self._ready.set()
                if self._worker_enabled:
                    self._worker = Thread(
                        target=self._run_worker,
                        name="self-cognition-worker",
                        daemon=False,
                    )
                    self._worker.start()
            except Exception:
                self._ready.clear()
                for resource in reversed(started_resources):
                    close = getattr(resource, "close", None)
                    if callable(close):
                        close()
                raise

    def wait_ready(self, timeout: float | None = None) -> bool:
        return self._ready.wait(timeout) and self._worker_error_type is None

    def stop(self, timeout: float | None = None) -> None:
        with self._lock:
            self._ready.clear()
            for resource in self._resources:
                stop_accepting = getattr(resource, "stop_accepting", None)
                if callable(stop_accepting):
                    stop_accepting()
            self._stop_requested.set()
            worker = self._worker
        if worker is not None:
            worker.join(timeout)
            if worker.is_alive():
                raise TimeoutError("worker did not stop before the timeout")
        with self._lock:
            self._worker = None
            for resource in reversed(self._resources):
                close = getattr(resource, "close", None)
                if callable(close):
                    close()

    def __enter__(self) -> "ApplicationLifecycle":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()

    def _run_worker(self) -> None:
        try:
            while not self._stop_requested.is_set():
                self._event_bus.drain()
                self._stop_requested.wait(self._poll_interval)
        except Exception as error:
            self._worker_error_type = type(error).__name__
            self._ready.clear()
            self._stop_requested.set()
