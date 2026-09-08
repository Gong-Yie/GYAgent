from __future__ import annotations

from threading import Event, Lock


class CancellationToken:
    def __init__(self, parent: "CancellationToken | None" = None) -> None:
        self._parent = parent
        self._event = Event()
        self._lock = Lock()
        self._reason: str | None = None

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set() or (
            self._parent is not None and self._parent.is_cancelled
        )

    @property
    def reason(self) -> str | None:
        if self._reason is not None:
            return self._reason
        return self._parent.reason if self._parent is not None else None

    def cancel(self, reason: str = "cancelled") -> None:
        with self._lock:
            if self._reason is None:
                self._reason = reason
            self._event.set()

    def child(self) -> "CancellationToken":
        return CancellationToken(self)
