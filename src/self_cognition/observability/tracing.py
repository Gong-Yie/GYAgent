from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from time import perf_counter
from typing import Mapping
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class TraceSpan:
    name: str
    span_id: UUID = field(default_factory=uuid4)
    trace_id: UUID = field(default_factory=uuid4)
    parent_id: UUID | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_seconds: float | None = None
    status: str = "running"
    attributes: Mapping[str, object] = field(default_factory=dict)

    def finish(self, *, status: str = "completed", duration_seconds: float = 0.0) -> "TraceSpan":
        return TraceSpan(
            self.name,
            self.span_id,
            self.trace_id,
            self.parent_id,
            self.started_at,
            duration_seconds,
            status,
            dict(self.attributes),
        )


class TraceRecorder:
    def __init__(self) -> None:
        self._spans: list[TraceSpan] = []
        self._lock = RLock()

    def start(self, name: str, *, parent_id: UUID | None = None, trace_id: UUID | None = None, **attributes: object) -> TraceSpan:
        if not name.strip():
            raise ValueError("span name must be non-empty")
        return TraceSpan(name, parent_id=parent_id, trace_id=trace_id or uuid4(), attributes=attributes)

    def finish(self, span: TraceSpan, *, status: str = "completed", duration_seconds: float = 0.0) -> TraceSpan:
        completed = span.finish(status=status, duration_seconds=duration_seconds)
        with self._lock:
            self._spans.append(completed)
        return completed

    def spans(self, trace_id: UUID | None = None) -> tuple[TraceSpan, ...]:
        with self._lock:
            if trace_id is None:
                return tuple(self._spans)
            return tuple(span for span in self._spans if span.trace_id == trace_id)

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()

    def span(self, name: str, **attributes: object) -> "SpanTimer":
        return SpanTimer(self, self.start(name, **attributes))


class SpanTimer:
    def __init__(self, recorder: TraceRecorder, span: TraceSpan) -> None:
        self._recorder = recorder
        self.span = span
        self._started = perf_counter()

    def __enter__(self) -> TraceSpan:
        return self.span

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._recorder.finish(
            self.span,
            status="failed" if exc_type is not None else "completed",
            duration_seconds=perf_counter() - self._started,
        )
