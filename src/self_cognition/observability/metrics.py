from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from time import perf_counter
from typing import Iterator, Mapping


def _percentile(values: tuple[float, ...], q: float) -> float | None:
    if not 0.0 <= q <= 1.0:
        raise ValueError("percentile must be between 0 and 1")
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


@dataclass(frozen=True, slots=True)
class MetricSnapshot:
    counters: Mapping[str, int]
    gauges: Mapping[str, float]
    timings: Mapping[str, tuple[float, ...]]

    def percentile(self, name: str, q: float) -> float | None:
        return _percentile(self.timings.get(name, ()), q)


class MetricsRegistry:
    """Small process-local metrics registry with deterministic snapshots."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        self._gauges: dict[str, float] = {}
        self._timings: dict[str, list[float]] = {}
        self._lock = RLock()

    def increment(self, name: str, value: int = 1) -> int:
        if not name.strip() or value < 0:
            raise ValueError("metric name must be non-empty and value non-negative")
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value
            return self._counters[name]

    def set_gauge(self, name: str, value: float) -> None:
        if not name.strip():
            raise ValueError("metric name must be non-empty")
        with self._lock:
            self._gauges[name] = float(value)

    def observe(self, name: str, seconds: float) -> None:
        if not name.strip() or seconds < 0:
            raise ValueError("timing metric must be non-negative")
        with self._lock:
            self._timings.setdefault(name, []).append(float(seconds))

    @contextmanager
    def time(self, name: str) -> Iterator[None]:
        started = perf_counter()
        try:
            yield
        finally:
            self.observe(name, perf_counter() - started)

    def percentile(self, name: str, q: float) -> float | None:
        with self._lock:
            values = tuple(self._timings.get(name, ()))
        return _percentile(values, q)

    def snapshot(self) -> MetricSnapshot:
        with self._lock:
            return MetricSnapshot(
                dict(self._counters),
                dict(self._gauges),
                {name: tuple(values) for name, values in self._timings.items()},
            )

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._timings.clear()
