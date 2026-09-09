"""Process-local diagnostics used by the runtime and later entry points."""

from self_cognition.observability.logging import LogContext, log_event
from self_cognition.observability.metrics import MetricsRegistry
from self_cognition.observability.tracing import TraceRecorder, TraceSpan

__all__ = ["LogContext", "MetricsRegistry", "TraceRecorder", "TraceSpan", "log_event"]
