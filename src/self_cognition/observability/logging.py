from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping
from uuid import UUID


_SENSITIVE = ("password", "secret", "token", "api_key", "private_key", "raw_body")


@dataclass(frozen=True, slots=True)
class LogContext:
    run_id: UUID | None = None
    correlation_id: UUID | None = None
    event_id: UUID | None = None
    subject_id: str | None = None
    module_id: str | None = None

    def as_extra(self) -> dict[str, str | None]:
        return {
            "run_id": _text(self.run_id),
            "correlation_id": _text(self.correlation_id),
            "event_id": _text(self.event_id),
            "subject_id": self.subject_id,
            "module_id": self.module_id,
        }


def log_event(
    logger: logging.Logger,
    level: int,
    message: str,
    context: LogContext | None = None,
    **fields: object,
) -> None:
    """Write structured fields without copying sensitive content."""

    safe_fields = {
        key: _redact(value, key) for key, value in fields.items() if not _sensitive(key)
    }
    extra = context.as_extra() if context is not None else {}
    extra.update(safe_fields)
    logger.log(level, message, extra=extra)


def redact_mapping(values: Mapping[str, object]) -> dict[str, object]:
    return {
        str(key): "[REDACTED]" if _sensitive(str(key)) else _redact(value, str(key))
        for key, value in values.items()
    }


def _redact(value: object, key: str) -> object:
    if isinstance(value, Mapping):
        return redact_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_redact(item, key) for item in value]
    return "[REDACTED]" if _sensitive(key) else value


def _sensitive(key: str) -> bool:
    normalized = key.lower()
    return any(token in normalized for token in _SENSITIVE)


def _text(value: object) -> str | None:
    return str(value) if value is not None else None
