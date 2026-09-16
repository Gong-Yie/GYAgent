from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ComponentHealth:
    name: str
    status: str
    detail: str | None = None
    details: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class HealthReport:
    ready: bool
    components: tuple[ComponentHealth, ...]

    @property
    def degraded(self) -> bool:
        return any(item.status != "healthy" for item in self.components)

    def as_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "degraded": self.degraded,
            "components": [
                {
                    "name": item.name,
                    "status": item.status,
                    "detail": item.detail,
                    "details": [dict(detail) for detail in item.details],
                }
                for item in self.components
            ],
        }

    @classmethod
    def from_dict(cls, data: object) -> "HealthReport":
        if not isinstance(data, dict):
            raise ValueError("health report must be an object")
        ready = data.get("ready")
        if type(ready) is not bool:
            raise ValueError("health report ready must be a boolean")
        raw_components = data.get("components")
        if not isinstance(raw_components, list):
            raise ValueError("health report components must be an array")
        components: list[ComponentHealth] = []
        for raw in raw_components:
            if not isinstance(raw, dict):
                raise ValueError("health component must be an object")
            name = raw.get("name")
            status = raw.get("status")
            detail = raw.get("detail")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("health component name must be non-blank")
            if not isinstance(status, str) or not status.strip():
                raise ValueError("health component status must be non-blank")
            if detail is not None and not isinstance(detail, str):
                raise ValueError("health component detail must be string or null")
            raw_details = raw.get("details", ())
            if not isinstance(raw_details, list):
                raise ValueError("health component details must be an array")
            details: list[dict[str, object]] = []
            for value in raw_details:
                if not isinstance(value, dict):
                    raise ValueError("health component detail must be an object")
                details.append(dict(value))
            components.append(
                ComponentHealth(name, status, detail, tuple(details))
            )
        return cls(ready, tuple(components))


@dataclass(frozen=True, slots=True)
class HealthHistoryEntry:
    checked_at: datetime
    report: HealthReport

    def as_dict(self) -> dict[str, object]:
        outbox = _component(self.report.as_dict(), "outbox")
        if outbox is not None:
            outbox = dict(outbox)
            for part in str(outbox.get("detail") or "").split(";"):
                if "=" not in part:
                    continue
                key, value = part.split("=", 1)
                try:
                    outbox[key] = int(value)
                except ValueError:
                    outbox[key] = value
        models = _component(self.report.as_dict(), "models")
        modules = _component(self.report.as_dict(), "modules")
        return {
            "checked_at": self.checked_at.isoformat(),
            "ready": self.report.ready,
            "degraded": self.report.degraded,
            "outbox": outbox,
            "models": models,
            "modules": modules,
        }

    def to_storage_dict(self) -> dict[str, object]:
        return {
            "checked_at": self.checked_at.isoformat(),
            "report": self.report.as_dict(),
        }

    @classmethod
    def from_storage_dict(cls, data: object) -> "HealthHistoryEntry":
        if not isinstance(data, dict):
            raise ValueError("health history entry must be an object")
        raw_checked_at = data.get("checked_at")
        if not isinstance(raw_checked_at, str) or not raw_checked_at.strip():
            raise ValueError("health history checked_at must be non-blank")
        try:
            checked_at = datetime.fromisoformat(raw_checked_at)
        except ValueError as error:
            raise ValueError("health history checked_at is invalid") from error
        if checked_at.tzinfo is None or checked_at.utcoffset() is None:
            checked_at = checked_at.replace(tzinfo=timezone.utc)
        return cls(
            checked_at,
            HealthReport.from_dict(data.get("report")),
        )


class HealthHistory:
    """Bounded in-process history of health snapshots for UI diagnostics."""

    def __init__(
        self,
        *,
        max_entries: int = 500,
        on_record: (
            Callable[[tuple[dict[str, object], ...]], None] | None
        ) = None,
    ) -> None:
        if max_entries < 1:
            raise ValueError("health history max entries must be positive")
        self._entries: deque[HealthHistoryEntry] = deque(maxlen=max_entries)
        self._on_record = on_record

    def record(
        self,
        report: HealthReport,
        *,
        checked_at: datetime | None = None,
    ) -> HealthHistoryEntry:
        if not isinstance(report, HealthReport):
            raise TypeError("report must be a HealthReport")
        entry = HealthHistoryEntry(
            checked_at=checked_at or datetime.now(timezone.utc),
            report=report,
        )
        self._entries.append(entry)
        if self._on_record is not None:
            try:
                self._on_record(self.storage_snapshots())
            except Exception:
                pass
        return entry

    def restore(self, snapshots: Iterable[dict[str, object]]) -> None:
        for snapshot in snapshots:
            try:
                entry = HealthHistoryEntry.from_storage_dict(snapshot)
            except (TypeError, ValueError):
                continue
            self._entries.append(entry)

    def storage_snapshots(self) -> tuple[dict[str, object], ...]:
        return tuple(entry.to_storage_dict() for entry in self._entries)

    def snapshots(self, *, limit: int = 50) -> tuple[HealthHistoryEntry, ...]:
        if limit < 1:
            raise ValueError("limit must be positive")
        entries = tuple(self._entries)
        return entries[-limit:]


class HealthService:
    """Read-only health aggregation for local files, runtime and registries."""

    def __init__(
        self,
        *,
        data_dir: str | Path,
        event_bus: object,
        lifecycle: object,
        module_registry: object,
        capability_registry: object,
        run_repository: object | None = None,
        model_router: object | None = None,
        history: HealthHistory | None = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._event_bus = event_bus
        self._lifecycle = lifecycle
        self._modules = module_registry
        self._capabilities = capability_registry
        self._run_repository = run_repository
        self._model_router = model_router
        self._history = history or HealthHistory()

    def check(self) -> HealthReport:
        components = [
            self._files(),
            self._queue(),
            self._worker(),
            self._modules_health(),
            self._tools(),
        ]
        if self._model_router is not None:
            components.append(self._models_health())
        report = HealthReport(
            ready=all(item.status == "healthy" for item in components),
            components=tuple(components),
        )
        self._history.record(report)
        return report

    def history(self, *, limit: int = 50) -> tuple[dict[str, object], ...]:
        return tuple(
            entry.as_dict() for entry in self._history.snapshots(limit=limit)
        )

    def failures(self, *, limit: int = 50) -> tuple[dict[str, object], ...]:
        if limit < 1:
            raise ValueError("limit must be positive")
        reader = getattr(self._run_repository, "read_recent_failures", None)
        if not callable(reader):
            return ()
        try:
            records = reader(limit=limit)
        except Exception:
            return ()
        return tuple(_failure_to_dict(record) for record in records)

    def _files(self) -> ComponentHealth:
        return ComponentHealth(
            "file_storage",
            "healthy" if self._data_dir.exists() else "failed",
            str(self._data_dir),
        )

    def _queue(self) -> ComponentHealth:
        try:
            backlog = len(self._event_bus.backlog())
            dead_letters = len(self._event_bus.dead_letters())
            status = "degraded" if dead_letters else "healthy"
            return ComponentHealth(
                "outbox",
                status,
                f"backlog={backlog};dead_letters={dead_letters}",
            )
        except Exception as error:
            return ComponentHealth("outbox", "failed", type(error).__name__)

    def _worker(self) -> ComponentHealth:
        enabled = bool(getattr(self._lifecycle, "_worker_enabled", False))
        ready = bool(getattr(self._lifecycle, "is_ready", False))
        return ComponentHealth(
            "worker",
            "healthy" if ready or not enabled else "degraded",
            getattr(self._lifecycle, "worker_error_type", None),
        )

    def _modules_health(self) -> ComponentHealth:
        statuses = tuple(self._modules.statuses())
        degraded = [item for item in statuses if item.health.value == "degraded"]
        disabled = [item for item in statuses if item.health.value == "disabled"]
        details = tuple(
            {
                "module_id": item.module_id,
                "category": item.category,
                "version": item.version,
                "subscriptions": sorted(item.subscriptions),
                "health": item.health.value,
                "degraded_reason": item.degraded_reason,
            }
            for item in statuses
        )
        return ComponentHealth(
            "modules",
            "degraded" if degraded else "healthy",
            (
                f"healthy={len(statuses) - len(degraded) - len(disabled)};"
                f"degraded={len(degraded)};disabled={len(disabled)}"
            ),
            details,
        )

    def _tools(self) -> ComponentHealth:
        return ComponentHealth(
            "tools",
            "healthy",
            f"registered={len(self._capabilities.registrations())}",
        )

    def _models_health(self) -> ComponentHealth:
        statuses = tuple(self._model_router.statuses())
        unavailable = [
            item for item in statuses if not item.enabled or not item.healthy
        ]
        recovering = [
            item
            for item in statuses
            if item.enabled
            and not item.healthy
            and item.degraded_until is not None
        ]
        details = tuple(
            {
                "task": item.task,
                "provider_id": item.provider_id,
                "enabled": item.enabled,
                "healthy": item.healthy,
                "degraded_reason": item.degraded_reason,
                "consecutive_failures": item.consecutive_failures,
                "degraded_until": _isoformat(item.degraded_until),
                "last_failure_at": _isoformat(item.last_failure_at),
                "last_success_at": _isoformat(item.last_success_at),
            }
            for item in statuses
        )
        return ComponentHealth(
            "models",
            "degraded" if unavailable else "healthy",
            (
                f"total={len(statuses)};unavailable={len(unavailable)};"
                f"recovering={len(recovering)}"
            ),
            details,
        )


def _isoformat(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _component(report: dict[str, object], name: str) -> dict[str, object] | None:
    components = report.get("components")
    if not isinstance(components, list):
        return None
    for item in components:
        if isinstance(item, dict) and item.get("name") == name:
            return item
    return None


def _failure_to_dict(record: object) -> dict[str, object]:
    subject = getattr(record, "subject", None)
    subject_payload: dict[str, object] | None = None
    if subject is not None:
        try:
            subject_payload = {
                "mind_id": subject.mind.mind_id,
                "kind": subject.subject.kind.value,
                "subject_id": subject.subject.subject_id,
            }
        except AttributeError:
            subject_payload = None
    return {
        "run_id": str(getattr(record, "run_id", "")),
        "correlation_id": str(getattr(record, "correlation_id", "")),
        "kind": _enum_value(getattr(record, "kind", None)),
        "status": _enum_value(getattr(record, "status", None)),
        "subject": subject_payload,
        "updated_at": _value_isoformat(getattr(record, "updated_at", None)),
        "error_type": getattr(record, "error_type", None),
        "termination_reason": getattr(record, "termination_reason", None),
        "input_event_ids": [
            str(item) for item in getattr(record, "input_event_ids", ())
        ],
    }


def _enum_value(value: object) -> object:
    return getattr(value, "value", value)


def _value_isoformat(value: object) -> object:
    return value.isoformat() if isinstance(value, datetime) else value
