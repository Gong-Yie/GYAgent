from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
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
        model_router: object | None = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._event_bus = event_bus
        self._lifecycle = lifecycle
        self._modules = module_registry
        self._capabilities = capability_registry
        self._model_router = model_router

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
        return HealthReport(
            ready=all(item.status == "healthy" for item in components),
            components=tuple(components),
        )

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
