from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ComponentHealth:
    name: str
    status: str
    detail: str | None = None


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
                {"name": item.name, "status": item.status, "detail": item.detail}
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
        return ComponentHealth(
            "modules",
            "degraded" if degraded else "healthy",
            f"degraded={len(degraded)};disabled={len(disabled)}",
        )

    def _tools(self) -> ComponentHealth:
        return ComponentHealth(
            "tools",
            "healthy",
            f"registered={len(self._capabilities.registrations())}",
        )

    def _models_health(self) -> ComponentHealth:
        unavailable = [
            item
            for item in self._model_router.statuses()
            if not item.enabled or not item.healthy
        ]
        return ComponentHealth(
            "models",
            "degraded" if unavailable else "healthy",
            f"unavailable={len(unavailable)}",
        )
