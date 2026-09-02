from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import RLock

from self_cognition.core.protocols import CognitiveModule


class ModuleHealth(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DISABLED = "disabled"


@dataclass(frozen=True, slots=True)
class ModuleRegistration:
    module_id: str
    category: str
    version: str
    module: CognitiveModule
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.module_id.strip():
            raise ValueError("module_id must not be blank")
        if not self.category.strip():
            raise ValueError("module category must not be blank")
        if not self.version.strip():
            raise ValueError("module version must not be blank")


@dataclass(frozen=True, slots=True)
class ModuleStatus:
    module_id: str
    category: str
    version: str
    subscriptions: frozenset[str]
    health: ModuleHealth
    degraded_reason: str | None = None


class CognitiveModuleRegistry:
    def __init__(self, registrations: tuple[ModuleRegistration, ...]) -> None:
        by_id = {
            registration.module_id: registration
            for registration in registrations
        }
        if len(by_id) != len(registrations):
            raise ValueError("cognitive module IDs must be unique")
        self._registrations = by_id
        self._health = {
            module_id: (
                ModuleHealth.HEALTHY
                if registration.enabled
                else ModuleHealth.DISABLED
            )
            for module_id, registration in by_id.items()
        }
        self._degraded_reasons: dict[str, str] = {}
        self._ids_by_object = {
            id(registration.module): module_id
            for module_id, registration in by_id.items()
        }
        self._lock = RLock()

    def active_modules(self) -> tuple[CognitiveModule, ...]:
        with self._lock:
            return tuple(
                registration.module
                for module_id, registration in self._registrations.items()
                if self._health[module_id] is not ModuleHealth.DISABLED
            )

    def all_modules(self) -> tuple[CognitiveModule, ...]:
        with self._lock:
            return tuple(
                registration.module
                for registration in self._registrations.values()
            )

    def enable(self, module_id: str) -> None:
        with self._lock:
            self._require(module_id)
            self._health[module_id] = ModuleHealth.HEALTHY
            self._degraded_reasons.pop(module_id, None)

    def disable(self, module_id: str) -> None:
        with self._lock:
            self._require(module_id)
            self._health[module_id] = ModuleHealth.DISABLED
            self._degraded_reasons.pop(module_id, None)

    def mark_healthy(self, module: CognitiveModule) -> None:
        with self._lock:
            module_id = self._ids_by_object.get(id(module))
            if (
                module_id is None
                or self._health[module_id] is ModuleHealth.DISABLED
            ):
                return
            self._health[module_id] = ModuleHealth.HEALTHY
            self._degraded_reasons.pop(module_id, None)

    def mark_degraded(self, module: CognitiveModule, error: Exception) -> None:
        with self._lock:
            module_id = self._ids_by_object.get(id(module))
            if (
                module_id is None
                or self._health[module_id] is ModuleHealth.DISABLED
            ):
                return
            self._health[module_id] = ModuleHealth.DEGRADED
            self._degraded_reasons[module_id] = type(error).__name__

    def statuses(self) -> tuple[ModuleStatus, ...]:
        with self._lock:
            return tuple(
                ModuleStatus(
                    module_id=module_id,
                    category=registration.category,
                    version=registration.version,
                    subscriptions=registration.module.subscriptions,
                    health=self._health[module_id],
                    degraded_reason=self._degraded_reasons.get(module_id),
                )
                for module_id, registration in self._registrations.items()
            )

    def _require(self, module_id: str) -> ModuleRegistration:
        try:
            return self._registrations[module_id]
        except KeyError as error:
            raise KeyError(f"unknown cognitive module: {module_id}") from error
