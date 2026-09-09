from __future__ import annotations

from dataclasses import dataclass, replace
from threading import RLock
from typing import Callable, TypeVar

from self_cognition.core.errors import ModelTimeoutError, RunBudgetExceededError
from self_cognition.runtime.run_context import RunContext


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ModelRegistration:
    task: str
    provider_id: str
    model: object
    cost_per_call: float = 0.0
    enabled: bool = True
    healthy: bool = True
    degraded_reason: str | None = None


class ModelRouter:
    def __init__(self, registrations: tuple[ModelRegistration, ...] = ()) -> None:
        self._registrations: dict[tuple[str, str], ModelRegistration] = {}
        self._lock = RLock()
        for registration in registrations:
            self.register(registration)

    def register(self, registration: ModelRegistration) -> None:
        if not registration.task.strip() or not registration.provider_id.strip():
            raise ValueError("model task and provider must not be blank")
        if registration.cost_per_call < 0:
            raise ValueError("model cost must be non-negative")
        key = registration.task, registration.provider_id
        with self._lock:
            if key in self._registrations:
                raise ValueError(f"model registration already exists: {key}")
            self._registrations[key] = registration

    def select(self, task: str, context: RunContext | None = None) -> ModelRegistration:
        if context is not None:
            limit = context.budget.max_model_calls
            if limit is not None and context.usage.model_calls >= limit:
                raise RunBudgetExceededError("model call budget exceeded")
        with self._lock:
            candidates = [
                item
                for item in self._registrations.values()
                if item.task == task and item.enabled and item.healthy
            ]
        if not candidates:
            raise LookupError(f"no healthy model is registered for task: {task}")
        return min(candidates, key=lambda item: (item.cost_per_call, item.provider_id))

    def disable(self, provider_id: str, task: str | None = None) -> None:
        self._update(provider_id, task, enabled=False, degraded_reason="disabled")

    def enable(self, provider_id: str, task: str | None = None) -> None:
        self._update(provider_id, task, enabled=True, healthy=True, degraded_reason=None)

    def mark_degraded(self, provider_id: str, task: str, reason: str) -> None:
        self._update(provider_id, task, healthy=False, degraded_reason=reason)

    def mark_healthy(self, provider_id: str, task: str) -> None:
        self._update(provider_id, task, healthy=True, degraded_reason=None)

    def statuses(self) -> tuple[ModelRegistration, ...]:
        with self._lock:
            return tuple(self._registrations[key] for key in sorted(self._registrations))

    def _update(self, provider_id: str, task: str | None, **changes: object) -> None:
        with self._lock:
            keys = [
                key
                for key in self._registrations
                if key[1] == provider_id and (task is None or key[0] == task)
            ]
            if not keys:
                raise LookupError(f"unknown model provider: {provider_id}")
            for key in keys:
                self._registrations[key] = replace(self._registrations[key], **changes)


def call_with_retry(call: Callable[[], T], *, attempts: int = 2) -> T:
    if attempts < 1:
        raise ValueError("attempts must be positive")
    for index in range(attempts):
        try:
            return call()
        except (ModelTimeoutError, OSError):
            if index + 1 == attempts:
                raise
    raise AssertionError("unreachable")


def _call_registration(
    router: ModelRouter,
    registration: ModelRegistration,
    call: Callable[[], T],
    *,
    attempts: int,
) -> T:
    try:
        result = call_with_retry(call, attempts=attempts)
    except Exception as error:
        router.mark_degraded(
            registration.provider_id,
            registration.task,
            type(error).__name__,
        )
        raise
    router.mark_healthy(registration.provider_id, registration.task)
    return result


class RoutedDialogueModel:
    def __init__(self, router: ModelRouter, *, attempts: int = 2) -> None:
        self._router = router
        self._attempts = attempts

    def generate(self, workspace, context):
        registration = self._router.select("dialogue", context)
        return _call_registration(
            self._router,
            registration,
            lambda: registration.model.generate(workspace, context),
            attempts=self._attempts,
        )

    def review(self, workspace, draft, context):
        registration = self._router.select("dialogue", context)
        return _call_registration(
            self._router,
            registration,
            lambda: registration.model.review(workspace, draft, context),
            attempts=self._attempts,
        )


class RoutedPlanningModel:
    def __init__(self, router: ModelRouter, *, attempts: int = 2) -> None:
        self._router = router
        self._attempts = attempts

    def create(self, goal, budget, workspace, capabilities, context):
        registration = self._router.select("planning", context)
        return _call_registration(
            self._router,
            registration,
            lambda: registration.model.create(goal, budget, workspace, capabilities, context),
            attempts=self._attempts,
        )

    def replan(self, goal, plan, progress, workspace, capabilities, context):
        registration = self._router.select("planning", context)
        return _call_registration(
            self._router,
            registration,
            lambda: registration.model.replan(goal, plan, progress, workspace, capabilities, context),
            attempts=self._attempts,
        )


class RoutedActionModel:
    def __init__(self, router: ModelRouter, *, attempts: int = 2) -> None:
        self._router = router
        self._attempts = attempts

    def propose(self, plan, step, workspace, tools, context):
        registration = self._router.select("action", context)
        return _call_registration(
            self._router,
            registration,
            lambda: registration.model.propose(plan, step, workspace, tools, context),
            attempts=self._attempts,
        )

    def decide(self, request, workspace, context):
        registration = self._router.select("action", context)
        return _call_registration(
            self._router,
            registration,
            lambda: registration.model.decide(request, workspace, context),
            attempts=self._attempts,
        )
