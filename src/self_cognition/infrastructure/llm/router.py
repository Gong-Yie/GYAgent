from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from threading import RLock
from typing import Callable, TypeVar

from self_cognition.core.errors import (
    ModelTimeoutError,
    RunBudgetExceededError,
    RunCancelledError,
)
from self_cognition.core.proactivity import MotiveProposal
from self_cognition.core.time import SYSTEM_CLOCK, Clock
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
    consecutive_failures: int = 0
    degraded_until: datetime | None = None
    last_failure_at: datetime | None = None
    last_success_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.task.strip() or not self.provider_id.strip():
            raise ValueError("model task and provider must not be blank")
        if self.cost_per_call < 0:
            raise ValueError("model cost must be non-negative")
        if self.consecutive_failures < 0:
            raise ValueError("model consecutive failures must be non-negative")


class ModelRouter:
    def __init__(
        self,
        registrations: tuple[ModelRegistration, ...] = (),
        *,
        disabled_provider_loader: Callable[[object], frozenset[str]] | None = None,
        failure_cooldown: timedelta = timedelta(seconds=30),
        max_failure_cooldown: timedelta = timedelta(minutes=15),
        clock: Clock = SYSTEM_CLOCK,
    ) -> None:
        if failure_cooldown <= timedelta(0):
            raise ValueError("failure cooldown must be positive")
        if max_failure_cooldown < failure_cooldown:
            raise ValueError("maximum failure cooldown must not be shorter than base")
        self._registrations: dict[tuple[str, str], ModelRegistration] = {}
        self._lock = RLock()
        self._disabled_provider_loader = disabled_provider_loader
        self._failure_cooldown = failure_cooldown
        self._max_failure_cooldown = max_failure_cooldown
        self._clock = clock
        for registration in registrations:
            self.register(registration)

    def register(self, registration: ModelRegistration) -> None:
        if not isinstance(registration, ModelRegistration):
            raise TypeError("registration must be a ModelRegistration")
        key = registration.task, registration.provider_id
        with self._lock:
            if key in self._registrations:
                raise ValueError(f"model registration already exists: {key}")
            self._registrations[key] = registration

    def select(
        self,
        task: str,
        context: RunContext | None = None,
        *,
        subject: object | None = None,
    ) -> ModelRegistration:
        if context is not None:
            limit = context.budget.max_model_calls
            if limit is not None and context.usage.model_calls >= limit:
                raise RunBudgetExceededError("model call budget exceeded")
        now = self._clock.now()
        with self._lock:
            disabled = (
                self._disabled_provider_loader(subject)
                if subject is not None and self._disabled_provider_loader is not None
                else frozenset()
            )
            enabled = [
                item
                for item in self._registrations.values()
                if item.task == task
                and item.enabled
                and item.provider_id not in disabled
            ]
            candidates = [item for item in enabled if item.healthy]
            if not candidates:
                # A bounded half-open probe is allowed after cooldown.  A single
                # transient provider failure must not permanently lock out a task.
                candidates = [
                    item
                    for item in enabled
                    if not item.healthy
                    and item.degraded_until is not None
                    and item.degraded_until <= now
                ]
        if not candidates:
            raise LookupError(f"no healthy model is registered for task: {task}")
        return min(candidates, key=lambda item: (item.cost_per_call, item.provider_id))

    def disable(self, provider_id: str, task: str | None = None) -> None:
        self._update(provider_id, task, enabled=False, degraded_reason="disabled")

    def enable(self, provider_id: str, task: str | None = None) -> None:
        self._update(
            provider_id,
            task,
            enabled=True,
            healthy=True,
            degraded_reason=None,
            consecutive_failures=0,
            degraded_until=None,
        )

    def mark_degraded(self, provider_id: str, task: str, reason: str) -> None:
        now = self._clock.now()
        with self._lock:
            keys = self._matching_keys(provider_id, task)
            for key in keys:
                item = self._registrations[key]
                failures = item.consecutive_failures + 1
                self._registrations[key] = replace(
                    item,
                    healthy=False,
                    degraded_reason=reason,
                    consecutive_failures=failures,
                    degraded_until=now + self._cooldown_for(failures),
                    last_failure_at=now,
                )

    def mark_healthy(self, provider_id: str, task: str) -> None:
        now = self._clock.now()
        with self._lock:
            keys = self._matching_keys(provider_id, task)
            for key in keys:
                item = self._registrations[key]
                self._registrations[key] = replace(
                    item,
                    healthy=True,
                    degraded_reason=None,
                    consecutive_failures=0,
                    degraded_until=None,
                    last_success_at=now,
                )

    def statuses(self) -> tuple[ModelRegistration, ...]:
        with self._lock:
            return tuple(self._registrations[key] for key in sorted(self._registrations))

    def _cooldown_for(self, failures: int) -> timedelta:
        exponent = min(max(failures - 1, 0), 16)
        seconds = min(
            self._failure_cooldown.total_seconds() * (2**exponent),
            self._max_failure_cooldown.total_seconds(),
        )
        return timedelta(seconds=seconds)

    def _update(self, provider_id: str, task: str | None, **changes: object) -> None:
        with self._lock:
            keys = self._matching_keys(provider_id, task)
            for key in keys:
                self._registrations[key] = replace(self._registrations[key], **changes)

    def _matching_keys(self, provider_id: str, task: str | None) -> list[tuple[str, str]]:
        keys = [
            key
            for key in self._registrations
            if key[1] == provider_id and (task is None or key[0] == task)
        ]
        if not keys:
            raise LookupError(f"unknown model provider: {provider_id}")
        return keys


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
    except RunCancelledError:
        # A user or run-budget cancellation is not evidence that the provider
        # is unhealthy.
        raise
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
        registration = self._router.select("dialogue", context, subject=getattr(workspace, "subject", None))
        return _call_registration(
            self._router,
            registration,
            lambda: registration.model.generate(workspace, context),
            attempts=self._attempts,
        )

    def repair_review(self, workspace, draft, previous, error, context):
        registration = self._router.select("dialogue", context, subject=getattr(workspace, "subject", None))
        repair = getattr(registration.model, "repair_review", None)
        if repair is None:
            return None
        result = _call_registration(
            self._router,
            registration,
            lambda: repair(workspace, draft, previous, error, context),
            attempts=1,
        )
        if getattr(result, "error_type", None) is None:
            return result
        # A provider may return an empty/incomplete repair response. Retry the
        # bounded repair once; deterministic validation still decides validity.
        return _call_registration(
            self._router,
            registration,
            lambda: repair(workspace, draft, previous, error, context),
            attempts=1,
        )

    def repair(self, workspace, previous, error, context):
        registration = self._router.select("dialogue", context, subject=getattr(workspace, "subject", None))
        repair = getattr(registration.model, "repair", None)
        if repair is None:
            return None
        result = _call_registration(
            self._router,
            registration,
            lambda: repair(workspace, previous, error, context),
            attempts=1,
        )
        if getattr(result, "error_type", None) is None:
            return result
        # A provider may return an empty/incomplete repair response. Retry the
        # bounded repair once; deterministic validation still decides validity.
        return _call_registration(
            self._router,
            registration,
            lambda: repair(workspace, previous, error, context),
            attempts=1,
        )

    def review(self, workspace, draft, context):
        registration = self._router.select("dialogue", context, subject=getattr(workspace, "subject", None))
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

    def repair(self, goal, budget, workspace, capabilities, previous, error, context):
        registration = self._router.select("planning", context, subject=getattr(workspace, "subject", None))
        repair = getattr(registration.model, "repair", None)
        if repair is None:
            return None
        return _call_registration(
            self._router,
            registration,
            lambda: repair(goal, budget, workspace, capabilities, previous, error, context),
            attempts=1,
        )

    def create(self, goal, budget, workspace, capabilities, context):
        registration = self._router.select("planning", context, subject=getattr(workspace, "subject", None))
        return _call_registration(
            self._router,
            registration,
            lambda: registration.model.create(goal, budget, workspace, capabilities, context),
            attempts=self._attempts,
        )

    def replan(self, goal, plan, progress, workspace, capabilities, context):
        registration = self._router.select("planning", context, subject=getattr(workspace, "subject", None))
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
        registration = self._router.select("action", context, subject=getattr(workspace, "subject", None))
        return _call_registration(
            self._router,
            registration,
            lambda: registration.model.propose(plan, step, workspace, tools, context),
            attempts=self._attempts,
        )

    def decide(self, request, workspace, context):
        registration = self._router.select("action", context, subject=getattr(workspace, "subject", None))
        return _call_registration(
            self._router,
            registration,
            lambda: registration.model.decide(request, workspace, context),
            attempts=self._attempts,
        )


class RoutedProactivityModel:
    def __init__(self, router: ModelRouter, *, attempts: int = 2) -> None:
        self._router = router
        self._attempts = attempts

    @property
    def model(self) -> object:
        return self._router.select("proactive").model

    def propose(self, event, workspace, context) -> MotiveProposal:
        try:
            registration = self._router.select(
                "proactive",
                context,
                subject=getattr(workspace, "subject", None),
            )
        except LookupError:
            # Proactivity is optional.  During the bounded cooldown of a
            # failing provider the slow loop should remain silent rather than
            # raise and block later fast-path work.
            return MotiveProposal(False, "", "", "", 0.0, 0, 1, ())
        return _call_registration(
            self._router,
            registration,
            lambda: registration.model.propose(event, workspace, context),
            attempts=self._attempts,
        )

    def express(self, intention, workspace, context) -> str:
        registration = self._router.select(
            "proactive",
            context,
            subject=getattr(workspace, "subject", None),
        )
        return _call_registration(
            self._router,
            registration,
            lambda: registration.model.express(intention, workspace, context),
            attempts=1,
        )
