from __future__ import annotations

import inspect
from dataclasses import dataclass, replace
from datetime import datetime

from self_cognition.blackboard.service import CognitiveSpaceService
from self_cognition.cognition.registry import (
    CognitiveModuleRegistry,
    ModuleRegistration,
)
from self_cognition.core.cognition import (
    CognitionFailureType,
    CognitionModuleResult,
    CognitionRequest,
    CognitionResultStatus,
)
from self_cognition.core.contributions import CognitiveContribution
from self_cognition.core.errors import (
    ContractValidationError,
    ModelOutputError,
    ModelTimeoutError,
    PersistedCognitionFailureError,
    RunCancelledError,
)
from self_cognition.core.events import EventEnvelope
from self_cognition.core.protocols import CognitiveModule
from self_cognition.core.state import SubjectState
from self_cognition.core.workspace import WorkspaceBuilder, WorkspaceRunInfo
from self_cognition.runtime.cognition_context import ReadOnlyCognitionContext
from self_cognition.runtime.run_context import RunContext


@dataclass(frozen=True, slots=True)
class _ModuleBinding:
    module: object
    module_id: str
    module_version: str
    deterministic: bool
    strict_metadata: bool


class CognitionEngine:
    def __init__(
        self,
        modules: tuple[CognitiveModule, ...],
        cognitive_space: CognitiveSpaceService,
        module_registry: CognitiveModuleRegistry | None = None,
        workspace_builder: WorkspaceBuilder | None = None,
    ) -> None:
        self._modules = modules
        self._cognitive_space = cognitive_space
        self._module_registry = module_registry
        self._workspace_builder = workspace_builder or WorkspaceBuilder()

    def analyze(
        self,
        event: EventEnvelope,
        state: SubjectState,
        context: RunContext | None = None,
        existing_results: tuple[CognitionModuleResult, ...] = (),
    ) -> tuple[CognitionModuleResult, ...]:
        request = CognitionRequest(
            event=event,
            context=ReadOnlyCognitionContext(
                builder=self._workspace_builder,
                state=state,
                as_of=event.recorded_at,
                run_info=_run_info(context),
            ),
            run_context=context,
        )
        results: list[CognitionModuleResult] = []
        existing_by_module = {
            result.module_id: result for result in existing_results
        }
        for binding in self._active_bindings():
            module = binding.module
            subscriptions = getattr(module, "subscriptions")
            if event.event_type not in subscriptions:
                continue
            existing = existing_by_module.get(binding.module_id)
            if existing is not None:
                results.append(existing)
                continue
            if context is not None and context.is_cancelled:
                raise RunCancelledError("run cancelled before cognitive module")
            try:
                contributions = self._invoke(module, request)
                if context is not None and context.is_cancelled:
                    raise RunCancelledError(
                        "run cancelled after cognitive module"
                    )
                module_id, module_version = _result_metadata(
                    binding,
                    contributions,
                )
                bound = tuple(
                    replace(contribution, target_version=state.version)
                    if contribution.target_version is None
                    else contribution
                    for contribution in contributions
                )
                result = CognitionModuleResult(
                    module_id=module_id,
                    module_version=module_version,
                    deterministic=binding.deterministic,
                    status=CognitionResultStatus.SUCCEEDED,
                    contributions=bound,
                    emitted_events=_drain_events(context),
                )
                if self._module_registry is not None:
                    self._module_registry.mark_healthy(module)
            except RunCancelledError as error:
                result = _failed_result(
                    binding,
                    CognitionResultStatus.CANCELLED,
                    CognitionFailureType.CANCELLED,
                    error,
                    _drain_events(context),
                )
            except Exception as error:
                if self._module_registry is not None:
                    self._module_registry.mark_degraded(module, error)
                result = _failed_result(
                    binding,
                    CognitionResultStatus.FAILED,
                    _failure_type(error),
                    error,
                    _drain_events(context),
                )
            results.append(result)
            if result.status is CognitionResultStatus.CANCELLED:
                break
        return tuple(results)

    def reduce(
        self,
        state: SubjectState,
        results: tuple[CognitionModuleResult, ...],
        *,
        decided_at: datetime,
        rebind_target_version: bool = False,
    ) -> SubjectState:
        contributions = tuple(
            replace(contribution, target_version=state.version)
            if rebind_target_version
            else contribution
            for result in results
            if result.status is CognitionResultStatus.SUCCEEDED
            for contribution in result.contributions
        )
        return self._cognitive_space.submit(
            state,
            contributions,
            decided_at=decided_at,
        )

    def process(
        self,
        event: EventEnvelope,
        state: SubjectState,
        context: RunContext | None = None,
    ) -> SubjectState:
        results = self.analyze(event, state, context)
        raise_terminal_failure(results)
        if context is not None and context.is_cancelled:
            raise RunCancelledError("run cancelled before state reduction")
        return self.reduce(state, results, decided_at=event.recorded_at)

    def requires_persisted_result(self, event_type: str) -> bool:
        return any(
            event_type in getattr(binding.module, "subscriptions")
            and not binding.deterministic
            for binding in self._active_bindings()
        )

    def _active_bindings(self) -> tuple[_ModuleBinding, ...]:
        if self._module_registry is not None:
            return tuple(
                _binding_from_registration(registration)
                for registration in self._module_registry.active_registrations()
            )
        return tuple(_binding_from_module(module) for module in self._modules)

    @staticmethod
    def _invoke(
        module: object,
        request: CognitionRequest,
    ) -> tuple[CognitiveContribution, ...]:
        run = getattr(module, "run", None)
        if callable(run):
            return tuple(run(request))

        # Compatibility for pre-stage-15 modules; new modules implement run().
        process = getattr(module, "process")
        if len(inspect.signature(process).parameters) == 2:
            if request.run_context is None:
                raise ValueError(
                    "contextual cognitive module requires RunContext"
                )
            return tuple(process(request.event, request.run_context))
        return tuple(process(request.event))


def raise_terminal_failure(
    results: tuple[CognitionModuleResult, ...],
) -> None:
    cancelled = next(
        (
            result
            for result in results
            if result.status is CognitionResultStatus.CANCELLED
        ),
        None,
    )
    if cancelled is not None:
        if cancelled.error is not None:
            raise cancelled.error
        raise RunCancelledError("persisted cognition result was cancelled")
    if results and all(
        result.status is CognitionResultStatus.FAILED for result in results
    ):
        first = results[0]
        if first.error is not None:
            raise first.error
        raise PersistedCognitionFailureError(
            first.error_type or "CognitionModuleError"
        )


def _binding_from_registration(
    registration: ModuleRegistration,
) -> _ModuleBinding:
    return _ModuleBinding(
        module=registration.module,
        module_id=registration.module_id,
        module_version=registration.version,
        deterministic=bool(registration.deterministic),
        strict_metadata=True,
    )


def _binding_from_module(module: object) -> _ModuleBinding:
    module_id = getattr(module, "module_id", None)
    module_version = getattr(module, "module_version", None)
    strict = isinstance(module_id, str) and isinstance(module_version, str)
    return _ModuleBinding(
        module=module,
        module_id=(
            module_id
            if isinstance(module_id, str)
            else f"{type(module).__module__}.{type(module).__qualname__}"
        ),
        module_version=module_version if isinstance(module_version, str) else "1",
        deterministic=bool(getattr(module, "deterministic", True)),
        strict_metadata=strict,
    )


def _result_metadata(
    binding: _ModuleBinding,
    contributions: tuple[CognitiveContribution, ...],
) -> tuple[str, str]:
    if binding.strict_metadata or not contributions:
        return binding.module_id, binding.module_version
    modules = {
        (contribution.source_module, contribution.module_version)
        for contribution in contributions
    }
    if len(modules) != 1:
        raise ContractValidationError(
            "legacy module contributions use inconsistent metadata"
        )
    return modules.pop()


def _failed_result(
    binding: _ModuleBinding,
    status: CognitionResultStatus,
    failure_type: CognitionFailureType,
    error: Exception,
    emitted_events: tuple[EventEnvelope, ...],
) -> CognitionModuleResult:
    return CognitionModuleResult(
        module_id=binding.module_id,
        module_version=binding.module_version,
        deterministic=binding.deterministic,
        status=status,
        emitted_events=emitted_events,
        failure_type=failure_type,
        error_type=type(error).__name__,
        error=error,
    )


def _failure_type(error: Exception) -> CognitionFailureType:
    if isinstance(error, ModelTimeoutError):
        return CognitionFailureType.TIMEOUT
    if isinstance(error, ModelOutputError):
        return CognitionFailureType.INVALID_OUTPUT
    return CognitionFailureType.EXECUTION


def _drain_events(context: RunContext | None) -> tuple[EventEnvelope, ...]:
    return context.drain_emitted_events() if context is not None else ()


def _run_info(context: RunContext | None) -> WorkspaceRunInfo | None:
    if context is None:
        return None
    return WorkspaceRunInfo(
        run_id=context.run_id,
        correlation_id=context.correlation_id,
        deadline=context.deadline,
        cancelled=context.cancelled,
    )
