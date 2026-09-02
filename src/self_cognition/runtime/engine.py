import inspect
from dataclasses import replace

from self_cognition.blackboard.service import CognitiveSpaceService
from self_cognition.cognition.registry import CognitiveModuleRegistry
from self_cognition.core.contributions import CognitiveContribution
from self_cognition.core.errors import RunCancelledError
from self_cognition.core.events import EventEnvelope
from self_cognition.core.protocols import CognitiveModule
from self_cognition.core.state import SubjectState
from self_cognition.runtime.run_context import RunContext


class CognitionEngine:
    def __init__(
        self,
        modules: tuple[CognitiveModule, ...],
        cognitive_space: CognitiveSpaceService,
        module_registry: CognitiveModuleRegistry | None = None,
    ) -> None:
        self._modules = modules
        self._cognitive_space = cognitive_space
        self._module_registry = module_registry

    def process(
        self,
        event: EventEnvelope,
        state: SubjectState,
        context: RunContext | None = None,
    ) -> SubjectState:
        contributions: list[CognitiveContribution] = []
        module_errors: list[Exception] = []
        modules = (
            self._module_registry.active_modules()
            if self._module_registry is not None
            else self._modules
        )
        for module in modules:
            if context is not None and context.is_cancelled:
                raise RunCancelledError("run cancelled before cognitive module")
            if event.event_type not in module.subscriptions:
                continue
            parameters = inspect.signature(module.process).parameters
            try:
                if len(parameters) == 2:
                    if context is None:
                        raise ValueError(
                            "contextual cognitive module requires RunContext"
                        )
                    contributions.extend(module.process(event, context))
                else:
                    contributions.extend(module.process(event))
                if self._module_registry is not None:
                    self._module_registry.mark_healthy(module)
            except RunCancelledError:
                raise
            except Exception as error:
                if self._module_registry is not None:
                    self._module_registry.mark_degraded(module, error)
                module_errors.append(error)
                continue

        if not contributions and module_errors:
            raise module_errors[0]

        if context is not None and context.is_cancelled:
            raise RunCancelledError("run cancelled before state reduction")

        bound_contributions = tuple(
            replace(contribution, target_version=state.version)
            if contribution.target_version is None
            else contribution
            for contribution in contributions
        )
        return self._cognitive_space.submit(
            state,
            bound_contributions,
            decided_at=event.recorded_at,
        )
