import inspect

from self_cognition.core.contributions import Contribution
from self_cognition.core.errors import RunCancelledError
from self_cognition.core.events import Event
from self_cognition.core.protocols import CognitiveModule
from self_cognition.core.state import SubjectState
from self_cognition.runtime.reducer import StateReducer
from self_cognition.runtime.run_context import RunContext


class CognitionEngine:
    def __init__(
        self,
        modules: tuple[CognitiveModule, ...],
        reducer: StateReducer,
    ) -> None:
        self._modules = modules
        self._reducer = reducer

    def process(
        self,
        event: Event,
        state: SubjectState,
        context: RunContext | None = None,
    ) -> SubjectState:
        contributions: list[Contribution] = []
        module_errors: list[Exception] = []
        for module in self._modules:
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
            except RunCancelledError:
                raise
            except Exception as error:
                module_errors.append(error)
                continue

        if not contributions and module_errors:
            raise module_errors[0]

        if context is not None and context.is_cancelled:
            raise RunCancelledError("run cancelled before state reduction")

        return self._reducer.apply_many(state, tuple(contributions))
