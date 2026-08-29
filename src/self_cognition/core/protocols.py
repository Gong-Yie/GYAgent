from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Protocol, runtime_checkable

from self_cognition.core.contributions import Contribution
from self_cognition.core.events import Event
from self_cognition.core.model_outputs import ModelExtractionResult
from self_cognition.core.state import SubjectState

if TYPE_CHECKING:
    from self_cognition.runtime.run_context import RunContext


@runtime_checkable
class CognitiveModule(Protocol):
    subscriptions: frozenset[str]

    def process(self, event: Event) -> tuple[Contribution, ...]: ...


@runtime_checkable
class ContextualCognitiveModule(Protocol):
    subscriptions: frozenset[str]

    def process(
        self,
        event: Event,
        context: RunContext,
    ) -> tuple[Contribution, ...]: ...


@runtime_checkable
class CognitionModel(Protocol):
    def extract(
        self,
        event: Event,
        context: RunContext,
    ) -> ModelExtractionResult: ...


@runtime_checkable
class EventStore(Protocol):
    def append(self, event: Event) -> None: ...

    def read_by_subject(self, subject_id: str) -> tuple[Event, ...]: ...


@runtime_checkable
class StateRepository(Protocol):
    def load(self, subject_id: str) -> SubjectState | None: ...

    def save(self, state: SubjectState, expected_version: int) -> None: ...
