from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Protocol, runtime_checkable
from uuid import UUID

from self_cognition.core.contributions import CognitiveContribution
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.events import EventEnvelope
from self_cognition.core.model_outputs import ModelExtractionResult
from self_cognition.core.scopes import SubjectScope
from self_cognition.core.state import SubjectState

if TYPE_CHECKING:
    from self_cognition.runtime.run_context import RunContext


@runtime_checkable
class CognitiveModule(Protocol):
    subscriptions: frozenset[str]

    def process(
        self,
        event: EventEnvelope,
    ) -> tuple[CognitiveContribution, ...]: ...


@runtime_checkable
class ContextualCognitiveModule(Protocol):
    subscriptions: frozenset[str]

    def process(
        self,
        event: EventEnvelope,
        context: RunContext,
    ) -> tuple[CognitiveContribution, ...]: ...


@runtime_checkable
class CognitionModel(Protocol):
    def extract(
        self,
        event: EventEnvelope,
        context: RunContext,
    ) -> ModelExtractionResult: ...


@runtime_checkable
class EventStore(Protocol):
    def append(self, event: EventEnvelope) -> None: ...

    def read_by_subject(
        self,
        subject: SubjectScope,
    ) -> tuple[EventEnvelope, ...]: ...


@runtime_checkable
class EvidenceRepository(Protocol):
    def append(self, evidence: EvidenceRef) -> None: ...

    def get(
        self,
        subject: SubjectScope,
        evidence_id: UUID,
    ) -> EvidenceRef | None: ...

    def read_by_subject(
        self,
        subject: SubjectScope,
    ) -> tuple[EvidenceRef, ...]: ...


@runtime_checkable
class StateRepository(Protocol):
    def load(self, subject: SubjectScope | str) -> SubjectState | None: ...

    def save(self, state: SubjectState, expected_version: int) -> None: ...
