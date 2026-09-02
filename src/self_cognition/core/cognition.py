from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from self_cognition.core.contributions import CognitiveContribution
from self_cognition.core.errors import ContractValidationError
from self_cognition.core.events import EventEnvelope
from self_cognition.core.workspace import RetrievalQuery, WorkspacePacket

if TYPE_CHECKING:
    from self_cognition.runtime.run_context import RunContext


class CognitionResultStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CognitionFailureType(str, Enum):
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    INVALID_OUTPUT = "invalid_output"
    EXECUTION = "execution"


@runtime_checkable
class CognitionContextQuery(Protocol):
    def query(self, query: RetrievalQuery) -> WorkspacePacket: ...


@dataclass(frozen=True, slots=True)
class CognitionRequest:
    event: EventEnvelope
    context: CognitionContextQuery
    run_context: RunContext | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.event, EventEnvelope):
            raise ContractValidationError("cognition request event is invalid")
        if not isinstance(self.context, CognitionContextQuery):
            raise ContractValidationError("cognition context query is invalid")


@dataclass(frozen=True, slots=True)
class CognitionModuleResult:
    module_id: str
    module_version: str
    deterministic: bool
    status: CognitionResultStatus
    contributions: tuple[CognitiveContribution, ...] = ()
    emitted_events: tuple[EventEnvelope, ...] = ()
    failure_type: CognitionFailureType | None = None
    error_type: str | None = None
    error: Exception | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.module_id.strip():
            raise ContractValidationError("module_id must not be blank")
        if not self.module_version.strip():
            raise ContractValidationError("module_version must not be blank")
        if not isinstance(self.deterministic, bool):
            raise ContractValidationError("deterministic must be a boolean")
        if not isinstance(self.status, CognitionResultStatus):
            raise ContractValidationError("cognition result status is invalid")
        if any(
            contribution.source_module != self.module_id
            or contribution.module_version != self.module_version
            for contribution in self.contributions
        ):
            raise ContractValidationError(
                "result contributions must match their module metadata"
            )
        failed = self.status is not CognitionResultStatus.SUCCEEDED
        if failed != (self.failure_type is not None):
            raise ContractValidationError(
                "failed cognition results require a failure type"
            )
        if failed != (self.error_type is not None):
            raise ContractValidationError(
                "failed cognition results require an error type"
            )
        if failed and self.contributions:
            raise ContractValidationError(
                "failed cognition results cannot contain contributions"
            )
