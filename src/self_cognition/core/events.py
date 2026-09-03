from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from uuid import UUID

from self_cognition.core.errors import ContractValidationError
from self_cognition.core.identity import (
    CapabilityRecord,
    GoalRecord,
    LimitationRecord,
    SelfModelAspect,
    SelfModelObservationValue,
)
from self_cognition.core.ids import new_event_id
from self_cognition.core.scopes import (
    ConversationScope,
    DataScope,
    DisclosureScope,
    SubjectRef,
    SubjectKind,
    SubjectScope,
    normalize_subject_scope,
)
from self_cognition.core.time import Clock, SYSTEM_CLOCK


EVENT_SCHEMA_VERSION = 2


class EventSource(str, Enum):
    USER = "user"
    TOOL = "tool"
    MODEL = "model"
    SYSTEM = "system"


@dataclass(frozen=True, slots=True)
class UserMessagePayload:
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ContractValidationError("user message text must not be blank")


@dataclass(frozen=True, slots=True)
class CognitionCorrectionPayload:
    target_field: str
    cognition_type: str
    value: object
    corrected_memory_id: UUID | None = None

    def __post_init__(self) -> None:
        _require_non_blank(self.target_field, "target_field")
        _require_non_blank(self.cognition_type, "cognition_type")
        if self.value is None or (
            isinstance(self.value, str) and not self.value.strip()
        ):
            raise ContractValidationError("correction value must not be empty")
        if self.corrected_memory_id is not None and not isinstance(
            self.corrected_memory_id,
            UUID,
        ):
            raise ContractValidationError("corrected_memory_id must be a UUID")


@dataclass(frozen=True, slots=True)
class SelfModelObservationPayload:
    aspect: SelfModelAspect
    field_id: str
    value: SelfModelObservationValue
    confidence: float
    explicitly_confirmed: bool = False
    expires_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.aspect, SelfModelAspect):
            raise ContractValidationError("self model aspect is invalid")
        _require_identifier(self.field_id, "self model field_id")
        if not 0.0 <= self.confidence <= 1.0:
            raise ContractValidationError(
                "self model confidence must be between zero and one"
            )
        if not isinstance(self.explicitly_confirmed, bool):
            raise ContractValidationError(
                "explicitly_confirmed must be a boolean"
            )
        if self.aspect in {SelfModelAspect.IDENTITY, SelfModelAspect.VALUE}:
            _require_non_blank(self.value, "self model value")
        elif self.aspect is SelfModelAspect.LIMITATION:
            if not isinstance(self.value, LimitationRecord):
                raise ContractValidationError(
                    "limitation observation requires a LimitationRecord"
                )
            if self.field_id != self.value.limitation_id:
                raise ContractValidationError(
                    "limitation field_id must match limitation_id"
                )
        elif not isinstance(self.value, GoalRecord):
            raise ContractValidationError(
                "goal observation requires a GoalRecord"
            )
        elif self.field_id != self.value.goal_id:
            raise ContractValidationError("goal field_id must match goal_id")
        if self.expires_at is not None:
            _require_aware(self.expires_at, "self model expires_at")


@dataclass(frozen=True, slots=True)
class CapabilityObservationPayload:
    capability: CapabilityRecord

    def __post_init__(self) -> None:
        if not isinstance(self.capability, CapabilityRecord):
            raise ContractValidationError(
                "capability observation requires a CapabilityRecord"
            )


@dataclass(frozen=True, slots=True)
class ModelResponsePayload:
    model: str
    response_id: str
    raw_output: str

    def __post_init__(self) -> None:
        _require_non_blank(self.model, "model")
        _require_non_blank(self.response_id, "response_id")
        _require_non_blank(self.raw_output, "raw_output")


@dataclass(frozen=True, slots=True)
class CognitionModuleResultPayload:
    module_id: str
    module_version: str
    deterministic: bool
    status: str
    contributions: tuple["CognitiveContribution", ...]
    response_event_ids: tuple[UUID, ...] = ()
    failure_type: str | None = None
    error_type: str | None = None

    def __post_init__(self) -> None:
        from self_cognition.core.contributions import CognitiveContribution

        _require_non_blank(self.module_id, "module_id")
        _require_non_blank(self.module_version, "module_version")
        if not isinstance(self.deterministic, bool):
            raise ContractValidationError("deterministic must be a boolean")
        if self.status not in {"succeeded", "failed", "cancelled"}:
            raise ContractValidationError("cognition result status is invalid")
        if any(
            not isinstance(contribution, CognitiveContribution)
            for contribution in self.contributions
        ):
            raise ContractValidationError(
                "cognition result contributions are invalid"
            )
        if any(
            contribution.source_module != self.module_id
            or contribution.module_version != self.module_version
            for contribution in self.contributions
        ):
            raise ContractValidationError(
                "cognition result contributions do not match module metadata"
            )
        if any(
            not isinstance(event_id, UUID) for event_id in self.response_event_ids
        ):
            raise ContractValidationError("response_event_ids must contain UUIDs")
        failed = self.status != "succeeded"
        if failed != (self.failure_type is not None):
            raise ContractValidationError(
                "failed cognition results require a failure type"
            )
        if failed != (self.error_type is not None):
            raise ContractValidationError(
                "failed cognition results require an error type"
            )
        if self.failure_type is not None:
            _require_non_blank(self.failure_type, "failure_type")
        if self.error_type is not None:
            _require_non_blank(self.error_type, "error_type")
        if failed and self.contributions:
            raise ContractValidationError(
                "failed cognition results cannot contain contributions"
            )


@dataclass(frozen=True, slots=True)
class StateReductionPayload:
    old_version: int
    new_version: int
    state_changed: bool
    applied_contribution_ids: tuple[UUID, ...]

    def __post_init__(self) -> None:
        if self.old_version < 0 or self.new_version < self.old_version:
            raise ContractValidationError("state reduction versions are invalid")
        if not isinstance(self.state_changed, bool):
            raise ContractValidationError("state_changed must be a boolean")
        if self.state_changed != (self.new_version != self.old_version):
            raise ContractValidationError(
                "state_changed must match the version change"
            )


@dataclass(frozen=True, slots=True)
class ProcessingFailurePayload:
    stage: str
    error_type: str

    def __post_init__(self) -> None:
        _require_non_blank(self.stage, "stage")
        _require_non_blank(self.error_type, "error_type")


EventPayload = (
    UserMessagePayload
    | CognitionCorrectionPayload
    | SelfModelObservationPayload
    | CapabilityObservationPayload
    | ModelResponsePayload
    | CognitionModuleResultPayload
    | StateReductionPayload
    | ProcessingFailurePayload
)


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    event_id: UUID
    event_type: str
    actor: SubjectRef | None
    subject: SubjectScope
    payload: EventPayload
    occurred_at: datetime
    recorded_at: datetime
    source: EventSource
    scope: DataScope
    causation_id: UUID | None = None
    correlation_id: UUID | None = None
    run_id: UUID | None = None
    schema_version: int = EVENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, UUID):
            raise ContractValidationError("event_id must be a UUID")
        if not isinstance(self.event_type, str) or not self.event_type.strip():
            raise ContractValidationError("event_type must not be blank")
        if not isinstance(self.subject, SubjectScope):
            raise ContractValidationError("subject must be a SubjectScope")
        if self.actor is not None and not isinstance(self.actor, SubjectRef):
            raise ContractValidationError("actor must be a SubjectRef or None")
        expected_payloads = {
            "user.message": UserMessagePayload,
            "user.correction": CognitionCorrectionPayload,
            "self_model.observation": SelfModelObservationPayload,
            "capability.observed": CapabilityObservationPayload,
            "model.response": ModelResponsePayload,
            "cognition.module_result": CognitionModuleResultPayload,
            "state.reduced": StateReductionPayload,
            "processing.failed": ProcessingFailurePayload,
        }
        payload_type = expected_payloads.get(self.event_type)
        if payload_type is None or not isinstance(self.payload, payload_type):
            raise ContractValidationError(
                "payload type does not match event_type"
            )
        _require_aware(self.occurred_at, "occurred_at")
        _require_aware(self.recorded_at, "recorded_at")
        if not isinstance(self.source, EventSource):
            raise ContractValidationError("source must be an EventSource")
        if not isinstance(self.scope, DataScope):
            raise ContractValidationError("scope must be a DataScope")
        if self.scope.owner != self.subject:
            raise ContractValidationError("scope owner must match event subject")
        if self.schema_version != EVENT_SCHEMA_VERSION:
            raise ContractValidationError(
                f"schema_version must be {EVENT_SCHEMA_VERSION}"
            )
        if self.event_type in {"user.message", "user.correction"}:
            if self.source is not EventSource.USER:
                raise ContractValidationError("user event source must be user")
            if self.actor != self.subject.subject:
                raise ContractValidationError(
                    "user event actor must match its subject"
                )
        elif self.event_type == "self_model.observation":
            self._validate_self_model_source()
        elif self.event_type == "capability.observed":
            if self.subject.subject.kind is not SubjectKind.MIND:
                raise ContractValidationError(
                    "capability observation must target a mind subject"
                )
            if self.source not in {EventSource.SYSTEM, EventSource.TOOL}:
                raise ContractValidationError(
                    "capability observation source must be system or tool"
                )
            if self.actor is not None:
                raise ContractValidationError(
                    "capability observation must not have a domain actor"
                )
        else:
            expected_source = (
                EventSource.MODEL
                if self.event_type == "model.response"
                else EventSource.SYSTEM
            )
            if self.source is not expected_source:
                raise ContractValidationError(
                    "event source does not match event_type"
                )
            if self.actor is not None:
                raise ContractValidationError(
                    "model and system events must not have a domain actor"
                )

    def _validate_self_model_source(self) -> None:
        if self.subject.subject.kind is not SubjectKind.MIND:
            raise ContractValidationError(
                "self model observation must target a mind subject"
            )
        if self.source is EventSource.SYSTEM:
            if self.actor is not None:
                raise ContractValidationError(
                    "system self model observation must not have an actor"
                )
            return
        if self.source is not EventSource.USER:
            raise ContractValidationError(
                "self model observation source must be user or system"
            )
        if self.actor is None or self.actor.kind is not SubjectKind.USER:
            raise ContractValidationError(
                "user self model observation requires a user actor"
            )

    @classmethod
    def user_message(
        cls,
        actor: SubjectScope | str,
        content: str,
        *,
        event_id: UUID | None = None,
        clock: Clock = SYSTEM_CLOCK,
        disclosure: DisclosureScope = DisclosureScope.PRIVATE,
        conversation: ConversationScope | None = None,
        correlation_id: UUID | None = None,
        causation_id: UUID | None = None,
        run_id: UUID | None = None,
    ) -> "EventEnvelope":
        actor_scope = normalize_subject_scope(actor)
        occurred_at = clock.now()
        recorded_at = clock.now()
        return cls(
            event_id=event_id or new_event_id(),
            event_type="user.message",
            actor=actor_scope.subject,
            subject=actor_scope,
            payload=UserMessagePayload(content),
            occurred_at=occurred_at,
            recorded_at=recorded_at,
            source=EventSource.USER,
            scope=DataScope(
                owner=actor_scope,
                disclosure=disclosure,
                conversation=conversation,
            ),
            causation_id=causation_id,
            correlation_id=correlation_id,
            run_id=run_id,
        )

    @classmethod
    def self_model_observation(
        cls,
        subject: SubjectScope,
        payload: SelfModelObservationPayload,
        *,
        actor: SubjectScope | None = None,
        event_id: UUID | None = None,
        clock: Clock = SYSTEM_CLOCK,
        correlation_id: UUID | None = None,
        causation_id: UUID | None = None,
        run_id: UUID | None = None,
    ) -> "EventEnvelope":
        if subject.subject.kind is not SubjectKind.MIND:
            raise ContractValidationError(
                "self model observation must target a mind subject"
            )
        if actor is not None:
            if actor.mind != subject.mind:
                raise ContractValidationError(
                    "self model actor must belong to the target mind"
                )
            if actor.subject.kind is not SubjectKind.USER:
                raise ContractValidationError(
                    "self model actor must be a user subject"
                )
        now = clock.now()
        return cls(
            event_id=event_id or new_event_id(),
            event_type="self_model.observation",
            actor=actor.subject if actor is not None else None,
            subject=subject,
            payload=payload,
            occurred_at=now,
            recorded_at=now,
            source=EventSource.USER if actor is not None else EventSource.SYSTEM,
            scope=DataScope(subject, DisclosureScope.MIND),
            causation_id=causation_id,
            correlation_id=correlation_id,
            run_id=run_id,
        )

    @classmethod
    def capability_observed(
        cls,
        subject: SubjectScope,
        capability: CapabilityRecord,
        *,
        source: EventSource,
        event_id: UUID | None = None,
        clock: Clock = SYSTEM_CLOCK,
        correlation_id: UUID | None = None,
        causation_id: UUID | None = None,
        run_id: UUID | None = None,
    ) -> "EventEnvelope":
        if source not in {EventSource.SYSTEM, EventSource.TOOL}:
            raise ContractValidationError(
                "capability observation source must be system or tool"
            )
        if subject.subject.kind is not SubjectKind.MIND:
            raise ContractValidationError(
                "capability observation must target a mind subject"
            )
        now = clock.now()
        return cls(
            event_id=event_id or new_event_id(),
            event_type="capability.observed",
            actor=None,
            subject=subject,
            payload=CapabilityObservationPayload(capability),
            occurred_at=now,
            recorded_at=now,
            source=source,
            scope=DataScope(subject, DisclosureScope.MIND),
            causation_id=causation_id,
            correlation_id=correlation_id,
            run_id=run_id,
        )

    @classmethod
    def processing_failed(
        cls,
        cause: "EventEnvelope",
        *,
        stage: str,
        error_type: str,
        clock: Clock,
        run_id: UUID,
        correlation_id: UUID,
    ) -> "EventEnvelope":
        now = clock.now()
        return cls(
            event_id=new_event_id(),
            event_type="processing.failed",
            actor=None,
            subject=cause.subject,
            payload=ProcessingFailurePayload(stage, error_type),
            occurred_at=now,
            recorded_at=now,
            source=EventSource.SYSTEM,
            scope=cause.scope,
            causation_id=cause.event_id,
            correlation_id=correlation_id,
            run_id=run_id,
        )

    @classmethod
    def correction(
        cls,
        actor: SubjectScope | str,
        *,
        target_field: str,
        cognition_type: str,
        value: object,
        corrected_memory_id: UUID | None = None,
        event_id: UUID | None = None,
        clock: Clock = SYSTEM_CLOCK,
        disclosure: DisclosureScope = DisclosureScope.PRIVATE,
        conversation: ConversationScope | None = None,
        correlation_id: UUID | None = None,
        causation_id: UUID | None = None,
        run_id: UUID | None = None,
    ) -> "EventEnvelope":
        actor_scope = normalize_subject_scope(actor)
        occurred_at = clock.now()
        recorded_at = clock.now()
        return cls(
            event_id=event_id or new_event_id(),
            event_type="user.correction",
            actor=actor_scope.subject,
            subject=actor_scope,
            payload=CognitionCorrectionPayload(
                target_field=target_field,
                cognition_type=cognition_type,
                value=value,
                corrected_memory_id=corrected_memory_id,
            ),
            occurred_at=occurred_at,
            recorded_at=recorded_at,
            source=EventSource.USER,
            scope=DataScope(
                owner=actor_scope,
                disclosure=disclosure,
                conversation=conversation,
            ),
            causation_id=causation_id,
            correlation_id=correlation_id,
            run_id=run_id,
        )

    @classmethod
    def model_response(
        cls,
        cause: "EventEnvelope",
        *,
        model: str,
        response_id: str,
        raw_output: str,
        clock: Clock,
        run_id: UUID,
        correlation_id: UUID,
    ) -> "EventEnvelope":
        now = clock.now()
        return cls(
            event_id=new_event_id(),
            event_type="model.response",
            actor=None,
            subject=cause.subject,
            payload=ModelResponsePayload(model, response_id, raw_output),
            occurred_at=now,
            recorded_at=now,
            source=EventSource.MODEL,
            scope=cause.scope,
            causation_id=cause.event_id,
            correlation_id=correlation_id,
            run_id=run_id,
        )

    @classmethod
    def cognition_module_result(
        cls,
        cause: "EventEnvelope",
        payload: CognitionModuleResultPayload,
        *,
        clock: Clock,
        run_id: UUID,
        correlation_id: UUID,
    ) -> "EventEnvelope":
        now = clock.now()
        return cls(
            event_id=new_event_id(),
            event_type="cognition.module_result",
            actor=None,
            subject=cause.subject,
            payload=payload,
            occurred_at=now,
            recorded_at=now,
            source=EventSource.SYSTEM,
            scope=cause.scope,
            causation_id=cause.event_id,
            correlation_id=correlation_id,
            run_id=run_id,
        )

    @classmethod
    def state_reduced(
        cls,
        cause: "EventEnvelope",
        payload: StateReductionPayload,
        *,
        clock: Clock,
        run_id: UUID,
        correlation_id: UUID,
    ) -> "EventEnvelope":
        now = clock.now()
        return cls(
            event_id=new_event_id(),
            event_type="state.reduced",
            actor=None,
            subject=cause.subject,
            payload=payload,
            occurred_at=now,
            recorded_at=now,
            source=EventSource.SYSTEM,
            scope=cause.scope,
            causation_id=cause.event_id,
            correlation_id=correlation_id,
            run_id=run_id,
        )


# Compatibility import only. New code should name the envelope explicitly.
Event = EventEnvelope


def _require_aware(value: datetime, name: str) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ContractValidationError(
            f"{name} must include timezone information"
        )


def _require_non_blank(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{name} must not be blank")


def _require_identifier(value: str, name: str) -> None:
    _require_non_blank(value, name)
    if any(character.isspace() for character in value):
        raise ContractValidationError(f"{name} must not contain whitespace")
