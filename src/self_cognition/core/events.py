from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from uuid import UUID

from self_cognition.core.actions import (
    ActionContextPayload,
    ActionDecisionPayload,
    ActionFailurePayload,
    ActionProposedPayload,
    ActionResultPayload,
)
from self_cognition.core.errors import ContractValidationError
from self_cognition.core.dialogue import (
    AssistantMessagePayload,
    DialogueContextPayload,
    DialogueFailurePayload,
    DialogueDraft,
    GroundingReview,
)
from self_cognition.core.identity import (
    CapabilityRecord,
    GoalRecord,
    GoalStatus,
    LimitationRecord,
    SelfModelAspect,
    SelfModelObservationValue,
)
from self_cognition.core.ids import new_event_id
from self_cognition.core.metacognition import ConflictReview
from self_cognition.core.plans import (
    GoalPlannedPayload,
    GoalPlanningRequest,
    GoalRequestedPayload,
    GoalStatusChangedPayload,
    PlanningContextPayload,
    PlanningFailurePayload,
    PlanRevisedPayload,
    PlanStepResultPayload,
)
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
class AssessmentRequestPayload:
    text: str
    source_event: EventEnvelope

    def __post_init__(self) -> None:
        _require_non_blank(self.text, "assessment task")
        if not isinstance(self.source_event, EventEnvelope):
            raise ContractValidationError("assessment requires a source event")
        if self.source_event.event_type == "cognition.assessment_requested":
            raise ContractValidationError("assessment requests cannot nest")


@dataclass(frozen=True, slots=True)
class ConflictReviewPayload:
    target_field: str
    review: ConflictReview

    def __post_init__(self) -> None:
        _require_non_blank(self.target_field, "conflict target field")
        if not isinstance(self.review, ConflictReview):
            raise ContractValidationError("conflict review payload is invalid")


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
    | DialogueContextPayload
    | AssistantMessagePayload
    | DialogueFailurePayload
    | AssessmentRequestPayload
    | ConflictReviewPayload
    | CognitionCorrectionPayload
    | SelfModelObservationPayload
    | CapabilityObservationPayload
    | ModelResponsePayload
    | CognitionModuleResultPayload
    | StateReductionPayload
    | ProcessingFailurePayload
    | GoalRequestedPayload
    | PlanningContextPayload
    | GoalPlannedPayload
    | PlanningFailurePayload
    | PlanRevisedPayload
    | PlanStepResultPayload
    | GoalStatusChangedPayload
    | ActionContextPayload
    | ActionProposedPayload
    | ActionDecisionPayload
    | ActionFailurePayload
    | ActionResultPayload
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
            "dialogue.started": DialogueContextPayload,
            "assistant.message": AssistantMessagePayload,
            "dialogue.failed": DialogueFailurePayload,
            "cognition.assessment_requested": AssessmentRequestPayload,
            "conflict.reviewed": ConflictReviewPayload,
            "user.correction": CognitionCorrectionPayload,
            "self_model.observation": SelfModelObservationPayload,
            "capability.observed": CapabilityObservationPayload,
            "model.response": ModelResponsePayload,
            "cognition.module_result": CognitionModuleResultPayload,
            "state.reduced": StateReductionPayload,
            "processing.failed": ProcessingFailurePayload,
            "goal.requested": GoalRequestedPayload,
            "planning.started": PlanningContextPayload,
            "goal.planned": GoalPlannedPayload,
            "planning.failed": PlanningFailurePayload,
            "plan.revised": PlanRevisedPayload,
            "plan.step_result": PlanStepResultPayload,
            "goal.status_changed": GoalStatusChangedPayload,
            "action.started": ActionContextPayload,
            "action.proposed": ActionProposedPayload,
            "action.decided": ActionDecisionPayload,
            "action.failed": ActionFailurePayload,
            "action.result": ActionResultPayload,
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
        elif self.event_type == "conflict.reviewed":
            if self.source is not EventSource.USER or self.actor is None:
                raise ContractValidationError(
                    "conflict confirmation requires a user actor"
                )
            if self.actor.kind is not SubjectKind.USER:
                raise ContractValidationError(
                    "conflict confirmation actor must be a user"
                )
            if (
                self.subject.subject.kind is not SubjectKind.MIND
                and self.actor != self.subject.subject
            ):
                raise ContractValidationError(
                    "conflict review must retain subject ownership"
                )
        elif self.event_type == "cognition.assessment_requested":
            if self.source is not EventSource.SYSTEM or self.actor is not None:
                raise ContractValidationError(
                    "assessment request must be a system event"
                )
            origin = self.payload.source_event
            if self.subject != SubjectScope.for_mind(origin.subject.mind.mind_id):
                raise ContractValidationError("assessment must target its source mind")
            if self.causation_id != origin.event_id:
                raise ContractValidationError("assessment must retain source causation")
            if self.scope != replace(origin.scope, owner=self.subject):
                raise ContractValidationError(
                    "assessment must preserve disclosure context"
                )
        elif isinstance(
            self.payload,
            (
                DialogueContextPayload,
                AssistantMessagePayload,
                DialogueFailurePayload,
            ),
        ):
            payload = self.payload
            if not isinstance(payload.request_event_id, UUID):
                raise ContractValidationError("dialogue request ID must be a UUID")
            if not isinstance(payload.recipient, SubjectScope):
                raise ContractValidationError("dialogue recipient must be a subject")
            if self.subject != SubjectScope.for_mind(payload.recipient.mind.mind_id):
                raise ContractValidationError(
                    "dialogue must belong to the recipient's mind"
                )
            if self.causation_id is None:
                raise ContractValidationError("dialogue must retain its causal chain")
            if isinstance(payload, AssistantMessagePayload):
                if not isinstance(payload.answer, DialogueDraft):
                    raise ContractValidationError("assistant answer must be structured")
                if payload.review is not None and (
                    not isinstance(payload.review, GroundingReview)
                    or not payload.review.supported
                ):
                    raise ContractValidationError(
                        "assistant answer must pass grounding"
                    )
                if payload.answer.claims and payload.review is None:
                    raise ContractValidationError(
                        "cognitive claims require a grounding review"
                    )
                if (
                    self.source is not EventSource.MODEL
                    or self.actor != self.subject.subject
                ):
                    raise ContractValidationError(
                        "assistant messages must retain the MIND actor"
                    )
                if payload.answer.disclosure.scope != self.scope.disclosure:
                    raise ContractValidationError(
                        "assistant disclosure scope does not match"
                    )
            elif self.source is not EventSource.SYSTEM or self.actor is not None:
                raise ContractValidationError(
                    "dialogue control events must be system events"
                )
            if isinstance(payload, DialogueFailurePayload):
                _require_non_blank(payload.stage, "dialogue stage")
                _require_non_blank(payload.error_type, "dialogue error type")
            else:
                if any(
                    ref.scope.owner.mind != self.subject.mind
                    for ref in payload.evidence_refs
                ):
                    raise ContractValidationError(
                        "dialogue evidence cannot cross minds"
                    )
            if isinstance(payload, DialogueContextPayload):
                _require_non_blank(payload.workspace_json, "dialogue workspace")
                if self.causation_id != payload.request_event_id:
                    raise ContractValidationError(
                        "dialogue context must follow its request"
                    )
            for version in (payload.old_version, payload.new_version):
                if version is None and isinstance(payload, DialogueFailurePayload):
                    continue
                if type(version) is not int or version < 0:
                    raise ContractValidationError("dialogue state version is invalid")
        elif isinstance(self.payload, GoalRequestedPayload):
            self._validate_goal_request()
        elif isinstance(self.payload, GoalStatusChangedPayload):
            self._validate_goal_status_change()
        elif isinstance(self.payload, PlanStepResultPayload):
            self._validate_planning_control({EventSource.SYSTEM, EventSource.TOOL})
        elif isinstance(
            self.payload,
            (
                PlanningContextPayload,
                GoalPlannedPayload,
                PlanningFailurePayload,
                PlanRevisedPayload,
            ),
        ):
            self._validate_planning_control({EventSource.SYSTEM})
        elif isinstance(
            self.payload,
            (
                ActionContextPayload,
                ActionProposedPayload,
                ActionDecisionPayload,
                ActionFailurePayload,
                ActionResultPayload,
            ),
        ):
            self._validate_action_control()
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

    def _validate_goal_request(self) -> None:
        payload = self.payload
        if self.subject.subject.kind is not SubjectKind.MIND:
            raise ContractValidationError("goal request must target a mind subject")
        if payload.goal.status is not GoalStatus.ACTIVE:
            raise ContractValidationError("new goals must start active")
        if self.source is EventSource.USER:
            if self.actor is None or self.actor != payload.requested_by:
                raise ContractValidationError("user goal request actor is invalid")
            if self.actor.kind is not SubjectKind.USER:
                raise ContractValidationError("goal requester must be a user")
        elif self.source is EventSource.SYSTEM:
            if self.actor is not None or payload.requested_by is not None:
                raise ContractValidationError("system goal request cannot name an actor")
        else:
            raise ContractValidationError("models and tools cannot request goals")

    def _validate_goal_status_change(self) -> None:
        payload = self.payload
        if self.subject.subject.kind is not SubjectKind.MIND:
            raise ContractValidationError("goal status must target a mind subject")
        if payload.changed_by is None:
            if self.source is not EventSource.SYSTEM or self.actor is not None:
                raise ContractValidationError("system goal status actor is invalid")
        elif (
            self.source is not EventSource.USER
            or self.actor != payload.changed_by
            or self.actor.kind is not SubjectKind.USER
        ):
            raise ContractValidationError("user goal status actor is invalid")

    def _validate_planning_control(self, sources: set[EventSource]) -> None:
        if self.subject.subject.kind is not SubjectKind.MIND:
            raise ContractValidationError("planning events must target a mind subject")
        if self.source not in sources or self.actor is not None:
            raise ContractValidationError("planning event source is invalid")

    def _validate_action_control(self) -> None:
        if self.subject.subject.kind is not SubjectKind.MIND:
            raise ContractValidationError("action events must target a mind subject")
        if self.actor is not None or self.causation_id is None:
            raise ContractValidationError(
                "action events require system causation without a domain actor"
            )
        if isinstance(self.payload, ActionResultPayload):
            if self.payload.result.owner != self.subject:
                raise ContractValidationError(
                    "action result owner must match its event subject"
                )
            if self.source not in {EventSource.SYSTEM, EventSource.TOOL}:
                raise ContractValidationError("action result source is invalid")
            return
        if self.source is not EventSource.SYSTEM:
            raise ContractValidationError("action control event source is invalid")
        if isinstance(self.payload, ActionFailurePayload):
            return
        request = self.payload.request
        if request.owner != self.subject:
            raise ContractValidationError("action event owner does not match")
        if isinstance(self.payload, ActionContextPayload):
            _require_non_blank(self.payload.workspace_json, "action workspace")
            if any(
                ref.scope.owner.mind != self.subject.mind
                for ref in self.payload.evidence_refs
            ):
                raise ContractValidationError("action evidence cannot cross minds")
        elif isinstance(self.payload, ActionDecisionPayload) and (
            self.payload.decision.action_id != self.payload.request.action_id
        ):
            raise ContractValidationError("action decision does not match its request")

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
    def goal_requested(
        cls,
        request: GoalPlanningRequest,
        *,
        clock: Clock = SYSTEM_CLOCK,
        correlation_id: UUID | None = None,
        run_id: UUID | None = None,
    ) -> "EventEnvelope":
        now = clock.now()
        return cls(
            event_id=request.request_id,
            event_type="goal.requested",
            actor=(
                request.requested_by.subject
                if request.requested_by is not None
                else None
            ),
            subject=request.owner,
            payload=GoalRequestedPayload(
                request.goal,
                request.budget,
                (
                    request.requested_by.subject
                    if request.requested_by is not None
                    else None
                ),
            ),
            occurred_at=now,
            recorded_at=now,
            source=(
                EventSource.USER
                if request.requested_by is not None
                else EventSource.SYSTEM
            ),
            scope=DataScope(request.owner, DisclosureScope.MIND),
            correlation_id=correlation_id,
            run_id=run_id,
        )

    @classmethod
    def assessment_requested(
        cls,
        source_event: EventEnvelope,
        text: str,
        *,
        event_id: UUID | None = None,
        clock: Clock = SYSTEM_CLOCK,
    ) -> EventEnvelope:
        subject = SubjectScope.for_mind(source_event.subject.mind.mind_id)
        now = clock.now()
        return cls(
            event_id=event_id or new_event_id(),
            event_type="cognition.assessment_requested",
            actor=None,
            subject=subject,
            payload=AssessmentRequestPayload(text, source_event),
            occurred_at=now,
            recorded_at=now,
            source=EventSource.SYSTEM,
            scope=replace(source_event.scope, owner=subject),
            causation_id=source_event.event_id,
        )

    @classmethod
    def conflict_reviewed(
        cls,
        subject: SubjectScope,
        payload: ConflictReviewPayload,
        *,
        actor: SubjectScope,
        event_id: UUID | None = None,
        clock: Clock = SYSTEM_CLOCK,
    ) -> EventEnvelope:
        if actor.mind != subject.mind:
            raise ContractValidationError(
                "conflict reviewer must belong to the same mind"
            )
        now = clock.now()
        return cls(
            event_id=event_id or new_event_id(),
            event_type="conflict.reviewed",
            actor=actor.subject,
            subject=subject,
            payload=payload,
            occurred_at=now,
            recorded_at=now,
            source=EventSource.USER,
            scope=DataScope(subject, DisclosureScope.PRIVATE),
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
