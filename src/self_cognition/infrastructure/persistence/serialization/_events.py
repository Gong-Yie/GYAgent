"""Codec for the event envelope and its typed payloads."""

from typing import Any

from self_cognition.core.actions import (
    ActionContextPayload,
    ActionDecisionPayload,
    ActionFailurePayload,
    ActionProposedPayload,
    ActionResultPayload,
    action_decision_from_dict,
    action_decision_to_dict,
    action_request_from_dict,
    action_request_to_dict,
    action_result_from_dict,
    action_result_to_dict,
    proposal_request_from_dict,
    proposal_request_to_dict,
    tool_descriptor_from_dict,
    tool_descriptor_to_dict,
)
from self_cognition.core.dialogue import (
    AssistantMessagePayload,
    DialogueContextPayload,
    DialogueFailurePayload,
    draft_from_dict,
    draft_to_dict,
    review_from_dict,
)
from self_cognition.core.errors import MalformedSerializedDataError, SerializationError
from self_cognition.core.events import (
    AssessmentRequestPayload,
    BehaviorModePayload,
    BehavioralDecisionPayload,
    CapabilityObservationPayload,
    CognitionCorrectionPayload,
    CognitionModuleResultPayload,
    ConflictReviewPayload,
    EVENT_SCHEMA_VERSION,
    EventEnvelope,
    MailboxAcknowledgedPayload,
    MailboxMessagePayload,
    ModelResponsePayload,
    MotiveFormedPayload,
    ProactiveIntentionPayload,
    ProactivityReassessmentPayload,
    ProcessingFailurePayload,
    SelfModelObservationPayload,
    StateReductionPayload,
    UserMessagePayload,
)
from self_cognition.core.identity import (
    CapabilityRecord,
    GoalRecord,
    GoalStatus,
    LimitationRecord,
    SelfModelAspect,
)
from self_cognition.core.metacognition import ConflictReview
from self_cognition.core.plans import (
    GoalPlannedPayload,
    GoalRequestedPayload,
    GoalStatusChangedPayload,
    PlanRevisedPayload,
    PlanStepResultPayload,
    PlanningContextPayload,
    PlanningFailurePayload,
    budget_from_dict,
    budget_to_dict,
    plan_from_dict,
    plan_to_dict,
    step_result_from_dict,
    step_result_to_dict,
)

from ._codec import (
    _from_json,
    _optional_uuid_to_string,
    _require_bool,
    _require_datetime,
    _require_event_source,
    _require_float,
    _require_int,
    _require_keys,
    _require_non_blank_string,
    _require_object,
    _require_optional_datetime,
    _require_optional_non_blank_string,
    _require_optional_uuid,
    _require_schema,
    _require_string,
    _require_uuid,
    _require_uuid_tuple,
    _to_json,
)
from ._refs import (
    _data_scope_from_dict,
    _data_scope_to_dict,
    _evidence_ref_from_dict,
    _evidence_ref_to_dict,
    _subject_ref_from_dict,
    _subject_ref_to_dict,
    _subject_scope_from_dict,
    _subject_scope_to_dict,
)
from ._contributions import _contribution_from_dict, _contribution_to_dict
from ._proactivity import (
    _decision_from_dict,
    _decision_to_dict,
    _intention_from_dict,
    _intention_to_dict,
    _motive_from_dict,
    _motive_to_dict,
)


def event_to_dict(event: EventEnvelope) -> dict[str, Any]:
    return {
        "schema_version": EVENT_SCHEMA_VERSION,
        "event_id": str(event.event_id),
        "event_type": event.event_type,
        "actor": (
            _subject_ref_to_dict(event.actor)
            if event.actor is not None
            else None
        ),
        "subject": _subject_scope_to_dict(event.subject),
        "payload": _event_payload_to_dict(event.payload),
        "occurred_at": event.occurred_at.isoformat(),
        "recorded_at": event.recorded_at.isoformat(),
        "source": event.source.value,
        "scope": _data_scope_to_dict(event.scope),
        "causation_id": _optional_uuid_to_string(event.causation_id),
        "correlation_id": _optional_uuid_to_string(event.correlation_id),
        "run_id": _optional_uuid_to_string(event.run_id),
    }


def event_from_dict(data: object) -> EventEnvelope:
    values = _require_object(data, "event")
    _require_schema(values, EVENT_SCHEMA_VERSION, "event")
    _require_keys(
        values,
        {
            "schema_version",
            "event_id",
            "event_type",
            "actor",
            "subject",
            "payload",
            "occurred_at",
            "recorded_at",
            "source",
            "scope",
            "causation_id",
            "correlation_id",
            "run_id",
        },
        "event",
    )
    occurred_at = _require_datetime(values["occurred_at"], "event.occurred_at")
    recorded_at = _require_datetime(values["recorded_at"], "event.recorded_at")
    actor_value = values["actor"]
    actor = (
        None
        if actor_value is None
        else _subject_ref_from_dict(actor_value, "event.actor")
    )
    subject = _subject_scope_from_dict(values["subject"], "event.subject")
    event_type = _require_string(values["event_type"], "event.event_type")

    try:
        return EventEnvelope(
            event_id=_require_uuid(values["event_id"], "event.event_id"),
            event_type=event_type,
            actor=actor,
            subject=subject,
            payload=_event_payload_from_dict(
                event_type,
                values["payload"],
                "event.payload",
            ),
            occurred_at=occurred_at,
            recorded_at=recorded_at,
            source=_require_event_source(values["source"], "event.source"),
            scope=_data_scope_from_dict(values["scope"], "event.scope"),
            causation_id=_require_optional_uuid(
                values["causation_id"],
                "event.causation_id",
            ),
            correlation_id=_require_optional_uuid(
                values["correlation_id"],
                "event.correlation_id",
            ),
            run_id=_require_optional_uuid(values["run_id"], "event.run_id"),
            schema_version=EVENT_SCHEMA_VERSION,
        )
    except SerializationError:
        raise
    except Exception as error:
        raise MalformedSerializedDataError("invalid event values") from error


def event_to_json(event: EventEnvelope) -> str:
    return _to_json(event_to_dict(event), "event")


def event_from_json(payload: str) -> EventEnvelope:
    return event_from_dict(_from_json(payload, "event"))


def _event_payload_to_dict(
    payload: (
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
    ),
) -> dict[str, object]:
    if isinstance(payload, ActionContextPayload):
        return {
            "request": proposal_request_to_dict(payload.request),
            "workspace_json": payload.workspace_json,
            "evidence_refs": [
                _evidence_ref_to_dict(ref) for ref in payload.evidence_refs
            ],
            "tools": [tool_descriptor_to_dict(tool) for tool in payload.tools],
        }
    if isinstance(payload, ActionProposedPayload):
        return {"request": action_request_to_dict(payload.request)}
    if isinstance(payload, ActionDecisionPayload):
        return {
            "request": action_request_to_dict(payload.request),
            "decision": action_decision_to_dict(payload.decision),
        }
    if isinstance(payload, ActionFailurePayload):
        return {
            "proposal_request_id": str(payload.proposal_request_id),
            "stage": payload.stage,
            "error_type": payload.error_type,
        }
    if isinstance(payload, ActionResultPayload):
        return {"result": action_result_to_dict(payload.result)}
    if isinstance(
        payload,
        (DialogueContextPayload, AssistantMessagePayload, DialogueFailurePayload),
    ):
        result = {
            "request_event_id": str(payload.request_event_id),
            "recipient": _subject_scope_to_dict(payload.recipient),
            "old_version": payload.old_version,
            "new_version": payload.new_version,
        }
        if isinstance(payload, DialogueFailurePayload):
            return {**result, "stage": payload.stage, "error_type": payload.error_type}
        result["evidence_refs"] = [
            _evidence_ref_to_dict(ref) for ref in payload.evidence_refs
        ]
        if isinstance(payload, DialogueContextPayload):
            return {**result, "workspace_json": payload.workspace_json}
        return {
            **result,
            "answer": draft_to_dict(payload.answer),
            "review": (
                None
                if payload.review is None
                else payload.review.to_state_value()
            ),
        }
    if isinstance(payload, AssessmentRequestPayload):
        return {
            "text": payload.text,
            "source_event": event_to_dict(payload.source_event),
        }
    if isinstance(payload, ConflictReviewPayload):
        return {
            "target_field": payload.target_field,
            "review": payload.review.to_state_value(),
        }
    if isinstance(payload, UserMessagePayload):
        return {"text": payload.text}
    if isinstance(payload, CognitionCorrectionPayload):
        return {
            "target_field": payload.target_field,
            "cognition_type": payload.cognition_type,
            "value": payload.value,
            "corrected_memory_id": _optional_uuid_to_string(
                payload.corrected_memory_id
            ),
        }
    if isinstance(payload, SelfModelObservationPayload):
        value = payload.value
        if isinstance(value, (LimitationRecord, GoalRecord)):
            value = value.to_state_value()
        return {
            "aspect": payload.aspect.value,
            "field_id": payload.field_id,
            "value": value,
            "confidence": payload.confidence,
            "explicitly_confirmed": payload.explicitly_confirmed,
            "expires_at": (
                payload.expires_at.isoformat()
                if payload.expires_at is not None
                else None
            ),
        }
    if isinstance(payload, CapabilityObservationPayload):
        return {"capability": payload.capability.to_state_value()}
    if isinstance(payload, GoalRequestedPayload):
        return {
            "goal": payload.goal.to_state_value(),
            "budget": budget_to_dict(payload.budget),
            "requested_by": (
                _subject_ref_to_dict(payload.requested_by)
                if payload.requested_by is not None
                else None
            ),
        }
    if isinstance(payload, PlanningContextPayload):
        return {
            "request_event_id": str(payload.request_event_id),
            "goal": payload.goal.to_state_value(),
            "budget": budget_to_dict(payload.budget),
            "workspace_json": payload.workspace_json,
            "evidence_refs": [
                _evidence_ref_to_dict(ref) for ref in payload.evidence_refs
            ],
            "state_version": payload.state_version,
        }
    if isinstance(payload, GoalPlannedPayload):
        return {
            "request_event_id": str(payload.request_event_id),
            "goal": payload.goal.to_state_value(),
            "plan": plan_to_dict(payload.plan),
        }
    if isinstance(payload, PlanningFailurePayload):
        return {
            "request_event_id": str(payload.request_event_id),
            "goal_id": payload.goal_id,
            "stage": payload.stage,
            "error_type": payload.error_type,
        }
    if isinstance(payload, PlanRevisedPayload):
        return {
            "request_event_id": str(payload.request_event_id),
            "plan": plan_to_dict(payload.plan),
            "failed_result_id": str(payload.failed_result_id),
        }
    if isinstance(payload, PlanStepResultPayload):
        return {"result": step_result_to_dict(payload.result)}
    if isinstance(payload, GoalStatusChangedPayload):
        return {
            "plan_id": str(payload.plan_id),
            "previous_status": payload.previous_status.value,
            "goal": payload.goal.to_state_value(),
            "changed_by": (
                _subject_ref_to_dict(payload.changed_by)
                if payload.changed_by is not None
                else None
            ),
            "reason": payload.reason,
        }
    if isinstance(payload, ModelResponsePayload):
        return {
            "model": payload.model,
            "response_id": payload.response_id,
            "raw_output": payload.raw_output,
        }
    if isinstance(payload, CognitionModuleResultPayload):
        return {
            "module_id": payload.module_id,
            "module_version": payload.module_version,
            "deterministic": payload.deterministic,
            "status": payload.status,
            "contributions": [
                _contribution_to_dict(contribution)
                for contribution in payload.contributions
            ],
            "response_event_ids": [
                str(event_id) for event_id in payload.response_event_ids
            ],
            "failure_type": payload.failure_type,
            "error_type": payload.error_type,
        }
    if isinstance(payload, ProcessingFailurePayload):
        return {"stage": payload.stage, "error_type": payload.error_type}
    if isinstance(payload, MotiveFormedPayload):
        return _motive_to_dict(payload.motive)
    if isinstance(payload, ProactiveIntentionPayload):
        return _intention_to_dict(payload.intention)
    if isinstance(payload, BehavioralDecisionPayload):
        return {
            "intention_id": str(payload.intention_id),
            "decision": _decision_to_dict(payload.decision),
        }
    if isinstance(payload, MailboxMessagePayload):
        return {
            "message_id": str(payload.message_id),
            "intention_id": str(payload.intention_id),
            "text": payload.text,
            "evidence_refs": [_evidence_ref_to_dict(ref) for ref in payload.evidence_refs],
            "created_at": payload.created_at.isoformat(),
            "valid_until": payload.valid_until.isoformat(),
        }
    if isinstance(payload, MailboxAcknowledgedPayload):
        return {
            "message_id": str(payload.message_id),
            "intention_id": str(payload.intention_id),
        }
    if isinstance(payload, BehaviorModePayload):
        return {"mode": payload.mode, "state_version": payload.state_version}
    if isinstance(payload, ProactivityReassessmentPayload):
        return {
            "wake_reason": payload.wake_reason,
            "previous_assessment_at": (
                payload.previous_assessment_at.isoformat()
                if payload.previous_assessment_at is not None else None
            ),
        }
    return {
        "old_version": payload.old_version,
        "new_version": payload.new_version,
        "state_changed": payload.state_changed,
        "applied_contribution_ids": [
            str(contribution_id)
            for contribution_id in payload.applied_contribution_ids
        ],
    }


def _event_payload_from_dict(
    event_type: str,
    value: object,
    path: str,
) -> (
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
):
    values = _require_object(value, path)
    if event_type == "action.started":
        _require_keys(
            values,
            {"request", "workspace_json", "evidence_refs", "tools"},
            path,
        )
        refs = values["evidence_refs"]
        tools = values["tools"]
        if not isinstance(refs, list) or not isinstance(tools, list):
            raise MalformedSerializedDataError(
                "action evidence and tools must be arrays"
            )
        return ActionContextPayload(
            proposal_request_from_dict(values["request"]),
            _require_non_blank_string(
                values["workspace_json"], f"{path}.workspace_json"
            ),
            tuple(_evidence_ref_from_dict(ref, path) for ref in refs),
            tuple(tool_descriptor_from_dict(tool) for tool in tools),
        )
    if event_type == "action.proposed":
        _require_keys(values, {"request"}, path)
        return ActionProposedPayload(action_request_from_dict(values["request"]))
    if event_type == "action.decided":
        _require_keys(values, {"request", "decision"}, path)
        return ActionDecisionPayload(
            action_request_from_dict(values["request"]),
            action_decision_from_dict(values["decision"]),
        )
    if event_type == "action.failed":
        _require_keys(
            values,
            {"proposal_request_id", "stage", "error_type"},
            path,
        )
        return ActionFailurePayload(
            _require_uuid(
                values["proposal_request_id"], f"{path}.proposal_request_id"
            ),
            _require_non_blank_string(values["stage"], f"{path}.stage"),
            _require_non_blank_string(
                values["error_type"], f"{path}.error_type"
            ),
        )
    if event_type == "action.result":
        _require_keys(values, {"result"}, path)
        return ActionResultPayload(action_result_from_dict(values["result"]))
    if event_type in {"dialogue.started", "assistant.message", "dialogue.failed"}:
        common = {"request_event_id", "recipient", "old_version", "new_version"}
        extra = {
            "dialogue.started": {"workspace_json", "evidence_refs"},
            "assistant.message": {"answer", "review", "evidence_refs"},
            "dialogue.failed": {"stage", "error_type"},
        }[event_type]
        _require_keys(values, common | extra, path)
        request_id = _require_uuid(
            values["request_event_id"], f"{path}.request_event_id"
        )
        recipient = _subject_scope_from_dict(values["recipient"], f"{path}.recipient")
        old_version, new_version = values["old_version"], values["new_version"]
        if event_type == "dialogue.failed":
            return DialogueFailurePayload(
                request_id,
                recipient,
                _require_non_blank_string(values["stage"], f"{path}.stage"),
                _require_non_blank_string(values["error_type"], f"{path}.error_type"),
                old_version,
                new_version,
            )
        if not isinstance(values["evidence_refs"], list):
            raise MalformedSerializedDataError("dialogue evidence must be an array")
        refs = tuple(
            _evidence_ref_from_dict(ref, path) for ref in values["evidence_refs"]
        )
        if event_type == "dialogue.started":
            return DialogueContextPayload(
                request_id,
                recipient,
                _require_non_blank_string(
                    values["workspace_json"], f"{path}.workspace_json"
                ),
                refs,
                old_version,
                new_version,
            )
        return AssistantMessagePayload(
            request_id,
            recipient,
            draft_from_dict(values["answer"]),
            refs,
            None if values["review"] is None else review_from_dict(values["review"]),
            old_version,
            new_version,
        )
    if event_type == "cognition.assessment_requested":
        _require_keys(values, {"text", "source_event"}, path)
        origin = _require_object(values["source_event"], f"{path}.source_event")
        if origin.get("event_type") == "cognition.assessment_requested":
            raise MalformedSerializedDataError("assessment requests cannot nest")
        return AssessmentRequestPayload(
            _require_string(values["text"], f"{path}.text"), event_from_dict(origin)
        )
    if event_type == "conflict.reviewed":
        _require_keys(values, {"target_field", "review"}, path)
        return ConflictReviewPayload(
            _require_string(values["target_field"], f"{path}.target_field"),
            ConflictReview.from_state_value(values["review"]),
        )
    if event_type == "user.message":
        _require_keys(values, {"text"}, path)
        return UserMessagePayload(
            _require_string(values["text"], f"{path}.text")
        )
    if event_type == "user.correction":
        _require_keys(
            values,
            {
                "target_field",
                "cognition_type",
                "value",
                "corrected_memory_id",
            },
            path,
        )
        return CognitionCorrectionPayload(
            target_field=_require_non_blank_string(
                values["target_field"],
                f"{path}.target_field",
            ),
            cognition_type=_require_non_blank_string(
                values["cognition_type"],
                f"{path}.cognition_type",
            ),
            value=values["value"],
            corrected_memory_id=_require_optional_uuid(
                values["corrected_memory_id"],
                f"{path}.corrected_memory_id",
            ),
        )
    if event_type == "self_model.observation":
        _require_keys(
            values,
            {
                "aspect",
                "field_id",
                "value",
                "confidence",
                "explicitly_confirmed",
                "expires_at",
            },
            path,
        )
        try:
            aspect = SelfModelAspect(
                _require_string(values["aspect"], f"{path}.aspect")
            )
        except ValueError as error:
            raise MalformedSerializedDataError(f"invalid {path}.aspect") from error
        field_id = _require_non_blank_string(values["field_id"], f"{path}.field_id")
        raw_value = values["value"]
        if aspect is SelfModelAspect.LIMITATION:
            raw_value = LimitationRecord.from_state_value(raw_value)
        elif aspect is SelfModelAspect.GOAL:
            raw_value = GoalRecord.from_state_value(raw_value)
        return SelfModelObservationPayload(
            aspect=aspect,
            field_id=field_id,
            value=raw_value,
            confidence=_require_float(values["confidence"], f"{path}.confidence"),
            explicitly_confirmed=_require_bool(
                values["explicitly_confirmed"],
                f"{path}.explicitly_confirmed",
            ),
            expires_at=_require_optional_datetime(
                values["expires_at"],
                f"{path}.expires_at",
            ),
        )
    if event_type == "capability.observed":
        _require_keys(values, {"capability"}, path)
        return CapabilityObservationPayload(
            CapabilityRecord.from_state_value(values["capability"])
        )
    if event_type == "goal.requested":
        _require_keys(values, {"goal", "budget", "requested_by"}, path)
        requester = values["requested_by"]
        return GoalRequestedPayload(
            GoalRecord.from_state_value(values["goal"]),
            budget_from_dict(values["budget"]),
            (
                None
                if requester is None
                else _subject_ref_from_dict(requester, f"{path}.requested_by")
            ),
        )
    if event_type == "planning.started":
        _require_keys(
            values,
            {
                "request_event_id",
                "goal",
                "budget",
                "workspace_json",
                "evidence_refs",
                "state_version",
            },
            path,
        )
        refs = values["evidence_refs"]
        if not isinstance(refs, list):
            raise MalformedSerializedDataError(
                "planning evidence must be an array"
            )
        return PlanningContextPayload(
            _require_uuid(values["request_event_id"], f"{path}.request_event_id"),
            GoalRecord.from_state_value(values["goal"]),
            budget_from_dict(values["budget"]),
            _require_non_blank_string(
                values["workspace_json"], f"{path}.workspace_json"
            ),
            tuple(_evidence_ref_from_dict(ref, path) for ref in refs),
            _require_int(values["state_version"], f"{path}.state_version"),
        )
    if event_type == "goal.planned":
        _require_keys(values, {"request_event_id", "goal", "plan"}, path)
        return GoalPlannedPayload(
            _require_uuid(values["request_event_id"], f"{path}.request_event_id"),
            GoalRecord.from_state_value(values["goal"]),
            plan_from_dict(values["plan"]),
        )
    if event_type == "planning.failed":
        _require_keys(
            values,
            {"request_event_id", "goal_id", "stage", "error_type"},
            path,
        )
        return PlanningFailurePayload(
            _require_uuid(values["request_event_id"], f"{path}.request_event_id"),
            _require_non_blank_string(values["goal_id"], f"{path}.goal_id"),
            _require_non_blank_string(values["stage"], f"{path}.stage"),
            _require_non_blank_string(values["error_type"], f"{path}.error_type"),
        )
    if event_type == "plan.revised":
        _require_keys(
            values,
            {"request_event_id", "plan", "failed_result_id"},
            path,
        )
        return PlanRevisedPayload(
            _require_uuid(values["request_event_id"], f"{path}.request_event_id"),
            plan_from_dict(values["plan"]),
            _require_uuid(values["failed_result_id"], f"{path}.failed_result_id"),
        )
    if event_type == "plan.step_result":
        _require_keys(values, {"result"}, path)
        return PlanStepResultPayload(step_result_from_dict(values["result"]))
    if event_type == "goal.status_changed":
        _require_keys(
            values,
            {"plan_id", "previous_status", "goal", "changed_by", "reason"},
            path,
        )
        changed_by = values["changed_by"]
        try:
            previous_status = GoalStatus(
                _require_string(
                    values["previous_status"], f"{path}.previous_status"
                )
            )
        except ValueError as error:
            raise MalformedSerializedDataError(
                f"invalid {path}.previous_status"
            ) from error
        return GoalStatusChangedPayload(
            _require_uuid(values["plan_id"], f"{path}.plan_id"),
            previous_status,
            GoalRecord.from_state_value(values["goal"]),
            (
                None
                if changed_by is None
                else _subject_ref_from_dict(changed_by, f"{path}.changed_by")
            ),
            _require_non_blank_string(values["reason"], f"{path}.reason"),
        )
    if event_type == "model.response":
        _require_keys(values, {"model", "response_id", "raw_output"}, path)
        return ModelResponsePayload(
            model=_require_non_blank_string(values["model"], f"{path}.model"),
            response_id=_require_non_blank_string(
                values["response_id"],
                f"{path}.response_id",
            ),
            raw_output=_require_non_blank_string(
                values["raw_output"],
                f"{path}.raw_output",
            ),
        )
    if event_type == "cognition.module_result":
        _require_keys(
            values,
            {
                "module_id",
                "module_version",
                "deterministic",
                "status",
                "contributions",
                "response_event_ids",
                "failure_type",
                "error_type",
            },
            path,
        )
        contributions = values["contributions"]
        if not isinstance(contributions, list):
            raise MalformedSerializedDataError(
                f"{path}.contributions must be an array"
            )
        response_event_ids = values["response_event_ids"]
        if not isinstance(response_event_ids, list):
            raise MalformedSerializedDataError(
                f"{path}.response_event_ids must be an array"
            )
        deterministic = values["deterministic"]
        if not isinstance(deterministic, bool):
            raise MalformedSerializedDataError(
                f"{path}.deterministic must be a boolean"
            )
        return CognitionModuleResultPayload(
            module_id=_require_non_blank_string(
                values["module_id"],
                f"{path}.module_id",
            ),
            module_version=_require_non_blank_string(
                values["module_version"],
                f"{path}.module_version",
            ),
            deterministic=deterministic,
            status=_require_non_blank_string(
                values["status"],
                f"{path}.status",
            ),
            contributions=tuple(
                _contribution_from_dict(
                    contribution,
                    f"{path}.contributions[{index}]",
                )
                for index, contribution in enumerate(contributions)
            ),
            response_event_ids=tuple(
                _require_uuid(event_id, f"{path}.response_event_ids[{index}]")
                for index, event_id in enumerate(response_event_ids)
            ),
            failure_type=_require_optional_non_blank_string(
                values["failure_type"],
                f"{path}.failure_type",
            ),
            error_type=_require_optional_non_blank_string(
                values["error_type"],
                f"{path}.error_type",
            ),
        )
    if event_type == "state.reduced":
        _require_keys(
            values,
            {
                "old_version",
                "new_version",
                "state_changed",
                "applied_contribution_ids",
            },
            path,
        )
        return StateReductionPayload(
            old_version=_require_int(
                values["old_version"],
                f"{path}.old_version",
            ),
            new_version=_require_int(
                values["new_version"],
                f"{path}.new_version",
            ),
            state_changed=_require_bool(
                values["state_changed"],
                f"{path}.state_changed",
            ),
            applied_contribution_ids=_require_uuid_tuple(
                values["applied_contribution_ids"],
                f"{path}.applied_contribution_ids",
            ),
        )
    if event_type == "processing.failed":
        _require_keys(values, {"stage", "error_type"}, path)
        return ProcessingFailurePayload(
            stage=_require_non_blank_string(values["stage"], f"{path}.stage"),
            error_type=_require_non_blank_string(
                values["error_type"],
                f"{path}.error_type",
            ),
        )
    if event_type == "motive.formed":
        return MotiveFormedPayload(_motive_from_dict(values, path))
    if event_type == "proactive.intention":
        return ProactiveIntentionPayload(_intention_from_dict(values, path))
    if event_type == "behavior.decided":
        _require_keys(values, {"intention_id", "decision"}, path)
        intention_id = _require_uuid(values["intention_id"], f"{path}.intention_id")
        return BehavioralDecisionPayload(
            intention_id,
            _decision_from_dict(values["decision"], f"{path}.decision"),
        )
    if event_type == "proactive.message.created":
        _require_keys(values, {"message_id", "intention_id", "text", "evidence_refs", "created_at", "valid_until"}, path)
        refs = values["evidence_refs"]
        if not isinstance(refs, list):
            raise MalformedSerializedDataError(f"{path}.evidence_refs must be an array")
        return MailboxMessagePayload(
            _require_uuid(values["message_id"], f"{path}.message_id"),
            _require_uuid(values["intention_id"], f"{path}.intention_id"),
            _require_non_blank_string(values["text"], f"{path}.text"),
            tuple(_evidence_ref_from_dict(item, f"{path}.evidence_refs[{index}]") for index, item in enumerate(refs)),
            _require_datetime(values["created_at"], f"{path}.created_at"),
            _require_datetime(values["valid_until"], f"{path}.valid_until"),
        )
    if event_type == "proactive.message.acknowledged":
        _require_keys(values, {"message_id", "intention_id"}, path)
        return MailboxAcknowledgedPayload(
            _require_uuid(values["message_id"], f"{path}.message_id"),
            _require_uuid(values["intention_id"], f"{path}.intention_id"),
        )
    if event_type == "behavior.mode_switched":
        _require_keys(values, {"mode", "state_version"}, path)
        return BehaviorModePayload(
            _require_non_blank_string(values["mode"], f"{path}.mode"),
            _require_int(values["state_version"], f"{path}.state_version"),
        )
    if event_type == "proactivity.reassessment":
        _require_keys(values, {"wake_reason", "previous_assessment_at"}, path)
        previous = values["previous_assessment_at"]
        return ProactivityReassessmentPayload(
            _require_non_blank_string(values["wake_reason"], f"{path}.wake_reason"),
            None if previous is None else _require_datetime(previous, f"{path}.previous_assessment_at"),
        )
    raise MalformedSerializedDataError(
        "event.event_type is not supported by this schema"
    )
