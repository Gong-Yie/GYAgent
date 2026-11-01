"""Codecs for motive, proactive intention and behavioural decision values."""

from self_cognition.core.affect import Motive
from self_cognition.core.errors import MalformedSerializedDataError
from self_cognition.core.proactivity import (
    BehavioralAction,
    BehavioralDecision,
    IntentionStatus,
    ProactiveIntention,
)

from ._codec import (
    _require_datetime,
    _require_float,
    _require_int,
    _require_keys,
    _require_non_blank_string,
    _require_object,
    _require_optional_datetime,
    _require_string,
    _require_uuid,
)
from ._refs import (
    _evidence_ref_from_dict,
    _evidence_ref_to_dict,
    _subject_scope_from_dict,
    _subject_scope_to_dict,
)


def _motive_to_dict(motive: Motive) -> dict[str, object]:
    return {
        "motive_id": str(motive.motive_id),
        "subject_id": motive.subject_id,
        "kind": motive.kind,
        "description": motive.description,
        "strength": motive.strength,
        "priority": motive.priority,
        "created_at": motive.created_at.isoformat(),
        "source_event_ids": [str(item) for item in motive.source_event_ids],
        "goal_ids": list(motive.goal_ids),
        "emotion_ids": [str(item) for item in motive.emotion_ids],
        "expires_at": motive.expires_at.isoformat() if motive.expires_at else None,
    }


def _motive_from_dict(value: object, path: str) -> Motive:
    values = _require_object(value, path)
    _require_keys(
        values,
        {"motive_id", "subject_id", "kind", "description", "strength", "priority", "created_at", "source_event_ids", "goal_ids", "emotion_ids", "expires_at"},
        path,
    )
    source_ids = values["source_event_ids"]
    goal_ids = values["goal_ids"]
    emotion_ids = values["emotion_ids"]
    if not isinstance(source_ids, list) or not isinstance(goal_ids, list) or not isinstance(emotion_ids, list):
        raise MalformedSerializedDataError(f"{path} references must be arrays")
    return Motive(
        _require_uuid(values["motive_id"], f"{path}.motive_id"),
        _require_non_blank_string(values["subject_id"], f"{path}.subject_id"),
        _require_non_blank_string(values["kind"], f"{path}.kind"),
        _require_non_blank_string(values["description"], f"{path}.description"),
        _require_float(values["strength"], f"{path}.strength"),
        _require_int(values["priority"], f"{path}.priority"),
        _require_datetime(values["created_at"], f"{path}.created_at"),
        tuple(_require_uuid(item, f"{path}.source_event_ids[{index}]") for index, item in enumerate(source_ids)),
        tuple(_require_non_blank_string(item, f"{path}.goal_ids[{index}]") for index, item in enumerate(goal_ids)),
        tuple(_require_uuid(item, f"{path}.emotion_ids[{index}]") for index, item in enumerate(emotion_ids)),
        _require_optional_datetime(values["expires_at"], f"{path}.expires_at"),
    )


def _intention_to_dict(intention: ProactiveIntention) -> dict[str, object]:
    return {
        "intention_id": str(intention.intention_id),
        "motive": _motive_to_dict(intention.motive),
        "target": _subject_scope_to_dict(intention.target),
        "expected_behavior": intention.expected_behavior,
        "priority": intention.priority,
        "state_version": intention.state_version,
        "evidence_refs": [_evidence_ref_to_dict(ref) for ref in intention.evidence_refs],
        "created_at": intention.created_at.isoformat(),
        "valid_until": intention.valid_until.isoformat(),
        "budget": intention.budget,
        "stop_conditions": list(intention.stop_conditions),
        "idempotency_key": intention.idempotency_key,
        "status": intention.status.value,
    }


def _intention_from_dict(value: object, path: str) -> ProactiveIntention:
    values = _require_object(value, path)
    _require_keys(
        values,
        {"intention_id", "motive", "target", "expected_behavior", "priority", "state_version", "evidence_refs", "created_at", "valid_until", "budget", "stop_conditions", "idempotency_key", "status"},
        path,
    )
    refs = values["evidence_refs"]
    stops = values["stop_conditions"]
    if not isinstance(refs, list) or not isinstance(stops, list):
        raise MalformedSerializedDataError(f"{path}.evidence_refs and stop_conditions must be arrays")
    return ProactiveIntention(
        _require_uuid(values["intention_id"], f"{path}.intention_id"),
        _motive_from_dict(values["motive"], f"{path}.motive"),
        _subject_scope_from_dict(values["target"], f"{path}.target"),
        _require_non_blank_string(values["expected_behavior"], f"{path}.expected_behavior"),
        _require_int(values["priority"], f"{path}.priority"),
        _require_int(values["state_version"], f"{path}.state_version"),
        tuple(_evidence_ref_from_dict(item, f"{path}.evidence_refs[{index}]") for index, item in enumerate(refs)),
        _require_datetime(values["created_at"], f"{path}.created_at"),
        _require_datetime(values["valid_until"], f"{path}.valid_until"),
        _require_int(values["budget"], f"{path}.budget"),
        tuple(_require_non_blank_string(item, f"{path}.stop_conditions[{index}]") for index, item in enumerate(stops)),
        _require_non_blank_string(values["idempotency_key"], f"{path}.idempotency_key"),
        IntentionStatus(_require_string(values["status"], f"{path}.status")),
    )


def _decision_to_dict(decision: BehavioralDecision) -> dict[str, object]:
    return {
        "decision_id": str(decision.decision_id),
        "intention_id": str(decision.intention_id),
        "action": decision.action.value,
        "reason": decision.reason,
        "evidence_refs": [_evidence_ref_to_dict(ref) for ref in decision.evidence_refs],
        "state_version": decision.state_version,
        "decided_at": decision.decided_at.isoformat(),
        "idempotency_key": decision.idempotency_key,
        "not_before": decision.not_before.isoformat() if decision.not_before else None,
        "merged_into": str(decision.merged_into) if decision.merged_into else None,
        "response": decision.response,
    }


def _decision_from_dict(value: object, path: str) -> BehavioralDecision:
    values = _require_object(value, path)
    _require_keys(
        values,
        {"decision_id", "intention_id", "action", "reason", "evidence_refs", "state_version", "decided_at", "idempotency_key", "not_before", "merged_into", "response"},
        path,
    )
    refs = values["evidence_refs"]
    if not isinstance(refs, list):
        raise MalformedSerializedDataError(f"{path}.evidence_refs must be an array")
    try:
        action = BehavioralAction(_require_string(values["action"], f"{path}.action"))
    except ValueError as error:
        raise MalformedSerializedDataError(f"invalid {path}.action") from error
    merged = values["merged_into"]
    return BehavioralDecision(
        _require_uuid(values["decision_id"], f"{path}.decision_id"),
        _require_uuid(values["intention_id"], f"{path}.intention_id"),
        action,
        _require_non_blank_string(values["reason"], f"{path}.reason"),
        tuple(_evidence_ref_from_dict(item, f"{path}.evidence_refs[{index}]") for index, item in enumerate(refs)),
        _require_int(values["state_version"], f"{path}.state_version"),
        _require_datetime(values["decided_at"], f"{path}.decided_at"),
        _require_non_blank_string(values["idempotency_key"], f"{path}.idempotency_key"),
        _require_optional_datetime(values["not_before"], f"{path}.not_before"),
        None if merged is None else _require_uuid(merged, f"{path}.merged_into"),
        None if values["response"] is None else _require_non_blank_string(values["response"], f"{path}.response"),
    )
