import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from self_cognition.core.contributions import (
    CognitiveContribution,
    CognitionType,
    ContributionOperation,
)
from self_cognition.core.errors import (
    MalformedSerializedDataError,
    SerializationError,
    UnsupportedSchemaVersionError,
)
from self_cognition.core.evidence import EvidenceRef, EvidenceSourceKind
from self_cognition.core.events import (
    CapabilityObservationPayload,
    CognitionModuleResultPayload,
    CognitionCorrectionPayload,
    EVENT_SCHEMA_VERSION,
    EventEnvelope,
    EventSource,
    ModelResponsePayload,
    ProcessingFailurePayload,
    SelfModelObservationPayload,
    StateReductionPayload,
    UserMessagePayload,
)
from self_cognition.core.identity import (
    CapabilityRecord,
    GoalRecord,
    LimitationRecord,
    SelfModelAspect,
)
from self_cognition.core.memories import (
    MemoryAccessRecord,
    MemoryConsolidationStatus,
    MemoryCues,
    MemoryLifecycleStatus,
    MemoryRecord,
    MemorySourceRef,
    MemoryType,
)
from self_cognition.core.scopes import (
    DEFAULT_MIND_ID,
    ConversationScope,
    DataScope,
    DisclosureScope,
    MindScope,
    SubjectKind,
    SubjectRef,
    SubjectScope,
)
from self_cognition.core.state import (
    ConflictRecord,
    StateAtom,
    StateChangeRecord,
    StateDecisionStatus,
    SubjectState,
)


STATE_SCHEMA_VERSION = 4
MEMORY_SCHEMA_VERSION = 4
MEMORY_ACCESS_SCHEMA_VERSION = 1
LEGACY_STATE_TIME = datetime(1970, 1, 1, tzinfo=timezone.utc)


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


def memory_to_dict(record: MemoryRecord) -> dict[str, Any]:
    return {
        "schema_version": MEMORY_SCHEMA_VERSION,
        "memory_id": str(record.memory_id),
        "memory_type": record.memory_type.value,
        "subject": _subject_scope_to_dict(record.subject),
        "scope": _data_scope_to_dict(record.scope),
        "content": record.content,
        "evidence_refs": [
            _evidence_ref_to_dict(evidence)
            for evidence in record.evidence_refs
        ],
        "confidence": record.confidence,
        "salience": record.salience,
        "stability": record.stability,
        "retrievability": record.retrievability,
        "version": record.version,
        "lifecycle_status": record.lifecycle_status.value,
        "created_at": record.created_at.isoformat(),
        "source_module": record.source_module,
        "source_module_version": record.source_module_version,
        "sources": [
            {
                "contribution_id": str(source.contribution_id),
                "old_state_version": source.old_state_version,
                "new_state_version": source.new_state_version,
                "target_field": source.target_field,
            }
            for source in record.sources
        ],
        "cues": {
            "people": list(record.cues.people),
            "topics": list(record.cues.topics),
            "time_keys": list(record.cues.time_keys),
            "relationships": list(record.cues.relationships),
            "tasks": list(record.cues.tasks),
        },
        "consolidation_status": record.consolidation_status.value,
        "expires_at": (
            record.expires_at.isoformat()
            if record.expires_at is not None
            else None
        ),
        "lifecycle_changed_at": (
            record.lifecycle_changed_at.isoformat()
            if record.lifecycle_changed_at is not None
            else None
        ),
        "lifecycle_reason": record.lifecycle_reason,
    }


def memory_from_dict(data: object) -> MemoryRecord:
    values = _require_object(data, "memory")
    schema_version = _require_supported_memory_schema(values)
    expected_keys = {
        "schema_version",
        "memory_id",
        "memory_type",
        "subject",
        "scope",
        "content",
        "evidence_refs",
        "confidence",
        "salience",
        "stability",
        "retrievability",
        "version",
        "lifecycle_status",
        "created_at",
        "source_module",
        "source_module_version",
    }
    if schema_version >= 2:
        expected_keys.add("sources")
    if schema_version >= 3:
        expected_keys.update({"cues", "consolidation_status"})
    else:
        expected_keys.update(
            key
            for key in ("cues", "consolidation_status")
            if key in values
        )
    if schema_version >= 4:
        expected_keys.update(
            {"expires_at", "lifecycle_changed_at", "lifecycle_reason"}
        )
    else:
        expected_keys.update(
            key
            for key in ("expires_at", "lifecycle_changed_at", "lifecycle_reason")
            if key in values
        )
    _require_keys(
        values,
        expected_keys,
        "memory",
    )
    try:
        return MemoryRecord(
            memory_id=_require_uuid(values["memory_id"], "memory.memory_id"),
            memory_type=MemoryType(
                _require_string(values["memory_type"], "memory.memory_type")
            ),
            subject=_subject_scope_from_dict(values["subject"], "memory.subject"),
            scope=_data_scope_from_dict(values["scope"], "memory.scope"),
            content=values["content"],
            evidence_refs=_require_evidence_ref_tuple(
                values["evidence_refs"],
                "memory.evidence_refs",
            ),
            confidence=_require_float(values["confidence"], "memory.confidence"),
            salience=_require_float(values["salience"], "memory.salience"),
            stability=_require_float(values["stability"], "memory.stability"),
            retrievability=_require_float(
                values["retrievability"],
                "memory.retrievability",
            ),
            version=_require_int(values["version"], "memory.version"),
            lifecycle_status=MemoryLifecycleStatus(
                _require_string(
                    values["lifecycle_status"],
                    "memory.lifecycle_status",
                )
            ),
            created_at=_require_datetime(values["created_at"], "memory.created_at"),
            source_module=_require_non_blank_string(
                values["source_module"],
                "memory.source_module",
            ),
            source_module_version=_require_non_blank_string(
                values["source_module_version"],
                "memory.source_module_version",
            ),
            sources=(
                _memory_sources_from_list(values["sources"])
                if schema_version >= 2
                else ()
            ),
            cues=(
                _memory_cues_from_dict(values["cues"])
                if "cues" in values
                else MemoryCues()
            ),
            consolidation_status=(
                MemoryConsolidationStatus(
                    _require_string(
                        values["consolidation_status"],
                        "memory.consolidation_status",
                    )
                )
                if "consolidation_status" in values
                else MemoryConsolidationStatus.RAW
            ),
            expires_at=(
                _require_optional_datetime(
                    values["expires_at"],
                    "memory.expires_at",
                )
                if schema_version >= 4
                else None
            ),
            lifecycle_changed_at=(
                _require_optional_datetime(
                    values["lifecycle_changed_at"],
                    "memory.lifecycle_changed_at",
                )
                if schema_version >= 4
                else None
            ),
            lifecycle_reason=(
                _require_optional_non_blank_string(
                    values["lifecycle_reason"],
                    "memory.lifecycle_reason",
                )
                if schema_version >= 4
                else None
            ),
        )
    except SerializationError:
        raise
    except Exception as error:
        raise MalformedSerializedDataError("invalid memory values") from error


def memory_to_json(record: MemoryRecord) -> str:
    return _to_json(memory_to_dict(record), "memory")


def memory_from_json(payload: str) -> MemoryRecord:
    return memory_from_dict(_from_json(payload, "memory"))


def _memory_sources_from_list(value: object) -> tuple[MemorySourceRef, ...]:
    if not isinstance(value, list):
        raise MalformedSerializedDataError("memory.sources must be an array")
    sources: list[MemorySourceRef] = []
    for index, raw_source in enumerate(value):
        path = f"memory.sources[{index}]"
        values = _require_object(raw_source, path)
        _require_keys(
            values,
            {
                "contribution_id",
                "old_state_version",
                "new_state_version",
                "target_field",
            },
            path,
        )
        try:
            sources.append(
                MemorySourceRef(
                    contribution_id=_require_uuid(
                        values["contribution_id"],
                        f"{path}.contribution_id",
                    ),
                    old_state_version=_require_int(
                        values["old_state_version"],
                        f"{path}.old_state_version",
                    ),
                    new_state_version=_require_int(
                        values["new_state_version"],
                        f"{path}.new_state_version",
                    ),
                    target_field=_require_non_blank_string(
                        values["target_field"],
                        f"{path}.target_field",
                    ),
                )
            )
        except SerializationError:
            raise
        except Exception as error:
            raise MalformedSerializedDataError(
                f"invalid {path} values"
            ) from error
    return tuple(sources)


def _memory_cues_from_dict(value: object) -> MemoryCues:
    values = _require_object(value, "memory.cues")
    names = {"people", "topics", "time_keys", "relationships", "tasks"}
    _require_keys(values, names, "memory.cues")
    return MemoryCues(
        **{
            name: _require_string_tuple(values[name], f"memory.cues.{name}")
            for name in names
        }
    )


def memory_access_to_dict(record: MemoryAccessRecord) -> dict[str, Any]:
    return {
        "schema_version": MEMORY_ACCESS_SCHEMA_VERSION,
        "access_id": str(record.access_id),
        "memory_id": str(record.memory_id),
        "subject": _subject_scope_to_dict(record.subject),
        "accessed_at": record.accessed_at.isoformat(),
        "purpose": record.purpose,
        "context": record.context,
    }


def memory_access_from_dict(data: object) -> MemoryAccessRecord:
    values = _require_object(data, "memory_access")
    _require_schema(values, MEMORY_ACCESS_SCHEMA_VERSION, "memory_access")
    _require_keys(
        values,
        {
            "schema_version",
            "access_id",
            "memory_id",
            "subject",
            "accessed_at",
            "purpose",
            "context",
        },
        "memory_access",
    )
    try:
        return MemoryAccessRecord(
            access_id=_require_uuid(values["access_id"], "memory_access.access_id"),
            memory_id=_require_uuid(values["memory_id"], "memory_access.memory_id"),
            subject=_subject_scope_from_dict(
                values["subject"], "memory_access.subject"
            ),
            accessed_at=_require_datetime(
                values["accessed_at"], "memory_access.accessed_at"
            ),
            purpose=_require_non_blank_string(
                values["purpose"], "memory_access.purpose"
            ),
            context=_require_non_blank_string(
                values["context"], "memory_access.context"
            ),
        )
    except SerializationError:
        raise
    except Exception as error:
        raise MalformedSerializedDataError(
            "invalid memory access values"
        ) from error


def memory_access_to_json(record: MemoryAccessRecord) -> str:
    return _to_json(memory_access_to_dict(record), "memory_access")


def memory_access_from_json(payload: str) -> MemoryAccessRecord:
    return memory_access_from_dict(_from_json(payload, "memory_access"))


def _event_payload_to_dict(
    payload: (
        UserMessagePayload
        | CognitionCorrectionPayload
        | SelfModelObservationPayload
        | CapabilityObservationPayload
        | ModelResponsePayload
        | CognitionModuleResultPayload
        | StateReductionPayload
        | ProcessingFailurePayload
    ),
) -> dict[str, object]:
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
    | CognitionCorrectionPayload
    | SelfModelObservationPayload
    | CapabilityObservationPayload
    | ModelResponsePayload
    | CognitionModuleResultPayload
    | StateReductionPayload
    | ProcessingFailurePayload
):
    values = _require_object(value, path)
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
    raise MalformedSerializedDataError(
        "event.event_type is not supported by this schema"
    )


def state_to_dict(state: SubjectState) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "mind_id": state.mind_id,
        "subject_kind": state.subject_kind.value,
        "subject_id": state.subject_id,
        "version": state.version,
        "entries": {
            target_field: {
                "value": entry.value,
                "cognition_type": entry.cognition_type.value,
                "confidence": entry.confidence,
                "scope": _data_scope_to_dict(entry.scope),
                "evidence_refs": [
                    _evidence_ref_to_dict(evidence)
                    for evidence in entry.evidence_refs
                ],
                "contribution_ids": [
                    str(contribution_id)
                    for contribution_id in entry.contribution_ids
                ],
                "created_at": entry.created_at.isoformat(),
                "valid_from": entry.valid_from.isoformat(),
                "expires_at": (
                    entry.expires_at.isoformat()
                    if entry.expires_at is not None
                    else None
                ),
            }
            for target_field, entry in sorted(state.entries.items())
        },
        "applied_contribution_ids": sorted(
            str(contribution_id)
            for contribution_id in state.applied_contribution_ids
        ),
        "conflicts": [
            {
                "target_field": conflict.target_field,
                "candidate_contribution_ids": [
                    str(contribution_id)
                    for contribution_id in conflict.candidate_contribution_ids
                ],
                "reason": conflict.reason,
            }
            for conflict in sorted(
                state.conflicts,
                key=lambda item: (
                    item.target_field,
                    tuple(item.candidate_contribution_ids),
                    item.reason,
                ),
            )
        ],
        "changes": [
            {
                "contribution": _contribution_to_dict(change.contribution),
                "status": change.status.value,
                "reason": change.reason,
                "old_version": change.old_version,
                "new_version": change.new_version,
                "decided_at": change.decided_at.isoformat(),
            }
            for change in state.changes
        ],
    }


def state_from_dict(data: object) -> SubjectState:
    values = _require_object(data, "state")
    schema_version = _require_supported_state_schema(values)
    scope_fields = (
        {"mind_id", "subject_kind"}
        if schema_version >= 2
        else set()
    )
    change_fields = {"changes"} if schema_version == STATE_SCHEMA_VERSION else set()
    _require_keys(
        values,
        {
            "schema_version",
            *scope_fields,
            "subject_id",
            "version",
            "entries",
            "applied_contribution_ids",
            "conflicts",
            *change_fields,
        },
        "state",
    )

    if schema_version == 1:
        mind_id = DEFAULT_MIND_ID
        subject_kind = SubjectKind.USER
    else:
        mind_id = _require_non_blank_string(
            values["mind_id"],
            "state.mind_id",
        )
        try:
            subject_kind = SubjectKind(
                _require_string(values["subject_kind"], "state.subject_kind")
            )
        except ValueError as error:
            raise MalformedSerializedDataError(
                "state.subject_kind must be a supported subject kind"
            ) from error

    subject_id = _require_non_blank_string(
        values["subject_id"],
        "state.subject_id",
    )
    if subject_kind is SubjectKind.MIND and subject_id != mind_id:
        raise MalformedSerializedDataError(
            "state mind subject ID must match state.mind_id"
        )
    version = _require_int(values["version"], "state.version")
    subject_scope = SubjectScope(
        mind=MindScope(mind_id),
        subject=SubjectRef(subject_kind, subject_id),
    )
    entries_value = values["entries"]
    if not isinstance(entries_value, dict):
        raise MalformedSerializedDataError("state.entries must be an object")

    entries: dict[str, StateAtom] = {}
    for target_field, raw_entry in entries_value.items():
        if not isinstance(target_field, str) or not target_field.strip():
            raise MalformedSerializedDataError(
                "state.entries keys must be non-blank strings"
            )
        entry = _require_object(raw_entry, f"state.entries[{target_field!r}]")
        evidence_field = (
            "evidence_refs" if schema_version >= 3 else "evidence_event_ids"
        )
        metadata_fields = (
            {
                "cognition_type",
                "scope",
                "created_at",
                "valid_from",
                "expires_at",
            }
            if schema_version == STATE_SCHEMA_VERSION
            else set()
        )
        _require_keys(
            entry,
            {
                "value",
                "confidence",
                evidence_field,
                "contribution_ids",
                *metadata_fields,
            },
            f"state.entries[{target_field!r}]",
        )
        if schema_version >= 3:
            evidence_refs = _require_evidence_ref_tuple(
                entry["evidence_refs"],
                f"state.entries[{target_field!r}].evidence_refs",
            )
        else:
            legacy_event_ids = _require_uuid_tuple(
                entry["evidence_event_ids"],
                f"state.entries[{target_field!r}].evidence_event_ids",
            )
            evidence_refs = tuple(
                _legacy_event_evidence(event_id, subject_scope)
                for event_id in legacy_event_ids
            )
        if schema_version == STATE_SCHEMA_VERSION:
            try:
                cognition_type = CognitionType(
                    _require_string(
                        entry["cognition_type"],
                        f"state.entries[{target_field!r}].cognition_type",
                    )
                )
            except ValueError as error:
                raise MalformedSerializedDataError(
                    f"state.entries[{target_field!r}].cognition_type is invalid"
                ) from error
            atom_scope = _data_scope_from_dict(
                entry["scope"],
                f"state.entries[{target_field!r}].scope",
            )
            created_at = _require_datetime(
                entry["created_at"],
                f"state.entries[{target_field!r}].created_at",
            )
            valid_from = _require_datetime(
                entry["valid_from"],
                f"state.entries[{target_field!r}].valid_from",
            )
            expires_at = _require_optional_datetime(
                entry["expires_at"],
                f"state.entries[{target_field!r}].expires_at",
            )
        else:
            cognition_type = CognitionType.UNKNOWN
            atom_scope = DataScope(
                owner=subject_scope,
                disclosure=DisclosureScope.PRIVATE,
            )
            created_at = LEGACY_STATE_TIME
            valid_from = LEGACY_STATE_TIME
            expires_at = None
        entries[target_field] = StateAtom(
            value=entry["value"],
            cognition_type=cognition_type,
            confidence=_require_float(
                entry["confidence"],
                f"state.entries[{target_field!r}].confidence",
            ),
            scope=atom_scope,
            evidence_refs=evidence_refs,
            contribution_ids=_require_uuid_tuple(
                entry["contribution_ids"],
                f"state.entries[{target_field!r}].contribution_ids",
            ),
            created_at=created_at,
            valid_from=valid_from,
            expires_at=expires_at,
        )

    applied_ids = frozenset(
        _require_uuid_tuple(
            values["applied_contribution_ids"],
            "state.applied_contribution_ids",
        )
    )
    conflicts_value = values["conflicts"]
    if not isinstance(conflicts_value, list):
        raise MalformedSerializedDataError("state.conflicts must be an array")
    conflicts: set[ConflictRecord] = set()
    for index, raw_conflict in enumerate(conflicts_value):
        conflict = _require_object(raw_conflict, f"state.conflicts[{index}]")
        _require_keys(
            conflict,
            {"target_field", "candidate_contribution_ids", "reason"},
            f"state.conflicts[{index}]",
        )
        conflicts.add(
            ConflictRecord(
                target_field=_require_string(
                    conflict["target_field"],
                    f"state.conflicts[{index}].target_field",
                ),
                candidate_contribution_ids=_require_uuid_tuple(
                    conflict["candidate_contribution_ids"],
                    f"state.conflicts[{index}].candidate_contribution_ids",
                ),
                reason=_require_string(
                    conflict["reason"],
                    f"state.conflicts[{index}].reason",
                ),
            )
        )

    changes = (
        _state_changes_from_list(values["changes"])
        if schema_version == STATE_SCHEMA_VERSION
        else ()
    )

    return SubjectState(
        subject_id=subject_id,
        version=version,
        entries=entries,
        applied_contribution_ids=applied_ids,
        conflicts=frozenset(conflicts),
        changes=changes,
        mind_id=mind_id,
        subject_kind=subject_kind,
    )


def state_to_json(state: SubjectState) -> str:
    return _to_json(state_to_dict(state), "state")


def state_from_json(payload: str) -> SubjectState:
    return state_from_dict(_from_json(payload, "state"))


def _contribution_to_dict(
    contribution: CognitiveContribution,
) -> dict[str, object]:
    return {
        "contribution_id": str(contribution.contribution_id),
        "target": _subject_scope_to_dict(contribution.target),
        "target_field": contribution.target_field,
        "operation": contribution.operation.value,
        "cognition_type": contribution.cognition_type.value,
        "value": contribution.value,
        "confidence": contribution.confidence,
        "evidence_refs": [
            _evidence_ref_to_dict(evidence)
            for evidence in contribution.evidence_refs
        ],
        "source_module": contribution.source_module,
        "module_version": contribution.module_version,
        "scope": _data_scope_to_dict(contribution.scope),
        "created_at": contribution.created_at.isoformat(),
        "valid_from": contribution.valid_from.isoformat(),
        "expires_at": (
            contribution.expires_at.isoformat()
            if contribution.expires_at is not None
            else None
        ),
        "target_version": contribution.target_version,
        "explicitly_confirmed": contribution.explicitly_confirmed,
    }


def _contribution_from_dict(value: object, path: str) -> CognitiveContribution:
    values = _require_object(value, path)
    _require_keys(
        values,
        {
            "contribution_id",
            "target",
            "target_field",
            "operation",
            "cognition_type",
            "value",
            "confidence",
            "evidence_refs",
            "source_module",
            "module_version",
            "scope",
            "created_at",
            "valid_from",
            "expires_at",
            "target_version",
            "explicitly_confirmed",
        },
        path,
    )
    try:
        return CognitiveContribution(
            contribution_id=_require_uuid(
                values["contribution_id"],
                f"{path}.contribution_id",
            ),
            target=_subject_scope_from_dict(values["target"], f"{path}.target"),
            target_field=_require_non_blank_string(
                values["target_field"],
                f"{path}.target_field",
            ),
            operation=ContributionOperation(
                _require_string(values["operation"], f"{path}.operation")
            ),
            cognition_type=CognitionType(
                _require_string(
                    values["cognition_type"],
                    f"{path}.cognition_type",
                )
            ),
            value=values["value"],
            confidence=_require_float(values["confidence"], f"{path}.confidence"),
            evidence_refs=_require_evidence_ref_tuple(
                values["evidence_refs"],
                f"{path}.evidence_refs",
            ),
            source_module=_require_non_blank_string(
                values["source_module"],
                f"{path}.source_module",
            ),
            module_version=_require_non_blank_string(
                values["module_version"],
                f"{path}.module_version",
            ),
            scope=_data_scope_from_dict(values["scope"], f"{path}.scope"),
            created_at=_require_datetime(
                values["created_at"],
                f"{path}.created_at",
            ),
            valid_from=_require_datetime(
                values["valid_from"],
                f"{path}.valid_from",
            ),
            expires_at=_require_optional_datetime(
                values["expires_at"],
                f"{path}.expires_at",
            ),
            target_version=_require_optional_int(
                values["target_version"],
                f"{path}.target_version",
            ),
            explicitly_confirmed=_require_bool(
                values["explicitly_confirmed"],
                f"{path}.explicitly_confirmed",
            ),
        )
    except SerializationError:
        raise
    except Exception as error:
        raise MalformedSerializedDataError(f"invalid {path} values") from error


def _state_changes_from_list(value: object) -> tuple[StateChangeRecord, ...]:
    if not isinstance(value, list):
        raise MalformedSerializedDataError("state.changes must be an array")
    changes: list[StateChangeRecord] = []
    for index, raw_change in enumerate(value):
        path = f"state.changes[{index}]"
        change = _require_object(raw_change, path)
        _require_keys(
            change,
            {
                "contribution",
                "status",
                "reason",
                "old_version",
                "new_version",
                "decided_at",
            },
            path,
        )
        try:
            changes.append(
                StateChangeRecord(
                    contribution=_contribution_from_dict(
                        change["contribution"],
                        f"{path}.contribution",
                    ),
                    status=StateDecisionStatus(
                        _require_string(change["status"], f"{path}.status")
                    ),
                    reason=_require_non_blank_string(
                        change["reason"],
                        f"{path}.reason",
                    ),
                    old_version=_require_int(
                        change["old_version"],
                        f"{path}.old_version",
                    ),
                    new_version=_require_int(
                        change["new_version"],
                        f"{path}.new_version",
                    ),
                    decided_at=_require_datetime(
                        change["decided_at"],
                        f"{path}.decided_at",
                    ),
                )
            )
        except SerializationError:
            raise
        except Exception as error:
            raise MalformedSerializedDataError(f"invalid {path} values") from error
    return tuple(changes)


def _to_json(data: dict[str, Any], kind: str) -> str:
    try:
        return json.dumps(
            data,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise SerializationError(f"{kind} contains non-JSON data") from error


def _from_json(payload: str, kind: str) -> object:
    if not isinstance(payload, str):
        raise MalformedSerializedDataError(f"{kind} JSON must be text")
    try:
        return json.loads(
            payload,
            parse_constant=lambda value: _reject_json_constant(value, kind),
        )
    except (json.JSONDecodeError, MalformedSerializedDataError) as error:
        if isinstance(error, MalformedSerializedDataError):
            raise
        raise MalformedSerializedDataError(f"malformed {kind} JSON") from error


def _require_object(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MalformedSerializedDataError(f"{path} must be an object")
    return value


def _require_schema(values: dict[str, Any], expected: int, path: str) -> None:
    version = values.get("schema_version")
    if version != expected:
        if isinstance(version, int) and not isinstance(version, bool):
            raise UnsupportedSchemaVersionError(
                f"unsupported {path} schema version: {version}"
            )
        raise MalformedSerializedDataError(
            f"{path}.schema_version must be {expected}"
        )


def _require_supported_memory_schema(values: dict[str, Any]) -> int:
    version = values.get("schema_version")
    if version in (1, 2, 3, MEMORY_SCHEMA_VERSION) and not isinstance(version, bool):
        return version
    if isinstance(version, int) and not isinstance(version, bool):
        raise UnsupportedSchemaVersionError(
            f"unsupported memory schema version: {version}"
        )
    raise MalformedSerializedDataError(
        "memory.schema_version must be 1, 2, 3 or 4"
    )


def _require_supported_state_schema(values: dict[str, Any]) -> int:
    version = values.get("schema_version")
    if version in (1, 2, 3, STATE_SCHEMA_VERSION) and not isinstance(version, bool):
        return version
    if isinstance(version, int) and not isinstance(version, bool):
        raise UnsupportedSchemaVersionError(
            f"unsupported state schema version: {version}"
        )
    raise MalformedSerializedDataError(
        "state.schema_version must be 1, 2, 3, or 4"
    )


def _require_keys(
    values: dict[str, Any],
    expected: set[str],
    path: str,
) -> None:
    actual = set(values)
    missing = expected - actual
    unknown = actual - expected
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing={sorted(missing)}")
        if unknown:
            details.append(f"unknown={sorted(unknown)}")
        raise MalformedSerializedDataError(
            f"invalid {path} fields ({', '.join(details)})"
        )


def _require_string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise MalformedSerializedDataError(f"{path} must be a string")
    return value


def _require_string_tuple(value: object, path: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise MalformedSerializedDataError(f"{path} must be an array")
    result = tuple(
        _require_non_blank_string(item, f"{path}[{index}]")
        for index, item in enumerate(value)
    )
    return result


def _require_non_blank_string(value: object, path: str) -> str:
    text = _require_string(value, path)
    if not text.strip():
        raise MalformedSerializedDataError(f"{path} must not be blank")
    return text


def _require_optional_non_blank_string(
    value: object,
    path: str,
) -> str | None:
    if value is None:
        return None
    return _require_non_blank_string(value, path)


def _require_int(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise MalformedSerializedDataError(f"{path} must be an integer")
    return value


def _require_optional_int(value: object, path: str) -> int | None:
    if value is None:
        return None
    return _require_int(value, path)


def _require_bool(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise MalformedSerializedDataError(f"{path} must be a boolean")
    return value


def _require_float(value: object, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise MalformedSerializedDataError(f"{path} must be a number")
    return float(value)


def _require_uuid(value: object, path: str) -> UUID:
    try:
        return UUID(_require_string(value, path))
    except (ValueError, TypeError) as error:
        raise MalformedSerializedDataError(f"{path} must be a UUID string") from error


def _require_uuid_tuple(value: object, path: str) -> tuple[UUID, ...]:
    if not isinstance(value, list):
        raise MalformedSerializedDataError(f"{path} must be an array")
    return tuple(
        _require_uuid(item, f"{path}[{index}]")
        for index, item in enumerate(value)
    )


def _require_optional_uuid(value: object, path: str) -> UUID | None:
    if value is None:
        return None
    return _require_uuid(value, path)


def _optional_uuid_to_string(value: UUID | None) -> str | None:
    return str(value) if value is not None else None


def _require_datetime(value: object, path: str) -> datetime:
    try:
        result = datetime.fromisoformat(_require_string(value, path))
    except ValueError as error:
        raise MalformedSerializedDataError(
            f"{path} must be a valid ISO 8601 datetime"
        ) from error
    if result.tzinfo is None or result.utcoffset() is None:
        raise MalformedSerializedDataError(
            f"{path} must include timezone information"
        )
    return result


def _require_optional_datetime(
    value: object,
    path: str,
) -> datetime | None:
    if value is None:
        return None
    return _require_datetime(value, path)


def _require_event_source(value: object, path: str) -> EventSource:
    try:
        return EventSource(_require_string(value, path))
    except ValueError as error:
        raise MalformedSerializedDataError(
            f"{path} must be a supported event source"
        ) from error


def _subject_ref_to_dict(subject: SubjectRef) -> dict[str, str]:
    return {"kind": subject.kind.value, "subject_id": subject.subject_id}


def _subject_ref_from_dict(value: object, path: str) -> SubjectRef:
    values = _require_object(value, path)
    _require_keys(values, {"kind", "subject_id"}, path)
    try:
        kind = SubjectKind(_require_string(values["kind"], f"{path}.kind"))
        return SubjectRef(
            kind=kind,
            subject_id=_require_non_blank_string(
                values["subject_id"],
                f"{path}.subject_id",
            ),
        )
    except ValueError as error:
        raise MalformedSerializedDataError(
            f"{path}.kind must be a supported subject kind"
        ) from error


def _subject_scope_to_dict(subject: SubjectScope) -> dict[str, str]:
    return {
        "mind_id": subject.mind.mind_id,
        "kind": subject.subject.kind.value,
        "subject_id": subject.subject.subject_id,
    }


def _subject_scope_from_dict(value: object, path: str) -> SubjectScope:
    values = _require_object(value, path)
    _require_keys(values, {"mind_id", "kind", "subject_id"}, path)
    try:
        return SubjectScope(
            mind=MindScope(
                _require_non_blank_string(
                    values["mind_id"],
                    f"{path}.mind_id",
                )
            ),
            subject=SubjectRef(
                SubjectKind(
                    _require_string(values["kind"], f"{path}.kind")
                ),
                _require_non_blank_string(
                    values["subject_id"],
                    f"{path}.subject_id",
                ),
            ),
        )
    except ValueError as error:
        raise MalformedSerializedDataError(
            f"{path}.kind must be a supported subject kind"
        ) from error


def _data_scope_to_dict(scope: DataScope) -> dict[str, object]:
    conversation = scope.conversation
    return {
        "owner": _subject_scope_to_dict(scope.owner),
        "disclosure": scope.disclosure.value,
        "conversation": (
            {
                "conversation_id": conversation.conversation_id,
                "group_id": conversation.group_id,
            }
            if conversation is not None
            else None
        ),
    }


def _data_scope_from_dict(value: object, path: str) -> DataScope:
    values = _require_object(value, path)
    _require_keys(values, {"owner", "disclosure", "conversation"}, path)
    conversation_value = values["conversation"]
    conversation: ConversationScope | None = None
    if conversation_value is not None:
        raw_conversation = _require_object(
            conversation_value,
            f"{path}.conversation",
        )
        _require_keys(
            raw_conversation,
            {"conversation_id", "group_id"},
            f"{path}.conversation",
        )
        group_id_value = raw_conversation["group_id"]
        conversation = ConversationScope(
            conversation_id=_require_non_blank_string(
                raw_conversation["conversation_id"],
                f"{path}.conversation.conversation_id",
            ),
            group_id=(
                None
                if group_id_value is None
                else _require_non_blank_string(
                    group_id_value,
                    f"{path}.conversation.group_id",
                )
            ),
        )
    try:
        disclosure = DisclosureScope(
            _require_string(values["disclosure"], f"{path}.disclosure")
        )
    except ValueError as error:
        raise MalformedSerializedDataError(
            f"{path}.disclosure must be a supported disclosure scope"
        ) from error
    return DataScope(
        owner=_subject_scope_from_dict(values["owner"], f"{path}.owner"),
        disclosure=disclosure,
        conversation=conversation,
    )


def _evidence_ref_to_dict(evidence: EvidenceRef) -> dict[str, object]:
    return {
        "evidence_id": str(evidence.evidence_id),
        "source_kind": evidence.source_kind.value,
        "source_ref": evidence.source_ref,
        "scope": _data_scope_to_dict(evidence.scope),
        "locator": evidence.locator,
        "excerpt": evidence.excerpt,
        "observed_at": (
            evidence.observed_at.isoformat()
            if evidence.observed_at is not None
            else None
        ),
        "reliability": evidence.reliability,
    }


def _evidence_ref_from_dict(value: object, path: str) -> EvidenceRef:
    values = _require_object(value, path)
    _require_keys(
        values,
        {
            "evidence_id",
            "source_kind",
            "source_ref",
            "scope",
            "locator",
            "excerpt",
            "observed_at",
            "reliability",
        },
        path,
    )
    try:
        source_kind = EvidenceSourceKind(
            _require_string(values["source_kind"], f"{path}.source_kind")
        )
    except ValueError as error:
        raise MalformedSerializedDataError(
            f"{path}.source_kind must be a supported evidence source kind"
        ) from error
    locator = values["locator"]
    excerpt = values["excerpt"]
    reliability_value = values["reliability"]
    try:
        return EvidenceRef(
            evidence_id=_require_uuid(
                values["evidence_id"],
                f"{path}.evidence_id",
            ),
            source_kind=source_kind,
            source_ref=_require_non_blank_string(
                values["source_ref"],
                f"{path}.source_ref",
            ),
            scope=_data_scope_from_dict(values["scope"], f"{path}.scope"),
            locator=(
                None
                if locator is None
                else _require_non_blank_string(locator, f"{path}.locator")
            ),
            excerpt=(
                None
                if excerpt is None
                else _require_non_blank_string(excerpt, f"{path}.excerpt")
            ),
            observed_at=_require_optional_datetime(
                values["observed_at"],
                f"{path}.observed_at",
            ),
            reliability=(
                None
                if reliability_value is None
                else _require_float(reliability_value, f"{path}.reliability")
            ),
        )
    except SerializationError:
        raise
    except Exception as error:
        raise MalformedSerializedDataError(f"invalid {path} values") from error


def _require_evidence_ref_tuple(
    value: object,
    path: str,
) -> tuple[EvidenceRef, ...]:
    if not isinstance(value, list):
        raise MalformedSerializedDataError(f"{path} must be an array")
    return tuple(
        _evidence_ref_from_dict(item, f"{path}[{index}]")
        for index, item in enumerate(value)
    )


def _legacy_event_evidence(
    event_id: UUID,
    subject: SubjectScope,
) -> EvidenceRef:
    return EvidenceRef.for_event_id(event_id, subject)


def _reject_json_constant(value: str, kind: str) -> None:
    raise MalformedSerializedDataError(
        f"{kind} JSON contains invalid numeric constant: {value}"
    )
