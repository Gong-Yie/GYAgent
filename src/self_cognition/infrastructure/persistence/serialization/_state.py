"""Codec for subject state, its atoms, conflicts and change records."""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from self_cognition.core.contributions import CognitionType
from self_cognition.core.errors import (
    MalformedSerializedDataError,
    SerializationError,
    UnsupportedSchemaVersionError,
)
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.metacognition import ConflictStatus
from self_cognition.core.scopes import (
    DEFAULT_MIND_ID,
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

from ._codec import (
    _from_json,
    _optional_uuid_to_string,
    _require_bool,
    _require_datetime,
    _require_float,
    _require_int,
    _require_keys,
    _require_non_blank_string,
    _require_object,
    _require_optional_datetime,
    _require_optional_non_blank_string,
    _require_optional_uuid,
    _require_string,
    _require_uuid,
    _require_uuid_tuple,
    _to_json,
)
from ._refs import (
    _data_scope_from_dict,
    _data_scope_to_dict,
    _evidence_ref_to_dict,
    _require_evidence_ref_tuple,
)
from ._contributions import _contribution_from_dict, _contribution_to_dict


STATE_SCHEMA_VERSION = 5

LEGACY_STATE_TIME = datetime(1970, 1, 1, tzinfo=timezone.utc)


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
                    _evidence_ref_to_dict(evidence) for evidence in entry.evidence_refs
                ],
                "contribution_ids": [
                    str(contribution_id) for contribution_id in entry.contribution_ids
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
            str(contribution_id) for contribution_id in state.applied_contribution_ids
        ),
        "conflicts": [
            {
                "target_field": conflict.target_field,
                "candidate_contribution_ids": [
                    str(contribution_id)
                    for contribution_id in conflict.candidate_contribution_ids
                ],
                "reason": conflict.reason,
                "conflict_id": str(conflict.conflict_id),
                "status": conflict.status.value,
                "evidence_refs": [
                    _evidence_ref_to_dict(ref) for ref in conflict.evidence_refs
                ],
                "requires_confirmation": conflict.requires_confirmation,
                "confirmed": conflict.confirmed,
                "resolution_reason": conflict.resolution_reason,
                "reviewed_by": _optional_uuid_to_string(conflict.reviewed_by),
                "selected_contribution_id": _optional_uuid_to_string(
                    conflict.selected_contribution_id
                ),
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
    change_fields = {"changes"} if schema_version >= 4 else set()
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
            if schema_version >= 4
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
        if schema_version >= 4:
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
            {
                "target_field",
                "candidate_contribution_ids",
                "reason",
                *(
                    {
                        "conflict_id",
                        "status",
                        "evidence_refs",
                        "requires_confirmation",
                        "confirmed",
                        "resolution_reason",
                        "reviewed_by",
                        "selected_contribution_id",
                    }
                    if schema_version >= 5
                    else set()
                ),
            },
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
                **(_conflict_metadata(conflict) if schema_version >= 5 else {}),
            )
        )

    changes = _state_changes_from_list(values["changes"]) if schema_version >= 4 else ()

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


def _conflict_metadata(value: dict[str, Any]) -> dict[str, Any]:
    record = ConflictRecord(
        target_field=value["target_field"],
        candidate_contribution_ids=_require_uuid_tuple(
            value["candidate_contribution_ids"], "conflict.candidates"
        ),
        reason=value["reason"],
    )
    if _require_uuid(value["conflict_id"], "conflict.id") != record.conflict_id:
        raise MalformedSerializedDataError("conflict ID does not match candidates")
    try:
        status = ConflictStatus(value["status"])
    except (ValueError, TypeError) as error:
        raise MalformedSerializedDataError("conflict status is invalid") from error
    return {
        "status": status,
        "evidence_refs": _require_evidence_ref_tuple(
            value["evidence_refs"], "conflict.evidence"
        ),
        "requires_confirmation": _require_bool(
            value["requires_confirmation"], "conflict.requires_confirmation"
        ),
        "confirmed": _require_bool(value["confirmed"], "conflict.confirmed"),
        "resolution_reason": _require_optional_non_blank_string(
            value["resolution_reason"], "conflict.resolution_reason"
        ),
        "reviewed_by": _require_optional_uuid(
            value["reviewed_by"], "conflict.reviewed_by"
        ),
        "selected_contribution_id": _require_optional_uuid(
            value["selected_contribution_id"], "conflict.selected_contribution_id"
        ),
    }


def state_from_json(payload: str) -> SubjectState:
    return state_from_dict(_from_json(payload, "state"))


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


def _require_supported_state_schema(values: dict[str, Any]) -> int:
    version = values.get("schema_version")
    if version in (1, 2, 3, 4, STATE_SCHEMA_VERSION) and not isinstance(version, bool):
        return version
    if isinstance(version, int) and not isinstance(version, bool):
        raise UnsupportedSchemaVersionError(
            f"unsupported state schema version: {version}"
        )
    raise MalformedSerializedDataError("state.schema_version must be 1, 2, 3, 4, or 5")


def _legacy_event_evidence(
    event_id: UUID,
    subject: SubjectScope,
) -> EvidenceRef:
    return EvidenceRef.for_event_id(event_id, subject)
