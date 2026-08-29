import json
from datetime import datetime
from typing import Any
from uuid import UUID

from self_cognition.core.errors import (
    MalformedSerializedDataError,
    SerializationError,
    UnsupportedSchemaVersionError,
)
from self_cognition.core.events import Event
from self_cognition.core.state import ConflictRecord, StateEntry, SubjectState


EVENT_SCHEMA_VERSION = 1
STATE_SCHEMA_VERSION = 1


def event_to_dict(event: Event) -> dict[str, Any]:
    return {
        "schema_version": EVENT_SCHEMA_VERSION,
        "event_id": str(event.event_id),
        "event_type": event.event_type,
        "actor_id": event.actor_id,
        "content": event.content,
        "occurred_at": event.occurred_at.isoformat(),
    }


def event_from_dict(data: object) -> Event:
    values = _require_object(data, "event")
    _require_schema(values, EVENT_SCHEMA_VERSION, "event")
    _require_keys(
        values,
        {
            "schema_version",
            "event_id",
            "event_type",
            "actor_id",
            "content",
            "occurred_at",
        },
        "event",
    )
    try:
        occurred_at = datetime.fromisoformat(
            _require_string(values["occurred_at"], "event.occurred_at")
        )
    except ValueError as error:
        raise MalformedSerializedDataError(
            "event.occurred_at must be a valid ISO 8601 datetime"
        ) from error
    if occurred_at.tzinfo is None or occurred_at.utcoffset() is None:
        raise MalformedSerializedDataError(
            "event.occurred_at must include timezone information"
        )

    try:
        return Event(
            event_id=_require_uuid(values["event_id"], "event.event_id"),
            event_type=_require_string(values["event_type"], "event.event_type"),
            actor_id=_require_string(values["actor_id"], "event.actor_id"),
            content=_require_string(values["content"], "event.content"),
            occurred_at=occurred_at,
        )
    except SerializationError:
        raise
    except Exception as error:
        raise MalformedSerializedDataError("invalid event values") from error


def event_to_json(event: Event) -> str:
    return _to_json(event_to_dict(event), "event")


def event_from_json(payload: str) -> Event:
    return event_from_dict(_from_json(payload, "event"))


def state_to_dict(state: SubjectState) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "subject_id": state.subject_id,
        "version": state.version,
        "entries": {
            target_field: {
                "value": entry.value,
                "confidence": entry.confidence,
                "evidence_event_ids": [
                    str(event_id) for event_id in entry.evidence_event_ids
                ],
                "contribution_ids": [
                    str(contribution_id)
                    for contribution_id in entry.contribution_ids
                ],
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
    }


def state_from_dict(data: object) -> SubjectState:
    values = _require_object(data, "state")
    _require_schema(values, STATE_SCHEMA_VERSION, "state")
    _require_keys(
        values,
        {
            "schema_version",
            "subject_id",
            "version",
            "entries",
            "applied_contribution_ids",
            "conflicts",
        },
        "state",
    )

    subject_id = _require_string(values["subject_id"], "state.subject_id")
    version = _require_int(values["version"], "state.version")
    entries_value = values["entries"]
    if not isinstance(entries_value, dict):
        raise MalformedSerializedDataError("state.entries must be an object")

    entries: dict[str, StateEntry] = {}
    for target_field, raw_entry in entries_value.items():
        if not isinstance(target_field, str) or not target_field.strip():
            raise MalformedSerializedDataError(
                "state.entries keys must be non-blank strings"
            )
        entry = _require_object(raw_entry, f"state.entries[{target_field!r}]")
        _require_keys(
            entry,
            {
                "value",
                "confidence",
                "evidence_event_ids",
                "contribution_ids",
            },
            f"state.entries[{target_field!r}]",
        )
        entries[target_field] = StateEntry(
            value=entry["value"],
            confidence=_require_float(
                entry["confidence"],
                f"state.entries[{target_field!r}].confidence",
            ),
            evidence_event_ids=_require_uuid_tuple(
                entry["evidence_event_ids"],
                f"state.entries[{target_field!r}].evidence_event_ids",
            ),
            contribution_ids=_require_uuid_tuple(
                entry["contribution_ids"],
                f"state.entries[{target_field!r}].contribution_ids",
            ),
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

    return SubjectState(
        subject_id=subject_id,
        version=version,
        entries=entries,
        applied_contribution_ids=applied_ids,
        conflicts=frozenset(conflicts),
    )


def state_to_json(state: SubjectState) -> str:
    return _to_json(state_to_dict(state), "state")


def state_from_json(payload: str) -> SubjectState:
    return state_from_dict(_from_json(payload, "state"))


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


def _require_int(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise MalformedSerializedDataError(f"{path} must be an integer")
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


def _reject_json_constant(value: str, kind: str) -> None:
    raise MalformedSerializedDataError(
        f"{kind} JSON contains invalid numeric constant: {value}"
    )
