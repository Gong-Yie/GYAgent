"""JSON encoding, schema checks and primitive field validation shared by the codecs."""

import json
from datetime import datetime
from typing import Any
from uuid import UUID

from self_cognition.core.errors import (
    MalformedSerializedDataError,
    SerializationError,
    UnsupportedSchemaVersionError,
)
from self_cognition.core.events import EventSource


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


def _reject_json_constant(value: str, kind: str) -> None:
    raise MalformedSerializedDataError(
        f"{kind} JSON contains invalid numeric constant: {value}"
    )
