"""Codec for memory records and memory access records."""

from typing import Any

from self_cognition.core.errors import (
    MalformedSerializedDataError,
    SerializationError,
    UnsupportedSchemaVersionError,
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

from ._codec import (
    _from_json,
    _require_datetime,
    _require_float,
    _require_int,
    _require_keys,
    _require_non_blank_string,
    _require_object,
    _require_optional_datetime,
    _require_optional_non_blank_string,
    _require_schema,
    _require_string,
    _require_string_tuple,
    _require_uuid,
    _to_json,
)
from ._refs import (
    _data_scope_from_dict,
    _data_scope_to_dict,
    _evidence_ref_to_dict,
    _require_evidence_ref_tuple,
    _subject_scope_from_dict,
    _subject_scope_to_dict,
)


MEMORY_SCHEMA_VERSION = 4

MEMORY_ACCESS_SCHEMA_VERSION = 1


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
