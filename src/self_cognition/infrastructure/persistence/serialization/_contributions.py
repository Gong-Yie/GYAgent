"""Codec for cognitive contribution values stored in events and state."""

from self_cognition.core.contributions import (
    CognitionType,
    CognitiveContribution,
    ContributionOperation,
)
from self_cognition.core.errors import MalformedSerializedDataError, SerializationError

from ._codec import (
    _require_bool,
    _require_datetime,
    _require_float,
    _require_keys,
    _require_non_blank_string,
    _require_object,
    _require_optional_datetime,
    _require_optional_int,
    _require_string,
    _require_uuid,
)
from ._refs import (
    _data_scope_from_dict,
    _data_scope_to_dict,
    _evidence_ref_to_dict,
    _require_evidence_ref_tuple,
    _subject_scope_from_dict,
    _subject_scope_to_dict,
)


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
