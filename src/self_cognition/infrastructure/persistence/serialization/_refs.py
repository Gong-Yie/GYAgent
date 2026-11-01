"""Codecs for subject, scope and evidence references embedded in persisted aggregates."""

from self_cognition.core.errors import MalformedSerializedDataError, SerializationError
from self_cognition.core.evidence import EvidenceRef, EvidenceSourceKind
from self_cognition.core.scopes import (
    ConversationScope,
    DataScope,
    DisclosureScope,
    MindScope,
    SubjectKind,
    SubjectRef,
    SubjectScope,
)

from ._codec import (
    _require_float,
    _require_keys,
    _require_non_blank_string,
    _require_object,
    _require_optional_datetime,
    _require_string,
    _require_uuid,
)


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
