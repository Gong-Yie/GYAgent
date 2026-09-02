from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Mapping

from self_cognition.core.memories import MemoryLifecycleStatus, MemoryRecord
from self_cognition.core.scopes import DEFAULT_MIND_ID, SubjectKind
from self_cognition.core.state import SubjectState


WORKSPACE_INDEX_SCHEMA_VERSION = 1
_WORD_PATTERN = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]+", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class WorkspaceIndex:
    """Rebuildable field, time and full-text references.

    Content, confidence and evidence are always read from authoritative state and
    memory records. The index can only narrow which references are inspected.
    """

    source_subject_id: str
    source_state_version: int
    fields_by_prefix: Mapping[str, tuple[str, ...]]
    time_index: tuple[tuple[datetime, str], ...]
    source_mind_id: str = DEFAULT_MIND_ID
    source_subject_kind: SubjectKind = SubjectKind.USER
    source_memory_versions: tuple[tuple[str, int], ...] = ()
    full_text_index: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    schema_version: int = WORKSPACE_INDEX_SCHEMA_VERSION

    @classmethod
    def build(
        cls,
        state: SubjectState,
        memories: tuple[MemoryRecord, ...] = (),
    ) -> WorkspaceIndex:
        fields_by_prefix: dict[str, list[str]] = {}
        timed_refs: list[tuple[datetime, str]] = []
        full_text: dict[str, set[str]] = {}

        for field_name in sorted(state.entries):
            parts = field_name.split(".")
            for end in range(1, len(parts) + 1):
                fields_by_prefix.setdefault(".".join(parts[:end]), []).append(
                    field_name
                )
            reference = f"state:{field_name}"
            value = state.entries[field_name].value
            _add_text(full_text, reference, field_name, value)
            parsed = _entry_time(field_name, value)
            if parsed is not None:
                timed_refs.append((parsed.astimezone(timezone.utc), reference))

        active_memories = tuple(
            record
            for record in memories
            if record.lifecycle_status is MemoryLifecycleStatus.ACTIVE
        )
        for record in active_memories:
            reference = f"memory:{record.memory_id}"
            _add_text(
                full_text,
                reference,
                record.content,
                record.memory_type.value,
                record.cues,
            )
            timed_refs.append((record.created_at.astimezone(timezone.utc), reference))

        return cls(
            source_subject_id=state.subject_id,
            source_state_version=state.version,
            fields_by_prefix=MappingProxyType(
                {
                    prefix: tuple(sorted(fields))
                    for prefix, fields in fields_by_prefix.items()
                }
            ),
            time_index=tuple(sorted(timed_refs, key=lambda item: (item[0], item[1]))),
            source_mind_id=state.mind_id,
            source_subject_kind=state.subject_kind,
            source_memory_versions=tuple(
                sorted(
                    (str(record.memory_id), record.version)
                    for record in active_memories
                )
            ),
            full_text_index=MappingProxyType(
                {
                    term: tuple(sorted(references))
                    for term, references in sorted(full_text.items())
                }
            ),
        )

    def is_compatible(
        self,
        state: SubjectState,
        memories: tuple[MemoryRecord, ...] = (),
    ) -> bool:
        active_versions = tuple(
            sorted(
                (str(record.memory_id), record.version)
                for record in memories
                if record.lifecycle_status is MemoryLifecycleStatus.ACTIVE
            )
        )
        return (
            self.schema_version == WORKSPACE_INDEX_SCHEMA_VERSION
            and self.source_subject_id == state.subject_id
            and self.source_mind_id == state.mind_id
            and self.source_subject_kind is state.subject_kind
            and self.source_state_version == state.version
            and self.source_memory_versions == active_versions
        )

    def fields_for_prefix(
        self,
        prefix: str,
        *,
        chronological: bool = False,
    ) -> tuple[str, ...]:
        normalized_prefix = prefix.rstrip(".")
        fields = self.fields_by_prefix.get(normalized_prefix, ())
        if not chronological or not fields:
            return fields

        state_refs = {f"state:{field_name}": field_name for field_name in fields}
        ordered = [
            state_refs[reference]
            for _, reference in self.time_index
            if reference in state_refs
        ]
        indexed_fields = set(ordered)
        ordered.extend(
            field_name for field_name in fields if field_name not in indexed_fields
        )
        return tuple(ordered)

    def references_for_text(self, text: str) -> tuple[str, ...]:
        terms = _terms(text)
        if not terms:
            return ()
        matches: set[str] = set()
        for term in terms:
            matches.update(self.full_text_index.get(term, ()))
        return tuple(sorted(matches))

    def references_for_time(
        self,
        time_from: datetime | None,
        time_to: datetime | None,
    ) -> tuple[str, ...]:
        if time_from is None and time_to is None:
            return ()
        return tuple(
            reference
            for occurred_at, reference in self.time_index
            if (time_from is None or occurred_at >= time_from.astimezone(timezone.utc))
            and (time_to is None or occurred_at <= time_to.astimezone(timezone.utc))
        )


def text_terms(*values: object) -> frozenset[str]:
    return frozenset(
        term
        for value in values
        for term in _terms(_searchable_text(value))
    )


def _add_text(
    index: dict[str, set[str]],
    reference: str,
    *values: object,
) -> None:
    for term in text_terms(*values):
        index.setdefault(term, set()).add(reference)


def _searchable_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _terms(text: str) -> frozenset[str]:
    result: set[str] = set()
    for match in _WORD_PATTERN.findall(text.casefold()):
        if all("\u4e00" <= character <= "\u9fff" for character in match):
            result.update(match[index : index + 2] for index in range(len(match) - 1))
            if len(match) == 1:
                result.add(match)
        else:
            result.add(match)
    return frozenset(result)


def _entry_time(field_name: str, value: object) -> datetime | None:
    occurred_at = value.get("occurred_at") if isinstance(value, dict) else None
    if not isinstance(occurred_at, str):
        for prefix in ("episodic.experience.", "narrative.chapter."):
            if not field_name.startswith(prefix):
                continue
            occurred_at = field_name[len(prefix) :].rsplit(".", 1)[0]
            break
    if not isinstance(occurred_at, str):
        return None
    try:
        parsed = datetime.fromisoformat(occurred_at)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed
