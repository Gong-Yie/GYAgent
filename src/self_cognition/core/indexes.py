from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Mapping

from self_cognition.core.state import SubjectState


@dataclass(frozen=True, slots=True)
class WorkspaceIndex:
    """Rebuildable field/time index derived from one state snapshot.

    The index stores only field names and ordering metadata.  Workspace content,
    confidence and evidence are always read from the authoritative state.
    """

    source_subject_id: str
    source_state_version: int
    fields_by_prefix: Mapping[str, tuple[str, ...]]
    time_index: tuple[tuple[datetime, str], ...]

    @classmethod
    def build(cls, state: SubjectState) -> "WorkspaceIndex":
        fields_by_prefix: dict[str, list[str]] = {}
        timed_fields: list[tuple[datetime, str]] = []

        for field in sorted(state.entries):
            parts = field.split(".")
            for end in range(1, len(parts) + 1):
                fields_by_prefix.setdefault(".".join(parts[:end]), []).append(field)

            value = state.entries[field].value
            parsed = _entry_time(field, value)
            if parsed is None:
                continue
            timed_fields.append((parsed.astimezone(timezone.utc), field))

        frozen_fields = MappingProxyType(
            {
                prefix: tuple(sorted(fields))
                for prefix, fields in fields_by_prefix.items()
            }
        )
        return cls(
            source_subject_id=state.subject_id,
            source_state_version=state.version,
            fields_by_prefix=frozen_fields,
            time_index=tuple(sorted(timed_fields, key=lambda item: (item[0], item[1]))),
        )

    def is_compatible(self, state: SubjectState) -> bool:
        return (
            self.source_subject_id == state.subject_id
            and self.source_state_version == state.version
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

        field_set = set(fields)
        ordered = [
            field
            for _, field in self.time_index
            if field in field_set
        ]
        indexed_fields = set(ordered)
        ordered.extend(field for field in fields if field not in indexed_fields)
        return tuple(ordered)


def _entry_time(field: str, value: object) -> datetime | None:
    """Extract time metadata without making the index authoritative."""
    occurred_at = value.get("occurred_at") if isinstance(value, dict) else None
    if not isinstance(occurred_at, str):
        for prefix in ("episodic.experience.", "narrative.chapter."):
            if not field.startswith(prefix):
                continue
            candidate = field[len(prefix):].rsplit(".", 1)[0]
            occurred_at = candidate
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
