import json
import os
from threading import RLock
from pathlib import Path
from uuid import UUID

from self_cognition.core.errors import MalformedSerializedDataError
from self_cognition.core.events import EventEnvelope
from self_cognition.core.scopes import SubjectScope
from self_cognition.infrastructure.persistence.serialization import (
    event_from_json,
    event_to_json,
)
from self_cognition.infrastructure.persistence.atomic_io import atomic_write_text


class FileEventStore:
    def __init__(
        self,
        path: str | Path,
        tombstone_path: str | Path | None = None,
    ) -> None:
        self._path = Path(path)
        self._tombstone_path = Path(
            tombstone_path or self._path.parent / "event_tombstones.jsonl"
        )
        self._tombstones = self._read_tombstones()
        self._events = list(self._read_events())
        self._event_ids = {event.event_id for event in self._events}
        self._lock = RLock()

    def append(self, event: EventEnvelope) -> None:
        with self._lock:
            if event.event_id in self._event_ids or event.event_id in self._tombstones:
                return

            self._path.parent.mkdir(parents=True, exist_ok=True)
            record = event_to_json(event) + "\n"
            with self._path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(record)
                handle.flush()
                os.fsync(handle.fileno())

            self._events.append(event)
            self._event_ids.add(event.event_id)

    def read_by_subject(
        self,
        subject: SubjectScope,
    ) -> tuple[EventEnvelope, ...]:
        if not isinstance(subject, SubjectScope):
            raise TypeError("subject must be a SubjectScope")
        with self._lock:
            return tuple(
                event for event in self._events if event.subject == subject
            )

    def redact(
        self,
        subject: SubjectScope,
        event_ids: tuple[UUID, ...],
        plan_id: UUID,
    ) -> None:
        targets = set(event_ids)
        with self._lock:
            if any(
                event.event_id in targets and event.subject != subject
                for event in self._events
            ):
                raise ValueError("cannot redact an event owned by another subject")
            new_tombstones = targets - self._tombstones
            if new_tombstones:
                self._append_tombstones(plan_id, new_tombstones)
                self._tombstones.update(new_tombstones)
            self._events = [
                event for event in self._events if event.event_id not in targets
            ]
            self._event_ids.difference_update(targets)
            payload = "".join(event_to_json(event) + "\n" for event in self._events)
            atomic_write_text(self._path, payload)

    def _read_events(self) -> tuple[EventEnvelope, ...]:
        if not self._path.exists():
            return ()

        events: list[EventEnvelope] = []
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        raise MalformedSerializedDataError(
                            f"blank event record at line {line_number}"
                        )
                    try:
                        event = event_from_json(line)
                        if event.event_id not in self._tombstones:
                            events.append(event)
                    except MalformedSerializedDataError as error:
                        raise MalformedSerializedDataError(
                            f"invalid event record at line {line_number}"
                        ) from error
        except UnicodeError as error:
            raise MalformedSerializedDataError(
                "event log is not valid UTF-8"
            ) from error
        return tuple(events)

    def _read_tombstones(self) -> set[UUID]:
        if not self._tombstone_path.exists():
            return set()
        tombstones = set()
        try:
            with self._tombstone_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    values = json.loads(line)
                    tombstones.add(UUID(values["event_id"]))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise MalformedSerializedDataError(
                f"invalid event tombstone at line {line_number}"
            ) from error
        return tombstones

    def _append_tombstones(self, plan_id: UUID, event_ids: set[UUID]) -> None:
        self._tombstone_path.parent.mkdir(parents=True, exist_ok=True)
        with self._tombstone_path.open(
            "a",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            for event_id in sorted(event_ids, key=lambda value: value.int):
                handle.write(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "plan_id": str(plan_id),
                            "event_id": str(event_id),
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
