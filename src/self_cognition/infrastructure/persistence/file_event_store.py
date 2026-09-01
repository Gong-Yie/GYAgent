import os
from pathlib import Path
from uuid import UUID

from self_cognition.core.errors import MalformedSerializedDataError
from self_cognition.core.events import EventEnvelope
from self_cognition.core.scopes import SubjectScope
from self_cognition.infrastructure.persistence.serialization import (
    event_from_json,
    event_to_json,
)


class FileEventStore:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._events = list(self._read_events())
        self._event_ids = {event.event_id for event in self._events}

    def append(self, event: EventEnvelope) -> None:
        if event.event_id in self._event_ids:
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
        return tuple(event for event in self._events if event.subject == subject)

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
                        events.append(event_from_json(line))
                    except MalformedSerializedDataError as error:
                        raise MalformedSerializedDataError(
                            f"invalid event record at line {line_number}"
                        ) from error
        except UnicodeError as error:
            raise MalformedSerializedDataError(
                "event log is not valid UTF-8"
            ) from error
        return tuple(events)
