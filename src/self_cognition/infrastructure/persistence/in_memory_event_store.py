from uuid import UUID
from threading import RLock

from self_cognition.core.events import EventEnvelope
from self_cognition.core.scopes import SubjectScope


class InMemoryEventStore:
    def __init__(self) -> None:
        self._events: list[EventEnvelope] = []
        self._event_ids: set[UUID] = set()
        self._lock = RLock()

    def append(self, event: EventEnvelope) -> None:
        with self._lock:
            if event.event_id in self._event_ids:
                return

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
