from uuid import UUID
from threading import RLock

from self_cognition.core.events import EventEnvelope
from self_cognition.core.scopes import SubjectScope


class InMemoryEventStore:
    def __init__(self) -> None:
        self._events: list[EventEnvelope] = []
        self._event_ids: set[UUID] = set()
        self._tombstones: set[UUID] = set()
        self._lock = RLock()

    def append(self, event: EventEnvelope) -> None:
        with self._lock:
            if event.event_id in self._event_ids or event.event_id in self._tombstones:
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

    def redact(
        self,
        subject: SubjectScope,
        event_ids: tuple[UUID, ...],
        plan_id: UUID,
    ) -> None:
        del plan_id
        targets = set(event_ids)
        with self._lock:
            if any(
                event.event_id in targets and event.subject != subject
                for event in self._events
            ):
                raise ValueError("cannot redact an event owned by another subject")
            self._tombstones.update(targets)
            self._events = [
                event for event in self._events if event.event_id not in targets
            ]
            self._event_ids.difference_update(targets)
