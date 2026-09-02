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
        self.append_many((event,))

    def append_many(self, events: tuple[EventEnvelope, ...]) -> None:
        with self._lock:
            pending = tuple(
                event
                for event in events
                if event.event_id not in self._event_ids
                and event.event_id not in self._tombstones
            )
            if not pending:
                return
            if len({event.event_id for event in pending}) != len(pending):
                raise ValueError("event batch contains duplicate IDs")

            self._events.extend(pending)
            self._event_ids.update(event.event_id for event in pending)

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
