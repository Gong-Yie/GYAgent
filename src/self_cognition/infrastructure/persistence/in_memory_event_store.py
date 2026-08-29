from uuid import UUID

from self_cognition.core.events import Event


class InMemoryEventStore:
    def __init__(self) -> None:
        self._events: list[Event] = []
        self._event_ids: set[UUID] = set()

    def append(self, event: Event) -> None:
        if event.event_id in self._event_ids:
            return

        self._events.append(event)
        self._event_ids.add(event.event_id)

    def read_all(self) -> tuple[Event, ...]:
        return tuple(self._events)

    def contains(self, event_id: UUID) -> bool:
        return event_id in self._event_ids

    def read_by_subject(self, subject_id: str) -> tuple[Event, ...]:
        return tuple(
            event for event in self._events if event.actor_id == subject_id
        )
