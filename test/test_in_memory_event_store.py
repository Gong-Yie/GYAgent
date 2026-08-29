from datetime import datetime, timezone
from uuid import UUID

from self_cognition.core.events import Event
from self_cognition.infrastructure.persistence.in_memory_event_store import (
    InMemoryEventStore,
)


def make_event(event_id: int, actor_id: str, content: str) -> Event:
    return Event(
        event_id=UUID(int=event_id),
        event_type="user.message",
        actor_id=actor_id,
        content=content,
        occurred_at=datetime(2026, 8, 13, event_id, tzinfo=timezone.utc),
    )


def test_appends_event_and_reports_membership():
    store = InMemoryEventStore()
    event = make_event(1, "user-1", "第一条消息")

    assert not store.contains(event.event_id)

    store.append(event)

    assert store.contains(event.event_id)
    assert store.read_all() == (event,)
    assert isinstance(store.read_all(), tuple)


def test_preserves_append_order():
    store = InMemoryEventStore()
    first = make_event(1, "user-1", "第一条消息")
    second = make_event(2, "user-1", "第二条消息")

    store.append(first)
    store.append(second)

    assert store.read_all() == (first, second)


def test_ignores_duplicate_id_and_preserves_first_event():
    store = InMemoryEventStore()
    first = make_event(1, "user-1", "首次保存的内容")
    duplicate = make_event(1, "user-1", "相同 ID 的不同内容")

    store.append(first)
    store.append(duplicate)

    assert store.read_all() == (first,)


def test_reads_events_by_subject_in_append_order():
    store = InMemoryEventStore()
    first_user_event = make_event(1, "user-1", "用户一的第一条消息")
    other_user_event = make_event(2, "user-2", "用户二的消息")
    second_user_event = make_event(3, "user-1", "用户一的第二条消息")

    store.append(first_user_event)
    store.append(other_user_event)
    store.append(second_user_event)

    assert store.read_by_subject("user-1") == (
        first_user_event,
        second_user_event,
    )
    assert store.read_by_subject("missing-user") == ()
