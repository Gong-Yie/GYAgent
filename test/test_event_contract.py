from datetime import datetime, timezone
from uuid import uuid4

import pytest

from self_cognition.core.errors import ContractValidationError
from self_cognition.core.events import Event


def make_event(**overrides: object) -> Event:
    values = {
        "event_id": uuid4(),
        "event_type": "user.message",
        "actor_id": "user-1",
        "content": "我喜欢晚上学习",
        "occurred_at": datetime.now(timezone.utc),
    }
    values.update(overrides)
    return Event(**values)


@pytest.mark.parametrize("event_type", ["", "  \t"])
def test_rejects_blank_event_type(event_type: str):
    with pytest.raises(ContractValidationError):
        make_event(event_type=event_type)


@pytest.mark.parametrize("actor_id", ["", "  \t"])
def test_rejects_blank_actor_id(actor_id: str):
    with pytest.raises(ContractValidationError):
        make_event(actor_id=actor_id)


@pytest.mark.parametrize("content", ["", "  \t"])
def test_rejects_blank_message_content(content: str):
    with pytest.raises(ContractValidationError):
        make_event(content=content)


def test_rejects_occurred_at_without_timezone():
    with pytest.raises(ContractValidationError):
        make_event(occurred_at=datetime.now())


def test_accepts_valid_event():
    event = make_event()

    assert event.occurred_at.utcoffset() is not None
