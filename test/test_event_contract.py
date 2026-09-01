from dataclasses import replace
from datetime import datetime

import pytest

from self_cognition.core.errors import ContractValidationError
from self_cognition.core.events import Event, UserMessagePayload
from self_cognition.core.scopes import SubjectKind, SubjectRef


def make_event(**overrides: object) -> Event:
    return replace(Event.user_message("user-1", "我喜欢晚上学习"), **overrides)


@pytest.mark.parametrize("event_type", ["", "  \t"])
def test_rejects_blank_event_type(event_type: str):
    with pytest.raises(ContractValidationError):
        make_event(event_type=event_type)


def test_rejects_actor_different_from_user_message_subject():
    with pytest.raises(ContractValidationError):
        make_event(actor=SubjectRef(SubjectKind.USER, "user-2"))


@pytest.mark.parametrize("content", ["", "  \t"])
def test_rejects_blank_message_content(content: str):
    with pytest.raises(ContractValidationError):
        make_event(payload=UserMessagePayload(content))


def test_rejects_occurred_at_without_timezone():
    with pytest.raises(ContractValidationError):
        make_event(occurred_at=datetime.now())


def test_rejects_recorded_at_without_timezone():
    with pytest.raises(ContractValidationError):
        make_event(recorded_at=datetime.now())


def test_accepts_valid_event():
    event = make_event()

    assert event.occurred_at.utcoffset() is not None
