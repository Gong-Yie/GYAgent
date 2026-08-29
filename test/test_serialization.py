import json
from datetime import datetime, timezone, timedelta
from uuid import UUID

import pytest

from self_cognition.core.errors import (
    MalformedSerializedDataError,
    UnsupportedSchemaVersionError,
)
from self_cognition.core.events import Event
from self_cognition.core.state import ConflictRecord, StateEntry, SubjectState
from self_cognition.infrastructure.persistence.serialization import (
    event_from_json,
    event_to_json,
    state_from_json,
    state_to_json,
)


def make_state() -> SubjectState:
    event_id = UUID(int=1)
    contribution_id = UUID(int=2)
    return SubjectState(
        subject_id="user-1",
        version=3,
        entries={
            "preferences.study_time": StateEntry(
                value="晚上",
                confidence=0.75,
                evidence_event_ids=(event_id,),
                contribution_ids=(contribution_id,),
            )
        },
        applied_contribution_ids=frozenset({contribution_id}),
        conflicts=frozenset(
            {
                ConflictRecord(
                    target_field="preferences.study_time",
                    candidate_contribution_ids=(contribution_id,),
                    reason="测试冲突",
                )
            }
        ),
    )


def test_event_json_roundtrip_preserves_timezone_uuid_and_chinese_text():
    event = Event(
        event_id=UUID(int=1),
        event_type="user.message",
        actor_id="user-1",
        content="我喜欢晚上学习，记住这句话",
        occurred_at=datetime(
            2026,
            8,
            13,
            20,
            30,
            tzinfo=timezone(timedelta(hours=8)),
        ),
    )

    payload = event_to_json(event)

    assert "schema_version" in payload
    assert "我喜欢晚上学习" in payload
    assert event_from_json(payload) == event


def test_state_json_roundtrip_preserves_all_snapshot_fields():
    state = make_state()

    payload = state_to_json(state)

    assert state_from_json(payload) == state


def test_unknown_fields_are_rejected():
    payload = json.loads(event_to_json(Event.user_message("user-1", "测试")))
    payload["unexpected"] = True

    with pytest.raises(MalformedSerializedDataError):
        event_from_json(json.dumps(payload, ensure_ascii=False))


def test_unsupported_schema_versions_are_rejected_explicitly():
    payload = json.loads(event_to_json(Event.user_message("user-1", "测试")))
    payload["schema_version"] = 0

    with pytest.raises(UnsupportedSchemaVersionError):
        event_from_json(json.dumps(payload))


def test_corrupt_json_and_naive_time_are_rejected():
    with pytest.raises(MalformedSerializedDataError):
        event_from_json("{not-json}")

    payload = json.loads(event_to_json(Event.user_message("user-1", "测试")))
    payload["occurred_at"] = "2026-08-13T12:00:00"
    with pytest.raises(MalformedSerializedDataError):
        event_from_json(json.dumps(payload))
