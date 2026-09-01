import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone, timedelta
from uuid import UUID

import pytest

from self_cognition.cognition.semantic.preference_extractor import (
    PreferenceExtractor,
)
from self_cognition.core.errors import (
    MalformedSerializedDataError,
    UnsupportedSchemaVersionError,
)
from self_cognition.blackboard.reducer import StateReducer
from self_cognition.core.contributions import CognitionType
from self_cognition.core.events import Event, StateReductionPayload
from self_cognition.core.state import ConflictRecord, SubjectState
from self_cognition.infrastructure.persistence.serialization import (
    event_from_json,
    event_to_json,
    state_from_json,
    state_to_json,
)


def make_state() -> SubjectState:
    clock = FixedClock(datetime(2026, 9, 1, 12, tzinfo=timezone.utc))
    event = Event.user_message("user-1", "我喜欢晚上学习", clock=clock)
    contribution = replace(
        PreferenceExtractor().process(event)[0],
        confidence=0.75,
        target_version=0,
    )
    state = StateReducer().apply(
        SubjectState.empty("user-1"),
        contribution,
        decided_at=event.recorded_at,
    )
    return replace(
        state,
        conflicts=frozenset(
            {
                ConflictRecord(
                    target_field="preferences.study_time",
                    candidate_contribution_ids=(contribution.contribution_id,),
                    reason="测试冲突",
                )
            }
        ),
    )


def test_event_json_roundtrip_preserves_timezone_uuid_and_chinese_text():
    occurred_at = datetime(
            2026,
            8,
            13,
            20,
            30,
            tzinfo=timezone(timedelta(hours=8)),
        )
    event = Event.user_message(
        "user-1",
        "我喜欢晚上学习，记住这句话",
        event_id=UUID(int=1),
        clock=FixedClock(occurred_at),
    )

    payload = event_to_json(event)

    assert "schema_version" in payload
    assert "我喜欢晚上学习" in payload
    assert event_from_json(payload) == event


@dataclass(frozen=True, slots=True)
class FixedClock:
    value: datetime

    def now(self) -> datetime:
        return self.value


def test_model_reduction_and_failure_events_roundtrip():
    clock = FixedClock(datetime(2026, 9, 1, 12, tzinfo=timezone.utc))
    cause = Event.user_message("user-1", "测试", clock=clock)
    events = (
        Event.model_response(
            cause,
            model="test-model",
            response_id="resp-1",
            raw_output='{"candidates":[]}',
            clock=clock,
            run_id=UUID(int=2),
            correlation_id=UUID(int=3),
        ),
        Event.state_reduced(
            cause,
            StateReductionPayload(0, 1, True, (UUID(int=4),)),
            clock=clock,
            run_id=UUID(int=2),
            correlation_id=UUID(int=3),
        ),
        Event.processing_failed(
            cause,
            stage="process_event",
            error_type="RuntimeError",
            clock=clock,
            run_id=UUID(int=2),
            correlation_id=UUID(int=3),
        ),
    )

    assert tuple(event_from_json(event_to_json(event)) for event in events) == events


def test_state_json_roundtrip_preserves_all_snapshot_fields():
    state = make_state()

    payload = state_to_json(state)

    assert state_from_json(payload) == state


@pytest.mark.parametrize("schema_version", [1, 2, 3])
def test_legacy_state_schemas_migrate_without_guessing_metadata(
    schema_version: int,
) -> None:
    current = json.loads(state_to_json(make_state()))
    current["schema_version"] = schema_version
    current.pop("changes")
    if schema_version == 1:
        current.pop("mind_id")
        current.pop("subject_kind")
    for entry in current["entries"].values():
        entry.pop("cognition_type")
        entry.pop("scope")
        entry.pop("created_at")
        entry.pop("valid_from")
        entry.pop("expires_at")
        if schema_version < 3:
            entry["evidence_event_ids"] = [
                item["evidence_id"] for item in entry.pop("evidence_refs")
            ]

    restored = state_from_json(json.dumps(current, ensure_ascii=False))

    atom = restored.get("preferences.study_time")
    assert atom.cognition_type is CognitionType.UNKNOWN
    assert atom.created_at == datetime(1970, 1, 1, tzinfo=timezone.utc)
    assert restored.changes == ()


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


def test_event_schema_v1_is_not_supported():
    payload = {
        "schema_version": 1,
        "event_id": str(UUID(int=1)),
        "event_type": "user.message",
        "actor_id": "user-1",
        "content": "旧消息",
        "occurred_at": "2026-08-13T12:00:00+00:00",
    }

    with pytest.raises(UnsupportedSchemaVersionError):
        event_from_json(json.dumps(payload, ensure_ascii=False))


def test_corrupt_json_and_naive_time_are_rejected():
    with pytest.raises(MalformedSerializedDataError):
        event_from_json("{not-json}")

    payload = json.loads(event_to_json(Event.user_message("user-1", "测试")))
    payload["occurred_at"] = "2026-08-13T12:00:00"
    with pytest.raises(MalformedSerializedDataError):
        event_from_json(json.dumps(payload))
