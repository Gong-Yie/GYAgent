from datetime import datetime, timedelta, timezone
from uuid import UUID

from self_cognition.application.process_event import ProcessEventService
from self_cognition.application.replay import ReplayService
from self_cognition.cognition.semantic.name_extractor import NameExtractor
from self_cognition.cognition.semantic.preference_extractor import (
    PreferenceExtractor,
)
from self_cognition.core.events import Event
from self_cognition.infrastructure.persistence.in_memory_event_store import (
    InMemoryEventStore,
)
from self_cognition.infrastructure.persistence.in_memory_evidence_repository import (
    InMemoryEvidenceRepository,
)
from self_cognition.infrastructure.persistence.in_memory_state_repository import (
    InMemoryStateRepository,
)
from self_cognition.runtime.engine import CognitionEngine
from self_cognition.blackboard.reducer import StateReducer
from self_cognition.blackboard.service import CognitiveSpaceService
from self_cognition.runtime.run_context import RunContext


def make_event(event_id: int, content: str) -> Event:
    return Event.user_message(
        "user-1",
        content,
        event_id=UUID(int=event_id),
        clock=FixedClock(datetime(2026, 8, 13, event_id, tzinfo=timezone.utc)),
    )


class FixedClock:
    def __init__(self, value: datetime) -> None:
        self._value = value

    def now(self) -> datetime:
        return self._value


def test_replay_rebuilds_the_same_state_without_changing_the_event_log():
    events = (
        make_event(1, "我喜欢晚上学习"),
        make_event(2, "我叫小明"),
        make_event(3, "我喜欢早上学习"),
    )
    event_store = InMemoryEventStore()
    state_repository = InMemoryStateRepository()
    engine = CognitionEngine(
        modules=(PreferenceExtractor(), NameExtractor()),
        cognitive_space=CognitiveSpaceService(StateReducer()),
    )
    process_service = ProcessEventService(
        event_store=event_store,
        evidence_repository=InMemoryEvidenceRepository(),
        state_repository=state_repository,
        engine=engine,
    )
    for event in events:
        result = process_service.process(
            event,
            RunContext(
                run_id=event.event_id,
                correlation_id=UUID(int=100),
                deadline=datetime.now(timezone.utc) + timedelta(minutes=1),
            ),
        )
        assert result.state is not None
        expected_state = result.state
    original_log = event_store.read_by_subject(events[0].subject)

    replay_service = ReplayService(event_store=event_store, engine=engine)
    first_replay = replay_service.replay(events[0].subject)
    second_replay = replay_service.replay(events[0].subject)

    assert first_replay == expected_state
    assert second_replay == first_replay
    assert first_replay.version == 3
    assert first_replay.get("preferences.study_time").value == "早上"
    assert first_replay.get("profile.name").value == "小明"
    assert event_store.read_by_subject(events[0].subject) == original_log
    user_events = tuple(
        event for event in original_log if event.event_type == "user.message"
    )
    assert tuple(event.event_id for event in user_events) == tuple(
        event.event_id for event in events
    )
