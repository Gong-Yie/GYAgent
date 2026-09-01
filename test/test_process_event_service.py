from datetime import datetime, timedelta, timezone
from uuid import UUID

from self_cognition.application.process_event import ProcessEventService
from self_cognition.application.results import ProcessEventStatus
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


def make_context(run_id: int) -> RunContext:
    return RunContext(
        run_id=UUID(int=run_id),
        correlation_id=UUID(int=100),
        deadline=datetime.now(timezone.utc) + timedelta(minutes=1),
    )


def test_processes_events_through_the_complete_application_service():
    event_store = InMemoryEventStore()
    state_repository = InMemoryStateRepository()
    service = ProcessEventService(
        event_store=event_store,
        evidence_repository=InMemoryEvidenceRepository(),
        state_repository=state_repository,
        engine=CognitionEngine(
            modules=(PreferenceExtractor(),),
            cognitive_space=CognitiveSpaceService(StateReducer()),
        ),
    )
    evening_event = Event.user_message("user-1", "我喜欢晚上学习")
    morning_event = Event.user_message("user-1", "我喜欢早上学习")

    first_result = service.process(evening_event, make_context(1))
    duplicate_result = service.process(evening_event, make_context(2))
    final_result = service.process(morning_event, make_context(3))

    assert first_result.status is ProcessEventStatus.SUCCEEDED
    assert first_result.old_version == 0
    assert first_result.new_version == 1
    assert first_result.state_changed is True
    assert duplicate_result.status is ProcessEventStatus.SUCCEEDED
    assert duplicate_result.old_version == 1
    assert duplicate_result.new_version == 1
    assert duplicate_result.state_changed is False
    assert final_result.status is ProcessEventStatus.SUCCEEDED
    assert final_result.old_version == 1
    assert final_result.new_version == 2
    assert final_result.state_changed is True

    first_state = first_result.state
    duplicate_state = duplicate_result.state
    final_state = final_result.state
    assert first_state is not None
    assert duplicate_state is not None
    assert final_state is not None

    assert first_state.version == 1
    assert duplicate_state.version == 1
    assert final_state.version == 2
    user_events = tuple(
        event for event in event_store.read_by_subject(evening_event.subject)
        if event.event_type == "user.message"
    )
    assert tuple(event.event_id for event in user_events) == (
        evening_event.event_id,
        morning_event.event_id,
    )
    assert state_repository.load("user-1") == final_state

    preference = final_state.get("preferences.study_time")
    assert preference.value == "早上"
    assert tuple(ref.evidence_id for ref in preference.evidence_refs) == (
        evening_event.event_id,
        morning_event.event_id,
    )
