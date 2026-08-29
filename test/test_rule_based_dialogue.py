from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from self_cognition.application.process_event import ProcessEventService
from self_cognition.cognition.semantic.preference_extractor import (
    PreferenceExtractor,
)
from self_cognition.core.events import Event
from self_cognition.core.workspace import Workspace, WorkspaceBuilder, WorkspaceItem
from self_cognition.executive.dialogue.rule_based import (
    DialogueResponse,
    RuleBasedDialogueModel,
)
from self_cognition.infrastructure.persistence.in_memory_event_store import (
    InMemoryEventStore,
)
from self_cognition.infrastructure.persistence.in_memory_state_repository import (
    InMemoryStateRepository,
)
from self_cognition.runtime.engine import CognitionEngine
from self_cognition.runtime.reducer import StateReducer
from self_cognition.runtime.run_context import RunContext


QUESTION = "我喜欢什么时候学习？"


def test_answers_from_the_remembered_preference_with_its_source():
    event = Event.user_message("user-1", "我喜欢晚上学习")
    service = ProcessEventService(
        event_store=InMemoryEventStore(),
        state_repository=InMemoryStateRepository(),
        engine=CognitionEngine(
            modules=(PreferenceExtractor(),),
            reducer=StateReducer(),
        ),
    )
    result = service.process(
        event,
        RunContext(
            run_id=UUID(int=1),
            correlation_id=UUID(int=100),
            deadline=datetime.now(timezone.utc) + timedelta(minutes=1),
        ),
    )
    assert result.state is not None
    state = result.state
    workspace = WorkspaceBuilder().build(QUESTION, state)

    response = RuleBasedDialogueModel().respond(QUESTION, workspace)

    assert response == DialogueResponse(
        text="你喜欢晚上学习。",
        source_event_ids=(event.event_id,),
    )
    assert "晚上" in response.text


def test_says_it_does_not_know_when_the_workspace_is_empty():
    workspace = Workspace(
        subject_id="user-1",
        state_version=0,
        items=(),
    )

    response = RuleBasedDialogueModel().respond(QUESTION, workspace)

    assert response.text == "我还不知道你喜欢什么时候学习。"
    assert response.source_event_ids == ()


def test_does_not_invent_a_preference_from_an_unrelated_item():
    workspace = Workspace(
        subject_id="user-1",
        state_version=1,
        items=(
            WorkspaceItem(
                target_field="profile.name",
                content="小明",
                evidence_event_ids=(uuid4(),),
                confidence=1.0,
                selection_reason="test unrelated item",
            ),
        ),
    )

    response = RuleBasedDialogueModel().respond(QUESTION, workspace)

    assert response.text == "我还不知道你喜欢什么时候学习。"
    assert "小明" not in response.text
    assert response.source_event_ids == ()
