from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import UUID

import pytest

from self_cognition.application.process_event import ProcessEventService
from self_cognition.application.results import ProcessEventStatus
from self_cognition.cognition.semantic.llm_extractor import LLMSemanticExtractor
from self_cognition.cognition.semantic.preference_extractor import (
    PreferenceExtractor,
)
from self_cognition.core.errors import ModelOutputError
from self_cognition.core.events import Event
from self_cognition.infrastructure.llm.openai_responses import (
    OpenAIResponsesCognitionModel,
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


def make_context() -> RunContext:
    return RunContext(
        run_id=UUID(int=1),
        correlation_id=UUID(int=100),
        deadline=datetime.now(timezone.utc) + timedelta(minutes=1),
    )


class FakeResponses:
    def __init__(self, response=None, error=None) -> None:
        self._response = response
        self._error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response


def make_model(output_text: str):
    responses = FakeResponses(
        SimpleNamespace(id="resp-test-1", output_text=output_text)
    )
    return OpenAIResponsesCognitionModel(
        SimpleNamespace(responses=responses),
        "test-model",
        timeout_seconds=15,
        max_output_tokens=128,
    ), responses


@pytest.mark.parametrize(
    ("content", "value"),
    [
        ("我喜欢晚上学习", "晚上"),
        ("我喜欢早上学习", "早上"),
    ],
)
def test_llm_module_matches_rule_module_for_simple_preferences(content, value):
    event = Event.user_message("user-1", content)
    model, responses = make_model(
        '{"candidates":[{"target_field":"preferences.study_time",'
        '"operation":"set","value":"'
        + value
        + '","confidence":1.0,"evidence_event_ids":["'
        + str(event.event_id)
        + '"]}]}'
    )

    llm_contribution = LLMSemanticExtractor(model).process(event, make_context())[0]
    rule_contribution = PreferenceExtractor().process(event)[0]

    assert llm_contribution.target_field == rule_contribution.target_field
    assert llm_contribution.value == rule_contribution.value
    assert llm_contribution.confidence == rule_contribution.confidence
    assert llm_contribution.evidence_event_ids == rule_contribution.evidence_event_ids
    assert llm_contribution.source_model_response_id == "resp-test-1"
    call = responses.calls[0]
    assert call["max_output_tokens"] == 128
    assert call["store"] is False
    assert 0 < call["timeout"] <= 15
    assert call["text"]["format"]["strict"] is True


@pytest.mark.parametrize(
    "output_text",
    [
        "not-json",
        '{"unexpected":[]}',
        '{"candidates":[{"target_field":"preferences.study_time"}]}',
    ],
)
def test_invalid_model_structure_never_reaches_reducer(output_text):
    model, _ = make_model(output_text)
    event_store = InMemoryEventStore()
    state_repository = InMemoryStateRepository()
    service = ProcessEventService(
        event_store,
        state_repository,
        CognitionEngine((LLMSemanticExtractor(model),), StateReducer()),
    )

    result = service.process(
        Event.user_message("user-1", "我喜欢晚上学习"),
        make_context(),
    )

    assert result.status is ProcessEventStatus.FAILED
    assert result.error_type == "ModelOutputError"
    assert state_repository.load("user-1") is None


def test_model_timeout_does_not_modify_state():
    model = OpenAIResponsesCognitionModel(
        SimpleNamespace(
            responses=FakeResponses(error=TimeoutError("simulated timeout"))
        ),
        "test-model",
    )
    state_repository = InMemoryStateRepository()
    service = ProcessEventService(
        InMemoryEventStore(),
        state_repository,
        CognitionEngine((LLMSemanticExtractor(model),), StateReducer()),
    )

    result = service.process(
        Event.user_message("user-1", "我喜欢晚上学习"),
        make_context(),
    )

    assert result.status is ProcessEventStatus.FAILED
    assert result.error_type == "ModelTimeoutError"
    assert state_repository.load("user-1") is None


def test_candidate_without_source_event_evidence_is_rejected():
    event = Event.user_message("user-1", "我喜欢晚上学习")
    model, _ = make_model(
        '{"candidates":[{"target_field":"preferences.study_time",'
        '"operation":"set","value":"晚上","confidence":1.0,'
        '"evidence_event_ids":["00000000-0000-0000-0000-000000000099"]}]}'
    )

    with pytest.raises(ModelOutputError, match="source event"):
        LLMSemanticExtractor(model).process(event, make_context())


def test_cancelled_run_does_not_call_the_model_client():
    model, responses = make_model('{"candidates":[]}')
    context = make_context()
    context.cancel()

    service = ProcessEventService(
        InMemoryEventStore(),
        InMemoryStateRepository(),
        CognitionEngine((LLMSemanticExtractor(model),), StateReducer()),
    )
    result = service.process(
        Event.user_message("user-1", "我喜欢晚上学习"),
        context,
    )

    assert result.status is ProcessEventStatus.CANCELLED
    assert responses.calls == []
