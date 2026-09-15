import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import UUID

import pytest

from self_cognition.application.process_event import ProcessEventService
from self_cognition.application.results import ProcessEventStatus
from self_cognition.cognition.affect.affect_extractor import AffectExtractor
from self_cognition.cognition.semantic.llm_extractor import LLMSemanticExtractor
from self_cognition.cognition.semantic.preference_extractor import (
    PreferenceExtractor,
)
from self_cognition.core.errors import ModelOutputError
from self_cognition.core.evidence import EvidenceSourceKind
from self_cognition.core.events import Event, ModelResponsePayload
from self_cognition.core.state import SubjectState
from self_cognition.infrastructure.llm.openai_responses import (
    OpenAIResponsesCognitionModel,
)
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


class SequenceResponses:
    def __init__(self, responses) -> None:
        self._responses = tuple(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        response = self._responses[index]
        if isinstance(response, Exception):
            raise response
        return response


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
        '"operation":"set","cognition_type":"preference","value":"'
        + value
        + '","confidence":1.0,"evidence_ids":["'
        + str(event.event_id)
        + '"]}]}'
    )

    context = make_context()
    llm_contribution = LLMSemanticExtractor(model).process(event, context)[0]
    rule_contribution = PreferenceExtractor().process(event)[0]

    assert llm_contribution.target_field == rule_contribution.target_field
    assert llm_contribution.value == rule_contribution.value
    assert llm_contribution.confidence == rule_contribution.confidence
    assert llm_contribution.evidence_refs[0] == rule_contribution.evidence_refs[0]
    response_evidence = llm_contribution.evidence_refs[1]
    assert response_evidence.source_kind is EvidenceSourceKind.MODEL_RESPONSE
    assert response_evidence.source_ref == "resp-test-1"
    call = responses.calls[0]
    assert call["max_output_tokens"] == 128
    assert call["store"] is False
    assert 0 < call["timeout"] <= 15
    assert call["text"]["format"]["strict"] is True
    candidate_schema = call["text"]["format"]["schema"]["properties"][
        "candidates"
    ]["items"]
    assert "cognition_type" in candidate_schema["required"]
    emitted = context.drain_emitted_events()
    assert len(emitted) == 1
    assert emitted[0].event_type == "model.response"
    assert emitted[0].causation_id == event.event_id
    assert isinstance(emitted[0].payload, ModelResponsePayload)
    assert emitted[0].payload.model == "test-model"
    assert emitted[0].payload.response_id == "resp-test-1"
    assert response_evidence.evidence_id == emitted[0].event_id


@pytest.mark.parametrize(
    "output_text",
    [
        "not-json",
        '{"unexpected":[]}',
        '{"candidates":[{"target_field":"preferences.study_time"}]}',
        '{"candidates":[{"target_field":"preferences.study_time",'
        '"operation":"set","cognition_type":"unsupported","value":"晚上",'
        '"confidence":1.0,"evidence_ids":["source"]}]}',
    ],
)
def test_invalid_model_structure_never_reaches_reducer(output_text):
    model, _ = make_model(output_text)
    event_store = InMemoryEventStore()
    state_repository = InMemoryStateRepository()
    service = ProcessEventService(
        event_store,
        InMemoryEvidenceRepository(),
        state_repository,
        CognitionEngine(
            (LLMSemanticExtractor(model),),
            CognitiveSpaceService(StateReducer()),
        ),
    )

    result = service.process(
        Event.user_message("user-1", "我喜欢晚上学习"),
        make_context(),
    )

    assert result.status is ProcessEventStatus.FAILED
    assert result.error_type == "ModelOutputError"
    assert state_repository.load("user-1") is None


def test_incomplete_empty_model_response_is_audited_before_failure():
    event = Event.user_message("user-1", "我喜欢晚上学习")
    response = SimpleNamespace(
        id="resp-incomplete",
        output_text="",
        status="incomplete",
        model_dump_json=lambda: '{"status":"incomplete"}',
    )
    model = OpenAIResponsesCognitionModel(
        SimpleNamespace(responses=FakeResponses(response=response)),
        "test-model",
    )

    context = make_context()
    with pytest.raises(ModelOutputError, match="invalid or incomplete"):
        LLMSemanticExtractor(model).process(event, context)

    emitted = context.drain_emitted_events()
    assert len(emitted) == 1
    assert emitted[0].payload.response_id == "resp-incomplete"
    assert emitted[0].payload.raw_output == '{"status":"incomplete"}'


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
        InMemoryEvidenceRepository(),
        state_repository,
        CognitionEngine(
            (LLMSemanticExtractor(model),),
            CognitiveSpaceService(StateReducer()),
        ),
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
        '"operation":"set","cognition_type":"preference",'
        '"value":"晚上","confidence":1.0,'
        '"evidence_ids":["00000000-0000-0000-0000-000000000099"]}]}'
    )

    with pytest.raises(ModelOutputError, match="not supplied to the model"):
        LLMSemanticExtractor(model).process(event, make_context())


def test_cancelled_run_does_not_call_the_model_client():
    model, responses = make_model('{"candidates":[]}')
    context = make_context()
    context.cancel()

    service = ProcessEventService(
        InMemoryEventStore(),
        InMemoryEvidenceRepository(),
        InMemoryStateRepository(),
        CognitionEngine(
            (LLMSemanticExtractor(model),),
            CognitiveSpaceService(StateReducer()),
        ),
    )
    result = service.process(
        Event.user_message("user-1", "我喜欢晚上学习"),
        context,
    )

    assert result.status is ProcessEventStatus.CANCELLED
    assert responses.calls == []


def _valid_semantic_output(event, evidence_id: str) -> str:
    return (
        '{"candidates":[{"target_field":"preferences.study_time",'
        '"operation":"set","cognition_type":"preference","value":"晚上",'
        '"confidence":1.0,"evidence_ids":["'
        + evidence_id
        + '"]}]}'
    )


def test_top_level_shape_is_normalized_without_repair():
    event = Event.user_message("user-1", "我喜欢晚上学习")
    candidate = (
        '{"candidates":[{"target_field":"preferences.study_time",'
        '"operation":"set","cognition_type":"preference","value":"晚上",'
        '"confidence":1.0,"evidence_ids":["'
        + str(event.event_id)
        + '"]}],"confidence":1.0}'
    )
    responses = SequenceResponses(
        [
            SimpleNamespace(
                id="resp-normalized",
                output_text=candidate,
                status="completed",
            )
        ]
    )
    model = OpenAIResponsesCognitionModel(
        SimpleNamespace(responses=responses),
        "test-model",
    )

    contributions = LLMSemanticExtractor(model).process(event, make_context())

    assert len(responses.calls) == 1
    assert contributions[0].target_field == "preferences.study_time"
    assert contributions[0].value == "晚上"


def test_invalid_evidence_id_is_filtered_without_repair():
    event = Event.user_message("user-1", "我喜欢晚上学习")
    invalid = SimpleNamespace(
        id="resp-invalid",
        output_text=(
            '{"candidates":[{"target_field":"preferences.study_time",'
            '"operation":"set","cognition_type":"preference","value":"晚上",'
            '"confidence":1.0,"evidence_ids":["'
            + str(event.event_id)
            + '","run:00000000-0000-0000-0000-000000000001"]}]}'
        ),
        status="completed",
    )
    responses = SequenceResponses([invalid])
    model = OpenAIResponsesCognitionModel(
        SimpleNamespace(responses=responses),
        "test-model",
    )

    contributions = LLMSemanticExtractor(model).process(event, make_context())

    assert len(responses.calls) == 1
    assert contributions[0].value == "晚上"


def test_invalid_json_repair_is_bounded_to_one_retry():
    event = Event.user_message("user-1", "我喜欢晚上学习")
    invalid = SimpleNamespace(
        id="resp-invalid",
        output_text="not-json",
        status="completed",
    )
    responses = SequenceResponses([invalid])
    model = OpenAIResponsesCognitionModel(
        SimpleNamespace(responses=responses),
        "test-model",
    )

    with pytest.raises(ModelOutputError, match="not valid JSON"):
        LLMSemanticExtractor(model).process(event, make_context())

    assert len(responses.calls) == 2


def test_affect_cognition_type_mismatch_is_repaired_once():
    event = Event.user_message("user-1", "我有点无聊")

    def output(cognition_type: str) -> str:
        return json.dumps(
            {
                "candidates": [
                    {
                        "target_field": "affect.current.interaction",
                        "operation": "set",
                        "cognition_type": cognition_type,
                        "value": {
                            "target": "user-1",
                            "goal_ids": [],
                            "emotion": "boredom",
                            "valence": "negative",
                            "scope": "interaction",
                            "initial_intensity": 0.7,
                            "assessed_at": event.occurred_at.isoformat(),
                            "half_life_seconds": 3600.0,
                            "active_threshold": 0.1,
                            "arousal": 0.1,
                            "control": 0.4,
                            "certainty": 0.7,
                        },
                        "confidence": 0.6,
                        "evidence_ids": [str(event.event_id)],
                    }
                ]
            },
            ensure_ascii=False,
        )

    responses = SequenceResponses(
        [
            SimpleNamespace(
                id="resp-invalid",
                output_text=output("preference"),
                status="completed",
            ),
            SimpleNamespace(
                id="resp-valid",
                output_text=output("affect"),
                status="completed",
            ),
        ]
    )
    model = OpenAIResponsesCognitionModel(
        SimpleNamespace(responses=responses),
        "test-model",
        assessment_kind="affect",
    )
    engine = CognitionEngine(
        (AffectExtractor(model),),
        CognitiveSpaceService(StateReducer()),
    )

    state = engine.process(
        event,
        SubjectState.empty("user-1"),
        make_context(),
    )

    assert len(responses.calls) == 2
    assert "affect.current.interaction" in state.entries
    assert state.entries["affect.current.interaction"].value["emotion"] == "boredom"


def test_invalid_json_is_repaired_once():
    event = Event.user_message("user-1", "我喜欢晚上学习")
    responses = SequenceResponses(
        [
            SimpleNamespace(
                id="resp-invalid",
                output_text="not-json",
                status="completed",
            ),
            SimpleNamespace(
                id="resp-valid",
                output_text=_valid_semantic_output(event, str(event.event_id)),
                status="completed",
            ),
        ]
    )
    model = OpenAIResponsesCognitionModel(
        SimpleNamespace(responses=responses),
        "test-model",
    )

    contributions = LLMSemanticExtractor(model).process(event, make_context())

    assert len(responses.calls) == 2
    assert contributions[0].value == "晚上"
    assert "previous_output=not-json" in responses.calls[1]["input"]

@pytest.mark.parametrize("status", ["incomplete", "failed", "cancelled"])
def test_non_completed_status_with_valid_structured_output_is_accepted(status):
    event = Event.user_message("user-1", "我喜欢晚上学习")
    response = SimpleNamespace(
        id="resp-non-completed-valid",
        output_text=_valid_semantic_output(event, str(event.event_id)),
        status=status,
    )
    responses = SequenceResponses([response])
    model = OpenAIResponsesCognitionModel(
        SimpleNamespace(responses=responses),
        "test-model",
    )

    contributions = LLMSemanticExtractor(model).process(event, make_context())

    assert len(responses.calls) == 1
    assert contributions[0].value == "晚上"

def test_single_candidate_recovers_top_level_confidence_and_evidence_ids():
    event = Event.user_message("user-1", "我喜欢晚上学习")
    output = json.dumps(
        {
            "candidates": [
                {
                    "target_field": "preferences.study_time",
                    "operation": "set",
                    "cognition_type": "preference",
                    "value": "晚上",
                }
            ],
            "confidence": 1.0,
            "evidence_ids": [str(event.event_id)],
        },
        ensure_ascii=False,
    )
    responses = SequenceResponses(
        [
            SimpleNamespace(
                id="resp-top-level-candidate-fields",
                output_text=output,
                status="completed",
            )
        ]
    )
    model = OpenAIResponsesCognitionModel(
        SimpleNamespace(responses=responses),
        "test-model",
    )

    contributions = LLMSemanticExtractor(model).process(event, make_context())

    assert len(responses.calls) == 1
    assert contributions[0].target_field == "preferences.study_time"
    assert contributions[0].value == "晚上"
