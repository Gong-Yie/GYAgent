from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import UUID

import pytest

from self_cognition.core.dialogue import DialogueModelOutput
from self_cognition.infrastructure.llm.action_responses import (
    OpenAIResponsesActionModel,
)
from self_cognition.infrastructure.llm.dialogue_responses import (
    OpenAIResponsesDialogueModel,
)
from self_cognition.infrastructure.llm.planning_responses import (
    OpenAIResponsesPlanningModel,
)
from self_cognition.infrastructure.llm.router import (
    ModelRegistration,
    ModelRouter,
    RoutedDialogueModel,
)
from self_cognition.runtime.run_context import RunContext


class FakeResponses:
    def __init__(self, response: object) -> None:
        self._response = response
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self._response


def fake_client(*, status: str | None, output_text: str) -> SimpleNamespace:
    return SimpleNamespace(
        responses=FakeResponses(
            SimpleNamespace(
                id="resp-status",
                output_text=output_text,
                status=status,
            )
        )
    )


def make_context() -> RunContext:
    return RunContext(
        run_id=UUID(int=11),
        correlation_id=UUID(int=12),
        deadline=datetime.now(timezone.utc) + timedelta(minutes=5),
    )


@pytest.mark.parametrize("status", ["incomplete", None, "unknown"])
def test_dialogue_accepts_usable_output_with_non_completed_status(status):
    model = OpenAIResponsesDialogueModel(
        fake_client(status=status, output_text='{"text":"ok"}'),
        "test-model",
    )

    output = model._call({}, "instructions", {}, "dialogue_answer", make_context())

    assert output.error_type is None


@pytest.mark.parametrize("status", ["incomplete", None, "unknown"])
def test_planning_accepts_usable_output_with_non_completed_status(status):
    model = OpenAIResponsesPlanningModel(
        fake_client(status=status, output_text='{"steps":[]}'),
        "test-model",
    )

    output = model._call({}, "instructions", "plan_create", make_context())

    assert output.error_type is None


@pytest.mark.parametrize("status", ["incomplete", None, "unknown"])
def test_action_accepts_usable_output_with_non_completed_status(status):
    model = OpenAIResponsesActionModel(
        fake_client(status=status, output_text='{"status":"allowed"}'),
        "test-model",
    )

    output = model._call(
        {},
        "instructions",
        "action_decision",
        {},
        make_context(),
    )

    assert output.error_type is None


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_dialogue_accepts_hard_failure_status_with_parseable_output(status):
    model = OpenAIResponsesDialogueModel(
        fake_client(status=status, output_text='{"text":"ok"}'),
        "test-model",
    )

    output = model._call({}, "instructions", {}, "dialogue_answer", make_context())

    assert output.error_type is None


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_dialogue_rejects_hard_failure_status_with_unparseable_output(status):
    model = OpenAIResponsesDialogueModel(
        fake_client(status=status, output_text="not-json"),
        "test-model",
    )

    output = model._call({}, "instructions", {}, "dialogue_answer", make_context())

    assert output.error_type == "ModelOutputError"


def test_dialogue_empty_output_is_still_invalid():
    model = OpenAIResponsesDialogueModel(
        fake_client(status="incomplete", output_text=""),
        "test-model",
    )

    output = model._call({}, "instructions", {}, "dialogue_answer", make_context())

    assert output.error_type == "ModelOutputError"

class EmptyThenValidRepairDialogue:
    def __init__(self) -> None:
        self.calls = 0

    def repair(self, workspace, previous, error, context):
        self.calls += 1
        if self.calls == 1:
            return DialogueModelOutput(
                "test-model",
                "resp-empty-repair",
                "{}",
                "ModelOutputError",
            )
        return DialogueModelOutput(
            "test-model",
            "resp-valid-repair",
            '{"text":"ok"}',
        )


def test_routed_dialogue_repair_retries_once_on_invalid_provider_output():
    stub = EmptyThenValidRepairDialogue()
    router = ModelRouter(
        (ModelRegistration("dialogue", "dialogue-test", stub),)
    )
    model = RoutedDialogueModel(router)

    repaired = model.repair(
        SimpleNamespace(subject=None),
        "previous",
        ValueError("invalid output"),
        make_context(),
    )

    assert stub.calls == 2
    assert repaired is not None
    assert repaired.error_type is None
    assert repaired.response_id == "resp-valid-repair"
