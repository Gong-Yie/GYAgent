import json
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest

from self_cognition.application.results import ProcessEventStatus
from self_cognition.bootstrap import build_container
from self_cognition.core.actions import (
    ActionDecisionDraft,
    ActionDecisionStatus,
    ActionDraft,
    ActionModelOutput,
    ActionProposalRequest,
    ActionResult,
    ActionResultStatus,
    ExpectedSideEffect,
    action_draft_to_dict,
    decision_draft_to_dict,
)
from self_cognition.core.errors import ContractValidationError, RunCancelledError
from self_cognition.core.identity import (
    CapabilityKind,
    CapabilityPermission,
    GoalPriority,
)
from self_cognition.core.plans import (
    GoalPlanningRequest,
    PlanBudget,
    PlanDraft,
    PlanStep,
    PlanningModelOutput,
    plan_draft_to_dict,
)
from self_cognition.core.scopes import MindScope, SubjectKind, SubjectRef, SubjectScope
from self_cognition.infrastructure.persistence.serialization import (
    event_from_json,
    event_to_json,
)
from self_cognition.infrastructure.llm.action_responses import (
    OpenAIResponsesActionModel,
)
from self_cognition.runtime.run_context import RunContext
from self_cognition.tools.registry import CapabilityRegistration


NOW = datetime(2026, 9, 8, 10, tzinfo=timezone.utc)
MIND = SubjectScope.for_mind("mind-22")
USER = SubjectScope(MindScope("mind-22"), SubjectRef(SubjectKind.USER, "alice"))
EFFECT = ExpectedSideEffect("file_write", "write one notes file", True)


@dataclass
class FixedClock:
    value: datetime = NOW

    def now(self) -> datetime:
        return self.value


class ToolPlanner:
    def create(self, goal, budget, workspace, capabilities, context):
        del budget, workspace, capabilities
        draft = PlanDraft(
            (
                PlanStep(
                    "write-notes",
                    "Write the requested notes",
                    required_tool_ids=("file.write",),
                    completion_conditions=goal.completion_conditions,
                ),
            )
        )
        return PlanningModelOutput(
            "tool-planner",
            f"plan-{context.run_id}",
            json.dumps(plan_draft_to_dict(draft)),
        )

    def replan(self, *args, **kwargs):
        raise AssertionError("replan is not part of stage 22")


class FixedActionModel:
    def __init__(
        self,
        status: ActionDecisionStatus = ActionDecisionStatus.ALLOWED,
        *,
        tool_id: str = "file.write",
        valid_until: datetime = NOW + timedelta(minutes=30),
    ) -> None:
        self.status = status
        self.tool_id = tool_id
        self.valid_until = valid_until
        self.calls: list[str] = []

    def propose(self, plan, step, workspace, tools, context):
        del plan, step, workspace, tools
        self.calls.append("propose")
        draft = ActionDraft(
            self.tool_id,
            {"path": "notes.txt", "content": "stage 22"},
            (EFFECT,),
        )
        return ActionModelOutput(
            "fixed-action",
            f"propose-{context.run_id}",
            json.dumps(action_draft_to_dict(draft)),
        )

    def decide(self, request, workspace, context):
        del request
        self.calls.append("decide")
        draft = ActionDecisionDraft(
            self.status,
            f"decision: {self.status.value}",
            ("finish the active goal",),
            "requested by the current user relationship",
            ("bounded file write",),
            tuple(ref.evidence_id for ref in workspace.evidence_refs),
            self.valid_until,
            (
                NOW + timedelta(minutes=10)
                if self.status is ActionDecisionStatus.DELAYED
                else None
            ),
            (
                "Confirm writing the notes file"
                if self.status is ActionDecisionStatus.CONFIRMATION_REQUIRED
                else None
            ),
        )
        return ActionModelOutput(
            "fixed-action",
            f"decide-{context.run_id}",
            json.dumps(decision_draft_to_dict(draft)),
        )


def context(*, cancelled: bool = False) -> RunContext:
    return RunContext(
        uuid4(),
        uuid4(),
        NOW + timedelta(hours=1),
        cancelled=cancelled,
        clock=FixedClock(),
    )


def goal_request() -> GoalPlanningRequest:
    return GoalPlanningRequest(
        uuid4(),
        MIND,
        USER,
        "goal-22",
        "Write notes with a registered tool",
        GoalPriority.HIGH,
        ("notes are written",),
        PlanBudget(max_steps=1, max_tool_steps=1),
    )


def container(tmp_path, model: FixedActionModel):
    result = build_container(
        tmp_path,
        planning_model=ToolPlanner(),
        action_model=model,
    )
    result.capability_registry.register(
        CapabilityRegistration(
            "file.write",
            "File writer",
            CapabilityKind.TOOL,
            CapabilityPermission.GRANTED,
            description="Write UTF-8 text inside an allowed root",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {"bytes_written": {"type": "integer"}},
                "required": ["bytes_written"],
                "additionalProperties": False,
            },
            expected_side_effects=(EFFECT,),
        )
    )
    return result


def create_action(container, run_context: RunContext | None = None):
    created = container.pursue_goal.create(goal_request(), context())
    assert created.plan is not None
    request = ActionProposalRequest(
        uuid4(),
        MIND,
        created.plan.plan_id,
        created.plan.version,
        created.plan.steps[0].step_id,
    )
    return request, container.action.prepare(request, run_context or context())


@pytest.mark.parametrize(
    "decision_status",
    tuple(ActionDecisionStatus),
)
def test_action_proposal_and_all_llm_decisions_are_audited(
    tmp_path,
    decision_status: ActionDecisionStatus,
) -> None:
    model = FixedActionModel(decision_status)
    app = container(tmp_path, model)

    request, outcome = create_action(app)

    assert outcome.status is ProcessEventStatus.SUCCEEDED
    assert outcome.request is not None
    assert outcome.decision is not None
    assert outcome.request.plan_id == request.plan_id
    assert outcome.request.plan_version == request.plan_version
    assert outcome.request.step_id == request.step_id
    assert outcome.request.tool_id == "file.write"
    assert outcome.decision.status is decision_status
    assert outcome.decision.one_time_scope == outcome.request.action_id
    assert model.calls == ["propose", "decide"]
    events = app.event_store.read_by_subject(MIND)
    assert [
        event.event_type for event in events if event.event_type.startswith("action.")
    ] == ["action.started", "action.proposed", "action.decided"]
    decided = next(event for event in events if event.event_type == "action.decided")
    assert event_from_json(event_to_json(decided)) == decided
    other_mind = SubjectScope.for_mind("other-mind")
    with pytest.raises(ContractValidationError, match="owner does not match"):
        replace(
            decided,
            subject=other_mind,
            scope=replace(decided.scope, owner=other_mind),
        )


def test_action_request_is_idempotent_and_checks_plan_scope_and_cancellation(
    tmp_path,
) -> None:
    model = FixedActionModel()
    app = container(tmp_path, model)
    request, first = create_action(app)

    second = app.action.prepare(request, context())

    assert second.reused is True
    assert second.request == first.request
    assert second.decision == first.decision
    assert model.calls == ["propose", "decide"]
    with pytest.raises(ContractValidationError, match="unknown plan"):
        app.action.prepare(
            replace(request, owner=SubjectScope.for_mind("other-mind")),
            context(),
        )
    with pytest.raises(ContractValidationError, match="stale"):
        app.action.prepare(
            replace(request, request_id=uuid4(), plan_version=2),
            context(),
        )
    with pytest.raises(RunCancelledError):
        app.action.prepare(
            replace(request, request_id=uuid4()),
            context(cancelled=True),
        )


def test_invalid_tool_or_expired_decision_fails_without_an_action_decision(
    tmp_path,
) -> None:
    wrong_tool = container(tmp_path / "wrong", FixedActionModel(tool_id="shell.run"))
    _, rejected = create_action(wrong_tool)
    assert rejected.status is ProcessEventStatus.FAILED
    assert rejected.error_type == "ContractValidationError"

    expired = container(
        tmp_path / "expired",
        FixedActionModel(valid_until=NOW - timedelta(seconds=1)),
    )
    _, rejected = create_action(expired)
    assert rejected.status is ProcessEventStatus.FAILED
    assert rejected.error_type in {"ContractValidationError", "ModelOutputError"}


@pytest.mark.parametrize("result_status", tuple(ActionResultStatus))
def test_standard_action_results_are_idempotent_and_conflicts_are_rejected(
    tmp_path,
    result_status: ActionResultStatus,
) -> None:
    app = container(tmp_path, FixedActionModel())
    _, prepared = create_action(app)
    assert prepared.request is not None
    result = ActionResult(
        uuid4(),
        prepared.request.action_id,
        MIND,
        result_status,
        f"tool result: {result_status.value}",
        (
            {"bytes_written": 8}
            if result_status is not ActionResultStatus.CANCELLED
            else None
        ),
        (EFFECT,),
        NOW,
        (
            "ToolFailure"
            if result_status
            in {ActionResultStatus.FAILED, ActionResultStatus.TIMED_OUT}
            else None
        ),
    )

    first = app.action.record_result(MIND, result, context())
    second = app.action.record_result(MIND, result, context())

    assert first.status is ProcessEventStatus.SUCCEEDED
    assert second.reused is True
    with pytest.raises(ContractValidationError, match="different recorded result"):
        app.action.record_result(
            MIND,
            replace(result, summary="conflicting result"),
            context(),
        )
    stored = next(
        event
        for event in app.event_store.read_by_subject(MIND)
        if event.event_type == "action.result"
    )
    assert event_from_json(event_to_json(stored)) == stored


def test_non_allowed_action_cannot_record_a_tool_result(tmp_path) -> None:
    app = container(tmp_path, FixedActionModel(ActionDecisionStatus.REJECTED))
    _, prepared = create_action(app)
    assert prepared.request is not None
    result = ActionResult(
        uuid4(),
        prepared.request.action_id,
        MIND,
        ActionResultStatus.SUCCEEDED,
        "must not be recorded",
        {},
        (),
        NOW,
    )
    with pytest.raises(ContractValidationError, match="only an allowed action"):
        app.action.record_result(MIND, result, context())


def test_responses_adapter_uses_two_stateless_structured_calls(tmp_path) -> None:
    class Response:
        def __init__(self, response_id: str, output_text: str) -> None:
            self.id = response_id
            self.output_text = output_text
            self.status = "completed"

    class Responses:
        def __init__(self) -> None:
            self.calls = []
            self.outputs = [
                Response(
                    "proposal-response",
                    json.dumps(
                        {
                            "tool_id": "file.write",
                            "arguments_json": json.dumps(
                                {"path": "notes.txt", "content": "stage 22"}
                            ),
                            "expected_side_effects": [
                                {
                                    "effect_type": EFFECT.effect_type,
                                    "description": EFFECT.description,
                                    "reversible": EFFECT.reversible,
                                }
                            ],
                        }
                    ),
                ),
                Response(
                    "decision-response",
                    json.dumps(
                        {
                            "status": "allowed",
                            "reason": "bounded action",
                            "value_basis": ["finish the active goal"],
                            "relationship_context": "current user request",
                            "risks": ["bounded file write"],
                            "evidence_ids": [],
                            "valid_until": (
                                NOW + timedelta(minutes=30)
                            ).isoformat(),
                            "not_before": None,
                            "confirmation_prompt": None,
                        }
                    ),
                ),
            ]

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return self.outputs.pop(0)

    class Client:
        def __init__(self) -> None:
            self.responses = Responses()

        def close(self) -> None:
            pass

    client = Client()
    app = container(
        tmp_path,
        OpenAIResponsesActionModel(client, "test-model"),
    )

    _, prepared = create_action(app)

    assert prepared.status is ProcessEventStatus.SUCCEEDED
    assert prepared.request is not None
    assert prepared.request.arguments == {
        "path": "notes.txt",
        "content": "stage 22",
    }
    assert len(client.responses.calls) == 2
    assert all(call["store"] is False for call in client.responses.calls)
    assert all("previous_response_id" not in call for call in client.responses.calls)
