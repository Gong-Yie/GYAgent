from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from self_cognition.application.results import ProcessEventStatus
from self_cognition.bootstrap import build_container
from self_cognition.core.events import EventSource
from self_cognition.core.identity import GoalPriority, GoalStatus
from self_cognition.core.plans import (
    GoalPlanningRequest,
    PlanBudget,
    PlanStepResult,
    PlanStepResultStatus,
    PlanningModelOutput,
)
from self_cognition.core.scopes import MindScope, SubjectKind, SubjectRef, SubjectScope
from self_cognition.infrastructure.persistence.serialization import (
    event_from_json,
    event_to_json,
)
from self_cognition.runtime.run_context import RunContext


NOW = datetime(2026, 9, 6, 10, tzinfo=timezone.utc)
MIND = SubjectScope.for_mind("mind-21")
USER = SubjectScope(MindScope("mind-21"), SubjectRef(SubjectKind.USER, "alice"))


@dataclass
class FixedClock:
    value: datetime = NOW

    def now(self) -> datetime:
        return self.value


def context() -> RunContext:
    return RunContext(uuid4(), uuid4(), NOW + timedelta(hours=1), clock=FixedClock())


def request(request_id: UUID | None = None) -> GoalPlanningRequest:
    return GoalPlanningRequest(
        request_id or uuid4(),
        MIND,
        USER,
        "goal-21",
        "完成阶段21目标、计划和进度",
        GoalPriority.HIGH,
        ("创建可执行计划", "完成进度验证"),
        PlanBudget(max_steps=4, max_tool_steps=0),
    )


def test_create_progress_pause_resume_and_complete(tmp_path) -> None:
    container = build_container(tmp_path)
    result = container.pursue_goal.create(request(), context())

    assert result.status is ProcessEventStatus.SUCCEEDED
    assert result.plan is not None
    assert result.progress is not None
    assert result.progress.status.value == "active"
    first, second = result.plan.steps

    paused = container.pursue_goal.pause(MIND, result.plan.plan_id, context())
    assert paused.goal is not None and paused.goal.status is GoalStatus.PAUSED
    resumed = container.pursue_goal.continue_goal(MIND, result.plan.plan_id, context())
    assert resumed.goal is not None and resumed.goal.status is GoalStatus.ACTIVE

    for step in (first, second):
        outcome = container.pursue_goal.record_step_result(
            MIND,
            PlanStepResult(
                uuid4(),
                result.plan.plan_id,
                result.plan.version,
                step.step_id,
                PlanStepResultStatus.SUCCEEDED,
                "recorded result",
                NOW,
            ),
            context(),
            source=EventSource.SYSTEM,
        )
        assert outcome.status is ProcessEventStatus.SUCCEEDED

    complete = container.pursue_goal.complete(MIND, result.plan.plan_id, context())
    assert complete.goal is not None and complete.goal.status is GoalStatus.COMPLETED


def test_plan_rejects_cycle(tmp_path) -> None:
    class BadPlanner:
        def create(self, goal, budget, workspace, capabilities, context):
            return PlanningModelOutput(
                "bad",
                "bad-1",
                '{"steps":[{"step_id":"a","description":"a","dependencies":["b"],"required_tool_ids":[],"completion_conditions":["创建可执行计划"],"checkpoint":true,"cancellable":true},{"step_id":"b","description":"b","dependencies":["a"],"required_tool_ids":[],"completion_conditions":["完成进度验证"],"checkpoint":true,"cancellable":true}]}',
            )

        def replan(self, *args, **kwargs):
            raise AssertionError("replan is not part of this case")

    result = build_container(tmp_path, planning_model=BadPlanner()).pursue_goal.create(
        request(), context()
    )
    assert result.status is ProcessEventStatus.FAILED
    assert result.error_type in {"ContractValidationError", "ModelOutputError"}


def test_replanning_replaces_failed_branch_and_keeps_success(tmp_path) -> None:
    container = build_container(tmp_path)
    result = container.pursue_goal.create(request(), context())
    assert result.plan is not None
    first, second = result.plan.steps
    container.pursue_goal.record_step_result(
        MIND,
        PlanStepResult(uuid4(), result.plan.plan_id, 1, first.step_id, PlanStepResultStatus.SUCCEEDED, "ok", NOW),
        context(),
        source=EventSource.TOOL,
    )
    container.pursue_goal.record_step_result(
        MIND,
        PlanStepResult(uuid4(), result.plan.plan_id, 1, second.step_id, PlanStepResultStatus.FAILED, "failed", NOW),
        context(),
        source=EventSource.TOOL,
    )

    revised = container.pursue_goal.replan(MIND, result.plan.plan_id, context())
    assert revised.status is ProcessEventStatus.SUCCEEDED
    assert revised.plan is not None and revised.plan.version == 2
    assert revised.plan.steps[0] == first
    assert revised.plan.steps[1].step_id != second.step_id


def test_goal_create_is_idempotent(tmp_path) -> None:
    container = build_container(tmp_path)
    goal_request = request(uuid4())
    first = container.pursue_goal.create(goal_request, context())
    second = container.pursue_goal.create(goal_request, context())

    assert first.plan == second.plan
    assert second.reused is True
    assert len(
        [
            event
            for event in container.event_store.read_by_subject(MIND)
            if event.event_type == "goal.requested"
        ]
    ) == 1


def test_planning_events_round_trip_through_json(tmp_path) -> None:
    container = build_container(tmp_path)
    result = container.pursue_goal.create(request(), context())
    assert result.plan is not None
    planned = next(
        event
        for event in container.event_store.read_by_subject(MIND)
        if event.event_type == "goal.planned"
    )
    assert event_from_json(event_to_json(planned)) == planned

    step = result.plan.steps[0]
    step_result = PlanStepResult(
        uuid4(),
        result.plan.plan_id,
        result.plan.version,
        step.step_id,
        PlanStepResultStatus.SUCCEEDED,
        "serialized result",
        NOW,
    )
    recorded = container.pursue_goal.record_step_result(
        MIND,
        step_result,
        context(),
        source=EventSource.TOOL,
    )
    assert recorded.event_id is not None
    stored = next(
        event
        for event in container.event_store.read_by_subject(MIND)
        if event.event_id == recorded.event_id
    )
    assert event_from_json(event_to_json(stored)) == stored
