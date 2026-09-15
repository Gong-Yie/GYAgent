import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from self_cognition.application.results import ProcessEventStatus
from self_cognition.bootstrap import build_container
from self_cognition.core.dialogue import DialogueModelOutput, DialogueRequest
from self_cognition.core.events import EventEnvelope
from self_cognition.core.identity import GoalPriority
from self_cognition.core.plans import (
    GoalPlanningRequest,
    PlanBudget,
    PlanningModelOutput,
)
from self_cognition.core.scopes import SubjectScope
from self_cognition.executive.planning.fake import RulePlanningModel
from self_cognition.runtime.run_context import RunContext


def _context() -> RunContext:
    return RunContext(
        uuid4(),
        uuid4(),
        datetime.now(timezone.utc) + timedelta(minutes=5),
    )


class RepairingDialogueModel:
    def __init__(self) -> None:
        self.repairs = 0

    def generate(self, workspace, context) -> DialogueModelOutput:
        return DialogueModelOutput(
            "fake",
            "dialogue-generate-1",
            json.dumps(
                {
                    "text": "你好",
                    "claims": [
                        {
                            "text": "用户发来问候。",
                            "evidence_ids": [],
                            "stance": "uncertain",
                        }
                    ],
                    "disclosure": {
                        "action": "withhold",
                        "scope": "private",
                        "reason": "greeting",
                        "evidence_ids": [],
                        "overrides_intent": False,
                    },
                }
            ),
        )

    def repair(self, workspace, previous, error, context) -> DialogueModelOutput:
        self.repairs += 1
        return DialogueModelOutput(
            "fake",
            "dialogue-repair-1",
            json.dumps(
                {
                    "text": "你好",
                    "claims": [],
                    "disclosure": {
                        "action": "withhold",
                        "scope": "private",
                        "reason": "greeting",
                        "evidence_ids": [],
                        "overrides_intent": False,
                    },
                }
            ),
        )

    def review(self, workspace, draft, context) -> DialogueModelOutput:
        return DialogueModelOutput(
            "fake",
            "dialogue-review-1",
            json.dumps({"supported": True, "reason": "ok"}),
        )


class RepairingPlanningModel:
    def __init__(self) -> None:
        self.repairs = 0

    def create(self, goal, budget, workspace, capabilities, context) -> PlanningModelOutput:
        return PlanningModelOutput(
            "fake",
            "planning-create-1",
            json.dumps({"steps": []}),
        )

    def repair(
        self,
        goal,
        budget,
        workspace,
        capabilities,
        previous,
        error,
        context,
    ) -> PlanningModelOutput:
        self.repairs += 1
        valid = RulePlanningModel().create(
            goal,
            budget,
            workspace,
            capabilities,
            context,
        )
        return PlanningModelOutput("fake", "planning-repair-1", valid.raw_output)

    def replan(self, *args, **kwargs):
        raise AssertionError("replan is not part of this test")


def test_dialogue_repairs_invalid_claim_text(tmp_path):
    model = RepairingDialogueModel()
    container = build_container(tmp_path, dialogue_model=model)
    try:
        user = SubjectScope.legacy_user("alice")
        result = container.converse.converse(
            DialogueRequest(EventEnvelope.user_message(user, "你好")),
            _context(),
        )
    finally:
        container.lifecycle.stop()

    assert result.status is ProcessEventStatus.SUCCEEDED
    assert model.repairs == 1


def test_planning_repairs_invalid_structure(tmp_path):
    model = RepairingPlanningModel()
    container = build_container(tmp_path, planning_model=model)
    try:
        user = SubjectScope.legacy_user("alice")
        owner = SubjectScope.for_mind(user.mind.mind_id)
        result = container.pursue_goal.create(
            GoalPlanningRequest(
                uuid4(),
                owner,
                user,
                "goal-stage32",
                "完成真实模型修复回归",
                GoalPriority.HIGH,
                ("生成一个可执行计划",),
                PlanBudget(max_steps=3, max_tool_steps=0),
            ),
            _context(),
        )
    finally:
        container.lifecycle.stop()

    assert result.status is ProcessEventStatus.SUCCEEDED
    assert result.plan is not None
    assert model.repairs == 1