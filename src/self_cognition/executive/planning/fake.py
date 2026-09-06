import json
from dataclasses import replace

from self_cognition.core.identity import CapabilityRecord, GoalRecord
from self_cognition.core.plans import (
    Plan,
    PlanBudget,
    PlanDraft,
    PlanProgress,
    PlanStep,
    PlanStepStatus,
    PlanningModelOutput,
    plan_draft_to_dict,
)
from self_cognition.core.workspace import WorkspacePacket
from self_cognition.runtime.run_context import RunContext


class RulePlanningModel:
    def create(
        self,
        goal: GoalRecord,
        budget: PlanBudget,
        workspace: WorkspacePacket,
        capabilities: tuple[CapabilityRecord, ...],
        context: RunContext,
    ) -> PlanningModelOutput:
        del budget, workspace, capabilities
        steps = []
        for index, condition in enumerate(goal.completion_conditions, start=1):
            step_id = f"step-{index}"
            steps.append(
                PlanStep(
                    step_id,
                    f"Complete: {condition}",
                    () if index == 1 else (f"step-{index - 1}",),
                    completion_conditions=(condition,),
                )
            )
        return self._output(PlanDraft(tuple(steps)), "create", context)

    def replan(
        self,
        goal: GoalRecord,
        plan: Plan,
        progress: PlanProgress,
        workspace: WorkspacePacket,
        capabilities: tuple[CapabilityRecord, ...],
        context: RunContext,
    ) -> PlanningModelOutput:
        del goal, workspace, capabilities
        failed = {
            item.step.step_id
            for item in progress.steps
            if item.status in {PlanStepStatus.FAILED, PlanStepStatus.CANCELLED}
        }
        affected = set(failed)
        while True:
            found = {
                step.step_id
                for step in plan.steps
                if set(step.dependencies) & affected
            }
            if found.issubset(affected):
                break
            affected.update(found)
        replacements = {
            step_id: f"retry-v{plan.version + 1}-{index}"
            for index, step_id in enumerate(
                (step.step_id for step in plan.steps if step.step_id in affected),
                start=1,
            )
        }
        steps = tuple(
            (
                replace(
                    step,
                    step_id=replacements[step.step_id],
                    description=f"Retry: {step.description}",
                    dependencies=tuple(
                        replacements.get(dependency, dependency)
                        for dependency in step.dependencies
                    ),
                )
                if step.step_id in affected
                else step
            )
            for step in plan.steps
        )
        return self._output(PlanDraft(steps), "replan", context)

    @staticmethod
    def _output(
        draft: PlanDraft,
        operation: str,
        context: RunContext,
    ) -> PlanningModelOutput:
        return PlanningModelOutput(
            "rule-planner-v1",
            f"rule-{operation}-{context.run_id}",
            json.dumps(plan_draft_to_dict(draft), ensure_ascii=False),
        )
