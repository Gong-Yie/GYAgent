from self_cognition.core.errors import ContractValidationError
from self_cognition.core.identity import (
    CapabilityKind,
    CapabilityRecord,
    GoalRecord,
    GoalStatus,
)
from self_cognition.core.plans import (
    Plan,
    PlanProgress,
    PlanStepStatus,
)


class PlanValidator:
    def validate(
        self,
        goal: GoalRecord,
        plan: Plan,
        capabilities: tuple[CapabilityRecord, ...],
    ) -> None:
        if plan.goal_id != goal.goal_id:
            raise ContractValidationError("plan goal does not match")
        if goal.status is not GoalStatus.ACTIVE:
            raise ContractValidationError("planning requires an active goal")
        expected = set(goal.completion_conditions)
        declared = {
            condition
            for step in plan.steps
            for condition in step.completion_conditions
        }
        if declared != expected:
            raise ContractValidationError(
                "plan must cover exactly the goal completion conditions"
            )
        available = {item.capability_id: item for item in capabilities}
        for step in plan.steps:
            for tool_id in step.required_tool_ids:
                record = available.get(tool_id)
                if record is None:
                    raise ContractValidationError(f"unknown plan tool: {tool_id}")
                if record.kind is not CapabilityKind.TOOL:
                    raise ContractValidationError(
                        f"plan capability is not a tool: {tool_id}"
                    )
                if not record.available:
                    raise ContractValidationError(
                        f"plan tool is unavailable: {tool_id}"
                    )

    def validate_revision(
        self,
        previous: Plan,
        revised: Plan,
        progress: PlanProgress,
    ) -> None:
        if revised.plan_id != previous.plan_id or revised.goal_id != previous.goal_id:
            raise ContractValidationError("replanning cannot change plan ownership")
        if revised.version != previous.version + 1:
            raise ContractValidationError("replanning must increment the version")
        if revised.budget != previous.budget:
            raise ContractValidationError("replanning cannot change the approved budget")
        if revised.created_at < previous.created_at:
            raise ContractValidationError("replanning time cannot move backwards")

        failed = {
            item.step.step_id
            for item in progress.steps
            if item.status in {PlanStepStatus.FAILED, PlanStepStatus.CANCELLED}
        }
        if not failed:
            raise ContractValidationError("replanning requires a failed step result")
        affected = _descendants(previous, failed)
        old_by_id = {step.step_id: step for step in previous.steps}
        new_by_id = {step.step_id: step for step in revised.steps}
        if affected & set(new_by_id):
            raise ContractValidationError(
                "replacement steps require new IDs"
            )
        for step_id in set(old_by_id) - affected:
            if new_by_id.get(step_id) != old_by_id[step_id]:
                raise ContractValidationError(
                    "replanning changed a step outside the failed downstream branch"
                )
        if not set(new_by_id) - set(old_by_id):
            raise ContractValidationError("replanning requires replacement steps")


def _descendants(plan: Plan, roots: set[str]) -> set[str]:
    affected = set(roots)
    changed = True
    while changed:
        found = {
            step.step_id
            for step in plan.steps
            if set(step.dependencies) & affected
        }
        changed = not found.issubset(affected)
        affected.update(found)
    return affected
