from self_cognition.core.identity import GoalRecord, GoalStatus
from self_cognition.core.plans import (
    Plan,
    PlanProgress,
    PlanProgressStatus,
    PlanStepProgress,
    PlanStepResult,
    PlanStepResultStatus,
    PlanStepStatus,
)


RESULT_TO_STEP_STATUS = {
    PlanStepResultStatus.SUCCEEDED: PlanStepStatus.SUCCEEDED,
    PlanStepResultStatus.FAILED: PlanStepStatus.FAILED,
    PlanStepResultStatus.CANCELLED: PlanStepStatus.CANCELLED,
    PlanStepResultStatus.PARTIAL: PlanStepStatus.PARTIAL,
}


def calculate_progress(
    plan: Plan,
    goal: GoalRecord,
    results: tuple[PlanStepResult, ...],
) -> PlanProgress:
    relevant = {
        result.step_id: result
        for result in sorted(
            (
                item
                for item in results
                if item.plan_id == plan.plan_id
                and item.plan_version <= plan.version
            ),
            key=lambda item: (item.recorded_at, item.result_id.int),
        )
        if result.step_id in {step.step_id for step in plan.steps}
    }
    statuses: dict[str, PlanStepStatus] = {}
    progress: list[PlanStepProgress] = []
    satisfied: set[str] = set()
    remaining = list(plan.steps)
    while remaining:
        progressed = False
        for step in tuple(remaining):
            if any(dependency not in statuses for dependency in step.dependencies):
                continue
            result = relevant.get(step.step_id)
            if result is not None:
                status = RESULT_TO_STEP_STATUS[result.status]
            elif any(
                statuses[dependency]
                in {
                    PlanStepStatus.FAILED,
                    PlanStepStatus.CANCELLED,
                    PlanStepStatus.BLOCKED,
                }
                for dependency in step.dependencies
            ):
                status = PlanStepStatus.BLOCKED
            elif all(
                statuses[dependency] is PlanStepStatus.SUCCEEDED
                for dependency in step.dependencies
            ):
                status = PlanStepStatus.READY
            else:
                status = PlanStepStatus.PENDING
            statuses[step.step_id] = status
            if status is PlanStepStatus.SUCCEEDED:
                satisfied.update(step.completion_conditions)
            progress.append(PlanStepProgress(step, status, result))
            remaining.remove(step)
            progressed = True
        if not progressed:
            raise RuntimeError("validated plan graph could not be ordered")

    ordered = tuple(
        next(item for item in progress if item.step.step_id == step.step_id)
        for step in plan.steps
    )
    if goal.status is GoalStatus.PAUSED:
        status = PlanProgressStatus.PAUSED
    elif goal.status is GoalStatus.COMPLETED:
        status = PlanProgressStatus.COMPLETED
    elif goal.status is GoalStatus.CANCELLED:
        status = PlanProgressStatus.CANCELLED
    elif any(
        item.status in {PlanStepStatus.FAILED, PlanStepStatus.CANCELLED}
        for item in ordered
    ):
        status = PlanProgressStatus.BLOCKED
    elif all(item.status is PlanStepStatus.SUCCEEDED for item in ordered):
        status = PlanProgressStatus.READY_TO_COMPLETE
    else:
        status = PlanProgressStatus.ACTIVE
    return PlanProgress(
        plan,
        status,
        ordered,
        tuple(
            condition
            for condition in goal.completion_conditions
            if condition in satisfied
        ),
    )
