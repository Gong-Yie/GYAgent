from __future__ import annotations

import json
from dataclasses import replace
from threading import RLock
from uuid import UUID, uuid5

from self_cognition.application.process_event import ProcessEventService
from self_cognition.application.results import (
    ProcessEventStatus,
    PursueGoalResult,
)
from self_cognition.core.errors import ContractValidationError, ModelOutputError
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.events import EventEnvelope, EventSource
from self_cognition.core.identity import GoalRecord, GoalStatus
from self_cognition.core.plans import (
    GoalPlanningRequest,
    GoalStatusChangedPayload,
    Plan,
    PlanProgress,
    PlanStepResult,
    PlanStepResultPayload,
    PlanningContextPayload,
    PlanningFailurePayload,
    PlanningModel,
    PlanRevisedPayload,
    GoalPlannedPayload,
    plan_draft_from_json,
    plan_to_dict,
)
from self_cognition.core.protocols import EventStore, StateRepository
from self_cognition.core.scopes import (
    DataScope,
    DisclosureScope,
    SubjectKind,
    SubjectRef,
    SubjectScope,
)
from self_cognition.core.state import SubjectState
from self_cognition.core.workspace import RetrievalBudget, WorkspaceBuilder, WorkspaceRunInfo
from self_cognition.executive.planning.validator import PlanValidator
from self_cognition.executive.planning.progress import calculate_progress
from self_cognition.runtime.run_context import RunContext
from self_cognition.tools.registry import CapabilityRegistry


class PursueGoalService:
    def __init__(
        self,
        process_event: ProcessEventService,
        event_store: EventStore,
        state_repository: StateRepository,
        workspace_builder: WorkspaceBuilder,
        planner: PlanningModel,
        capability_registry: CapabilityRegistry,
        validator: PlanValidator | None = None,
    ) -> None:
        self._process_event = process_event
        self._events = event_store
        self._states = state_repository
        self._workspace = workspace_builder
        self._planner = planner
        self._capabilities = capability_registry
        self._validator = validator or PlanValidator()
        self._lock = RLock()

    def create(
        self,
        request: GoalPlanningRequest,
        context: RunContext,
    ) -> PursueGoalResult:
        with self._lock:
            return self._create(request, context)

    def record_step_result(
        self,
        subject: SubjectScope,
        result: PlanStepResult,
        context: RunContext,
        *,
        source: EventSource = EventSource.TOOL,
    ) -> PursueGoalResult:
        if source not in {EventSource.SYSTEM, EventSource.TOOL}:
            raise ContractValidationError("step results require system or tool source")
        with self._lock:
            plan, goal = self._require_plan_goal(subject, result.plan_id)
            if result.plan_version != plan.version:
                raise ContractValidationError("step result plan version is stale")
            if result.step_id not in {step.step_id for step in plan.steps}:
                raise ContractValidationError("step result references an unknown step")
            event = self._planning_event(
                subject,
                "plan.step_result",
                PlanStepResultPayload(result),
                context,
                event_id=result.result_id,
                source=source,
            )
            self._events.append(event)
            progress = self.progress(subject, result.plan_id, goal)
            return PursueGoalResult(
                ProcessEventStatus.SUCCEEDED,
                context.run_id,
                context.correlation_id,
                goal,
                plan,
                progress,
                event.event_id,
            )

    def pause(
        self,
        subject: SubjectScope,
        plan_id: UUID,
        context: RunContext,
        *,
        reason: str = "paused by user",
        actor: SubjectScope | None = None,
    ) -> PursueGoalResult:
        return self._change_status(
            subject, plan_id, GoalStatus.PAUSED, reason, context, actor
        )

    def continue_goal(
        self,
        subject: SubjectScope,
        plan_id: UUID,
        context: RunContext,
        *,
        actor: SubjectScope | None = None,
    ) -> PursueGoalResult:
        return self._change_status(
            subject, plan_id, GoalStatus.ACTIVE, "resumed by user", context, actor
        )

    def cancel(
        self,
        subject: SubjectScope,
        plan_id: UUID,
        context: RunContext,
        *,
        reason: str = "cancelled by user",
        actor: SubjectScope | None = None,
    ) -> PursueGoalResult:
        return self._change_status(
            subject, plan_id, GoalStatus.CANCELLED, reason, context, actor
        )

    def complete(
        self,
        subject: SubjectScope,
        plan_id: UUID,
        context: RunContext,
        *,
        actor: SubjectScope | None = None,
    ) -> PursueGoalResult:
        with self._lock:
            plan, goal = self._require_plan_goal(subject, plan_id)
            progress = self.progress(subject, plan_id, goal)
            if progress.status.value != "ready_to_complete":
                raise ContractValidationError(
                    "goal cannot complete before all steps succeed"
                )
            return self._change_status(
                subject,
                plan_id,
                GoalStatus.COMPLETED,
                "all planned completion conditions satisfied",
                context,
                actor,
                plan=plan,
                goal=goal,
                progress=progress,
                allow_complete=True,
            )

    def progress(
        self,
        subject: SubjectScope,
        plan_id: UUID,
        goal: GoalRecord | None = None,
    ) -> PlanProgress:
        plan, stored_goal = self._require_plan_goal(subject, plan_id)
        return calculate_progress(
            plan,
            goal or stored_goal,
            tuple(
                event.payload.result
                for event in self._events.read_by_subject(subject)
                if event.event_type == "plan.step_result"
                and isinstance(event.payload, PlanStepResultPayload)
                and event.payload.result.plan_id == plan_id
            ),
        )

    def replan(
        self,
        subject: SubjectScope,
        plan_id: UUID,
        context: RunContext,
    ) -> PursueGoalResult:
        with self._lock:
            plan, goal = self._require_plan_goal(subject, plan_id)
            progress = self.progress(subject, plan_id, goal)
            if progress.status.value != "blocked":
                raise ContractValidationError("replanning requires a blocked plan")
            workspace = self._build_workspace(subject, goal, context)
            context.record_model_call()
            output = self._planner.replan(
                goal,
                plan,
                progress,
                workspace,
                self._capabilities.registrations(),
                context,
            )
            self._persist_model_response(
                subject,
                output,
                context,
                next(
                    event.event_id
                    for event in reversed(self._events.read_by_subject(subject))
                    if event.event_type in {"goal.planned", "plan.revised"}
                ),
            )
            if output.error_type is not None:
                return self._failure(
                    subject,
                    plan,
                    context,
                    "replan",
                    output.error_type,
                )
            try:
                draft = plan_draft_from_json(output.raw_output)
                revised = Plan(
                    plan.plan_id,
                    plan.goal_id,
                    plan.version + 1,
                    plan.budget,
                    draft.steps,
                    context.clock.now(),
                    plan.version,
                )
                self._validator.validate_revision(plan, revised, progress)
            except (ContractValidationError, ModelOutputError) as error:
                return self._failure(subject, plan, context, "replan", type(error).__name__)
            event = self._planning_event(
                subject,
                "plan.revised",
                PlanRevisedPayload(
                    uuid5(plan.plan_id, f"replan-request:{revised.version}"),
                    revised,
                    next(
                        item.result.result_id
                        for item in progress.steps
                        if item.result is not None
                        and item.status.value in {"failed", "cancelled"}
                    ),
                ),
                context,
                event_id=uuid5(plan.plan_id, f"plan-revised:{revised.version}"),
            )
            self._events.append(event)
            return PursueGoalResult(
                ProcessEventStatus.SUCCEEDED,
                context.run_id,
                context.correlation_id,
                goal,
                revised,
                calculate_progress(revised, goal, ()),
                event.event_id,
            )

    def _create(
        self,
        request: GoalPlanningRequest,
        context: RunContext,
    ) -> PursueGoalResult:
        if request.owner.subject.kind is not SubjectKind.MIND:
            raise ContractValidationError("goal owner must be a mind subject")
        existing = self._events.read_by_subject(request.owner)
        terminal = next(
            (
                event
                for event in existing
                if event.event_type == "goal.planned"
                and isinstance(event.payload, GoalPlannedPayload)
                and event.payload.request_event_id == request.request_id
            ),
            None,
        )
        if terminal is not None:
            payload = terminal.payload
            assert isinstance(payload, GoalPlannedPayload)
            return PursueGoalResult(
                ProcessEventStatus.SUCCEEDED,
                context.run_id,
                context.correlation_id,
                payload.goal,
                payload.plan,
                self.progress(request.owner, payload.plan.plan_id, payload.goal),
                terminal.event_id,
                reused=True,
            )
        request_event = EventEnvelope.goal_requested(
            request,
            clock=context.clock,
            correlation_id=context.correlation_id,
            run_id=context.run_id,
        )
        processed = self._process_event.process(request_event, context)
        if processed.status is not ProcessEventStatus.SUCCEEDED or processed.state is None:
            return PursueGoalResult(
                processed.status,
                context.run_id,
                context.correlation_id,
                error_type=processed.error_type,
            )
        workspace = self._build_workspace(request.owner, request.goal, context)
        context_event = self._planning_event(
            request.owner,
            "planning.started",
            PlanningContextPayload(
                request.request_id,
                request.goal,
                request.budget,
                json.dumps(
                    {
                        "task": workspace.task_context,
                        "items": [item.content for item in workspace.items],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                tuple(workspace.evidence_refs),
                processed.new_version or 0,
            ),
            context,
            event_id=uuid5(request.request_id, "planning.started"),
        )
        self._events.append(context_event)
        context.record_model_call()
        output = self._planner.create(
            request.goal,
            request.budget,
            workspace,
            self._capabilities.registrations(),
            context,
        )
        self._persist_model_response(
            request.owner, output, context, context_event.event_id
        )
        if output.error_type is not None:
            return self._failure(
                request.owner,
                None,
                context,
                "create",
                output.error_type,
                request.request_id,
                request.goal.goal_id,
            )
        try:
            draft = plan_draft_from_json(output.raw_output)
            plan = Plan(
                uuid5(request.request_id, "plan:1"),
                request.goal.goal_id,
                1,
                request.budget,
                draft.steps,
                context.clock.now(),
            )
            self._validator.validate(request.goal, plan, self._capabilities.registrations())
        except (ContractValidationError, ModelOutputError) as error:
            return self._failure(
                request.owner,
                None,
                context,
                "validate",
                type(error).__name__,
                request.request_id,
                request.goal.goal_id,
            )
        planned = self._planning_event(
            request.owner,
            "goal.planned",
            GoalPlannedPayload(request.request_id, request.goal, plan),
            context,
            event_id=uuid5(request.request_id, "goal.planned"),
        )
        processed = self._process_event.process(planned, context)
        if processed.status is not ProcessEventStatus.SUCCEEDED:
            return PursueGoalResult(
                processed.status,
                context.run_id,
                context.correlation_id,
                request.goal,
                plan,
                error_type=processed.error_type,
            )
        return PursueGoalResult(
            ProcessEventStatus.SUCCEEDED,
            context.run_id,
            context.correlation_id,
            request.goal,
            plan,
            self.progress(request.owner, plan.plan_id, request.goal),
            planned.event_id,
        )

    def _change_status(
        self,
        subject: SubjectScope,
        plan_id: UUID,
        status: GoalStatus,
        reason: str,
        context: RunContext,
        actor: SubjectScope | None,
        *,
        plan: Plan | None = None,
        goal: GoalRecord | None = None,
        progress: PlanProgress | None = None,
        allow_complete: bool = False,
    ) -> PursueGoalResult:
        with self._lock:
            plan, goal = (plan, goal) if plan is not None and goal is not None else self._require_plan_goal(subject, plan_id)
            if goal.status is status:
                raise ContractValidationError("goal is already in the requested status")
            if goal.status in {GoalStatus.COMPLETED, GoalStatus.CANCELLED}:
                raise ContractValidationError("terminal goal cannot change status")
            if status is GoalStatus.COMPLETED and not allow_complete:
                raise ContractValidationError("use complete after step results")
            updated = replace(goal, status=status)
            event = self._planning_event(
                subject,
                "goal.status_changed",
                GoalStatusChangedPayload(
                    plan.plan_id,
                    goal.status,
                    updated,
                    actor.subject if actor is not None else None,
                    reason,
                ),
                context,
                event_id=uuid5(plan.plan_id, f"status:{status.value}"),
                source=EventSource.USER if actor is not None else EventSource.SYSTEM,
                actor=actor.subject if actor is not None else None,
            )
            processed = self._process_event.process(event, context)
            if processed.status is not ProcessEventStatus.SUCCEEDED:
                return PursueGoalResult(
                    processed.status,
                    context.run_id,
                    context.correlation_id,
                    goal,
                    plan,
                    progress,
                    error_type=processed.error_type,
                )
            return PursueGoalResult(
                ProcessEventStatus.SUCCEEDED,
                context.run_id,
                context.correlation_id,
                updated,
                plan,
                progress or self.progress(subject, plan_id, updated),
                event.event_id,
            )

    def _require_plan_goal(
        self,
        subject: SubjectScope,
        plan_id: UUID,
    ) -> tuple[Plan, GoalRecord]:
        if subject.subject.kind is not SubjectKind.MIND:
            raise ContractValidationError("planning requires a mind subject")
        for event in reversed(self._events.read_by_subject(subject)):
            if event.event_type == "plan.revised" and isinstance(event.payload, PlanRevisedPayload) and event.payload.plan.plan_id == plan_id:
                plan = event.payload.plan
                break
            if event.event_type == "goal.planned" and isinstance(event.payload, GoalPlannedPayload) and event.payload.plan.plan_id == plan_id:
                plan = event.payload.plan
                break
        else:
            raise ContractValidationError("unknown plan")
        state = self._states.load(subject)
        if state is None:
            raise ContractValidationError("plan owner state is missing")
        goal_atom = state.entries.get(f"goals.{plan.goal_id}")
        if goal_atom is None:
            raise ContractValidationError("plan goal state is missing")
        return plan, GoalRecord.from_state_value(goal_atom.value)

    def _build_workspace(
        self,
        subject: SubjectScope,
        goal: GoalRecord,
        context: RunContext,
    ):
        state = self._states.load(subject)
        if state is None:
            raise ContractValidationError("planning owner state is missing")
        cause = next(
            (
                event
                for event in reversed(self._events.read_by_subject(subject))
                if event.event_type in {"goal.requested", "goal.planned"}
            ),
            None,
        )
        if cause is None:
            raise ContractValidationError("planning request evidence is missing")
        return self._workspace.build_shared(
            goal.description,
            (state,),
            subject,
            input_evidence=EvidenceRef.for_event(cause),
            budget=RetrievalBudget(1024, 16),
            run_info=WorkspaceRunInfo(
                context.run_id,
                context.correlation_id,
                context.deadline,
                context.is_cancelled,
            ),
            as_of=context.clock.now(),
        )

    def _persist_model_response(
        self,
        subject: SubjectScope,
        output,
        context: RunContext,
        cause_id: UUID,
    ) -> None:
        cause = next(
            event
            for event in self._events.read_by_subject(subject)
            if event.event_id == cause_id
        )
        self._events.append(
            EventEnvelope.model_response(
                cause,
                model=output.model,
                response_id=output.response_id,
                raw_output=output.raw_output,
                clock=context.clock,
                run_id=context.run_id,
                correlation_id=context.correlation_id,
            )
        )

    def _failure(
        self,
        subject: SubjectScope,
        plan: Plan | None,
        context: RunContext,
        stage: str,
        error_type: str,
        request_id: UUID | None = None,
        goal_id: str = "unknown",
    ) -> PursueGoalResult:
        request_id = request_id or uuid5(plan.plan_id, f"failure:{stage}")
        event = self._planning_event(
            subject,
            "planning.failed",
            PlanningFailurePayload(request_id, goal_id, stage, error_type),
            context,
            event_id=uuid5(request_id, f"planning.failed:{stage}"),
        )
        self._events.append(event)
        return PursueGoalResult(
            ProcessEventStatus.FAILED,
            context.run_id,
            context.correlation_id,
            error_type=error_type,
            event_id=event.event_id,
        )

    @staticmethod
    def _planning_event(
        subject: SubjectScope,
        event_type: str,
        payload: object,
        context: RunContext,
        *,
        event_id: UUID,
        source: EventSource = EventSource.SYSTEM,
        actor: SubjectRef | None = None,
    ) -> EventEnvelope:
        now = context.clock.now()
        return EventEnvelope(
            event_id,
            event_type,
            actor,
            subject,
            payload,
            now,
            now,
            source,
            DataScope(subject, DisclosureScope.MIND),
            correlation_id=context.correlation_id,
            run_id=context.run_id,
        )
