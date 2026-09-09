from __future__ import annotations

import json
import logging
from dataclasses import replace
from threading import RLock
from uuid import UUID, uuid5

from self_cognition.application.pursue_goal import PursueGoalService
from self_cognition.application.results import (
    ActionServiceResult,
    ProcessEventStatus,
)
from self_cognition.core.actions import (
    ActionContextPayload,
    ActionDecision,
    ActionDecisionPayload,
    ActionDecisionStatus,
    ActionFailurePayload,
    ActionModel,
    ActionModelOutput,
    ActionProposalRequest,
    ActionProposedPayload,
    ActionRequest,
    ActionResult,
    ActionResultPayload,
    ActionResultStatus,
    ToolDescriptor,
    action_draft_from_json,
    decision_draft_from_json,
)
from self_cognition.core.errors import (
    ContractValidationError,
    ModelOutputError,
    ModelTimeoutError,
    RunCancelledError,
)
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.events import EventEnvelope, EventSource
from self_cognition.core.ids import new_run_id
from self_cognition.core.identity import CapabilityKind
from self_cognition.core.plans import (
    GoalPlannedPayload,
    PlanRevisedPayload,
    PlanStep,
)
from self_cognition.core.runs import RunKind, RunStatus
from self_cognition.core.protocols import EventStore, StateRepository
from self_cognition.core.scopes import DataScope, DisclosureScope, SubjectScope
from self_cognition.core.workspace import (
    RetrievalBudget,
    WorkspaceBuilder,
    WorkspacePacket,
    WorkspaceRunInfo,
    workspace_model_context,
)
from self_cognition.executive.action.validator import ActionValidator
from self_cognition.runtime.run_context import RunContext
from self_cognition.runtime.run_service import RunLifecycle
from self_cognition.tools.registry import CapabilityRegistry
from self_cognition.tools.executor import ToolExecutor


logger = logging.getLogger(__name__)


class ActionService:
    def __init__(
        self,
        event_store: EventStore,
        state_repository: StateRepository,
        workspace_builder: WorkspaceBuilder,
        pursue_goal: PursueGoalService,
        capability_registry: CapabilityRegistry,
        model: ActionModel,
        executor: ToolExecutor | None = None,
        validator: ActionValidator | None = None,
        run_lifecycle: RunLifecycle | None = None,
        process_event: "ProcessEventService | None" = None,
    ) -> None:
        self._events = event_store
        self._states = state_repository
        self._workspace = workspace_builder
        self._goals = pursue_goal
        self._capabilities = capability_registry
        self._model = model
        self._executor = executor
        self._validator = validator or ActionValidator()
        self._run_lifecycle = run_lifecycle
        self._process_event = process_event
        self._lock = RLock()

    def execute(
        self,
        action: ActionRequest,
        decision: ActionDecision,
        context: RunContext,
    ) -> ActionServiceResult:
        """Execute an allowed action once and append its result event."""
        if self._executor is None:
            raise ContractValidationError("no tool executor is configured")
        with self._lock:
            run_record = (
                self._run_lifecycle.begin(
                    context,
                    RunKind.ACTION,
                    action.owner,
                    input_event_ids=(action.action_id,),
                )
                if self._run_lifecycle is not None
                else None
            )
            existing = next(
                (
                    event.payload.result
                    for event in self._events.read_by_subject(action.owner)
                    if isinstance(event.payload, ActionResultPayload)
                    and event.payload.result.action_id == action.action_id
                ),
                None,
            )
            if existing is not None:
                result = self.record_result(action.owner, existing, context)
                if run_record is not None:
                    run_record = self._run_lifecycle.checkpoint(
                        context,
                        run_record,
                        "action.result_recorded",
                        action_id=action.action_id,
                        result_event_id=result.event_id,
                    )
                    self._run_lifecycle.finish(
                        context,
                        run_record,
                        RunStatus.COMPLETED,
                    )
                return result
            if run_record is not None and run_record.status.is_terminal:
                return ActionServiceResult(
                    ProcessEventStatus.FAILED,
                    context.run_id,
                    context.correlation_id,
                    error_type="InterruptedAction",
                    reused=True,
                )
            if run_record is not None:
                run_record = self._run_lifecycle.checkpoint(
                    context,
                    run_record,
                    "action.execute_started",
                    action_id=action.action_id,
                )
            context.record_tool_call()
            result = self._executor.execute(action, decision, context)
            recorded = self.record_result(action.owner, result, context)
            if run_record is not None:
                run_record = self._run_lifecycle.checkpoint(
                    context,
                    run_record,
                    "action.result_recorded",
                    action_id=action.action_id,
                    result_event_id=recorded.event_id,
                )
                status = {
                    ActionResultStatus.CANCELLED: RunStatus.CANCELLED,
                    ActionResultStatus.TIMED_OUT: RunStatus.TIMED_OUT,
                    ActionResultStatus.FAILED: RunStatus.FAILED,
                    ActionResultStatus.PARTIAL: RunStatus.FAILED,
                }.get(result.status, RunStatus.COMPLETED)
                self._run_lifecycle.finish(
                    context,
                    run_record,
                    status,
                    reason=result.summary if status is not RunStatus.COMPLETED else None,
                    error_type=result.error_type,
                )
            return recorded

    def prepare(
        self,
        request: ActionProposalRequest,
        context: RunContext,
    ) -> ActionServiceResult:
        with self._lock:
            return self._prepare(request, context)

    def record_result(
        self,
        owner: SubjectScope,
        result: ActionResult,
        context: RunContext,
        *,
        source: EventSource = EventSource.TOOL,
    ) -> ActionServiceResult:
        if source not in {EventSource.SYSTEM, EventSource.TOOL}:
            raise ContractValidationError(
                "action results require system or tool source"
            )
        with self._lock:
            history = self._events.read_by_subject(owner)
            decision_event = next(
                (
                    event
                    for event in reversed(history)
                    if isinstance(event.payload, ActionDecisionPayload)
                    and event.payload.request.action_id == result.action_id
                ),
                None,
            )
            if decision_event is None:
                raise ContractValidationError(
                    "action result references an unknown action"
                )
            payload = decision_event.payload
            assert isinstance(payload, ActionDecisionPayload)
            if payload.request.owner != owner or result.owner != owner:
                raise ContractValidationError("action result owner does not match")
            if payload.decision.status is not ActionDecisionStatus.ALLOWED:
                raise ContractValidationError(
                    "only an allowed action may record a result"
                )
            if (
                result.status not in {
                    ActionResultStatus.CANCELLED,
                    ActionResultStatus.TIMED_OUT,
                }
                and context.clock.now() > payload.decision.valid_until
            ):
                raise ContractValidationError("action decision has expired")
            existing = next(
                (
                    event
                    for event in history
                    if isinstance(event.payload, ActionResultPayload)
                    and event.payload.result.action_id == result.action_id
                ),
                None,
            )
            if existing is not None:
                existing_result = existing.payload.result
                if existing_result != result:
                    raise ContractValidationError(
                        "action already has a different recorded result"
                    )
                return ActionServiceResult(
                    ProcessEventStatus.SUCCEEDED,
                    context.run_id,
                    context.correlation_id,
                    payload.request,
                    payload.decision,
                    existing_result,
                    existing.event_id,
                    reused=True,
                )
            if result.status not in {
                ActionResultStatus.CANCELLED,
                ActionResultStatus.TIMED_OUT,
            }:
                self._ensure_active(context)
            event = self._event(
                owner,
                "action.result",
                ActionResultPayload(result),
                context,
                event_id=result.result_id,
                causation_id=decision_event.event_id,
                source=source,
            )
            self._events.append(event)
            self._reflow_result(event, context)
            return ActionServiceResult(
                ProcessEventStatus.SUCCEEDED,
                context.run_id,
                context.correlation_id,
                payload.request,
                payload.decision,
                result,
                event.event_id,
            )

    def _reflow_result(self, event: EventEnvelope, context: RunContext) -> None:
        if self._process_event is None:
            return
        child = context.child(new_run_id(), correlation_id=context.correlation_id)
        try:
            self._process_event.process(
                replace(event, run_id=None, correlation_id=None),
                child,
            )
        except Exception:
            logger.exception("action result cognition reflow failed")

    def _prepare(
        self,
        request: ActionProposalRequest,
        context: RunContext,
    ) -> ActionServiceResult:
        history = self._events.read_by_subject(request.owner)
        started = next(
            (
                event
                for event in history
                if isinstance(event.payload, ActionContextPayload)
                and event.event_id == request.request_id
            ),
            None,
        )
        same_id = next(
            (event for event in history if event.event_id == request.request_id), None
        )
        if same_id is not None and started is None:
            raise ContractValidationError(
                "request ID already belongs to a different event"
            )
        if started is not None and started.payload.request != request:
            raise ContractValidationError(
                "request ID already belongs to a different action input"
            )
        terminal = self._terminal(history, request.request_id)
        if terminal is not None:
            return self._result(terminal, context, reused=True)
        if started is not None:
            return self._failure(
                request,
                started,
                context,
                "resume",
                "InterruptedAction",
            )

        progress = self._goals.progress(request.owner, request.plan_id)
        if progress.plan.version != request.plan_version:
            raise ContractValidationError("action request plan version is stale")
        step_progress = next(
            (
                item
                for item in progress.steps
                if item.step.step_id == request.step_id
            ),
            None,
        )
        if step_progress is None:
            raise ContractValidationError("action request references an unknown step")
        self._validator.validate_step(
            request,
            step_progress.status,
            step_progress.step,
        )
        tools = self._tools(step_progress.step)
        workspace, cause = self._build_workspace(request, step_progress.step, context)
        self._ensure_active(context)
        started = self._event(
            request.owner,
            "action.started",
            ActionContextPayload(
                request,
                json.dumps(
                    workspace_model_context(workspace),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                tuple(workspace.evidence_refs),
                tools,
            ),
            context,
            event_id=request.request_id,
            causation_id=cause.event_id,
        )
        self._events.append(started)
        stage = "propose"
        current = started
        try:
            context.record_model_call()
            output = self._model.propose(
                progress.plan,
                step_progress.step,
                workspace,
                tools,
                context,
            )
            output_event = self._save_output(current, output, context)
            self._ensure_active(context)
            draft = action_draft_from_json(output.raw_output)
            self._validator.validate_draft(
                step_progress.step,
                draft,
                self._capabilities.registrations(),
                tools,
            )
            now = context.clock.now()
            action_id = uuid5(request.request_id, "action")
            action = ActionRequest(
                action_id,
                request.request_id,
                request.owner,
                request.plan_id,
                request.plan_version,
                request.step_id,
                draft.tool_id,
                draft.arguments,
                draft.expected_side_effects,
                f"action:{action_id}",
                now,
            )
            current = self._event(
                request.owner,
                "action.proposed",
                ActionProposedPayload(action),
                context,
                event_id=uuid5(request.request_id, "action.proposed"),
                causation_id=output_event.event_id,
            )
            self._events.append(current)
            stage = "decide"
            context.record_model_call()
            output = self._model.decide(action, workspace, context)
            output_event = self._save_output(current, output, context)
            self._ensure_active(context)
            decision_draft = decision_draft_from_json(output.raw_output)
            self._validator.validate_decision(
                decision_draft,
                workspace,
                context.clock.now(),
            )
            decided_at = context.clock.now()
            decision = ActionDecision(
                uuid5(action.action_id, "decision"),
                action.action_id,
                decision_draft.status,
                decision_draft.reason,
                decision_draft.value_basis,
                decision_draft.relationship_context,
                decision_draft.risks,
                decision_draft.evidence_ids,
                decided_at,
                decision_draft.valid_until,
                action.action_id,
                decision_draft.not_before,
                decision_draft.confirmation_prompt,
            )
            event = self._event(
                request.owner,
                "action.decided",
                ActionDecisionPayload(action, decision),
                context,
                event_id=uuid5(request.request_id, "action.decided"),
                causation_id=output_event.event_id,
            )
            self._events.append(event)
            return ActionServiceResult(
                ProcessEventStatus.SUCCEEDED,
                context.run_id,
                context.correlation_id,
                action,
                decision,
                event_id=event.event_id,
            )
        except Exception as error:
            logger.warning(
                "action failed stage=%s error_type=%s",
                stage,
                type(error).__name__,
            )
            return self._failure(
                request,
                current,
                context,
                stage,
                type(error).__name__,
            )

    def _tools(self, step: PlanStep) -> tuple[ToolDescriptor, ...]:
        records = {
            record.capability_id: record
            for record in self._capabilities.registrations()
        }
        descriptors = {
            descriptor.tool_id: descriptor
            for descriptor in self._capabilities.tool_descriptors()
        }
        tools = []
        for tool_id in step.required_tool_ids:
            record = records.get(tool_id)
            if (
                record is None
                or record.kind is not CapabilityKind.TOOL
                or not record.available
            ):
                raise ContractValidationError(f"plan tool is unavailable: {tool_id}")
            descriptor = descriptors.get(tool_id)
            if descriptor is None:
                raise ContractValidationError(
                    f"plan tool descriptor is missing: {tool_id}"
                )
            tools.append(descriptor)
        return tuple(tools)

    def _build_workspace(
        self,
        request: ActionProposalRequest,
        step: PlanStep,
        context: RunContext,
    ) -> tuple[WorkspacePacket, EventEnvelope]:
        owner = request.owner
        history = self._events.read_by_mind(owner.mind)
        cause = next(
            (
                event
                for event in reversed(history)
                if (
                    isinstance(event.payload, GoalPlannedPayload)
                    and event.payload.plan.plan_id == request.plan_id
                    and event.payload.plan.version == request.plan_version
                )
                or isinstance(event.payload, PlanRevisedPayload)
                and event.payload.plan.plan_id == request.plan_id
                and event.payload.plan.version == request.plan_version
            ),
            None,
        )
        if cause is None:
            raise ContractValidationError("action plan evidence is missing")
        owners = {owner, *(event.subject for event in history)}
        states = tuple(
            state
            for subject in owners
            if (state := self._states.load(subject)) is not None
        )
        if not states:
            raise ContractValidationError("action workspace state is missing")
        workspace = self._workspace.build_shared(
            step.description,
            states,
            owner,
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
        return workspace, cause

    def _save_output(
        self,
        cause: EventEnvelope,
        output: ActionModelOutput,
        context: RunContext,
    ) -> EventEnvelope:
        event = EventEnvelope.model_response(
            cause,
            model=output.model,
            response_id=output.response_id,
            raw_output=output.raw_output,
            clock=context.clock,
            run_id=context.run_id,
            correlation_id=context.correlation_id,
        )
        self._events.append(event)
        if output.error_type is not None:
            raise ModelOutputError(output.error_type)
        return event

    def _failure(
        self,
        request: ActionProposalRequest,
        cause: EventEnvelope,
        context: RunContext,
        stage: str,
        error_type: str,
    ) -> ActionServiceResult:
        event = self._event(
            request.owner,
            "action.failed",
            ActionFailurePayload(request.request_id, stage, error_type),
            context,
            event_id=uuid5(request.request_id, f"action.failed:{stage}"),
            causation_id=cause.event_id,
        )
        self._events.append(event)
        return ActionServiceResult(
            ProcessEventStatus.FAILED,
            context.run_id,
            context.correlation_id,
            event_id=event.event_id,
            error_type=error_type,
        )

    @staticmethod
    def _terminal(
        history: tuple[EventEnvelope, ...],
        request_id: UUID,
    ) -> EventEnvelope | None:
        return next(
            (
                event
                for event in history
                if (
                    isinstance(event.payload, ActionDecisionPayload)
                    and event.payload.request.proposal_request_id == request_id
                )
                or (
                    isinstance(event.payload, ActionFailurePayload)
                    and event.payload.proposal_request_id == request_id
                )
            ),
            None,
        )

    @staticmethod
    def _result(
        event: EventEnvelope,
        context: RunContext,
        *,
        reused: bool,
    ) -> ActionServiceResult:
        if isinstance(event.payload, ActionDecisionPayload):
            return ActionServiceResult(
                ProcessEventStatus.SUCCEEDED,
                context.run_id,
                context.correlation_id,
                event.payload.request,
                event.payload.decision,
                event_id=event.event_id,
                reused=reused,
            )
        assert isinstance(event.payload, ActionFailurePayload)
        return ActionServiceResult(
            ProcessEventStatus.FAILED,
            context.run_id,
            context.correlation_id,
            event_id=event.event_id,
            error_type=event.payload.error_type,
            reused=reused,
        )

    @staticmethod
    def _ensure_active(context: RunContext) -> None:
        if context.cancelled:
            raise RunCancelledError("action was cancelled")
        if context.clock.now() >= context.deadline:
            raise ModelTimeoutError("action deadline reached")

    @staticmethod
    def _event(
        owner: SubjectScope,
        event_type: str,
        payload: object,
        context: RunContext,
        *,
        event_id: UUID,
        causation_id: UUID,
        source: EventSource = EventSource.SYSTEM,
    ) -> EventEnvelope:
        now = context.clock.now()
        return EventEnvelope(
            event_id,
            event_type,
            None,
            owner,
            payload,
            now,
            now,
            source,
            DataScope(owner, DisclosureScope.MIND),
            causation_id=causation_id,
            correlation_id=context.correlation_id,
            run_id=context.run_id,
        )
