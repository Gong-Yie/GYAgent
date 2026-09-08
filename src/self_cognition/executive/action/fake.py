import json

from self_cognition.core.actions import (
    ActionDecisionDraft,
    ActionDecisionStatus,
    ActionDraft,
    ActionModelOutput,
    ActionRequest,
    ToolDescriptor,
    action_draft_to_dict,
    decision_draft_to_dict,
)
from self_cognition.core.plans import Plan, PlanStep
from self_cognition.core.workspace import WorkspacePacket
from self_cognition.runtime.run_context import RunContext


class RuleActionModel:
    def propose(
        self,
        plan: Plan,
        step: PlanStep,
        workspace: WorkspacePacket,
        tools: tuple[ToolDescriptor, ...],
        context: RunContext,
    ) -> ActionModelOutput:
        del plan, workspace
        descriptor = next(
            tool for tool in tools if tool.tool_id in step.required_tool_ids
        )
        draft = ActionDraft(
            descriptor.tool_id,
            {},
            descriptor.expected_side_effects,
        )
        return ActionModelOutput(
            "rule-action-v1",
            f"rule-propose-{context.run_id}",
            json.dumps(action_draft_to_dict(draft), ensure_ascii=False),
        )

    def decide(
        self,
        request: ActionRequest,
        workspace: WorkspacePacket,
        context: RunContext,
    ) -> ActionModelOutput:
        requires_confirmation = any(
            not effect.reversible for effect in request.expected_side_effects
        )
        draft = ActionDecisionDraft(
            (
                ActionDecisionStatus.CONFIRMATION_REQUIRED
                if requires_confirmation
                else ActionDecisionStatus.ALLOWED
            ),
            (
                "Irreversible side effects require contextual confirmation."
                if requires_confirmation
                else "The bounded context supports the planned action."
            ),
            ("advance the active plan",),
            "No contrary relationship context was selected.",
            tuple(
                effect.description for effect in request.expected_side_effects
            )
            or ("no declared side effects",),
            tuple(ref.evidence_id for ref in workspace.evidence_refs),
            context.deadline,
            confirmation_prompt=(
                "Confirm this irreversible action."
                if requires_confirmation
                else None
            ),
        )
        return ActionModelOutput(
            "rule-action-v1",
            f"rule-decide-{context.run_id}",
            json.dumps(decision_draft_to_dict(draft), ensure_ascii=False),
        )
