from datetime import datetime

from self_cognition.core.actions import (
    ActionDecisionDraft,
    ActionDraft,
    ActionProposalRequest,
    ToolDescriptor,
)
from self_cognition.core.errors import ContractValidationError
from self_cognition.core.identity import CapabilityKind, CapabilityRecord
from self_cognition.core.plans import PlanStep, PlanStepStatus
from self_cognition.core.workspace import WorkspacePacket


class ActionValidator:
    def validate_step(
        self,
        request: ActionProposalRequest,
        step_status: PlanStepStatus,
        step: PlanStep,
    ) -> None:
        if step.step_id != request.step_id:
            raise ContractValidationError("action step does not match its request")
        if step_status is not PlanStepStatus.READY:
            raise ContractValidationError("actions require a ready plan step")
        if not step.required_tool_ids:
            raise ContractValidationError("action step does not require a tool")

    def validate_draft(
        self,
        step: PlanStep,
        draft: ActionDraft,
        capabilities: tuple[CapabilityRecord, ...],
        descriptors: tuple[ToolDescriptor, ...],
    ) -> ToolDescriptor:
        if draft.tool_id not in step.required_tool_ids:
            raise ContractValidationError(
                "action selected a tool outside its plan step"
            )
        records = {record.capability_id: record for record in capabilities}
        record = records.get(draft.tool_id)
        if record is None or record.kind is not CapabilityKind.TOOL:
            raise ContractValidationError("action selected an unknown tool")
        if not record.available:
            raise ContractValidationError("action selected an unavailable tool")
        descriptor = next(
            (item for item in descriptors if item.tool_id == draft.tool_id), None
        )
        if descriptor is None:
            raise ContractValidationError("action tool descriptor is missing")
        if draft.expected_side_effects != descriptor.expected_side_effects:
            raise ContractValidationError(
                "action side effects do not match the tool descriptor"
            )
        return descriptor

    def validate_decision(
        self,
        draft: ActionDecisionDraft,
        workspace: WorkspacePacket,
        now: datetime,
    ) -> None:
        if draft.valid_until <= now:
            raise ContractValidationError("action decision is already expired")
        available_evidence = {ref.evidence_id for ref in workspace.evidence_refs}
        if not set(draft.evidence_ids).issubset(available_evidence):
            raise ContractValidationError(
                "action decision cites evidence outside its workspace"
            )
