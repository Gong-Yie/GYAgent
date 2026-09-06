import json
from dataclasses import replace

from self_cognition.core.dialogue import (
    ClaimStance,
    DialogueClaim,
    DialogueDraft,
    DialogueModelOutput,
    DisclosureDecision,
    draft_to_dict,
)
from self_cognition.core.scopes import DisclosureScope, SubjectKind
from self_cognition.core.workspace import RetrievalSource, WorkspacePacket
from self_cognition.executive.dialogue.grounding import is_plain_smalltalk
from self_cognition.executive.dialogue.rule_based import RuleBasedDialogueModel
from self_cognition.runtime.run_context import RunContext


class RuleDialogueAdapter:
    def __init__(self, model: RuleBasedDialogueModel) -> None:
        self._model = model

    def generate(
        self, workspace: WorkspacePacket, context: RunContext
    ) -> DialogueModelOutput:
        draft = self._compose(workspace)
        return DialogueModelOutput(
            "rule-dialogue-v1",
            f"rule-generate-{context.run_id}",
            json.dumps(draft_to_dict(draft), ensure_ascii=False),
        )

    def review(
        self, workspace: WorkspacePacket, draft: DialogueDraft, context: RunContext
    ) -> DialogueModelOutput:
        return DialogueModelOutput(
            "rule-grounding-v1",
            f"rule-review-{context.run_id}",
            json.dumps(
                {
                    "supported": draft == self._compose(workspace),
                    "reason": "Deterministic fixture comparison; not a real-model quality evaluation.",
                }
            ),
        )

    def _compose(self, workspace: WorkspacePacket) -> DialogueDraft:
        disclosure = DisclosureDecision(
            "disclose",
            DisclosureScope.PRIVATE,
            "Offline rule fixture, not an LLM value decision.",
            (),
        )
        greeting = DialogueDraft("你好！", (), disclosure)
        if is_plain_smalltalk(workspace.task_context, greeting):
            return greeting
        self_question = workspace.task_context in {
            "你是谁？",
            "你能做什么？",
            "你不能做什么？",
            "你当前的目标是什么？",
        }
        items = tuple(
            item
            for item in workspace.items
            if item.subject is not None
            and (
                item.subject.subject.kind is SubjectKind.MIND
                if self_question
                else item.subject == workspace.subject
            )
        )
        selected = replace(workspace, items=items)
        response = self._model.respond(workspace.task_context, selected)
        ids = tuple(dict.fromkeys(ref.evidence_id for ref in response.evidence_refs))
        used = tuple(
            item
            for item in items
            if any(ref.evidence_id in ids for ref in item.evidence_refs)
        )
        stance = ClaimStance.SUPPORTED if ids else ClaimStance.UNKNOWN
        if any(
            item.confidence < 1.0 or item.source is RetrievalSource.CONFLICT
            for item in used
        ):
            stance = ClaimStance.UNCERTAIN
        return DialogueDraft(
            response.text,
            (DialogueClaim(response.text, ids, stance),),
            replace(disclosure, evidence_ids=ids),
        )
