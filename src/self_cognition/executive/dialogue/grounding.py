from self_cognition.core.dialogue import ClaimStance, DialogueDraft
from self_cognition.core.errors import ModelOutputError
from self_cognition.core.evidence import EvidenceRef, EvidenceSourceKind
from self_cognition.core.workspace import RetrievalSource, WorkspacePacket


def validate_grounding(
    draft: DialogueDraft, workspace: WorkspacePacket
) -> tuple[EvidenceRef, ...]:
    available = {ref.evidence_id: ref for ref in workspace.evidence_refs}
    if workspace.input_evidence is not None:
        available[workspace.input_evidence.evidence_id] = workspace.input_evidence
    cited = set(draft.disclosure.evidence_ids)
    for claim in draft.claims:
        if claim.text not in draft.text:
            raise ModelOutputError("claim text must occur in the answer")
        cited.update(claim.evidence_ids)
        if any(value not in available for value in claim.evidence_ids):
            raise ModelOutputError("claim cites evidence outside the workspace")
        if claim.stance is ClaimStance.SUPPORTED:
            if all(
                available[value].source_kind is EvidenceSourceKind.MODEL_RESPONSE
                for value in claim.evidence_ids
            ):
                raise ModelOutputError(
                    "model assertions cannot independently establish facts"
                )
            if any(
                (item.confidence < 1.0 or item.source is RetrievalSource.CONFLICT)
                and any(
                    ref.evidence_id in claim.evidence_ids for ref in item.evidence_refs
                )
                for item in workspace.items
            ):
                raise ModelOutputError(
                    "uncertain evidence cannot support a certain claim"
                )
    if not cited <= available.keys():
        raise ModelOutputError("disclosure cites evidence outside the workspace")
    return tuple(ref for value, ref in available.items() if value in cited)


def is_plain_smalltalk(question: str, draft: DialogueDraft) -> bool:
    utterances = {
        "你好",
        "您好",
        "嗨",
        "谢谢",
        "不客气",
        "再见",
        "hello",
        "hi",
        "thanks",
        "bye",
    }
    return (
        _normalize(question) in utterances
        and _normalize(draft.text) in utterances
        and not draft.claims
        and not draft.disclosure.evidence_ids
    )


def _normalize(text: str) -> str:
    return text.strip(" \t\r\n。！？.!?").lower()
