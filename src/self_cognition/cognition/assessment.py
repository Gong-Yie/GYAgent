from dataclasses import dataclass, field, replace
from typing import Literal

from self_cognition.core.affect import AffectAssessment
from self_cognition.core.cognition import CognitionContextQuery, CognitionRequest
from self_cognition.core.contributions import (
    CognitiveContribution,
    CognitionType,
    ContributionOperation,
)
from self_cognition.core.errors import (
    ContractValidationError,
    ModelOutputError,
    RunCancelledError,
)
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.events import AssessmentRequestPayload
from self_cognition.core.ids import contribution_id
from self_cognition.core.metacognition import (
    ConflictReview,
    EvidenceBasis,
    KnowledgeStatus,
    MetacognitiveAssessment,
)
from self_cognition.core.protocols import CognitionModel
from self_cognition.core.workspace import RetrievalQuery, WorkspacePacket


@dataclass
class _ObservedContext:
    source: CognitionContextQuery
    packets: list[WorkspacePacket] = field(default_factory=list)

    def query(self, query: RetrievalQuery) -> WorkspacePacket:
        packet = self.source.query(query)
        self.packets.append(packet)
        return packet


def extract_assessments(
    request: CognitionRequest,
    model: CognitionModel,
    *,
    kind: Literal["metacognition", "affect"],
    module_id: str,
    module_version: str,
) -> tuple[CognitiveContribution, ...]:
    if request.run_context is None:
        raise ModelOutputError("model assessments require a run context")
    if request.run_context.is_cancelled:
        raise RunCancelledError("assessment cancelled before model call")
    observed = _ObservedContext(request.context)
    result = model.extract(replace(request, context=observed))
    if request.run_context.is_cancelled:
        raise RunCancelledError("assessment cancelled after model call")
    event = request.event
    source_refs = [EvidenceRef.for_event(event)]
    if isinstance(event.payload, AssessmentRequestPayload):
        source_refs.append(EvidenceRef.for_event(event.payload.source_event))
    refs = {str(ref.evidence_id): ref for ref in source_refs}
    for packet in observed.packets:
        for item in packet.items:
            refs.update((str(ref.evidence_id), ref) for ref in item.evidence_refs)
    refs[str(result.response_evidence.evidence_id)] = result.response_evidence
    if result.response_evidence.scope.owner.mind != event.subject.mind:
        raise ModelOutputError("model response evidence belongs to another mind")
    known_goals = {
        item.target_field.removeprefix("goals.")
        for packet in observed.packets
        for item in packet.items
        if item.target_field.startswith("goals.")
    }
    contributions: list[CognitiveContribution] = []
    seen: set[str] = set()
    for candidate in result.candidates:
        if any(
            str(ref.evidence_id) not in candidate.evidence_ids for ref in source_refs
        ):
            raise ModelOutputError("assessment must cite the request and its source")
        if any(item not in refs for item in candidate.evidence_ids):
            raise ModelOutputError(
                "assessment cites evidence not supplied to the model"
            )
        try:
            if (
                kind == "metacognition"
                and candidate.operation is ContributionOperation.REVIEW_CONFLICT
            ):
                value = ConflictReview.from_state_value(
                    candidate.value
                ).to_state_value()
                cognition_type = CognitionType.INFERENCE
            elif kind == "metacognition" and candidate.target_field.startswith(
                "metacognition.assessments."
            ):
                assessment = MetacognitiveAssessment.from_state_value(candidate.value)
                value = assessment.to_state_value()
                cognition_type = (
                    CognitionType.UNKNOWN
                    if assessment.status is KnowledgeStatus.UNKNOWN
                    else CognitionType.INFERENCE
                )
                if (
                    assessment.basis is EvidenceBasis.DIRECT
                    and assessment.status is KnowledgeStatus.KNOWN
                ):
                    cognition_type = CognitionType.FACT
            elif kind == "affect" and candidate.target_field.startswith(
                "affect.current."
            ):
                affect = AffectAssessment.from_state_value(candidate.value)
                if not set(affect.goal_ids).issubset(known_goals):
                    raise ContractValidationError("affect refers to an unsupplied goal")
                if affect.assessed_at != event.occurred_at:
                    raise ContractValidationError(
                        "affect time must match its assessment event"
                    )
                value = affect.to_state_value()
                cognition_type = CognitionType.AFFECT
            else:
                raise ContractValidationError(
                    "assessment target is outside its module ownership"
                )
            if (
                kind == "affect"
                and candidate.operation is not ContributionOperation.SET
            ):
                raise ContractValidationError("affect cannot review conflicts")
            if candidate.cognition_type is not cognition_type:
                raise ContractValidationError(
                    "assessment cognition type contradicts its epistemic status"
                )
        except ContractValidationError as error:
            raise ModelOutputError(str(error)) from error
        discriminator = f"{candidate.operation.value}:{candidate.target_field}"
        if discriminator in seen:
            raise ModelOutputError("duplicate assessment target in one model result")
        seen.add(discriminator)
        contribution = CognitiveContribution.set_from_event(
            event,
            contribution_id=contribution_id(event.event_id, module_id, discriminator),
            target_field=candidate.target_field,
            cognition_type=cognition_type,
            value=value,
            confidence=candidate.confidence,
            evidence_refs=tuple(
                dict.fromkeys(
                    tuple(refs[item] for item in candidate.evidence_ids)
                    + (result.response_evidence,)
                )
            ),
            source_module=module_id,
            module_version=module_version,
        )
        contributions.append(replace(contribution, operation=candidate.operation))
    return tuple(contributions)
