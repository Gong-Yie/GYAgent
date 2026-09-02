from self_cognition.core.cognition import CognitionRequest
from self_cognition.core.contributions import CognitiveContribution
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.errors import ModelOutputError, RunCancelledError
from self_cognition.core.events import EventEnvelope
from self_cognition.core.ids import contribution_id
from self_cognition.core.protocols import CognitionModel
from self_cognition.core.state import SubjectState
from self_cognition.core.workspace import WorkspaceBuilder
from self_cognition.runtime.cognition_context import ReadOnlyCognitionContext
from self_cognition.runtime.run_context import RunContext


SOURCE_MODULE = "semantic.llm_extractor"
MODULE_VERSION = "1"


class LLMSemanticExtractor:
    subscriptions = frozenset({"user.message"})
    module_id = SOURCE_MODULE
    module_version = MODULE_VERSION
    deterministic = False

    def __init__(self, model: CognitionModel) -> None:
        self._model = model

    def process(
        self,
        event: EventEnvelope,
        context: RunContext,
    ) -> tuple[CognitiveContribution, ...]:
        return self.run(
            CognitionRequest(
                event=event,
                context=ReadOnlyCognitionContext(
                    builder=WorkspaceBuilder(),
                    state=SubjectState.empty(
                        event.subject.subject.subject_id,
                        mind_id=event.subject.mind.mind_id,
                        subject_kind=event.subject.subject.kind,
                    ),
                    as_of=event.recorded_at,
                ),
                run_context=context,
            )
        )

    def run(
        self,
        request: CognitionRequest,
    ) -> tuple[CognitiveContribution, ...]:
        context = request.run_context
        if context is None:
            raise ValueError("LLM cognition requires RunContext")
        if context.is_cancelled:
            raise RunCancelledError("run cancelled before cognition model")

        result = self._model.extract(request)

        if context.is_cancelled:
            raise RunCancelledError("run cancelled after cognition model")

        contributions: list[CognitiveContribution] = []
        event = request.event
        event_evidence = EvidenceRef.for_event(event)
        for candidate in result.candidates:
            if str(event_evidence.evidence_id) not in candidate.evidence_ids:
                raise ModelOutputError(
                    "candidate must cite the source evidence"
                )
            contributions.append(
                CognitiveContribution.set_from_event(
                    event,
                    contribution_id=contribution_id(
                        event.event_id,
                        SOURCE_MODULE,
                        candidate.target_field,
                    ),
                    target_field=candidate.target_field,
                    cognition_type=candidate.cognition_type,
                    value=candidate.value,
                    confidence=candidate.confidence,
                    evidence_refs=(event_evidence, result.response_evidence),
                    source_module=SOURCE_MODULE,
                    module_version=MODULE_VERSION,
                )
            )

        return tuple(contributions)
