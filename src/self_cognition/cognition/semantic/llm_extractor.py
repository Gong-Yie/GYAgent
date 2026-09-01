from self_cognition.core.contributions import CognitiveContribution
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.errors import ModelOutputError, RunCancelledError
from self_cognition.core.events import EventEnvelope
from self_cognition.core.ids import contribution_id
from self_cognition.core.protocols import CognitionModel
from self_cognition.runtime.run_context import RunContext


SOURCE_MODULE = "semantic.llm_extractor"
MODULE_VERSION = "1"


class LLMSemanticExtractor:
    subscriptions = frozenset({"user.message"})

    def __init__(self, model: CognitionModel) -> None:
        self._model = model

    def process(
        self,
        event: EventEnvelope,
        context: RunContext,
    ) -> tuple[CognitiveContribution, ...]:
        if context.is_cancelled:
            raise RunCancelledError("run cancelled before cognition model")

        result = self._model.extract(event, context)

        if context.is_cancelled:
            raise RunCancelledError("run cancelled after cognition model")

        contributions: list[CognitiveContribution] = []
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
