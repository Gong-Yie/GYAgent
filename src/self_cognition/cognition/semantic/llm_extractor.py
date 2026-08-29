from uuid import UUID

from self_cognition.core.contributions import Contribution
from self_cognition.core.errors import ModelOutputError, RunCancelledError
from self_cognition.core.events import Event
from self_cognition.core.ids import contribution_id
from self_cognition.core.protocols import CognitionModel
from self_cognition.runtime.run_context import RunContext


SOURCE_MODULE = "semantic.llm_extractor"


class LLMSemanticExtractor:
    subscriptions = frozenset({"user.message"})

    def __init__(self, model: CognitionModel) -> None:
        self._model = model

    def process(
        self,
        event: Event,
        context: RunContext,
    ) -> tuple[Contribution, ...]:
        if context.is_cancelled:
            raise RunCancelledError("run cancelled before cognition model")

        result = self._model.extract(event, context)

        if context.is_cancelled:
            raise RunCancelledError("run cancelled after cognition model")

        contributions: list[Contribution] = []
        for candidate in result.candidates:
            try:
                evidence_ids = tuple(
                    UUID(event_id) for event_id in candidate.evidence_event_ids
                )
            except ValueError as error:
                raise ModelOutputError(
                    "candidate evidence_event_ids must contain UUID strings"
                ) from error
            if event.event_id not in evidence_ids:
                raise ModelOutputError(
                    "candidate must cite the source event as evidence"
                )
            contributions.append(
                Contribution(
                    contribution_id=contribution_id(
                        event.event_id,
                        SOURCE_MODULE,
                        candidate.target_field,
                    ),
                    target_subject_id=event.actor_id,
                    target_field=candidate.target_field,
                    value=candidate.value,
                    confidence=candidate.confidence,
                    evidence_event_ids=evidence_ids,
                    source_event_id=event.event_id,
                    source_module=SOURCE_MODULE,
                    source_model_response_id=result.response_id,
                )
            )

        return tuple(contributions)
