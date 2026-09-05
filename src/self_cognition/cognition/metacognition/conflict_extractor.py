from self_cognition.core.cognition import CognitionRequest
from self_cognition.core.contributions import (
    CognitiveContribution,
    CognitionType,
    ContributionOperation,
)
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.events import ConflictReviewPayload, EventEnvelope
from self_cognition.core.ids import contribution_id
from self_cognition.core.protocols import CognitionModel

SOURCE_MODULE = "metacognition.conflict_extractor"
MODULE_VERSION = "1"
PREFERENCE_FIELD = "preferences.study_time"
UNCERTAINTY_FIELD = "metacognition.uncertainties.preferences.study_time"
CONFLICT_FIELD = "metacognition.conflicts.preferences.study_time"
UNCERTAIN_STATEMENT = "我不确定更喜欢早上还是晚上学习"
CONFLICT_STATEMENT = "我既喜欢早上学习，也喜欢晚上学习"


class ConflictMetacognitionExtractor:
    """Extracts the first explicit uncertainty and contradiction rules."""

    subscriptions = frozenset(
        {"user.message", "cognition.assessment_requested", "conflict.reviewed"}
    )
    module_id = SOURCE_MODULE
    module_version = MODULE_VERSION
    deterministic = True

    def __init__(self, model: CognitionModel | None = None) -> None:
        self._model = model
        self.deterministic = model is None
        self.module_version = "2" if model is not None else MODULE_VERSION

    def run(
        self,
        request: CognitionRequest,
    ) -> tuple[CognitiveContribution, ...]:
        event = request.event
        if isinstance(event.payload, ConflictReviewPayload):
            contribution = CognitiveContribution.set_from_event(
                event,
                contribution_id=contribution_id(
                    event.event_id, SOURCE_MODULE, event.payload.target_field
                ),
                target_field=event.payload.target_field,
                cognition_type=CognitionType.INFERENCE,
                value=event.payload.review.to_state_value(),
                confidence=1.0,
                evidence_refs=(EvidenceRef.for_event(event),),
                source_module=SOURCE_MODULE,
                module_version=self.module_version,
                explicitly_confirmed=True,
            )
            return (
                replace(contribution, operation=ContributionOperation.REVIEW_CONFLICT),
            )
        if self._model is not None:
            return extract_assessments(
                request,
                self._model,
                kind="metacognition",
                module_id=self.module_id,
                module_version=self.module_version,
            )
        if event.event_type != "user.message":
            return ()
        return self.process(event)

    def process(self, event: EventEnvelope) -> tuple[CognitiveContribution, ...]:
        if event.payload.text == UNCERTAIN_STATEMENT:
            return (self._contribution(event, UNCERTAINTY_FIELD, "早上或晚上"),)
        if event.payload.text != CONFLICT_STATEMENT:
            return ()

        return (
            self._contribution(event, PREFERENCE_FIELD, "早上"),
            self._contribution(event, PREFERENCE_FIELD, "晚上"),
            self._contribution(event, CONFLICT_FIELD, "早上和晚上"),
        )

    @staticmethod
    def _contribution(
        event: EventEnvelope,
        target_field: str,
        value: str,
    ) -> CognitiveContribution:
        cognition_type = (
            CognitionType.PREFERENCE
            if target_field == PREFERENCE_FIELD
            else CognitionType.INFERENCE
        )
        return CognitiveContribution.set_from_event(
            event,
            contribution_id=contribution_id(
                event.event_id,
                SOURCE_MODULE,
                f"{target_field}:{value}",
            ),
            target_field=target_field,
            cognition_type=cognition_type,
            value=value,
            confidence=1.0,
            evidence_refs=(EvidenceRef.for_event(event),),
            source_module=SOURCE_MODULE,
            module_version=MODULE_VERSION,
        )
from dataclasses import replace

from self_cognition.cognition.assessment import extract_assessments
