from self_cognition.core.contributions import CognitiveContribution, CognitionType
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.events import EventEnvelope
from self_cognition.core.ids import contribution_id


SOURCE_MODULE = "metacognition.conflict_extractor"
MODULE_VERSION = "1"
PREFERENCE_FIELD = "preferences.study_time"
UNCERTAINTY_FIELD = "metacognition.uncertainties.preferences.study_time"
CONFLICT_FIELD = "metacognition.conflicts.preferences.study_time"
UNCERTAIN_STATEMENT = "我不确定更喜欢早上还是晚上学习"
CONFLICT_STATEMENT = "我既喜欢早上学习，也喜欢晚上学习"


class ConflictMetacognitionExtractor:
    """Extracts the first explicit uncertainty and contradiction rules."""

    subscriptions = frozenset({"user.message"})

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
