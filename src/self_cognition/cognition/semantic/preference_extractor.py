from self_cognition.core.contributions import CognitiveContribution, CognitionType
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.events import EventEnvelope
from self_cognition.core.ids import contribution_id


SOURCE_MODULE = "semantic.preference_extractor"
MODULE_VERSION = "1"
TARGET_FIELD = "preferences.study_time"
STUDY_TIME_PREFERENCES = {
    "喜欢晚上学习": "晚上",
    "喜欢早上学习": "早上",
}


class PreferenceExtractor:
    subscriptions = frozenset({"user.message"})

    def process(self, event: EventEnvelope) -> tuple[CognitiveContribution, ...]:
        matched_values = {
            preference_value
            for phrase, preference_value in STUDY_TIME_PREFERENCES.items()
            if phrase in event.payload.text
        }
        if len(matched_values) != 1:
            return ()
        value = matched_values.pop()

        return (
            CognitiveContribution.set_from_event(
                event,
                contribution_id=contribution_id(
                    event.event_id,
                    SOURCE_MODULE,
                    TARGET_FIELD,
                ),
                target_field=TARGET_FIELD,
                cognition_type=CognitionType.PREFERENCE,
                value=value,
                confidence=1.0,
                evidence_refs=(EvidenceRef.for_event(event),),
                source_module=SOURCE_MODULE,
                module_version=MODULE_VERSION,
            ),
        )
