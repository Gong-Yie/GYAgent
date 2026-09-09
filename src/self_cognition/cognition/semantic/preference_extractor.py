import re

from self_cognition.core.cognition import CognitionRequest
from self_cognition.core.contributions import CognitiveContribution, CognitionType
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.events import EventEnvelope
from self_cognition.core.ids import contribution_id


SOURCE_MODULE = "semantic.preference_extractor"
MODULE_VERSION = "1"
TARGET_FIELD = "preferences.study_time"
STUDY_TIME_PATTERN = re.compile(r"(早上|早晨|晚上|夜里|夜间)(?:学习|读书|学|了|$)")
STUDY_TIME_VALUES = {
    "早上": "早上",
    "早晨": "早上",
    "晚上": "晚上",
    "夜里": "晚上",
    "夜间": "晚上",
}


class PreferenceExtractor:
    subscriptions = frozenset({"user.message"})
    module_id = SOURCE_MODULE
    module_version = MODULE_VERSION
    deterministic = True

    def run(
        self,
        request: CognitionRequest,
    ) -> tuple[CognitiveContribution, ...]:
        return self.process(request.event)

    def process(self, event: EventEnvelope) -> tuple[CognitiveContribution, ...]:
        matches = list(STUDY_TIME_PATTERN.finditer(event.payload.text))
        if not matches:
            return ()
        value = STUDY_TIME_VALUES[matches[-1].group(1)]

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
