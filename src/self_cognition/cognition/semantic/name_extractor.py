from self_cognition.core.contributions import CognitiveContribution, CognitionType
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.events import EventEnvelope
from self_cognition.core.ids import contribution_id


SOURCE_MODULE = "semantic.name_extractor"
MODULE_VERSION = "1"
TARGET_FIELD = "profile.name"
SUPPORTED_STATEMENT = "我叫小明"


class NameExtractor:
    subscriptions = frozenset({"user.message"})

    def process(self, event: EventEnvelope) -> tuple[CognitiveContribution, ...]:
        if event.payload.text != SUPPORTED_STATEMENT:
            return ()

        return (
            CognitiveContribution.set_from_event(
                event,
                contribution_id=contribution_id(
                    event.event_id,
                    SOURCE_MODULE,
                    TARGET_FIELD,
                ),
                target_field=TARGET_FIELD,
                cognition_type=CognitionType.FACT,
                value="小明",
                confidence=1.0,
                evidence_refs=(EvidenceRef.for_event(event),),
                source_module=SOURCE_MODULE,
                module_version=MODULE_VERSION,
            ),
        )
