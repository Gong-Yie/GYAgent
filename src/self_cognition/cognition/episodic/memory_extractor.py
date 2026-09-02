from datetime import timezone

from self_cognition.core.cognition import CognitionRequest
from self_cognition.core.contributions import CognitiveContribution, CognitionType
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.events import EventEnvelope
from self_cognition.core.ids import contribution_id


SOURCE_MODULE = "episodic.memory_extractor"
MODULE_VERSION = "1"
TIME_CUES = ("今天", "昨天", "刚刚", "前天", "上周")


class EpisodicMemoryExtractor:
    """Records one concrete, time-cued user experience as one contribution."""

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
        if not event.payload.text.startswith(TIME_CUES):
            return ()

        occurred_at = event.occurred_at.astimezone(timezone.utc).isoformat()
        target_field = (
            f"episodic.experience.{occurred_at}.{event.event_id}"
        )
        return (
            CognitiveContribution.set_from_event(
                event,
                contribution_id=contribution_id(
                    event.event_id,
                    SOURCE_MODULE,
                    target_field,
                ),
                target_field=target_field,
                cognition_type=CognitionType.FACT,
                value=event.payload.text,
                confidence=1.0,
                evidence_refs=(EvidenceRef.for_event(event),),
                source_module=SOURCE_MODULE,
                module_version=MODULE_VERSION,
            ),
        )
