from self_cognition.core.contributions import CognitiveContribution, CognitionType
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.events import CognitionCorrectionPayload, EventEnvelope
from self_cognition.core.ids import contribution_id


SOURCE_MODULE = "metacognition.user_correction"
MODULE_VERSION = "1"


class UserCorrectionModule:
    subscriptions = frozenset({"user.correction"})

    def process(self, event: EventEnvelope) -> tuple[CognitiveContribution, ...]:
        payload = event.payload
        if not isinstance(payload, CognitionCorrectionPayload):
            return ()
        return (
            CognitiveContribution.set_from_event(
                event,
                contribution_id=contribution_id(
                    event.event_id,
                    SOURCE_MODULE,
                    payload.target_field,
                ),
                target_field=payload.target_field,
                cognition_type=CognitionType(payload.cognition_type),
                value=payload.value,
                confidence=1.0,
                evidence_refs=(EvidenceRef.for_event(event),),
                source_module=SOURCE_MODULE,
                module_version=MODULE_VERSION,
                explicitly_confirmed=True,
            ),
        )
