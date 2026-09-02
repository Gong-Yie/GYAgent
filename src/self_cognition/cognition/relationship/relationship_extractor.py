from self_cognition.core.cognition import CognitionRequest
from self_cognition.core.contributions import CognitiveContribution, CognitionType
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.events import EventEnvelope
from self_cognition.core.ids import contribution_id


SOURCE_MODULE = "relationship.relationship_extractor"
MODULE_VERSION = "1"
RELATIONSHIP_STATEMENTS = {
    "小明是我的朋友": ("小明", "朋友"),
    "小红是我的同事": ("小红", "同事"),
}


class RelationshipExtractor:
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
        relationship = RELATIONSHIP_STATEMENTS.get(event.payload.text)
        if relationship is None:
            return ()

        related_subject, role = relationship
        target_field = f"relationships.{related_subject}.role"
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
                value=role,
                confidence=1.0,
                evidence_refs=(EvidenceRef.for_event(event),),
                source_module=SOURCE_MODULE,
                module_version=MODULE_VERSION,
            ),
        )
