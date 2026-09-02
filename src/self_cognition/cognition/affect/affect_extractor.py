from self_cognition.core.cognition import CognitionRequest
from self_cognition.core.contributions import CognitiveContribution, CognitionType
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.events import EventEnvelope
from self_cognition.core.ids import contribution_id


SOURCE_MODULE = "affect.affect_extractor"
MODULE_VERSION = "1"
HALF_LIFE_SECONDS = 3600.0
AFFECT_STATEMENTS = {
    "这次考试通过了，我很开心": (
        "exam",
        "这次考试",
        "开心",
        "positive",
        0.8,
    ),
    "这次考试没通过，我很失望": (
        "exam",
        "这次考试",
        "失望",
        "negative",
        0.8,
    ),
    "这个项目失败了，我很失望": (
        "project",
        "这个项目",
        "失望",
        "negative",
        0.8,
    ),
}


class AffectExtractor:
    """Produces one structured, scoped affect assessment per known event."""

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
        assessment = AFFECT_STATEMENTS.get(event.payload.text)
        if assessment is None:
            return ()

        scope, target, emotion, valence, intensity = assessment
        target_field = f"affect.current.{scope}"
        value = {
            "emotion": emotion,
            "valence": valence,
            "target": target,
            "scope": scope,
            "initial_intensity": intensity,
            "assessed_at": event.occurred_at.isoformat(),
            "half_life_seconds": HALF_LIFE_SECONDS,
        }
        return (
            CognitiveContribution.set_from_event(
                event,
                contribution_id=contribution_id(
                    event.event_id,
                    SOURCE_MODULE,
                    target_field,
                ),
                target_field=target_field,
                cognition_type=CognitionType.AFFECT,
                value=value,
                confidence=1.0,
                evidence_refs=(EvidenceRef.for_event(event),),
                source_module=SOURCE_MODULE,
                module_version=MODULE_VERSION,
            ),
        )
