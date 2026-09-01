from self_cognition.core.contributions import CognitiveContribution, CognitionType
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.events import EventEnvelope
from self_cognition.core.ids import contribution_id


SOURCE_MODULE = "narrative.narrative_extractor"
MODULE_VERSION = "1"
NARRATIVE_EVENTS = {
    "我开始准备研究项目": ("启动", "开始准备研究项目"),
    "今天我开始准备研究项目": ("启动", "开始准备研究项目"),
    "我完成了研究项目": ("完成", "完成研究项目"),
}


class NarrativeExtractor:
    """Creates one evidence-backed chapter for each known project stage."""

    subscriptions = frozenset({"user.message"})

    def process(self, event: EventEnvelope) -> tuple[CognitiveContribution, ...]:
        chapter = NARRATIVE_EVENTS.get(event.payload.text)
        if chapter is None:
            return ()

        stage, summary = chapter
        occurred_at = event.occurred_at.isoformat()
        target_field = (
            f"narrative.chapter.{occurred_at}.{event.event_id}"
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
                cognition_type=CognitionType.INFERENCE,
                value={
                    "theme": "研究项目",
                    "stage": stage,
                    "summary": summary,
                    "occurred_at": occurred_at,
                },
                confidence=1.0,
                evidence_refs=(EvidenceRef.for_event(event),),
                source_module=SOURCE_MODULE,
                module_version=MODULE_VERSION,
            ),
        )
