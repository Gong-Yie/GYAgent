from self_cognition.core.cognition import CognitionRequest
from self_cognition.core.contributions import CognitiveContribution, CognitionType
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.events import EventEnvelope
from self_cognition.core.ids import contribution_id
from self_cognition.core.narratives import NarrativeLayer, NarrativeRecord


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
    module_id = SOURCE_MODULE
    module_version = MODULE_VERSION
    deterministic = True

    def run(
        self,
        request: CognitionRequest,
    ) -> tuple[CognitiveContribution, ...]:
        return self.process(request.event)

    def process(self, event: EventEnvelope) -> tuple[CognitiveContribution, ...]:
        chapter = NARRATIVE_EVENTS.get(event.payload.text)
        if chapter is not None:
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

        record = _structured_narrative(event)
        if record is None:
            return ()
        occurred_at = event.occurred_at.isoformat()
        target_field = (
            f"narrative.{record.layer.value}.{occurred_at}.{event.event_id}"
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
                value=record.to_state_value(),
                confidence=1.0,
                evidence_refs=(EvidenceRef.for_event(event),),
                source_module=SOURCE_MODULE,
                module_version=MODULE_VERSION,
            ),
        )


def _structured_narrative(event: EventEnvelope) -> NarrativeRecord | None:
    text = event.payload.text.strip(" ，,。")
    layer: NarrativeLayer | None = None
    theme = ""
    stage = ""
    summary = text
    unknowns: tuple[str, ...] = ()

    if text.startswith("任务："):
        layer, theme, stage = NarrativeLayer.TASK, text[3:], "进行"
    elif text.startswith("这段时间"):
        layer, theme, stage = NarrativeLayer.PERIOD, "近期", "进行"
    elif text.startswith("我从") and "变成" in text:
        layer, theme, stage = NarrativeLayer.IDENTITY, "身份变化", "转折"
    elif text.startswith("我和") and "一起" in text:
        layer, theme, stage = NarrativeLayer.RELATIONSHIP, "共同经历", "进行"
    elif "转折的原因" in text and "不知道" in text:
        layer, theme, stage = NarrativeLayer.TASK, "未知转折", "转折"
        unknowns = ("转折原因",)
    else:
        return None

    return NarrativeRecord(
        narrative_id=f"{layer.value}:{event.event_id}",
        layer=layer,
        subject=event.subject,
        theme=theme.strip() or "未命名主题",
        stage=stage,
        summary=summary,
        occurred_at=event.occurred_at.isoformat(),
        evidence_event_ids=(event.event_id,),
        unknowns=unknowns,
    )
