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
        value = _structure_experience(event.payload.text)
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
                value=value,
                confidence=1.0,
                evidence_refs=(EvidenceRef.for_event(event),),
                source_module=SOURCE_MODULE,
                module_version=MODULE_VERSION,
            ),
        )


def _structure_experience(text: str) -> dict[str, object]:
    time_key = next(cue for cue in TIME_CUES if text.startswith(cue))
    remainder = text[len(time_key) :].strip(" ，,。")
    environment = ""
    action = remainder
    if "去了" in remainder:
        person, environment = remainder.split("去了", 1)
        action = "去了"
    elif "在" in remainder:
        person, environment = remainder.split("在", 1)
        action = "在"
    else:
        person = remainder.split("读完了", 1)[0] if "读完了" in remainder else "我"
        if "读完了" in remainder:
            action = "读完了"
    result = (
        remainder
        if any(marker in remainder for marker in ("完", "成功", "失败", "完成"))
        else ""
    )
    salience = (
        0.8
        if result or any(marker in remainder for marker in ("第一次", "重要"))
        else 0.5
    )
    return {
        "text": text,
        "people": (person.strip() or "我",),
        "time": time_key,
        "environment": environment,
        "action": action,
        "result": result,
        "salience": salience,
    }
