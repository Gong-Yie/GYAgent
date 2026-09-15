from uuid import NAMESPACE_URL, uuid5

from self_cognition.core.affect import (
    MOOD_FIELD,
    EmotionState,
    MoodState,
    accumulate_mood,
)
from self_cognition.core.cognition import CognitionRequest
from self_cognition.core.contributions import CognitiveContribution, CognitionType
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.events import EventEnvelope
from self_cognition.core.ids import contribution_id

SOURCE_MODULE = "affect.fast_reaction"
MODULE_VERSION = "1"
HALF_LIFE_SECONDS = 3600.0

POSITIVE_MARKERS = (
    "开心",
    "高兴",
    "成功",
    "谢谢",
    "太好了",
    "松了一口气",
    "满意",
)
NEGATIVE_MARKERS = (
    "难过",
    "失望",
    "生气",
    "失败",
    "焦虑",
    "担心",
    "无聊",
    "孤独",
    "害怕",
    "压力",
)
BOREDOM_MARKERS = ("无聊", "孤独", "没意思", "无事可做")


class FastAffectExtractor:
    """Deterministic low-latency affect reaction and mood accumulation."""

    subscriptions = frozenset(
        {"user.message", "action.result", "goal.status_changed"}
    )
    module_id = SOURCE_MODULE
    module_version = MODULE_VERSION
    deterministic = True

    def run(
        self,
        request: CognitionRequest,
    ) -> tuple[CognitiveContribution, ...]:
        event = request.event
        text = getattr(event.payload, "text", "")
        if not isinstance(text, str):
            text = ""
        emotion = self._classify(text, event)
        if emotion is None:
            return ()
        scope, emotion_name, valence, arousal, control, certainty, intensity = emotion
        emotion_state = EmotionState(
            emotion_id=uuid5(
                event.event_id,
                f"fast-affect:{scope}:{emotion_name}",
            ),
            target=event.subject.subject.subject_id,
            emotion=emotion_name,
            valence=valence,
            scope=scope,
            intensity=intensity,
            assessed_at=event.occurred_at,
            half_life_seconds=HALF_LIFE_SECONDS,
            arousal=arousal,
            control=control,
            certainty=certainty,
            cause=event.event_type,
        )
        reaction_field = f"affect.reaction.{scope}"
        contributions = [
            CognitiveContribution.set_from_event(
                event,
                contribution_id=contribution_id(
                    event.event_id,
                    SOURCE_MODULE,
                    reaction_field,
                ),
                target_field=reaction_field,
                cognition_type=CognitionType.AFFECT,
                value=emotion_state.to_state_value(),
                confidence=0.6,
                evidence_refs=(EvidenceRef.for_event(event),),
                source_module=SOURCE_MODULE,
                module_version=MODULE_VERSION,
            )
        ]
        prior_mood = self._prior_mood(request)
        mood = accumulate_mood(
            prior_mood,
            emotion_state,
            as_of=event.occurred_at,
        )
        contributions.append(
            CognitiveContribution.set_from_event(
                event,
                contribution_id=contribution_id(
                    event.event_id,
                    SOURCE_MODULE,
                    MOOD_FIELD,
                ),
                target_field=MOOD_FIELD,
                cognition_type=CognitionType.AFFECT,
                value=mood.to_state_value(),
                confidence=0.6,
                evidence_refs=(EvidenceRef.for_event(event),),
                source_module=SOURCE_MODULE,
                module_version=MODULE_VERSION,
            )
        )
        return tuple(contributions)

    @staticmethod
    def _prior_mood(request: CognitionRequest) -> MoodState | None:
        state = getattr(request.context, "state", None)
        if state is None:
            return None
        entry = state.entries.get(MOOD_FIELD)
        if entry is None or not isinstance(entry.value, dict):
            return None
        try:
            return MoodState.from_state_value(entry.value)
        except Exception:
            return None

    @staticmethod
    def _classify(
        text: str,
        event: EventEnvelope,
    ) -> tuple[str, str, str, float, float, float, float] | None:
        if any(marker in text for marker in BOREDOM_MARKERS):
            return ("interaction", "boredom", "negative", 0.1, 0.4, 0.7, 0.7)
        if any(marker in text for marker in POSITIVE_MARKERS):
            return ("interaction", "positive", "positive", 0.6, 0.6, 0.6, 0.7)
        if any(marker in text for marker in NEGATIVE_MARKERS):
            return ("interaction", "negative", "negative", 0.5, 0.3, 0.5, 0.7)
        if event.event_type == "action.result":
            result = getattr(event.payload, "result", None)
            status = getattr(result, "status", None)
            value = getattr(status, "value", None)
            if value == "succeeded":
                return ("task", "satisfaction", "positive", 0.5, 0.7, 0.7, 0.6)
            if value in {"failed", "timed_out", "cancelled", "partial"}:
                return ("task", "frustration", "negative", 0.6, 0.3, 0.5, 0.7)
        return None