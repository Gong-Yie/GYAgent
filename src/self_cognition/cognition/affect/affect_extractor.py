from self_cognition.core.contributions import Contribution
from self_cognition.core.events import Event
from self_cognition.core.ids import contribution_id


SOURCE_MODULE = "affect.affect_extractor"
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

    def process(self, event: Event) -> tuple[Contribution, ...]:
        assessment = AFFECT_STATEMENTS.get(event.content)
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
            Contribution(
                contribution_id=contribution_id(
                    event.event_id,
                    SOURCE_MODULE,
                    target_field,
                ),
                target_subject_id=event.actor_id,
                target_field=target_field,
                value=value,
                confidence=1.0,
                evidence_event_ids=(event.event_id,),
                source_event_id=event.event_id,
                source_module=SOURCE_MODULE,
            ),
        )
