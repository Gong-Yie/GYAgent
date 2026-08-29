from datetime import timezone

from self_cognition.core.contributions import Contribution
from self_cognition.core.events import Event
from self_cognition.core.ids import contribution_id


SOURCE_MODULE = "episodic.memory_extractor"
TIME_CUES = ("今天", "昨天", "刚刚", "前天", "上周")


class EpisodicMemoryExtractor:
    """Records one concrete, time-cued user experience as one contribution."""

    subscriptions = frozenset({"user.message"})

    def process(self, event: Event) -> tuple[Contribution, ...]:
        if not event.content.startswith(TIME_CUES):
            return ()

        occurred_at = event.occurred_at.astimezone(timezone.utc).isoformat()
        target_field = (
            f"episodic.experience.{occurred_at}.{event.event_id}"
        )
        return (
            Contribution(
                contribution_id=contribution_id(
                    event.event_id,
                    SOURCE_MODULE,
                    target_field,
                ),
                target_subject_id=event.actor_id,
                target_field=target_field,
                value=event.content,
                confidence=1.0,
                evidence_event_ids=(event.event_id,),
                source_event_id=event.event_id,
                source_module=SOURCE_MODULE,
            ),
        )
