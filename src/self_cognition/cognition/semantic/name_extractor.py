from self_cognition.core.contributions import Contribution
from self_cognition.core.events import Event
from self_cognition.core.ids import contribution_id


SOURCE_MODULE = "semantic.name_extractor"
TARGET_FIELD = "profile.name"
SUPPORTED_STATEMENT = "我叫小明"


class NameExtractor:
    subscriptions = frozenset({"user.message"})

    def process(self, event: Event) -> tuple[Contribution, ...]:
        if event.content != SUPPORTED_STATEMENT:
            return ()

        return (
            Contribution(
                contribution_id=contribution_id(
                    event.event_id,
                    SOURCE_MODULE,
                    TARGET_FIELD,
                ),
                target_subject_id=event.actor_id,
                target_field=TARGET_FIELD,
                value="小明",
                confidence=1.0,
                evidence_event_ids=(event.event_id,),
                source_event_id=event.event_id,
                source_module=SOURCE_MODULE,
            ),
        )
