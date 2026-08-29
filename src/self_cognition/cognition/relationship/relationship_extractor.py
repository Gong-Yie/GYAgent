from self_cognition.core.contributions import Contribution
from self_cognition.core.events import Event
from self_cognition.core.ids import contribution_id


SOURCE_MODULE = "relationship.relationship_extractor"
RELATIONSHIP_STATEMENTS = {
    "小明是我的朋友": ("小明", "朋友"),
    "小红是我的同事": ("小红", "同事"),
}


class RelationshipExtractor:
    subscriptions = frozenset({"user.message"})

    def process(self, event: Event) -> tuple[Contribution, ...]:
        relationship = RELATIONSHIP_STATEMENTS.get(event.content)
        if relationship is None:
            return ()

        related_subject, role = relationship
        target_field = f"relationships.{related_subject}.role"
        return (
            Contribution(
                contribution_id=contribution_id(
                    event.event_id,
                    SOURCE_MODULE,
                    target_field,
                ),
                target_subject_id=event.actor_id,
                target_field=target_field,
                value=role,
                confidence=1.0,
                evidence_event_ids=(event.event_id,),
                source_event_id=event.event_id,
                source_module=SOURCE_MODULE,
            ),
        )
