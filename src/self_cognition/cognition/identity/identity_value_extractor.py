from self_cognition.core.contributions import Contribution
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.events import EventEnvelope
from self_cognition.core.ids import contribution_id


SOURCE_MODULE = "identity.identity_value_extractor"
STATEMENT_PREFIXES = (
    ("我确认将最重视的原则改为", "values.principle", True),
    ("我确认将角色改为", "identity.role", True),
    ("我最重视", "values.principle", False),
    ("我的角色是", "identity.role", False),
)


class IdentityValueExtractor:
    """Extracts explicit role and value statements with confirmation intent."""

    subscriptions = frozenset({"user.message"})

    def process(self, event: EventEnvelope) -> tuple[Contribution, ...]:
        if event.payload.text.endswith(("？", "?")):
            return ()
        for prefix, target_field, explicitly_confirmed in STATEMENT_PREFIXES:
            if not event.payload.text.startswith(prefix):
                continue
            value = event.payload.text[len(prefix):].strip()
            if not value:
                return ()
            return (
                Contribution(
                    contribution_id=contribution_id(
                        event.event_id,
                        SOURCE_MODULE,
                        f"{target_field}:{value}:{explicitly_confirmed}",
                    ),
                    target_subject_id=event.subject.subject.subject_id,
                    target_field=target_field,
                    value=value,
                    confidence=1.0,
                    evidence_refs=(EvidenceRef.for_event(event),),
                    source_module=SOURCE_MODULE,
                    explicitly_confirmed=explicitly_confirmed,
                ),
            )
        return ()
