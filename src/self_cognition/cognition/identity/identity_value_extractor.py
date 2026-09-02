from self_cognition.core.cognition import CognitionRequest
from self_cognition.core.contributions import (
    CognitiveContribution,
    CognitionType,
)
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.events import EventEnvelope
from self_cognition.core.ids import contribution_id


SOURCE_MODULE = "identity.identity_value_extractor"
MODULE_VERSION = "1"
STATEMENT_PREFIXES = (
    (
        "我确认将最重视的原则改为",
        "values.principle",
        CognitionType.PREFERENCE,
        True,
    ),
    ("我确认将角色改为", "identity.role", CognitionType.FACT, True),
    ("我最重视", "values.principle", CognitionType.PREFERENCE, False),
    ("我的角色是", "identity.role", CognitionType.FACT, False),
)


class IdentityValueExtractor:
    """Extracts explicit role and value statements with confirmation intent."""

    subscriptions = frozenset({"user.message"})
    module_id = SOURCE_MODULE
    module_version = MODULE_VERSION
    deterministic = True

    def run(
        self,
        request: CognitionRequest,
    ) -> tuple[CognitiveContribution, ...]:
        return self.process(request.event)

    def process(
        self,
        event: EventEnvelope,
    ) -> tuple[CognitiveContribution, ...]:
        if event.payload.text.endswith(("？", "?")):
            return ()
        for (
            prefix,
            target_field,
            cognition_type,
            explicitly_confirmed,
        ) in STATEMENT_PREFIXES:
            if not event.payload.text.startswith(prefix):
                continue
            value = event.payload.text[len(prefix):].strip()
            if not value:
                return ()
            return (
                CognitiveContribution.set_from_event(
                    event,
                    contribution_id=contribution_id(
                        event.event_id,
                        SOURCE_MODULE,
                        f"{target_field}:{value}:{explicitly_confirmed}",
                    ),
                    target_field=target_field,
                    cognition_type=cognition_type,
                    value=value,
                    confidence=1.0,
                    evidence_refs=(EvidenceRef.for_event(event),),
                    source_module=SOURCE_MODULE,
                    module_version=MODULE_VERSION,
                    explicitly_confirmed=explicitly_confirmed,
                ),
            )
        return ()
