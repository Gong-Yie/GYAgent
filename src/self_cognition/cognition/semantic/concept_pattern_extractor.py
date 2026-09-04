import re

from self_cognition.core.cognition import CognitionRequest
from self_cognition.core.contributions import CognitiveContribution, CognitionType
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.events import EventEnvelope
from self_cognition.core.ids import contribution_id
from self_cognition.core.memories import MemoryType
from self_cognition.core.workspace import RetrievalQuery


SOURCE_MODULE = "semantic.concept_pattern_extractor"
MODULE_VERSION = "1"


class ConceptPatternExtractor:
    subscriptions = frozenset({"user.message"})
    module_id = SOURCE_MODULE
    module_version = MODULE_VERSION
    deterministic = True

    def run(
        self,
        request: CognitionRequest,
    ) -> tuple[CognitiveContribution, ...]:
        return self.process(request)

    def process(
        self,
        request: CognitionRequest,
    ) -> tuple[CognitiveContribution, ...]:
        event = request.event
        concept = _concept_contribution(event)
        if concept is not None:
            return (concept,)
        if request.context is None:
            return ()
        action = _action(event.payload.text)
        if action is None:
            return ()
        packet = request.context.query(
            RetrievalQuery(
                subject=event.subject,
                task="cross-experience semantic pattern",
                field_patterns=("episodic.experience.*",),
                memory_types=frozenset({MemoryType.EPISODIC}),
            )
        )
        matches = tuple(
            item
            for item in packet.items
            if isinstance(item.content, dict)
            and item.content.get("action") == action
        )
        if not matches:
            return ()
        evidence = tuple(
            dict.fromkeys(
                (EvidenceRef.for_event(event),)
                + tuple(ref for item in matches for ref in item.evidence_refs)
            )
        )
        target_field = f"semantic.pattern.{action}"
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
                value={"pattern": action, "evidence_count": len(evidence)},
                confidence=min(1.0, 0.5 + 0.2 * len(matches)),
                evidence_refs=evidence,
                source_module=SOURCE_MODULE,
                module_version=MODULE_VERSION,
            ),
        )


def _concept_contribution(event: EventEnvelope) -> CognitiveContribution | None:
    match = re.fullmatch(
        r"(.+?)是(?:一种|一个)(.+?)[。.!！]?$",
        event.payload.text,
    )
    if match is None:
        return None
    subject, concept = (part.strip() for part in match.groups())
    target_field = f"semantic.concept.{subject}"
    return CognitiveContribution.set_from_event(
        event,
        contribution_id=contribution_id(
            event.event_id,
            SOURCE_MODULE,
            target_field,
        ),
        target_field=target_field,
        cognition_type=CognitionType.FACT,
        value=concept,
        confidence=1.0,
        evidence_refs=(EvidenceRef.for_event(event),),
        source_module=SOURCE_MODULE,
        module_version=MODULE_VERSION,
    )


def _action(text: str) -> str | None:
    for marker in ("去了", "读完了", "完成了", "在"):
        if marker in text:
            return marker
    return None
