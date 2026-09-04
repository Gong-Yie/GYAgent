from self_cognition.core.cognition import CognitionRequest
from self_cognition.core.contributions import CognitiveContribution, CognitionType
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.events import EventEnvelope
from self_cognition.core.ids import contribution_id
from self_cognition.core.relationships import RelationshipState
from self_cognition.core.scopes import SubjectKind, SubjectRef, SubjectScope


SOURCE_MODULE = "relationship.relationship_extractor"
MODULE_VERSION = "1"
RELATIONSHIP_STATEMENTS = {
    "小明是我的朋友": ("小明", "朋友"),
    "小红是我的同事": ("小红", "同事"),
}


class RelationshipExtractor:
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
        relationship = RELATIONSHIP_STATEMENTS.get(event.payload.text)
        if relationship is not None:
            related_subject, role = relationship
            target_field = f"relationships.{related_subject}.role"
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
                    value=role,
                    confidence=1.0,
                    evidence_refs=(EvidenceRef.for_event(event),),
                    source_module=SOURCE_MODULE,
                    module_version=MODULE_VERSION,
                ),
            )

        record = _structured_relationship(event)
        if record is None:
            return ()
        related_subject = record.target.subject.subject_id
        context = record.context.replace(".", " ")
        target_field = f"relationships.edge.{related_subject}.{context}"
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
                value=record.to_state_value(),
                confidence=record.confidence,
                evidence_refs=(EvidenceRef.for_event(event),),
                source_module=SOURCE_MODULE,
                module_version=MODULE_VERSION,
            ),
        )


def _structured_relationship(event: EventEnvelope) -> RelationshipState | None:
    text = event.payload.text.strip(" ，,。")
    target: str | None = None
    relation = ""
    context = "一般互动"
    boundaries: tuple[str, ...] = ()
    commitments: tuple[str, ...] = ()

    if text.startswith("我和") and text.endswith("中合作过"):
        body = text[2:-4]
        if "在" not in body:
            return None
        target, context = body.split("在", 1)
        relation = "合作"
    elif text.startswith("我和") and "的边界是" in text:
        target, boundary = text[2:].split("的边界是", 1)
        boundaries = (boundary,)
        relation = "协作"
    elif "答应" in text:
        target, commitment = text.split("答应", 1)
        commitments = (commitment,)
        relation = "承诺"
    elif "不希望" in text:
        target, boundary = text.split("不希望", 1)
        boundaries = (boundary,)
        relation = "边界"
    else:
        return None

    target = target.removeprefix("我和").strip()
    if not target or not relation:
        return None
    source = event.subject
    target_scope = SubjectScope(
        source.mind,
        SubjectRef(SubjectKind.USER, target),
    )
    return RelationshipState(
        source=source,
        target=target_scope,
        relation=relation,
        context=context.strip() or "一般互动",
        scope=event.scope,
        shared_experience_ids=(event.event_id,) if relation == "合作" else (),
        boundaries=boundaries,
        commitments=commitments,
        confidence=1.0,
    )
