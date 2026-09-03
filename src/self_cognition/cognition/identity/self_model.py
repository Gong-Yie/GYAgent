from self_cognition.core.cognition import CognitionRequest
from self_cognition.core.contributions import CognitiveContribution, CognitionType
from self_cognition.core.evidence import EvidenceRef, EvidenceSourceKind
from self_cognition.core.events import (
    CapabilityObservationPayload,
    EventEnvelope,
    EventSource,
    SelfModelObservationPayload,
)
from self_cognition.core.identity import (
    GoalRecord,
    LimitationRecord,
    SelfModelAspect,
)
from self_cognition.core.ids import contribution_id


SOURCE_MODULE = "identity.self_model"
MODULE_VERSION = "1"
FIELD_PREFIXES = {
    SelfModelAspect.IDENTITY: "identity",
    SelfModelAspect.VALUE: "values",
    SelfModelAspect.LIMITATION: "limitations",
    SelfModelAspect.GOAL: "goals",
}
COGNITION_TYPES = {
    SelfModelAspect.IDENTITY: CognitionType.FACT,
    SelfModelAspect.VALUE: CognitionType.PREFERENCE,
    SelfModelAspect.LIMITATION: CognitionType.LIMITATION,
    SelfModelAspect.GOAL: CognitionType.GOAL,
}


class SelfModelCognitionModule:
    subscriptions = frozenset(
        {"self_model.observation", "capability.observed"}
    )
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
        payload = event.payload
        if isinstance(payload, SelfModelObservationPayload):
            target_field = f"{FIELD_PREFIXES[payload.aspect]}.{payload.field_id}"
            value = payload.value
            if isinstance(value, (LimitationRecord, GoalRecord)):
                value = value.to_state_value()
            return (
                CognitiveContribution.set_from_event(
                    event,
                    contribution_id=contribution_id(
                        event.event_id,
                        SOURCE_MODULE,
                        target_field,
                    ),
                    target_field=target_field,
                    cognition_type=COGNITION_TYPES[payload.aspect],
                    value=value,
                    confidence=payload.confidence,
                    evidence_refs=(_self_model_evidence(event),),
                    source_module=SOURCE_MODULE,
                    module_version=MODULE_VERSION,
                    expires_at=payload.expires_at,
                    explicitly_confirmed=payload.explicitly_confirmed,
                ),
            )
        if isinstance(payload, CapabilityObservationPayload):
            target_field = f"capabilities.{payload.capability.capability_id}"
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
                    value=payload.capability.to_state_value(),
                    confidence=1.0,
                    evidence_refs=(EvidenceRef.for_event(event),),
                    source_module=SOURCE_MODULE,
                    module_version=MODULE_VERSION,
                ),
            )
        return ()


def _self_model_evidence(event: EventEnvelope) -> EvidenceRef:
    if event.source is not EventSource.SYSTEM:
        return EvidenceRef.for_event(event)
    return EvidenceRef(
        evidence_id=event.event_id,
        source_kind=EvidenceSourceKind.SYSTEM_PRIOR,
        source_ref=str(event.event_id),
        scope=event.scope,
        locator="payload.value",
        observed_at=event.occurred_at,
        reliability=1.0,
    )
