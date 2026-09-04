from self_cognition.core.cognition import CognitionRequest
from self_cognition.core.contributions import CognitiveContribution, CognitionType
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.events import (
    CapabilityObservationPayload,
    EventEnvelope,
    EventSource,
)
from self_cognition.core.ids import contribution_id


SOURCE_MODULE = "procedural.execution_extractor"
MODULE_VERSION = "1"


class ProceduralExecutionExtractor:
    """Records only tool-backed execution outcomes as procedural memory."""

    subscriptions = frozenset({"capability.observed"})
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
        if event.source is not EventSource.TOOL:
            return ()
        payload = event.payload
        if not isinstance(payload, CapabilityObservationPayload):
            return ()
        capability = payload.capability
        target_field = f"procedural.execution.{capability.capability_id}"
        value = {
            "capability_id": capability.capability_id,
            "strategy": f"invoke:{capability.name}",
            "applicability": f"capability:{capability.kind.value}",
            "outcome": capability.execution_status.value,
            "failure_mode": capability.reason,
        }
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
                value=value,
                confidence=1.0,
                evidence_refs=(EvidenceRef.for_event(event),),
                source_module=SOURCE_MODULE,
                module_version=MODULE_VERSION,
            ),
        )
