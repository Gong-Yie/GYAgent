from __future__ import annotations

from datetime import datetime
from typing import Mapping
from uuid import NAMESPACE_URL, UUID, uuid5

from self_cognition.core.actions import (
    ActionDecisionPayload,
    ActionProposedPayload,
    ActionResultPayload,
)
from self_cognition.core.affect import MOOD_FIELD, EmotionState, MoodState
from self_cognition.core.contributions import CognitiveContribution
from self_cognition.core.errors import ContractValidationError
from self_cognition.core.events import (
    CognitionModuleResultPayload,
    EventEnvelope,
    EventSource,
)
from self_cognition.core.memories import MemoryRecord
from self_cognition.core.narratives import NarrativeRecord
from self_cognition.core.relationships import RelationshipState
from self_cognition.core.provenance import (
    ProvenanceEdge,
    ProvenanceEdgeKind,
    ProvenanceGraph,
    ProvenanceNode,
    ProvenanceNodeKind,
)
from self_cognition.core.protocols import EventStore, MemoryRepository
from self_cognition.core.scopes import MindScope


def build_provenance_graph(
    mind: MindScope,
    events: tuple[EventEnvelope, ...],
    memories: tuple[MemoryRecord, ...],
) -> ProvenanceGraph:
    if not isinstance(mind, MindScope):
        raise TypeError("mind must be a MindScope")
    event_nodes: dict[UUID, ProvenanceNode] = {}
    nodes: dict[UUID, ProvenanceNode] = {}
    edges: dict[UUID, ProvenanceEdge] = {}

    for event in sorted(events, key=lambda item: item.event_id.int):
        if event.subject.mind.mind_id != mind.mind_id:
            raise ContractValidationError("provenance event belongs to another mind")
        node = ProvenanceNode(
            node_id=event.event_id,
            kind=ProvenanceNodeKind.EVENT,
            subject=event.subject,
            recorded_at=event.recorded_at,
            attributes={
                "event_type": event.event_type,
                "source": event.source.value,
                "occurred_at": event.occurred_at.isoformat(),
                "recorded_at": event.recorded_at.isoformat(),
                "run_id": _uuid_text(event.run_id),
                "causation_id": _uuid_text(event.causation_id),
                "correlation_id": _uuid_text(event.correlation_id),
                "disclosure": event.scope.disclosure.value,
            },
        )
        event_nodes[event.event_id] = node
        nodes[event.event_id] = node

    for event in events:
        if event.causation_id is not None and event.causation_id in event_nodes:
            _add_edge(
                edges,
                kind=ProvenanceEdgeKind.CAUSED_BY,
                source=event.event_id,
                target=event.causation_id,
                subject=event.subject,
                recorded_at=event.recorded_at,
                attributes={},
            )

    contributions: dict[UUID, CognitiveContribution] = {}
    for event in events:
        payload = event.payload
        if not isinstance(payload, CognitionModuleResultPayload):
            continue
        for contribution in payload.contributions:
            if contribution.target.mind.mind_id != mind.mind_id:
                raise ContractValidationError(
                    "provenance contribution belongs to another mind"
                )
            contributions[contribution.contribution_id] = contribution
            nodes[contribution.contribution_id] = ProvenanceNode(
                node_id=contribution.contribution_id,
                kind=ProvenanceNodeKind.CONTRIBUTION,
                subject=contribution.target,
                recorded_at=contribution.created_at,
                attributes={
                    "target_field": contribution.target_field,
                    "operation": contribution.operation.value,
                    "cognition_type": contribution.cognition_type.value,
                    "confidence": contribution.confidence,
                    "source_module": contribution.source_module,
                    "module_version": contribution.module_version,
                    "target_version": contribution.target_version,
                    "created_at": contribution.created_at.isoformat(),
                    "valid_from": contribution.valid_from.isoformat(),
                    "expires_at": (
                        contribution.expires_at.isoformat()
                        if contribution.expires_at is not None
                        else None
                    ),
                    "explicitly_confirmed": contribution.explicitly_confirmed,
                },
            )
            for evidence in contribution.evidence_refs:
                target_id = _evidence_target(
                    nodes,
                    event_nodes,
                    evidence,
                    contribution.target,
                    contribution.created_at,
                )
                _add_edge(
                    edges,
                    kind=ProvenanceEdgeKind.DERIVED_FROM,
                    source=contribution.contribution_id,
                    target=target_id,
                    subject=contribution.target,
                    recorded_at=contribution.created_at,
                    attributes={"source_kind": evidence.source_kind.value},
                )

    for memory in sorted(memories, key=lambda item: item.memory_id.int):
        if memory.subject.mind.mind_id != mind.mind_id:
            raise ContractValidationError("provenance memory belongs to another mind")
        nodes[memory.memory_id] = ProvenanceNode(
            node_id=memory.memory_id,
            kind=ProvenanceNodeKind.MEMORY,
            subject=memory.subject,
            recorded_at=memory.created_at,
            attributes={
                "memory_type": memory.memory_type.value,
                "version": memory.version,
                "confidence": memory.confidence,
                "salience": memory.salience,
                "stability": memory.stability,
                "retrievability": memory.retrievability,
                "lifecycle_status": memory.lifecycle_status.value,
                "source_module": memory.source_module,
                "source_module_version": memory.source_module_version,
                "created_at": memory.created_at.isoformat(),
                "target_fields": sorted(
                    {source.target_field for source in memory.sources}
                ),
            },
        )
        for source in memory.sources:
            _ensure_contribution_placeholder(
                nodes,
                contributions,
                source.contribution_id,
                memory.subject,
                memory.created_at,
            )
            _add_edge(
                edges,
                kind=ProvenanceEdgeKind.DERIVED_FROM,
                source=memory.memory_id,
                target=source.contribution_id,
                subject=memory.subject,
                recorded_at=memory.created_at,
                attributes={
                    "old_state_version": source.old_state_version,
                    "new_state_version": source.new_state_version,
                    "target_field": source.target_field,
                },
            )
        for evidence in memory.evidence_refs:
            target_id = _evidence_target(
                nodes,
                event_nodes,
                evidence,
                memory.subject,
                memory.created_at,
            )
            _add_edge(
                edges,
                kind=ProvenanceEdgeKind.DERIVED_FROM,
                source=memory.memory_id,
                target=target_id,
                subject=memory.subject,
                recorded_at=memory.created_at,
                attributes={"source_kind": evidence.source_kind.value},
            )

    for contribution in contributions.values():
        relationship_node = _relationship_node(contribution)
        if relationship_node is not None:
            nodes[relationship_node.node_id] = relationship_node
            _link_contribution_node(
                nodes,
                edges,
                relationship_node,
                contribution,
                event_nodes,
            )
        narrative_node = _narrative_node(contribution)
        if narrative_node is not None:
            nodes[narrative_node.node_id] = narrative_node
            _link_contribution_node(
                nodes,
                edges,
                narrative_node,
                contribution,
                event_nodes,
            )

    for contribution in contributions.values():
        affect_node = _affect_node(contribution)
        if affect_node is None:
            continue
        nodes[affect_node.node_id] = affect_node
        _add_edge(
            edges,
            kind=ProvenanceEdgeKind.DERIVED_FROM,
            source=affect_node.node_id,
            target=contribution.contribution_id,
            subject=contribution.target,
            recorded_at=contribution.created_at,
            attributes={},
        )
        for evidence in contribution.evidence_refs:
            target_id = _evidence_target(
                nodes,
                event_nodes,
                evidence,
                contribution.target,
                contribution.created_at,
            )
            _add_edge(
                edges,
                kind=ProvenanceEdgeKind.DERIVED_FROM,
                source=affect_node.node_id,
                target=target_id,
                subject=contribution.target,
                recorded_at=contribution.created_at,
                attributes={"source_kind": evidence.source_kind.value},
            )

    for contribution in contributions.values():
        if contribution.target_field != MOOD_FIELD:
            continue
        mood_node = _mood_node(contribution)
        if mood_node is None:
            continue
        nodes[mood_node.node_id] = mood_node
        _add_edge(
            edges,
            kind=ProvenanceEdgeKind.DERIVED_FROM,
            source=mood_node.node_id,
            target=contribution.contribution_id,
            subject=contribution.target,
            recorded_at=contribution.created_at,
            attributes={},
        )
        for emotion_id in _mood_source_emotion_ids(contribution):
            emotion_node_id = _emotion_node_id(emotion_id)
            if emotion_node_id in nodes:
                _add_edge(
                    edges,
                    kind=ProvenanceEdgeKind.DERIVED_FROM,
                    source=mood_node.node_id,
                    target=emotion_node_id,
                    subject=contribution.target,
                    recorded_at=contribution.created_at,
                    attributes={},
                )

    decision_by_action: dict[UUID, UUID] = {}
    for event in events:
        payload = event.payload
        if isinstance(payload, ActionProposedPayload):
            request = payload.request
            nodes[request.action_id] = ProvenanceNode(
                node_id=request.action_id,
                kind=ProvenanceNodeKind.ACTION_REQUEST,
                subject=request.owner,
                recorded_at=request.requested_at,
                attributes={
                    "plan_id": str(request.plan_id),
                    "plan_version": request.plan_version,
                    "step_id": request.step_id,
                    "tool_id": request.tool_id,
                    "idempotency_key": request.idempotency_key,
                    "requested_at": request.requested_at.isoformat(),
                    "argument_keys": sorted(str(key) for key in request.arguments),
                    "expected_side_effect_count": len(request.expected_side_effects),
                },
            )
            _add_edge(
                edges,
                kind=ProvenanceEdgeKind.DERIVED_FROM,
                source=request.action_id,
                target=event.event_id,
                subject=request.owner,
                recorded_at=event.recorded_at,
                attributes={},
            )
        elif isinstance(payload, ActionDecisionPayload):
            request = payload.request
            decision = payload.decision
            decision_by_action[request.action_id] = decision.decision_id
            nodes[decision.decision_id] = ProvenanceNode(
                node_id=decision.decision_id,
                kind=ProvenanceNodeKind.ACTION_DECISION,
                subject=request.owner,
                recorded_at=decision.decided_at,
                attributes={
                    "status": decision.status.value,
                    "reason": decision.reason,
                    "value_basis": list(decision.value_basis),
                    "relationship_context": decision.relationship_context,
                    "risks": list(decision.risks),
                    "evidence_ids": [str(value) for value in decision.evidence_ids],
                    "valid_until": decision.valid_until.isoformat(),
                    "not_before": (
                        decision.not_before.isoformat()
                        if decision.not_before is not None
                        else None
                    ),
                    "confirmation_prompt": decision.confirmation_prompt,
                },
            )
            _ensure_action_request_placeholder(
                nodes,
                request.action_id,
                request.owner,
                request.requested_at,
            )
            for target_id in (request.action_id, event.event_id):
                _add_edge(
                    edges,
                    kind=ProvenanceEdgeKind.DERIVED_FROM,
                    source=decision.decision_id,
                    target=target_id,
                    subject=request.owner,
                    recorded_at=decision.decided_at,
                    attributes={},
                )
            for evidence_id in decision.evidence_ids:
                _ensure_placeholder_evidence_node(
                    nodes,
                    evidence_id,
                    request.owner,
                    decision.decided_at,
                )
                _add_edge(
                    edges,
                    kind=ProvenanceEdgeKind.DERIVED_FROM,
                    source=decision.decision_id,
                    target=evidence_id,
                    subject=request.owner,
                    recorded_at=decision.decided_at,
                    attributes={"source_kind": "event"},
                )
        elif isinstance(payload, ActionResultPayload):
            result = payload.result
            nodes[result.result_id] = ProvenanceNode(
                node_id=result.result_id,
                kind=ProvenanceNodeKind.ACTION_RESULT,
                subject=result.owner,
                recorded_at=result.recorded_at,
                attributes={
                    "status": result.status.value,
                    "summary": result.summary,
                    "error_type": result.error_type,
                    "actual_side_effect_count": len(result.actual_side_effects),
                    "source": event.source.value,
                    "tool_result": event.source is EventSource.TOOL,
                },
            )
            _ensure_action_request_placeholder(
                nodes,
                result.action_id,
                result.owner,
                result.recorded_at,
            )
            _add_edge(
                edges,
                kind=ProvenanceEdgeKind.DERIVED_FROM,
                source=result.result_id,
                target=result.action_id,
                subject=result.owner,
                recorded_at=result.recorded_at,
                attributes={},
            )
            decision_id = decision_by_action.get(result.action_id)
            if decision_id is not None:
                _add_edge(
                    edges,
                    kind=ProvenanceEdgeKind.DERIVED_FROM,
                    source=result.result_id,
                    target=decision_id,
                    subject=result.owner,
                    recorded_at=result.recorded_at,
                    attributes={},
                )
            _add_edge(
                edges,
                kind=ProvenanceEdgeKind.DERIVED_FROM,
                source=result.result_id,
                target=event.event_id,
                subject=result.owner,
                recorded_at=event.recorded_at,
                attributes={"tool_result": event.source is EventSource.TOOL},
            )

    return ProvenanceGraph(
        mind_id=mind.mind_id,
        nodes=tuple(sorted(nodes.values(), key=lambda item: item.node_id.int)),
        edges=tuple(sorted(edges.values(), key=lambda item: item.edge_id.int)),
    )


class ProvenanceGraphService:
    def __init__(
        self,
        event_store: EventStore,
        memory_repository: MemoryRepository,
        store: object,
    ) -> None:
        self._event_store = event_store
        self._memory_repository = memory_repository
        self._store = store

    def rebuild(self, mind: MindScope) -> ProvenanceGraph:
        graph = build_provenance_graph(
            mind,
            self._event_store.read_by_mind(mind),
            self._memory_repository.read_by_mind(mind),
        )
        self._store.write(graph)
        return graph

    def load(self, mind: MindScope) -> ProvenanceGraph | None:
        return self._store.read(mind)

    def delete(self, mind: MindScope) -> None:
        self._store.delete(mind)

    def rebuild_all(self) -> tuple[ProvenanceGraph, ...]:
        known = set(self._known_mind_ids())
        for stale_mind_id in set(self._store.known_mind_ids()) - known:
            self._store.delete(MindScope(stale_mind_id))
        return tuple(
            self.rebuild(MindScope(mind_id))
            for mind_id in sorted(known)
        )

    def delete_all(self) -> None:
        self._store.delete_all()

    def verify(self) -> bool:
        known = set(self._known_mind_ids())
        stored = set(self._store.known_mind_ids())
        if known != stored:
            return False
        for mind_id in known:
            graph = self.load(MindScope(mind_id))
            if graph is None or graph.mind_id != mind_id:
                return False
        return True

    def source_events(self, mind: MindScope, node_id: UUID):
        graph = self.load(mind)
        if graph is None:
            graph = self.rebuild(mind)
        return graph.source_events(node_id)

    def _known_mind_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    event.subject.mind.mind_id
                    for event in self._event_store.read_all()
                }
            )
        )


def _affect_node(contribution: CognitiveContribution) -> ProvenanceNode | None:
    if not (
        contribution.target_field.startswith("affect.current.")
        or contribution.target_field.startswith("affect.reaction.")
    ):
        return None
    if not isinstance(contribution.value, Mapping):
        return None
    try:
        emotion = EmotionState.from_state_value(dict(contribution.value))
    except ContractValidationError:
        return None
    return ProvenanceNode(
        node_id=_emotion_node_id(emotion.emotion_id),
        kind=ProvenanceNodeKind.EMOTION,
        subject=contribution.target,
        recorded_at=emotion.assessed_at,
        attributes={
            "emotion": emotion.emotion,
            "valence": emotion.valence,
            "intensity": emotion.intensity,
            "arousal": emotion.arousal,
            "control": emotion.control,
            "certainty": emotion.certainty,
            "cause": emotion.cause,
            "target": emotion.target,
            "assessed_at": emotion.assessed_at.isoformat(),
        },
    )


def _mood_node(contribution: CognitiveContribution) -> ProvenanceNode | None:
    if not isinstance(contribution.value, Mapping):
        return None
    try:
        mood = MoodState.from_state_value(dict(contribution.value))
    except ContractValidationError:
        return None
    return ProvenanceNode(
        node_id=mood.mood_id,
        kind=ProvenanceNodeKind.MOOD,
        subject=contribution.target,
        recorded_at=mood.updated_at,
        attributes={
            "valence": mood.valence,
            "intensity": mood.intensity,
            "arousal": mood.arousal,
            "control": mood.control,
            "certainty": mood.certainty,
            "target": mood.target,
            "updated_at": mood.updated_at.isoformat(),
        },
    )


def _mood_source_emotion_ids(contribution: CognitiveContribution) -> tuple[UUID, ...]:
    if not isinstance(contribution.value, Mapping):
        return ()
    raw = contribution.value.get("source_emotion_ids")
    if not isinstance(raw, list):
        return ()
    result: list[UUID] = []
    for item in raw:
        try:
            result.append(UUID(str(item)))
        except (TypeError, ValueError):
            continue
    return tuple(result)


def _link_contribution_node(
    nodes: dict[UUID, ProvenanceNode],
    edges: dict[UUID, ProvenanceEdge],
    node: ProvenanceNode,
    contribution: CognitiveContribution,
    event_nodes: Mapping[UUID, ProvenanceNode],
) -> None:
    _add_edge(
        edges,
        kind=ProvenanceEdgeKind.DERIVED_FROM,
        source=node.node_id,
        target=contribution.contribution_id,
        subject=node.subject,
        recorded_at=node.recorded_at,
        attributes={},
    )
    for evidence in contribution.evidence_refs:
        target_id = _evidence_target(
            nodes,
            event_nodes,
            evidence,
            node.subject,
            node.recorded_at,
        )
        _add_edge(
            edges,
            kind=ProvenanceEdgeKind.DERIVED_FROM,
            source=node.node_id,
            target=target_id,
            subject=node.subject,
            recorded_at=node.recorded_at,
            attributes={"source_kind": evidence.source_kind.value},
        )


def _relationship_node(
    contribution: CognitiveContribution,
) -> ProvenanceNode | None:
    if not contribution.target_field.startswith("relationships."):
        return None
    value = contribution.value
    if isinstance(value, Mapping):
        try:
            record = RelationshipState.from_state_value(dict(value))
        except ContractValidationError:
            record = None
        if record is not None:
            node_id = uuid5(
                NAMESPACE_URL,
                "provenance:relationship:"
                f"{record.source.mind.mind_id}:"
                f"{record.source.subject.subject_id}:"
                f"{record.target.subject.subject_id}:"
                f"{record.relation}:{record.context}",
            )
            return ProvenanceNode(
                node_id=node_id,
                kind=ProvenanceNodeKind.RELATIONSHIP,
                subject=record.source,
                recorded_at=contribution.created_at,
                attributes={
                    "relation": record.relation,
                    "context": record.context,
                    "source_subject_id": record.source.subject.subject_id,
                    "target_subject_id": record.target.subject.subject_id,
                    "target_kind": record.target.subject.kind.value,
                    "shared_experience_ids": [
                        str(value) for value in record.shared_experience_ids
                    ],
                    "boundaries": list(record.boundaries),
                    "commitments": list(record.commitments),
                    "disclosure_decision": record.disclosure_decision.value,
                    "confidence": record.confidence,
                },
            )
    if isinstance(value, str) and value.strip():
        parts = contribution.target_field.split(".")
        target_subject_id = parts[1] if len(parts) >= 3 else contribution.target_field
        node_id = uuid5(
            NAMESPACE_URL,
            f"provenance:relationship-role:{contribution.contribution_id}",
        )
        return ProvenanceNode(
            node_id=node_id,
            kind=ProvenanceNodeKind.RELATIONSHIP,
            subject=contribution.target,
            recorded_at=contribution.created_at,
            attributes={
                "relation": value,
                "target_subject_id": target_subject_id,
                "target_field": contribution.target_field,
            },
        )
    return None


def _narrative_node(
    contribution: CognitiveContribution,
) -> ProvenanceNode | None:
    target_field = contribution.target_field
    if not (
        target_field.startswith("narrative.")
        or target_field.startswith("narratives.")
    ):
        return None
    if not isinstance(contribution.value, Mapping):
        return None
    value = contribution.value
    try:
        record = NarrativeRecord.from_state_value(dict(value))
    except ContractValidationError:
        record = None
    if record is not None:
        node_id = uuid5(
            NAMESPACE_URL,
            f"provenance:narrative:{record.narrative_id}",
        )
        return ProvenanceNode(
            node_id=node_id,
            kind=ProvenanceNodeKind.NARRATIVE,
            subject=record.subject,
            recorded_at=contribution.created_at,
            attributes={
                "layer": record.layer.value,
                "theme": record.theme,
                "stage": record.stage,
                "summary": record.summary,
                "occurred_at": record.occurred_at,
                "unknowns": list(record.unknowns),
                "revision_of": record.revision_of,
                "version": record.version,
            },
        )
    node_id = uuid5(
        NAMESPACE_URL,
        f"provenance:narrative-contribution:{contribution.contribution_id}",
    )
    return ProvenanceNode(
        node_id=node_id,
        kind=ProvenanceNodeKind.NARRATIVE,
        subject=contribution.target,
        recorded_at=contribution.created_at,
        attributes={
            "theme": value.get("theme"),
            "stage": value.get("stage"),
            "summary": value.get("summary"),
            "occurred_at": value.get("occurred_at"),
        },
    )


def _ensure_action_request_placeholder(
    nodes: dict[UUID, ProvenanceNode],
    action_id: UUID,
    subject,
    recorded_at: datetime,
) -> None:
    if action_id in nodes:
        return
    nodes[action_id] = ProvenanceNode(
        node_id=action_id,
        kind=ProvenanceNodeKind.ACTION_REQUEST,
        subject=subject,
        recorded_at=recorded_at,
        attributes={"placeholder": True},
    )


def _evidence_target(
    nodes: dict[UUID, ProvenanceNode],
    event_nodes: Mapping[UUID, ProvenanceNode],
    evidence,
    fallback_subject,
    fallback_recorded_at: datetime,
) -> UUID:
    if evidence.evidence_id in event_nodes:
        return evidence.evidence_id
    if evidence.evidence_id not in nodes:
        nodes[evidence.evidence_id] = ProvenanceNode(
            node_id=evidence.evidence_id,
            kind=ProvenanceNodeKind.EVIDENCE,
            subject=evidence.scope.owner,
            recorded_at=evidence.observed_at or fallback_recorded_at,
            attributes={
                "source_kind": evidence.source_kind.value,
                "source_ref": evidence.source_ref,
                "locator": evidence.locator,
                "reliability": evidence.reliability,
            },
        )
    return evidence.evidence_id


def _ensure_placeholder_evidence_node(
    nodes: dict[UUID, ProvenanceNode],
    evidence_id: UUID,
    subject,
    recorded_at: datetime,
) -> None:
    if evidence_id in nodes:
        return
    nodes[evidence_id] = ProvenanceNode(
        node_id=evidence_id,
        kind=ProvenanceNodeKind.EVIDENCE,
        subject=subject,
        recorded_at=recorded_at,
        attributes={"placeholder": True, "evidence_id": str(evidence_id)},
    )


def _emotion_node_id(emotion_id: UUID) -> UUID:
    return uuid5(NAMESPACE_URL, f"provenance:emotion:{emotion_id}")


def _ensure_contribution_placeholder(
    nodes: dict[UUID, ProvenanceNode],
    contributions: Mapping[UUID, CognitiveContribution],
    contribution_id: UUID,
    subject,
    recorded_at: datetime,
) -> None:
    if contribution_id in contributions or contribution_id in nodes:
        return
    nodes[contribution_id] = ProvenanceNode(
        node_id=contribution_id,
        kind=ProvenanceNodeKind.CONTRIBUTION,
        subject=subject,
        recorded_at=recorded_at,
        attributes={"placeholder": True},
    )


def _add_edge(
    edges: dict[UUID, ProvenanceEdge],
    *,
    kind: ProvenanceEdgeKind,
    source: UUID,
    target: UUID,
    subject,
    recorded_at: datetime,
    attributes: Mapping[str, object],
) -> None:
    edge_id = uuid5(
        NAMESPACE_URL,
        f"provenance:edge:{kind.value}:{source}:{target}",
    )
    edges[edge_id] = ProvenanceEdge(
        edge_id=edge_id,
        kind=kind,
        source_id=source,
        target_id=target,
        subject=subject,
        recorded_at=recorded_at,
        attributes=dict(attributes),
    )


def _uuid_text(value: UUID | None) -> str | None:
    return str(value) if value is not None else None
