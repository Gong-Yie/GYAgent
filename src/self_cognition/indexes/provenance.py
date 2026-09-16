from __future__ import annotations

from datetime import datetime
from typing import Mapping
from uuid import NAMESPACE_URL, UUID, uuid5

from self_cognition.core.affect import MOOD_FIELD, EmotionState, MoodState
from self_cognition.core.contributions import CognitiveContribution
from self_cognition.core.errors import ContractValidationError
from self_cognition.core.events import (
    CognitionModuleResultPayload,
    EventEnvelope,
)
from self_cognition.core.memories import MemoryRecord
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
                if evidence.evidence_id not in event_nodes:
                    continue
                _add_edge(
                    edges,
                    kind=ProvenanceEdgeKind.DERIVED_FROM,
                    source=contribution.contribution_id,
                    target=evidence.evidence_id,
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
            if evidence.evidence_id not in event_nodes:
                continue
            _add_edge(
                edges,
                kind=ProvenanceEdgeKind.DERIVED_FROM,
                source=memory.memory_id,
                target=evidence.evidence_id,
                subject=memory.subject,
                recorded_at=memory.created_at,
                attributes={"source_kind": evidence.source_kind.value},
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
            if evidence.evidence_id not in event_nodes:
                continue
            _add_edge(
                edges,
                kind=ProvenanceEdgeKind.DERIVED_FROM,
                source=affect_node.node_id,
                target=evidence.evidence_id,
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

    def source_events(self, mind: MindScope, node_id: UUID):
        graph = self.load(mind)
        if graph is None:
            graph = self.rebuild(mind)
        return graph.source_events(node_id)


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
