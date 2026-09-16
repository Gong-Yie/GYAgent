from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from self_cognition.bootstrap import build_container
from self_cognition.core.events import EventEnvelope
from self_cognition.core.errors import ContractValidationError
from self_cognition.core.provenance import (
    ProvenanceEdgeKind,
    ProvenanceGraph,
    ProvenanceNodeKind,
)
from self_cognition.core.scopes import MindScope, SubjectScope
from self_cognition.core.time import SYSTEM_CLOCK
from self_cognition.indexes.provenance import build_provenance_graph
from self_cognition.runtime.run_context import RunContext
from self_cognition.settings import ApplicationSettings


def _context() -> RunContext:
    return RunContext(
        uuid4(),
        uuid4(),
        SYSTEM_CLOCK.now() + timedelta(minutes=5),
    )


def _process(container, subject: SubjectScope, text: str) -> None:
    context = _context()
    event = EventEnvelope.user_message(
        subject,
        text,
        run_id=context.run_id,
        correlation_id=context.correlation_id,
    )
    result = container.process_event.process(event, context)
    assert result.status.value == "succeeded"


def test_provenance_graph_rebuilds_events_contributions_memories_and_emotions(
    tmp_path: Path,
) -> None:
    container = build_container(
        tmp_path,
        settings=ApplicationSettings(data_dir=tmp_path, worker_enabled=False),
        dotenv_path=tmp_path / "missing.env",
    )
    subject = SubjectScope.legacy_user("provenance-user")

    _process(container, subject, "我喜欢晚上学习")
    _process(container, subject, "好无聊，想找个人聊聊天")

    mind = MindScope(subject.mind.mind_id)
    graph = container.provenance.rebuild(mind)

    assert graph.mind_id == mind.mind_id
    assert graph.nodes
    assert graph.edges
    assert {node.kind for node in graph.nodes} >= {
        ProvenanceNodeKind.EVENT,
        ProvenanceNodeKind.CONTRIBUTION,
        ProvenanceNodeKind.MEMORY,
        ProvenanceNodeKind.EMOTION,
        ProvenanceNodeKind.MOOD,
    }

    memory_nodes = [
        node for node in graph.nodes if node.kind is ProvenanceNodeKind.MEMORY
    ]
    emotion_nodes = [
        node for node in graph.nodes if node.kind is ProvenanceNodeKind.EMOTION
    ]
    mood_nodes = [
        node for node in graph.nodes if node.kind is ProvenanceNodeKind.MOOD
    ]

    assert memory_nodes
    assert emotion_nodes
    assert mood_nodes

    memory_events = graph.source_events(memory_nodes[0].node_id)
    assert memory_events
    memory_chain = graph.provenance_chain(memory_nodes[0].node_id)
    assert any(node.kind is ProvenanceNodeKind.CONTRIBUTION for node in memory_chain)
    assert any(
        edge.kind is ProvenanceEdgeKind.DERIVED_FROM
        for edge in graph.outgoing(memory_nodes[0].node_id)
    )

    emotion_events = graph.source_events(emotion_nodes[0].node_id)
    assert emotion_events
    assert any(
        node.kind is ProvenanceNodeKind.CONTRIBUTION
        for node in graph.provenance_chain(emotion_nodes[0].node_id)
    )

    loaded = container.provenance.load(mind)
    assert loaded == graph

    container.provenance.delete(mind)
    assert container.provenance.load(mind) is None
    rebuilt = container.provenance.rebuild(mind)
    assert rebuilt == graph
    assert container.provenance.load(mind) == graph


def test_provenance_graph_serialization_round_trip() -> None:
    graph = ProvenanceGraph(
        mind_id="mind-1",
        nodes=(),
        edges=(),
    )

    assert ProvenanceGraph.from_state_value(graph.to_state_value()) == graph


def test_provenance_builder_rejects_cross_mind_event() -> None:
    event = EventEnvelope.user_message("user-1", "hello")

    with pytest.raises(ContractValidationError, match="another mind"):
        build_provenance_graph(
            MindScope("other-mind"),
            (event,),
            (),
        )
