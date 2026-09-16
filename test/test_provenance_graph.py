from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from self_cognition.bootstrap import build_container
from self_cognition.core.actions import (
    ActionDecision,
    ActionDecisionPayload,
    ActionDecisionStatus,
    ActionProposedPayload,
    ActionRequest,
    ActionResult,
    ActionResultPayload,
    ActionResultStatus,
)
from self_cognition.core.contributions import (
    CognitiveContribution,
    CognitionType,
    ContributionOperation,
)
from self_cognition.core.errors import ContractValidationError
from self_cognition.core.evidence import EvidenceRef, EvidenceSourceKind
from self_cognition.core.events import (
    CognitionModuleResultPayload,
    EventEnvelope,
    EventSource,
)
from self_cognition.core.provenance import (
    ProvenanceEdgeKind,
    ProvenanceGraph,
    ProvenanceNodeKind,
)
from self_cognition.core.scopes import (
    DataScope,
    DisclosureScope,
    MindScope,
    SubjectScope,
)
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
    _process(container, subject, "小明是我的朋友")
    _process(container, subject, "我开始准备研究项目")

    mind = MindScope(subject.mind.mind_id)
    graph = container.provenance.rebuild(mind)

    assert graph.mind_id == mind.mind_id
    assert graph.nodes
    assert graph.edges
    assert {node.kind for node in graph.nodes} >= {
        ProvenanceNodeKind.EVENT,
        ProvenanceNodeKind.CONTRIBUTION,
        ProvenanceNodeKind.MEMORY,
        ProvenanceNodeKind.RELATIONSHIP,
        ProvenanceNodeKind.NARRATIVE,
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

def _action_fixture(owner: SubjectScope):
    now = SYSTEM_CLOCK.now()
    cause_id = uuid4()
    run_id = uuid4()
    correlation_id = uuid4()
    request = ActionRequest(
        action_id=uuid4(),
        proposal_request_id=uuid4(),
        owner=owner,
        plan_id=uuid4(),
        plan_version=1,
        step_id="step-1",
        tool_id="file.write",
        arguments={"path": "notes.txt"},
        expected_side_effects=(),
        idempotency_key="idem-1",
        requested_at=now,
    )
    decision = ActionDecision(
        decision_id=uuid4(),
        action_id=request.action_id,
        status=ActionDecisionStatus.ALLOWED,
        reason="allowed for test",
        value_basis=("test",),
        relationship_context="test context",
        risks=("bounded",),
        evidence_ids=(uuid4(),),
        decided_at=now,
        valid_until=now + timedelta(minutes=5),
        one_time_scope=request.action_id,
    )
    result = ActionResult(
        result_id=uuid4(),
        action_id=request.action_id,
        owner=owner,
        status=ActionResultStatus.SUCCEEDED,
        summary="ok",
        output={"ok": True},
        actual_side_effects=(),
        recorded_at=now,
    )
    scope = DataScope(owner, DisclosureScope.MIND)
    events = (
        EventEnvelope(
            event_id=uuid4(),
            event_type="action.proposed",
            actor=None,
            subject=owner,
            payload=ActionProposedPayload(request),
            occurred_at=now,
            recorded_at=now,
            source=EventSource.SYSTEM,
            scope=scope,
            causation_id=cause_id,
            correlation_id=correlation_id,
            run_id=run_id,
        ),
        EventEnvelope(
            event_id=uuid4(),
            event_type="action.decided",
            actor=None,
            subject=owner,
            payload=ActionDecisionPayload(request, decision),
            occurred_at=now,
            recorded_at=now,
            source=EventSource.SYSTEM,
            scope=scope,
            causation_id=cause_id,
            correlation_id=correlation_id,
            run_id=run_id,
        ),
        EventEnvelope(
            event_id=result.result_id,
            event_type="action.result",
            actor=None,
            subject=owner,
            payload=ActionResultPayload(result),
            occurred_at=now,
            recorded_at=now,
            source=EventSource.TOOL,
            scope=scope,
            causation_id=cause_id,
            correlation_id=correlation_id,
            run_id=run_id,
        ),
    )
    return events, request, decision, result


def test_provenance_graph_includes_action_tool_result_and_full_lifecycle(
    tmp_path: Path,
) -> None:
    container = build_container(
        tmp_path,
        settings=ApplicationSettings(data_dir=tmp_path, worker_enabled=False),
        dotenv_path=tmp_path / "missing.env",
    )
    owner = SubjectScope.for_mind("default-mind")
    events, request, decision, result = _action_fixture(owner)
    for event in events:
        container.event_store.append(event)

    graphs = container.provenance.rebuild_all()
    assert len(graphs) == 1
    graph = graphs[0]
    assert {node.kind for node in graph.nodes} >= {
        ProvenanceNodeKind.ACTION_REQUEST,
        ProvenanceNodeKind.ACTION_DECISION,
        ProvenanceNodeKind.ACTION_RESULT,
        ProvenanceNodeKind.EVIDENCE,
    }

    result_node = graph.node(result.result_id)
    assert result_node.attributes["tool_result"] is True
    chain_ids = {
        node.node_id for node in graph.provenance_chain(result.result_id)
    }
    assert request.action_id in chain_ids
    assert decision.decision_id in chain_ids
    assert graph.source_events(result.result_id)

    assert container.provenance.verify()
    container.provenance.delete_all()
    assert container.provenance.load(MindScope("default-mind")) is None
    assert not container.provenance.verify()

    rebuilt = container.provenance.rebuild_all()
    assert len(rebuilt) == 1
    assert container.provenance.verify()

def test_provenance_graph_builds_evidence_node_for_system_prior() -> None:
    source_event = EventEnvelope.user_message("user-1", "hello")
    subject = source_event.subject
    prior_ref = EvidenceRef(
        evidence_id=uuid4(),
        source_kind=EvidenceSourceKind.SYSTEM_PRIOR,
        source_ref="policy:v1",
        scope=DataScope(subject, DisclosureScope.PRIVATE),
        locator="prior.value",
        reliability=0.9,
    )
    contribution = CognitiveContribution(
        contribution_id=uuid4(),
        target=subject,
        target_field="values.principle",
        operation=ContributionOperation.SET,
        cognition_type=CognitionType.INFERENCE,
        value="evidence-backed principle",
        confidence=0.7,
        evidence_refs=(prior_ref,),
        source_module="test.evidence",
        module_version="1",
        scope=DataScope(subject, DisclosureScope.PRIVATE),
        created_at=source_event.recorded_at,
        valid_from=source_event.occurred_at,
    )
    result_event = EventEnvelope(
        event_id=uuid4(),
        event_type="cognition.module_result",
        actor=None,
        subject=subject,
        payload=CognitionModuleResultPayload(
            module_id="test.evidence",
            module_version="1",
            deterministic=True,
            status="succeeded",
            contributions=(contribution,),
        ),
        occurred_at=source_event.occurred_at,
        recorded_at=source_event.recorded_at,
        source=EventSource.SYSTEM,
        scope=DataScope(subject, DisclosureScope.PRIVATE),
        causation_id=source_event.event_id,
        correlation_id=uuid4(),
        run_id=uuid4(),
    )

    graph = build_provenance_graph(
        MindScope(subject.mind.mind_id),
        (source_event, result_event),
        (),
    )
    evidence_node = graph.node(prior_ref.evidence_id)

    assert evidence_node.kind is ProvenanceNodeKind.EVIDENCE
    assert evidence_node.attributes["source_kind"] == "system_prior"
    assert any(
        edge.kind is ProvenanceEdgeKind.DERIVED_FROM
        and edge.target_id == prior_ref.evidence_id
        for edge in graph.outgoing(contribution.contribution_id)
    )
