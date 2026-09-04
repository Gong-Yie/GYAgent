from dataclasses import replace
from datetime import datetime, timezone
from uuid import UUID

from self_cognition.blackboard.reducer import StateReducer
from self_cognition.blackboard.service import CognitiveSpaceService
from self_cognition.core.cognition import CognitionRequest
from self_cognition.cognition.procedural.execution_extractor import (
    ProceduralExecutionExtractor,
)
from self_cognition.cognition.episodic.memory_extractor import EpisodicMemoryExtractor
from self_cognition.cognition.semantic.concept_pattern_extractor import (
    ConceptPatternExtractor,
)
from self_cognition.core.contributions import CognitiveContribution, CognitionType
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.identity import (
    CapabilityKind,
    CapabilityPermission,
)
from self_cognition.core.memories import MemoryType
from self_cognition.core.scopes import SubjectScope
from self_cognition.core.state import SubjectState
from self_cognition.core.workspace import RetrievalQuery, WorkspaceBuilder
from self_cognition.infrastructure.persistence.file_memory_repository import (
    FileMemoryRepository,
)
from self_cognition.memory.encoder import StateChangeMemoryEncoder
from self_cognition.workspace.retrieval import HybridWorkspaceRetriever
from self_cognition.runtime.cognition_context import ReadOnlyCognitionContext
from self_cognition.tools.registry import CapabilityRegistration, CapabilityRegistry
from self_cognition.core.events import EventSource
from self_cognition.core.events import Event


NOW = datetime(2026, 9, 4, tzinfo=timezone.utc)
MIND = SubjectScope.for_mind("mind-17")


class FixedClock:
    def now(self):
        return NOW


def _tool_event(succeeded: bool = True):
    registry = CapabilityRegistry(
        (
            CapabilityRegistration(
                capability_id="filesystem.read",
                name="读取工作区文件",
                kind=CapabilityKind.TOOL,
                permission=CapabilityPermission.GRANTED,
            ),
        )
    )
    return registry.record_execution(
        "filesystem.read",
        MIND,
        succeeded=succeeded,
        reason=None if succeeded else "permission denied",
        event_id=UUID(int=1701 if succeeded else 1702),
    )


def test_procedural_memory_requires_tool_execution_and_preserves_failure_mode():
    extractor = ProceduralExecutionExtractor()
    success = extractor.process(_tool_event())
    failure = extractor.process(_tool_event(False))

    assert len(success) == len(failure) == 1
    assert success[0].target.subject.kind.value == "mind"
    assert success[0].target_field == "procedural.execution.filesystem.read"
    assert success[0].value["outcome"] == "succeeded"
    assert failure[0].value["failure_mode"] == "permission denied"
    assert success[0].evidence_refs[0].source_kind.value == "tool_result"


def test_procedural_memory_is_encoded_and_can_be_filtered(tmp_path):
    event = _tool_event()
    contribution = ProceduralExecutionExtractor().process(event)[0]
    semantic = CognitiveContribution.set_from_event(
        event,
        contribution_id=UUID(int=1703),
        target_field="profile.name",
        cognition_type=CognitionType.FACT,
        value="认知助手",
        confidence=1.0,
        evidence_refs=(EvidenceRef.for_event(event),),
        source_module="test.semantic",
        module_version="1",
    )
    state = CognitiveSpaceService(StateReducer()).submit(
        SubjectState.empty("mind-17", mind_id="mind-17", subject_kind=MIND.subject.kind),
        (replace(contribution, target_version=0), replace(semantic, target_version=0)),
        decided_at=event.recorded_at,
    )
    repository = FileMemoryRepository(tmp_path / "memories", tmp_path / "indexes")
    procedural_change = next(
        change
        for change in state.changes
        if change.contribution.target_field.startswith("procedural.")
    )
    record = StateChangeMemoryEncoder().encode(procedural_change)
    assert record is not None
    repository.save(record, expected_version=0)
    assert record.memory_type is MemoryType.PROCEDURAL

    query = RetrievalQuery(
        subject=MIND,
        task="执行文件读取",
        field_patterns=("procedural.*", "profile.*"),
        memory_types=frozenset({MemoryType.PROCEDURAL}),
    )
    assert query.memory_types == frozenset({MemoryType.PROCEDURAL})
    retrieved = HybridWorkspaceRetriever(repository).retrieve(
        query,
        SubjectState.empty(
            "mind-17",
            mind_id="mind-17",
            subject_kind=MIND.subject.kind,
        ),
        as_of=event.recorded_at,
    )
    assert retrieved.candidates[0].target_field == "procedural.execution.filesystem.read"
    packet = WorkspaceBuilder().build("执行文件读取", state, query=query, as_of=event.recorded_at)
    assert tuple(item.target_field for item in packet.items) == (
        "procedural.execution.filesystem.read",
    )


def test_system_capability_registration_does_not_create_procedural_memory():
    registry = CapabilityRegistry(
        (
            CapabilityRegistration(
                capability_id="filesystem.read",
                name="读取工作区文件",
                kind=CapabilityKind.TOOL,
                permission=CapabilityPermission.GRANTED,
            ),
        )
    )
    event = registry.registration_event("filesystem.read", MIND)
    assert event.source is EventSource.SYSTEM
    assert ProceduralExecutionExtractor().process(event) == ()


def test_semantic_module_separates_explicit_concepts_and_cross_experience_patterns():
    concept_event = Event.user_message("user-17", "编程是一种创造活动", clock=FixedClock())
    concept = ConceptPatternExtractor().process(
        CognitionRequest(
            event=concept_event,
            context=ReadOnlyCognitionContext(
                builder=WorkspaceBuilder(),
                state=SubjectState.empty("user-17"),
                as_of=concept_event.recorded_at,
            ),
        )
    )
    assert concept[0].target_field == "semantic.concept.编程"
    assert concept[0].cognition_type is CognitionType.FACT

    old_state = SubjectState.empty("user-17")
    episodic = EpisodicMemoryExtractor()
    for text in ("昨天我去了公园", "前天我去了公园"):
        contribution = replace(
            episodic.process(Event.user_message("user-17", text, clock=FixedClock()))[0],
            target_version=old_state.version,
        )
        old_state = CognitiveSpaceService(StateReducer()).submit(
            old_state,
            (contribution,),
            decided_at=NOW,
        )
    current = Event.user_message("user-17", "今天我去了公园", clock=FixedClock())
    pattern = ConceptPatternExtractor().process(
        CognitionRequest(
            event=current,
            context=ReadOnlyCognitionContext(
                builder=WorkspaceBuilder(),
                state=old_state,
                as_of=NOW,
            ),
        )
    )
    assert pattern[0].target_field == "semantic.pattern.去了"
    assert pattern[0].cognition_type is CognitionType.INFERENCE
    assert len(pattern[0].evidence_refs) >= 2
