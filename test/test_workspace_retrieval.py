from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

from self_cognition.blackboard.reducer import StateReducer
from self_cognition.core.contributions import (
    CognitiveContribution,
    CognitionType,
    ContributionOperation,
)
from self_cognition.core.errors import ContractValidationError
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.indexes import WorkspaceIndex
from self_cognition.core.memories import (
    MemoryLifecycleStatus,
    MemoryRecord,
    MemoryType,
)
from self_cognition.core.scopes import DataScope, DisclosureScope, SubjectScope
from self_cognition.core.state import ConflictRecord, SubjectState
from self_cognition.core.workspace import (
    RetrievalBudget,
    RetrievalQuery,
    RetrievalSource,
    WorkspaceBuilder,
    WorkspaceFixedContext,
    WorkspaceRunInfo,
)
from self_cognition.infrastructure.persistence.file_memory_repository import (
    FileMemoryRepository,
)
from self_cognition.workspace.retrieval import HybridWorkspaceRetriever


NOW = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)
SUBJECT = SubjectScope.legacy_user("user-1")


def make_state() -> SubjectState:
    contribution = CognitiveContribution(
        contribution_id=UUID(int=1),
        target=SUBJECT,
        target_field="episodic.experience.park",
        operation=ContributionOperation.SET,
        cognition_type=CognitionType.FACT,
        value="今天我去了公园",
        confidence=0.9,
        evidence_refs=(EvidenceRef.for_event_id(UUID(int=101), SUBJECT),),
        source_module="test.workspace",
        module_version="1",
        scope=DataScope(SUBJECT, DisclosureScope.PRIVATE),
        created_at=NOW,
        valid_from=NOW,
        target_version=0,
    )
    return StateReducer().apply(
        SubjectState.empty("user-1"),
        contribution,
        decided_at=NOW,
    )


def save_distinct_memory(repository: FileMemoryRepository) -> MemoryRecord:
    record = MemoryRecord(
        memory_id=UUID(int=2),
        memory_type=MemoryType.EPISODIC,
        subject=SUBJECT,
        scope=DataScope(SUBJECT, DisclosureScope.PRIVATE),
        content="上周我在公园完成了项目讨论",
        evidence_refs=(EvidenceRef.for_event_id(UUID(int=102), SUBJECT),),
        confidence=0.8,
        salience=0.8,
        stability=0.8,
        retrievability=1.0,
        version=1,
        lifecycle_status=MemoryLifecycleStatus.ACTIVE,
        created_at=NOW - timedelta(days=7),
        source_module="test.workspace",
        source_module_version="1",
    )
    repository.save(record, expected_version=0)
    return record


def make_builder(tmp_path: Path) -> tuple[WorkspaceBuilder, FileMemoryRepository]:
    repository = FileMemoryRepository(
        tmp_path / "memories",
        tmp_path / "indexes",
    )
    return WorkspaceBuilder(HybridWorkspaceRetriever(repository)), repository


def test_packet_selects_full_text_state_memory_and_run_with_provenance(
    tmp_path: Path,
) -> None:
    builder, repository = make_builder(tmp_path)
    state = make_state()
    memory = save_distinct_memory(repository)
    run_info = WorkspaceRunInfo(
        run_id=UUID(int=10),
        correlation_id=UUID(int=11),
        deadline=NOW + timedelta(minutes=1),
        cancelled=False,
    )

    packet = builder.build(
        "公园里发生了什么？",
        state,
        as_of=NOW,
        fixed_context=WorkspaceFixedContext(
            identity=("我是可审计的认知 Agent",),
            current_goal="回答当前问题",
            safety_rules=("不得跨 mind_id 读取",),
        ),
        run_info=run_info,
    )

    assert {item.source for item in packet.items} == {
        RetrievalSource.STATE,
        RetrievalSource.MEMORY,
        RetrievalSource.RUN,
    }
    assert any(str(memory.memory_id) in item.source_ref for item in packet.items)
    assert packet.used_tokens <= packet.budget.max_tokens
    assert all(decision.selected for decision in packet.decisions)
    assert packet.state_version == state.version
    assert packet.run_info == run_info


def test_budget_records_why_lower_ranked_candidates_are_excluded(
    tmp_path: Path,
) -> None:
    builder, repository = make_builder(tmp_path)
    save_distinct_memory(repository)

    packet = builder.build(
        "公园里发生了什么？",
        make_state(),
        as_of=NOW,
        budget=RetrievalBudget(max_tokens=512, max_items=1),
    )

    assert len(packet.items) == 1
    assert any(
        not decision.selected and decision.reason == "excluded: item budget exhausted"
        for decision in packet.decisions
    )
    assert packet.used_tokens <= 512

    with pytest.raises(ContractValidationError, match="exceeds token budget"):
        builder.build(
            "公园里发生了什么？",
            make_state(),
            as_of=NOW,
            budget=RetrievalBudget(max_tokens=1, max_items=1),
        )


def test_token_budget_and_open_conflict_are_explicit(tmp_path: Path) -> None:
    builder, repository = make_builder(tmp_path)
    memory = save_distinct_memory(repository)
    repository.save(
        replace(memory, content="公园" * 500, version=2),
        expected_version=1,
    )
    state = make_state()
    state = replace(
        state,
        conflicts=frozenset(
            {
                ConflictRecord(
                    target_field="episodic.experience.park",
                    candidate_contribution_ids=(UUID(int=1),),
                    reason="公园经历存在冲突",
                )
            }
        ),
    )

    packet = builder.build(
        "公园里发生了什么？",
        state,
        as_of=NOW,
        budget=RetrievalBudget(max_tokens=160, max_items=10),
    )

    assert RetrievalSource.CONFLICT in {item.source for item in packet.items}
    assert any(
        not decision.selected and decision.reason == "excluded: token budget exhausted"
        for decision in packet.decisions
    )


def test_compatible_index_and_authoritative_fallback_are_equivalent(
    tmp_path: Path,
) -> None:
    builder, repository = make_builder(tmp_path)
    state = make_state()
    memory = save_distinct_memory(repository)
    index = WorkspaceIndex.build(state, (memory,))

    indexed = builder.build("公园里发生了什么？", state, as_of=NOW, index=index)
    scanned = builder.build("公园里发生了什么？", state, as_of=NOW)
    stale = replace(index, source_state_version=state.version + 1)
    fallback = builder.build("公园里发生了什么？", state, as_of=NOW, index=stale)
    rebuilt = WorkspaceIndex.build(state, (memory,))

    assert indexed.index_status == "used"
    assert fallback.index_status == "fallback_incompatible"
    assert tuple((item.source, item.content) for item in indexed.items) == tuple(
        (item.source, item.content) for item in scanned.items
    )
    assert tuple((item.source, item.content) for item in fallback.items) == tuple(
        (item.source, item.content) for item in scanned.items
    )
    assert rebuilt == index

    time_query = RetrievalQuery(
        subject=SUBJECT,
        task="按时间检索",
        time_from=NOW - timedelta(days=8),
        time_to=NOW - timedelta(days=6),
    )
    indexed_time = builder.build(
        time_query.task,
        state,
        as_of=NOW,
        query=time_query,
        index=index,
    )
    scanned_time = builder.build(
        time_query.task,
        state,
        as_of=NOW,
        query=time_query,
    )
    assert tuple(item.content for item in indexed_time.items) == tuple(
        item.content for item in scanned_time.items
    )


def test_each_packet_receives_fresh_fixed_context_and_enforces_scope(
    tmp_path: Path,
) -> None:
    builder, _ = make_builder(tmp_path)
    state = make_state()
    first = builder.build(
        "公园",
        state,
        as_of=NOW,
        fixed_context=WorkspaceFixedContext(current_goal="第一个目标"),
    )
    second = builder.build(
        "公园",
        state,
        as_of=NOW,
        fixed_context=WorkspaceFixedContext(current_goal="第二个目标"),
    )

    assert first.fixed_context.current_goal == "第一个目标"
    assert second.fixed_context.current_goal == "第二个目标"
    with pytest.raises(ContractValidationError, match="subjects do not match"):
        builder.build(
            "公园",
            state,
            query=RetrievalQuery(
                subject=SubjectScope.legacy_user("user-2"),
                task="公园",
            ),
        )
