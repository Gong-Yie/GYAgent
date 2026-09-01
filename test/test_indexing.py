from datetime import datetime, timezone
from uuid import UUID

from self_cognition.core.contributions import (
    CognitiveContribution,
    CognitionType,
    ContributionOperation,
)
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.indexes import WorkspaceIndex
from self_cognition.core.scopes import DataScope, DisclosureScope, SubjectScope
from self_cognition.core.state import SubjectState
from self_cognition.core.workspace import WorkspaceBuilder
from self_cognition.infrastructure.persistence.serialization import (
    state_from_json,
    state_to_json,
)
from self_cognition.blackboard.reducer import StateReducer


def make_contribution(
    contribution_id: int,
    field: str,
    value: object,
    event_id: int,
) -> CognitiveContribution:
    event_uuid = UUID(int=event_id)
    subject = SubjectScope.legacy_user("user-1")
    return CognitiveContribution(
        contribution_id=UUID(int=contribution_id),
        target=subject,
        target_field=field,
        operation=ContributionOperation.SET,
        cognition_type=CognitionType.FACT,
        value=value,
        confidence=1.0,
        evidence_refs=(EvidenceRef.for_event_id(event_uuid, subject),),
        source_module="test.indexing",
        module_version="1",
        scope=DataScope(subject, DisclosureScope.PRIVATE),
        created_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        valid_from=datetime(2026, 9, 1, tzinfo=timezone.utc),
        target_version=0,
    )


def make_state() -> SubjectState:
    reducer = StateReducer()
    state = SubjectState.empty("user-1")
    return reducer.apply_many(
        state,
        (
            make_contribution(
                1,
                "episodic.experience.2026-08-13T09:00:00+00:00.1",
                "今天完成阅读",
                1,
            ),
            make_contribution(
                5,
                "episodic.experience.2026-08-13T08:00:00+00:00.5",
                "今天开始阅读",
                5,
            ),
            make_contribution(
                2,
                "narrative.chapter.2026-08-13T09:00:00+00:00.2",
                {
                    "theme": "研究项目",
                    "stage": "完成",
                    "summary": "完成研究项目",
                    "occurred_at": "2026-08-13T09:00:00+00:00",
                },
                2,
            ),
            make_contribution(
                3,
                "narrative.chapter.2026-08-13T07:00:00+00:00.3",
                {
                    "theme": "研究项目",
                    "stage": "启动",
                    "summary": "开始准备研究项目",
                    "occurred_at": "2026-08-13T07:00:00+00:00",
                },
                3,
            ),
            make_contribution(4, "profile.name", "小明", 4),
        ),
        decided_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
    )


def test_index_records_source_and_returns_prefixes_in_time_order():
    state = make_state()
    index = WorkspaceIndex.build(state)

    assert index.source_subject_id == "user-1"
    assert index.source_state_version == state.version
    assert index.is_compatible(state)
    assert index.fields_for_prefix("narrative.chapter.", chronological=True) == (
        "narrative.chapter.2026-08-13T07:00:00+00:00.3",
        "narrative.chapter.2026-08-13T09:00:00+00:00.2",
    )
    assert index.fields_for_prefix("episodic.experience.", chronological=True) == (
        "episodic.experience.2026-08-13T08:00:00+00:00.5",
        "episodic.experience.2026-08-13T09:00:00+00:00.1",
    )


def test_indexed_workspace_matches_unindexed_workspace_and_reads_state_content():
    state = make_state()
    index = WorkspaceIndex.build(state)
    builder = WorkspaceBuilder()

    for question in ("我经历过什么？", "我的项目经历如何发展？"):
        assert builder.build(question, state) == builder.build(
            question,
            state,
            index=index,
        )


def test_stale_or_foreign_index_falls_back_without_leaking_indexed_fields():
    state = make_state()
    index = WorkspaceIndex.build(state)
    changed_state = SubjectState(
        subject_id=state.subject_id,
        version=state.version + 1,
        entries=state.entries,
        applied_contribution_ids=state.applied_contribution_ids,
        conflicts=state.conflicts,
    )
    foreign_state = SubjectState(
        subject_id="user-2",
        version=state.version,
        entries={},
        applied_contribution_ids=frozenset(),
        conflicts=frozenset(),
    )

    assert index.is_compatible(changed_state) is False
    assert index.is_compatible(foreign_state) is False
    assert WorkspaceBuilder(index=index).build("我经历过什么？", changed_state).items
    foreign_workspace = WorkspaceBuilder(index=index).build(
        "我经历过什么？",
        foreign_state,
    )
    assert foreign_workspace.subject_id == "user-2"
    assert foreign_workspace.items == ()


def test_index_rebuilds_identically_from_the_authoritative_state_snapshot():
    state = make_state()
    restored_state = state_from_json(state_to_json(state))

    assert WorkspaceIndex.build(restored_state) == WorkspaceIndex.build(state)


def test_index_field_names_cannot_supply_content_or_evidence():
    state = make_state()
    valid_index = WorkspaceIndex.build(state)
    tampered_index = WorkspaceIndex(
        source_subject_id=state.subject_id,
        source_state_version=state.version,
        fields_by_prefix={
            "episodic.experience": ("episodic.experience.phantom",),
        },
        time_index=(),
    )

    assert valid_index.is_compatible(state)
    workspace = WorkspaceBuilder(index=tampered_index).build(
        "我经历过什么？",
        state,
    )

    assert workspace.items == ()
