from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

from self_cognition.core.evidence import EvidenceRef, EvidenceSourceKind
from self_cognition.core.memories import (
    MemoryConsolidationStatus,
    MemoryCues,
    MemoryLifecycleStatus,
    MemoryRecord,
    MemorySourceRef,
    MemoryType,
)
from self_cognition.core.scopes import DataScope, DisclosureScope, SubjectScope
from self_cognition.infrastructure.persistence.file_memory_repository import (
    FileMemoryRepository,
)
from self_cognition.memory.behavior import (
    MemoryConsolidationService,
    MemoryRetrievalService,
    derive_recall_view,
)


NOW = datetime(2026, 9, 2, tzinfo=timezone.utc)
SUBJECT = SubjectScope.legacy_user("user-1")


def make_record(
    number: int,
    *,
    created_at: datetime = NOW,
    content: object = "morning study",
    memory_type: MemoryType = MemoryType.EPISODIC,
    cues: MemoryCues | None = None,
    source_kind: EvidenceSourceKind = EvidenceSourceKind.EVENT,
) -> MemoryRecord:
    evidence = EvidenceRef(
        evidence_id=UUID(int=100 + number),
        source_kind=source_kind,
        source_ref=f"source-{number}",
        scope=DataScope(SUBJECT, DisclosureScope.PRIVATE),
        observed_at=created_at,
        reliability=1.0,
    )
    return MemoryRecord(
        memory_id=UUID(int=number),
        memory_type=memory_type,
        subject=SUBJECT,
        scope=DataScope(SUBJECT, DisclosureScope.PRIVATE),
        content=content,
        evidence_refs=(evidence,),
        confidence=0.9,
        salience=0.8,
        stability=0.5,
        retrievability=1.0,
        version=1,
        lifecycle_status=MemoryLifecycleStatus.ACTIVE,
        created_at=created_at,
        source_module="test.memory",
        source_module_version="1",
        sources=(
            MemorySourceRef(
                contribution_id=UUID(int=200 + number),
                old_state_version=number - 1,
                new_state_version=number,
                target_field="episodic.experience",
            ),
        ),
        cues=cues or MemoryCues(people=("user-1",), topics=("study",)),
    )


def test_recall_uses_cues_and_read_time_decay_without_mutating_record() -> None:
    record = make_record(1, created_at=NOW - timedelta(days=2))

    matched = derive_recall_view(record, MemoryCues(topics=("study",)), (), NOW)
    unmatched = derive_recall_view(record, MemoryCues(topics=("work",)), (), NOW)
    old = derive_recall_view(
        record,
        MemoryCues(topics=("study",)),
        (),
        NOW + timedelta(days=90),
    )

    assert matched.score > unmatched.score
    assert old.score < matched.score
    assert record.retrievability == 1.0


def test_spaced_access_reinforces_more_than_same_session_accesses() -> None:
    record = make_record(1)
    same_session = derive_recall_view(
        record,
        MemoryCues(),
        (NOW + timedelta(hours=1), NOW + timedelta(hours=2)),
        NOW + timedelta(days=2),
    )
    spaced = derive_recall_view(
        record,
        MemoryCues(),
        (NOW + timedelta(days=1), NOW + timedelta(days=3)),
        NOW + timedelta(days=4),
    )

    assert spaced.effective_stability > same_session.effective_stability


def test_interference_is_explained_and_does_not_delete_candidates() -> None:
    first = make_record(1, content="morning study")
    second = make_record(2, content="evening study")

    view = derive_recall_view(
        first,
        MemoryCues(topics=("study",)),
        (),
        NOW,
        (first, second),
    )

    assert view.interference[0].memory_id == second.memory_id
    assert "cue overlap" in view.interference[0].reason


def test_consolidation_requires_spaced_evidence_and_is_idempotent(tmp_path: Path) -> None:
    repository = FileMemoryRepository(tmp_path / "memories", tmp_path / "indexes")
    records = tuple(
        make_record(index, created_at=NOW + timedelta(days=index - 1))
        for index in range(1, 4)
    )
    for record in records:
        repository.save(record, expected_version=0)

    service = MemoryConsolidationService(repository)
    first = service.consolidate(SUBJECT, now=NOW + timedelta(days=4))
    second = service.consolidate(SUBJECT, now=NOW + timedelta(days=4))

    assert len(first) == 1
    assert first[0].memory_type is MemoryType.SEMANTIC
    assert first[0].consolidation_status is MemoryConsolidationStatus.CONSOLIDATED
    assert len(first[0].evidence_refs) == 3
    assert second == ()


def test_retrieval_returns_only_active_records(tmp_path: Path) -> None:
    repository = FileMemoryRepository(tmp_path / "memories", tmp_path / "indexes")
    active = make_record(1)
    invalidated = replace(
        make_record(2, content="other"),
        lifecycle_status=MemoryLifecycleStatus.INVALIDATED,
    )
    repository.save(active, expected_version=0)
    repository.save(invalidated, expected_version=0)

    views = MemoryRetrievalService(repository).retrieve(
        SUBJECT,
        MemoryCues(topics=("study",)),
        now=NOW,
    )

    assert tuple(view.record.memory_id for view in views) == (active.memory_id,)
