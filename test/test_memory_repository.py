import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

from self_cognition.bootstrap import build_container
from self_cognition.core.errors import (
    ContractValidationError,
    MalformedSerializedDataError,
    VersionConflictError,
)
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.memories import (
    MemoryLifecycleStatus,
    MemoryRecord,
    MemoryType,
)
from self_cognition.core.scopes import (
    DataScope,
    DisclosureScope,
    MindScope,
    SubjectKind,
    SubjectRef,
    SubjectScope,
)
from self_cognition.infrastructure.persistence.file_memory_repository import (
    FileMemoryRepository,
)
from self_cognition.infrastructure.persistence.serialization import (
    memory_from_json,
    memory_to_json,
)


NOW = datetime(2026, 9, 2, tzinfo=timezone.utc)


def make_subject(mind_id: str = "mind-1") -> SubjectScope:
    return SubjectScope(
        MindScope(mind_id),
        SubjectRef(SubjectKind.USER, "user-1"),
    )


def make_record(
    *,
    memory_type: MemoryType = MemoryType.SEMANTIC,
    subject: SubjectScope | None = None,
    evidence_subject: SubjectScope | None = None,
    evidence_refs: tuple[EvidenceRef, ...] | None = None,
) -> MemoryRecord:
    owner = subject or make_subject()
    evidence_owner = evidence_subject or owner
    evidence = EvidenceRef.for_event_id(UUID(int=1), evidence_owner)
    return MemoryRecord(
        memory_id=UUID(int=10),
        memory_type=memory_type,
        subject=owner,
        scope=DataScope(owner, DisclosureScope.PRIVATE),
        content={"preference": "morning"},
        evidence_refs=(evidence,) if evidence_refs is None else evidence_refs,
        confidence=0.9,
        salience=0.8,
        stability=0.7,
        retrievability=0.6,
        version=1,
        lifecycle_status=MemoryLifecycleStatus.ACTIVE,
        created_at=NOW,
        source_module="test.memory",
        source_module_version="1",
    )


@pytest.mark.parametrize("memory_type", tuple(MemoryType))
def test_memory_record_round_trips_all_declared_types(
    memory_type: MemoryType,
) -> None:
    record = make_record(memory_type=memory_type)

    assert memory_from_json(memory_to_json(record)) == record


def test_file_repository_preserves_versions_and_rebuilds_derived_index(
    tmp_path: Path,
) -> None:
    memory_root = tmp_path / "memories"
    index_root = tmp_path / "indexes" / "memories"
    repository = FileMemoryRepository(memory_root, index_root)
    first = make_record()
    second = replace(
        first,
        content={"preference": "early morning"},
        version=2,
        created_at=NOW + timedelta(minutes=1),
    )

    repository.save(first, expected_version=0)
    repository.save(first, expected_version=0)
    with pytest.raises(VersionConflictError):
        repository.save(second, expected_version=0)
    repository.save(second, expected_version=1)

    assert repository.load(first.subject, first.memory_id) == second
    assert repository.read_history(first.subject, first.memory_id) == (
        first,
        second,
    )
    assert len(tuple(memory_root.rglob("*.json"))) == 2

    index_path = next(index_root.rglob("*.json"))
    index_path.unlink()
    repository.rebuild_index(first.subject)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["records"] == [
        {
            "latest_version": 2,
            "lifecycle_status": "active",
            "memory_id": str(first.memory_id),
            "memory_type": "semantic",
        }
    ]


def test_repository_rejects_working_memory_and_invalid_provenance(
    tmp_path: Path,
) -> None:
    repository = FileMemoryRepository(
        tmp_path / "memories",
        tmp_path / "indexes" / "memories",
    )

    with pytest.raises(ContractValidationError, match="run checkpoints"):
        repository.save(
            make_record(memory_type=MemoryType.WORKING),
            expected_version=0,
        )
    with pytest.raises(ContractValidationError, match="must not be empty"):
        make_record(evidence_refs=())
    with pytest.raises(ContractValidationError, match="same mind"):
        make_record(evidence_subject=make_subject("mind-2"))


def test_repository_isolates_minds_and_rejects_corrupt_authority(
    tmp_path: Path,
) -> None:
    repository = FileMemoryRepository(
        tmp_path / "memories",
        tmp_path / "indexes" / "memories",
    )
    first_mind = make_record()
    second_mind = make_record(subject=make_subject("mind-2"))
    repository.save(first_mind, expected_version=0)
    repository.save(second_mind, expected_version=0)

    assert repository.read_by_subject(first_mind.subject) == (first_mind,)
    assert repository.read_by_subject(second_mind.subject) == (second_mind,)
    assert repository.load(second_mind.subject, first_mind.memory_id) == second_mind

    first_path = next(
        path
        for path in (tmp_path / "memories").rglob("*.json")
        if json.loads(path.read_text(encoding="utf-8"))["subject"]["mind_id"]
        == "mind-1"
    )
    first_path.write_text("{broken}", encoding="utf-8")
    with pytest.raises(MalformedSerializedDataError):
        repository.load(first_mind.subject, first_mind.memory_id)


def test_bootstrap_exposes_file_memory_repository(tmp_path: Path) -> None:
    container = build_container(
        tmp_path,
        dotenv_path=tmp_path / "missing.env",
    )

    assert isinstance(container.memory_repository, FileMemoryRepository)
