from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

from self_cognition.bootstrap import build_container
from self_cognition.blackboard.reducer import StateReducer
from self_cognition.core.contributions import (
    CognitiveContribution,
    CognitionType,
    ContributionOperation,
)
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.errors import ContractValidationError
from self_cognition.core.memories import MemoryType
from self_cognition.core.scopes import (
    DataScope,
    DisclosureScope,
    MindScope,
    SubjectScope,
)
from self_cognition.core.state import StateDecisionStatus, SubjectState
from self_cognition.core.events import Event
from self_cognition.infrastructure.persistence.file_memory_repository import (
    FileMemoryRepository,
)
from self_cognition.infrastructure.persistence.serialization import (
    memory_from_dict,
    memory_to_dict,
)
from self_cognition.memory.encoder import StateChangeMemoryEncoder
from self_cognition.memory.service import MemoryAccessService, MemoryEncodingService
from self_cognition.runtime.run_context import RunContext


NOW = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)


def make_contribution(
    contribution_id: int,
    target_field: str,
    *,
    target: SubjectScope | None = None,
) -> CognitiveContribution:
    owner = target or SubjectScope.legacy_user("user-1")
    return CognitiveContribution(
        contribution_id=UUID(int=contribution_id),
        target=owner,
        target_field=target_field,
        operation=ContributionOperation.SET,
        cognition_type=CognitionType.FACT,
        value={"field": target_field, "value": contribution_id},
        confidence=0.9,
        evidence_refs=(EvidenceRef.for_event_id(UUID(int=100 + contribution_id), owner),),
        source_module="test.module",
        module_version="1",
        scope=DataScope(owner, DisclosureScope.PRIVATE),
        created_at=NOW,
        valid_from=NOW,
        target_version=0,
    )


def accepted_state(target_field: str = "preferences.study_time") -> SubjectState:
    contribution = make_contribution(1, target_field)
    return StateReducer().apply(
        SubjectState.empty("user-1"),
        contribution,
        decided_at=NOW,
    )


@pytest.mark.parametrize(
    ("target_field", "expected_type"),
    (
        ("episodic.experience.1", MemoryType.EPISODIC),
        ("relationships.小明.role", MemoryType.RELATIONSHIP),
        ("narrative.chapter.1", MemoryType.NARRATIVE),
        ("profile.name", MemoryType.SEMANTIC),
        ("preferences.study_time", MemoryType.SEMANTIC),
        ("identity.role", MemoryType.SEMANTIC),
        ("values.principle", MemoryType.SEMANTIC),
    ),
)
def test_encoder_maps_accepted_changes_to_memory_types(
    target_field: str,
    expected_type: MemoryType,
) -> None:
    record = StateChangeMemoryEncoder().encode(accepted_state(target_field).changes[0])

    assert record is not None
    assert record.memory_type is expected_type
    assert record.sources[0].contribution_id == UUID(int=1)
    assert record.sources[0].old_state_version == 0
    assert record.sources[0].new_state_version == 1
    assert record.sources[0].target_field == target_field
    assert record.evidence_refs[0].evidence_id == UUID(int=101)


def test_encoder_skips_unaccepted_and_unknown_changes() -> None:
    state = accepted_state()
    rejected = replace(
        state.changes[0],
        status=StateDecisionStatus.REJECTED,
    )
    unknown = replace(
        state.changes[0],
        contribution=make_contribution(2, "unclassified.field", target=state.subject_scope),
    )
    encoder = StateChangeMemoryEncoder()

    assert encoder.encode(rejected) is None
    assert encoder.encode(unknown) is None


def test_encoding_service_is_idempotent_and_access_is_separate(
    tmp_path: Path,
) -> None:
    repository = FileMemoryRepository(
        tmp_path / "memories",
        tmp_path / "indexes",
        tmp_path / "access",
    )
    encoding = MemoryEncodingService(repository, StateChangeMemoryEncoder())
    state = accepted_state()

    first = encoding.encode_changes(state.changes)
    second = encoding.encode_changes(state.changes)

    assert len(first) == len(second) == 1
    assert repository.read_by_subject(state.subject_scope) == first
    assert len(tuple((tmp_path / "memories").rglob("*.json"))) == 1

    access = MemoryAccessService(repository).record_access(
        state.subject_scope,
        first[0].memory_id,
        accessed_at=NOW,
        purpose="workspace retrieval",
        context="answer:user question",
    )
    assert repository.read_access_history(
        state.subject_scope,
        first[0].memory_id,
    ) == (access,)
    assert first[0] == repository.load(state.subject_scope, first[0].memory_id)


def test_memory_schema_v1_reads_without_provenance() -> None:
    state = accepted_state()
    record = StateChangeMemoryEncoder().encode(state.changes[0])
    assert record is not None
    legacy = memory_to_dict(record)
    legacy["schema_version"] = 1
    legacy.pop("sources")

    migrated = memory_from_dict(legacy)

    assert migrated == replace(record, sources=())


def test_access_audit_rejects_cross_mind_memory(tmp_path: Path) -> None:
    repository = FileMemoryRepository(
        tmp_path / "memories",
        tmp_path / "indexes",
        tmp_path / "access",
    )
    state = accepted_state()
    record = MemoryEncodingService(repository, StateChangeMemoryEncoder()).encode_changes(
        state.changes
    )[0]
    other_subject = SubjectScope(
        mind=MindScope("mind-2"),
        subject=state.subject_scope.subject,
    )

    with pytest.raises(ContractValidationError, match="unknown memory"):
        MemoryAccessService(repository).record_access(
            other_subject,
            record.memory_id,
            accessed_at=NOW,
            purpose="workspace retrieval",
            context="cross mind",
        )


def test_process_event_automatically_encodes_accepted_changes(tmp_path: Path) -> None:
    container = build_container(
        tmp_path,
        dotenv_path=tmp_path / "missing.env",
    )
    event = Event.user_message("user-1", "我喜欢晚上学习", clock=_FixedClock(NOW))
    context = RunContext(
        run_id=UUID(int=50),
        correlation_id=UUID(int=51),
        deadline=NOW + timedelta(minutes=1),
        clock=_FixedClock(NOW),
    )

    result = container.process_event.process(event, context)

    assert result.status.value == "succeeded"
    memories = container.memory_repository.read_by_subject(event.subject)
    assert len(memories) == 1
    assert memories[0].sources[0].new_state_version == 1


class _FixedClock:
    def __init__(self, value: datetime) -> None:
        self._value = value

    def now(self) -> datetime:
        return self._value
