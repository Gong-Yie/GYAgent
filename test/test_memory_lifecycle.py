from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

from self_cognition.bootstrap import ApplicationContainer, build_container
from self_cognition.core.contributions import CognitionType
from self_cognition.core.deletions import (
    DeletionSelector,
    DeletionStatus,
)
from self_cognition.core.events import EventEnvelope
from self_cognition.core.memories import MemoryLifecycleStatus, MemoryType
from self_cognition.core.scopes import MindScope, SubjectKind, SubjectRef, SubjectScope
from self_cognition.runtime.run_context import RunContext


NOW = datetime(2026, 9, 2, 8, tzinfo=timezone.utc)


class FixedClock:
    def __init__(self, value: datetime) -> None:
        self._value = value

    def now(self) -> datetime:
        return self._value


def make_subject(mind_id: str = "mind-1") -> SubjectScope:
    return SubjectScope(
        MindScope(mind_id),
        SubjectRef(SubjectKind.USER, "user-1"),
    )


def process(
    container: ApplicationContainer,
    event: EventEnvelope,
    run_id: int,
) -> None:
    result = container.process_event.process(
        event,
        RunContext(
            run_id=UUID(int=run_id),
            correlation_id=UUID(int=100),
            deadline=NOW + timedelta(minutes=5),
            clock=FixedClock(event.recorded_at),
        ),
    )
    assert result.error_type is None


def test_correction_supersedes_old_memory_through_the_contribution_chain(
    tmp_path: Path,
) -> None:
    container = build_container(tmp_path, dotenv_path=tmp_path / "missing.env")
    subject = make_subject()
    original = EventEnvelope.user_message(
        subject,
        "我喜欢晚上学习",
        clock=FixedClock(NOW),
    )
    process(container, original, 1)
    old_memory = next(
        memory
        for memory in container.memory_repository.read_by_subject(subject)
        if memory.content == "晚上"
    )
    correction = EventEnvelope.correction(
        subject,
        target_field="preferences.study_time",
        cognition_type=CognitionType.PREFERENCE.value,
        value="早上",
        corrected_memory_id=old_memory.memory_id,
        clock=FixedClock(NOW + timedelta(minutes=1)),
    )

    process(container, correction, 2)

    state = container.state_repository.load(subject)
    assert state is not None
    assert state.get("preferences.study_time").value == "早上"
    old_latest = container.memory_repository.load(subject, old_memory.memory_id)
    assert old_latest is not None
    assert old_latest.content == "晚上"
    assert old_latest.lifecycle_status is MemoryLifecycleStatus.SUPERSEDED
    assert old_latest.lifecycle_reason == f"corrected_by:{correction.event_id}"
    active = tuple(
        memory
        for memory in container.memory_repository.read_by_subject(subject)
        if memory.lifecycle_status is MemoryLifecycleStatus.ACTIVE
    )
    assert any(memory.content == "早上" for memory in active)


def test_archive_and_explicit_expiration_create_explainable_versions(
    tmp_path: Path,
) -> None:
    container = build_container(tmp_path, dotenv_path=tmp_path / "missing.env")
    subject = make_subject()
    first = EventEnvelope.user_message(
        subject,
        "我喜欢晚上学习",
        clock=FixedClock(NOW),
    )
    process(container, first, 1)
    memory = container.memory_repository.read_by_subject(subject)[0]

    archived = container.memory_lifecycle.archive(
        subject,
        memory.memory_id,
        changed_at=NOW + timedelta(minutes=1),
    )

    assert archived.version == memory.version + 1
    assert archived.lifecycle_status is MemoryLifecycleStatus.ARCHIVED
    assert archived.lifecycle_reason == "user_archive"

    expiring = replace(
        memory,
        memory_id=UUID(int=999),
        expires_at=NOW + timedelta(days=1),
    )
    container.memory_repository.save(expiring, expected_version=0)
    expired = container.memory_lifecycle.expire_due(
        subject,
        now=NOW + timedelta(days=2),
    )

    assert len(expired) == 1
    assert expired[0].lifecycle_status is MemoryLifecycleStatus.EXPIRED
    assert expired[0].lifecycle_reason == "retention_expired"


def test_dry_run_then_delete_removes_authority_and_rebuilds_state(
    tmp_path: Path,
) -> None:
    container = build_container(tmp_path, dotenv_path=tmp_path / "missing.env")
    subject = make_subject()
    preference_event = EventEnvelope.user_message(
        subject,
        "我喜欢晚上学习",
        clock=FixedClock(NOW),
    )
    name_event = EventEnvelope.user_message(
        subject,
        "我叫小明",
        clock=FixedClock(NOW + timedelta(minutes=1)),
    )
    process(container, preference_event, 1)
    process(container, name_event, 2)
    preference_memory = next(
        memory
        for memory in container.memory_repository.read_by_subject(subject)
        if memory.content == "晚上"
    )
    plan = container.forget.dry_run(
        DeletionSelector(subject, memory_id=preference_memory.memory_id),
        now=NOW + timedelta(minutes=2),
    )

    assert preference_memory.memory_id in plan.memory_ids
    assert preference_event.event_id in plan.event_ids
    assert name_event.event_id not in plan.event_ids
    result = container.forget.execute(
        plan,
        now=NOW + timedelta(minutes=3),
    )

    assert result.status is DeletionStatus.COMPLETED
    assert result.reason == "user_request"
    assert result.cache_result == "not_applicable"
    assert result.export_result == "not_applicable"
    assert container.memory_repository.load(
        subject,
        preference_memory.memory_id,
    ) is None
    state = container.state_repository.load(subject)
    assert state is not None
    assert state.get("profile.name").value == "小明"
    assert "preferences.study_time" not in state.entries
    event_text = (tmp_path / "events" / "events.jsonl").read_text(
        encoding="utf-8"
    )
    assert "喜欢晚上学习" not in event_text
    container.event_store.append(preference_event)
    assert all(
        event.event_id != preference_event.event_id
        for event in container.event_store.read_by_subject(subject)
    )


def test_subject_deletion_isolated_by_mind_and_executing_plan_recovers(
    tmp_path: Path,
) -> None:
    first = build_container(tmp_path, dotenv_path=tmp_path / "missing.env")
    deleted_subject = make_subject("mind-1")
    retained_subject = make_subject("mind-2")
    deleted_event = EventEnvelope.user_message(
        deleted_subject,
        "我喜欢晚上学习",
        clock=FixedClock(NOW),
    )
    retained_event = EventEnvelope.user_message(
        retained_subject,
        "我喜欢早上学习",
        clock=FixedClock(NOW),
    )
    process(first, deleted_event, 1)
    process(first, retained_event, 2)
    plan = first.forget.dry_run(
        DeletionSelector(deleted_subject, delete_subject=True),
        now=NOW + timedelta(minutes=1),
    )
    first.deletion_repository.save(
        plan.with_status(
            DeletionStatus.EXECUTING,
            NOW + timedelta(minutes=2),
        )
    )

    recovered = build_container(tmp_path, dotenv_path=tmp_path / "missing.env")

    stored = recovered.deletion_repository.get(plan.plan_id)
    assert stored is not None
    assert stored.status is DeletionStatus.COMPLETED
    assert recovered.state_repository.load(deleted_subject) is None
    retained_state = recovered.state_repository.load(retained_subject)
    assert retained_state is not None
    assert retained_state.get("preferences.study_time").value == "早上"
