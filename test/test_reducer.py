from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from self_cognition.blackboard.reducer import (
    CONFIRMATION_REQUIRED_REASON,
    CONFLICT_REASON,
    EXPIRED_REASON,
    LOW_CONFIDENCE_REASON,
    NOT_YET_VALID_REASON,
    STALE_VERSION_REASON,
    StateReducer,
)
from self_cognition.core.contributions import (
    CognitiveContribution,
    CognitionType,
    ContributionOperation,
)
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.errors import ScopeMismatchError, SubjectMismatchError
from self_cognition.core.scopes import (
    DataScope,
    DisclosureScope,
    MindScope,
    SubjectKind,
    SubjectRef,
    SubjectScope,
)
from self_cognition.core.state import (
    ConflictRecord,
    StateDecisionStatus,
    SubjectState,
)


NOW = datetime(2026, 9, 1, 12, tzinfo=timezone.utc)
SUBJECT = SubjectScope.legacy_user("user-1")


def make_contribution(
    contribution_id: int,
    *,
    target: SubjectScope = SUBJECT,
    target_field: str = "preferences.study_time",
    value: object = "晚上",
    confidence: float = 1.0,
    target_version: int = 0,
    valid_from: datetime = NOW,
    expires_at: datetime | None = None,
    explicitly_confirmed: bool = False,
) -> CognitiveContribution:
    scope = DataScope(target, DisclosureScope.PRIVATE)
    return CognitiveContribution(
        contribution_id=UUID(int=contribution_id),
        target=target,
        target_field=target_field,
        operation=ContributionOperation.SET,
        cognition_type=CognitionType.PREFERENCE,
        value=value,
        confidence=confidence,
        evidence_refs=(EvidenceRef.for_event_id(UUID(int=100 + contribution_id), target),),
        source_module="test.reducer",
        module_version="1",
        scope=scope,
        created_at=NOW,
        valid_from=valid_from,
        expires_at=expires_at,
        target_version=target_version,
        explicitly_confirmed=explicitly_confirmed,
    )


def test_accepts_once_and_builds_typed_atom_with_audit_record() -> None:
    contribution = make_contribution(1)
    reducer = StateReducer()

    first = reducer.apply(
        SubjectState.empty("user-1"),
        contribution,
        decided_at=NOW,
    )
    replayed = reducer.apply(first, contribution, decided_at=NOW)

    assert replayed is first
    assert first.version == 1
    assert first.applied_contribution_ids == frozenset({UUID(int=1)})
    assert first.get("preferences.study_time").cognition_type is CognitionType.PREFERENCE
    assert first.changes[0].status is StateDecisionStatus.ACCEPTED
    assert first.changes[0].contribution == contribution


def test_sequential_updates_merge_evidence_and_contribution_ids() -> None:
    reducer = StateReducer()
    first = make_contribution(1)
    second = make_contribution(2, value="早上", target_version=1)

    state = reducer.apply(
        SubjectState.empty("user-1"),
        first,
        decided_at=NOW,
    )
    state = reducer.apply(state, second, decided_at=NOW)

    atom = state.get("preferences.study_time")
    assert atom.value == "早上"
    assert tuple(ref.evidence_id for ref in atom.evidence_refs) == (
        UUID(int=101),
        UUID(int=102),
    )
    assert atom.contribution_ids == (UUID(int=1), UUID(int=2))


def test_batch_conflict_is_pending_and_safe_to_replay() -> None:
    evening = make_contribution(1)
    morning = make_contribution(2, value="早上")
    reducer = StateReducer()

    state = reducer.apply_many(
        SubjectState.empty("user-1"),
        (morning, evening),
        decided_at=NOW,
    )
    replayed = reducer.apply_many(
        state,
        (evening, morning),
        decided_at=NOW,
    )

    assert replayed is state
    assert state.version == 2
    assert "preferences.study_time" not in state.entries
    assert tuple(change.status for change in state.changes) == (
        StateDecisionStatus.PENDING,
        StateDecisionStatus.PENDING,
    )
    assert state.conflicts == frozenset(
        {
            ConflictRecord(
                target_field="preferences.study_time",
                candidate_contribution_ids=(UUID(int=1), UUID(int=2)),
                reason=CONFLICT_REASON,
                evidence_refs=evening.evidence_refs + morning.evidence_refs,
            )
        }
    )


@pytest.mark.parametrize(
    ("contribution", "status", "reason"),
    [
        (
            make_contribution(
                1,
                target_field="identity.role",
                confidence=0.5,
            ),
            StateDecisionStatus.REJECTED,
            LOW_CONFIDENCE_REASON,
        ),
        (
            make_contribution(2, target_version=9),
            StateDecisionStatus.REJECTED,
            STALE_VERSION_REASON,
        ),
        (
            make_contribution(3, valid_from=NOW + timedelta(seconds=1)),
            StateDecisionStatus.PENDING,
            NOT_YET_VALID_REASON,
        ),
        (
            make_contribution(
                4,
                valid_from=NOW - timedelta(seconds=2),
                expires_at=NOW - timedelta(seconds=1),
            ),
            StateDecisionStatus.REJECTED,
            EXPIRED_REASON,
        ),
    ],
)
def test_non_applied_decisions_increment_version_and_are_audited(
    contribution: CognitiveContribution,
    status: StateDecisionStatus,
    reason: str,
) -> None:
    state = StateReducer().apply(
        SubjectState.empty("user-1"),
        contribution,
        decided_at=NOW,
    )

    assert state.version == 1
    assert state.entries == {}
    assert state.changes[0].status is status
    assert state.changes[0].reason == reason


def test_protected_update_waits_for_confirmation_then_applies() -> None:
    reducer = StateReducer()
    initial = make_contribution(1, target_field="identity.role", value="助手")
    unconfirmed = make_contribution(
        2,
        target_field="identity.role",
        value="研究助手",
        target_version=1,
    )
    confirmed = make_contribution(
        3,
        target_field="identity.role",
        value="研究助手",
        target_version=2,
        explicitly_confirmed=True,
    )

    state = reducer.apply(
        SubjectState.empty("user-1"),
        initial,
        decided_at=NOW,
    )
    state = reducer.apply(state, unconfirmed, decided_at=NOW)
    assert state.version == 2
    assert state.get("identity.role").value == "助手"
    assert state.changes[-1].status is StateDecisionStatus.PENDING
    assert state.changes[-1].reason == CONFIRMATION_REQUIRED_REASON

    state = reducer.apply(state, confirmed, decided_at=NOW)
    assert state.version == 3
    assert state.get("identity.role").value == "研究助手"


def test_batch_order_does_not_change_result() -> None:
    contributions = (
        make_contribution(11),
        make_contribution(12, value="早上"),
        make_contribution(13, target_field="profile.name", value="小明"),
    )
    reducer = StateReducer()

    forward = reducer.apply_many(
        SubjectState.empty("user-1"),
        contributions,
        decided_at=NOW,
    )
    reverse = reducer.apply_many(
        SubjectState.empty("user-1"),
        tuple(reversed(contributions)),
        decided_at=NOW,
    )

    assert forward == reverse
    assert forward.version == 3
    assert forward.get("profile.name").value == "小明"


@pytest.mark.parametrize(
    ("target", "error_type"),
    [
        (
            SubjectScope.legacy_user("user-2"),
            SubjectMismatchError,
        ),
        (
            SubjectScope(
                MindScope("mind-2"),
                SubjectRef(SubjectKind.USER, "user-1"),
            ),
            ScopeMismatchError,
        ),
    ],
)
def test_rejects_foreign_scope_without_changing_state(
    target: SubjectScope,
    error_type: type[Exception],
) -> None:
    old_state = SubjectState.empty("user-1")

    with pytest.raises(error_type):
        StateReducer().apply(
            old_state,
            make_contribution(1, target=target),
            decided_at=NOW,
        )

    assert old_state == SubjectState.empty("user-1")
