from dataclasses import replace
from datetime import datetime
from uuid import UUID

from self_cognition.core.contributions import (
    CognitiveContribution,
    ContributionOperation,
)
from self_cognition.core.evidence import EvidenceSourceKind
from self_cognition.core.errors import (
    ContractValidationError,
    ScopeMismatchError,
    SubjectMismatchError,
)
from self_cognition.core.metacognition import ConflictReview, ConflictStatus
from self_cognition.core.state import (
    ConflictRecord,
    StateAtom,
    StateChangeRecord,
    StateDecisionStatus,
    SubjectState,
)
from self_cognition.core.scopes import SubjectKind


CONFLICT_REASON = "different values for the same field in one batch"
PROTECTED_FIELD_PREFIXES = ("identity.", "values.")
PROTECTED_FIELD_MIN_CONFIDENCE = 0.9

ACCEPTED_REASON = "contribution applied"
STALE_VERSION_REASON = "target version does not match current state"
NOT_YET_VALID_REASON = "contribution is not yet valid"
EXPIRED_REASON = "contribution has expired"
LOW_CONFIDENCE_REASON = "protected field confidence is below minimum"
CONFIRMATION_REQUIRED_REASON = (
    "protected field update requires explicit confirmation"
)


class StateReducer:
    def apply(
        self,
        state: SubjectState,
        contribution: CognitiveContribution,
        *,
        decided_at: datetime,
    ) -> SubjectState:
        return self.apply_many(state, (contribution,), decided_at=decided_at)

    def apply_many(
        self,
        state: SubjectState,
        contributions: tuple[CognitiveContribution, ...],
        *,
        decided_at: datetime,
    ) -> SubjectState:
        unique: dict[UUID, CognitiveContribution] = {}
        for contribution in contributions:
            self._require_matching_scope(state, contribution)
            unique.setdefault(contribution.contribution_id, contribution)

        pending = sorted(
            (
                contribution
                for contribution_id, contribution in unique.items()
                if contribution_id not in state.decided_contribution_ids
            ),
            key=lambda contribution: (
                contribution.target_field,
                contribution.contribution_id.int,
            ),
        )
        if not pending:
            return state

        decisions: dict[UUID, tuple[StateDecisionStatus, str]] = {}
        eligible_by_field: dict[str, list[CognitiveContribution]] = {}
        reviews: list[CognitiveContribution] = []
        for contribution in pending:
            decision = self._preliminary_decision(
                state,
                contribution,
                decided_at,
            )
            if decision is None:
                if contribution.operation is ContributionOperation.REVIEW_CONFLICT:
                    reviews.append(contribution)
                    continue
                eligible_by_field.setdefault(
                    contribution.target_field,
                    [],
                ).append(contribution)
            else:
                decisions[contribution.contribution_id] = decision

        conflicts = set(state.conflicts)
        for target_field in sorted(eligible_by_field):
            candidates = eligible_by_field[target_field]
            first_value = candidates[0].value
            if any(
                candidate.value != first_value
                for candidate in candidates[1:]
            ):
                conflicts.add(
                    ConflictRecord(
                        target_field=target_field,
                        candidate_contribution_ids=tuple(
                            candidate.contribution_id for candidate in candidates
                        ),
                        reason=CONFLICT_REASON,
                        evidence_refs=tuple(
                            dict.fromkeys(
                                ref
                                for candidate in candidates
                                for ref in candidate.evidence_refs
                            )
                        ),
                    )
                )
                for candidate in candidates:
                    decisions[candidate.contribution_id] = (
                        StateDecisionStatus.PENDING,
                        CONFLICT_REASON,
                    )
            else:
                for candidate in candidates:
                    decisions[candidate.contribution_id] = (
                        StateDecisionStatus.ACCEPTED,
                        ACCEPTED_REASON,
                    )

        result = state
        final_conflicts = frozenset(conflicts)
        for contribution in pending:
            if contribution in reviews:
                continue
            status, reason = decisions[contribution.contribution_id]
            result = self._record_decision(
                result,
                contribution,
                status,
                reason,
                decided_at,
                final_conflicts,
            )
        for contribution in reviews:
            result, status, reason = self._review_conflict(
                result, contribution, decided_at
            )
            result = self._record_decision(
                result, contribution, status, reason, decided_at, result.conflicts
            )
        for contribution in pending:
            if (
                contribution.source_module != "metacognition.user_correction"
                or not contribution.explicitly_confirmed
                or contribution.contribution_id not in result.applied_contribution_ids
            ):
                continue
            conflicts = set(result.conflicts)
            for conflict in result.conflicts:
                if (
                    conflict.status is not ConflictStatus.OPEN
                    or conflict.target_field != contribution.target_field
                ):
                    continue
                selected_id = next(
                    (
                        change.contribution.contribution_id
                        for change in result.changes
                        if change.contribution.contribution_id
                        in conflict.candidate_contribution_ids
                        and change.contribution.value == contribution.value
                    ),
                    None,
                )
                conflicts.remove(conflict)
                conflicts.add(
                    replace(
                        conflict,
                        status=(
                            ConflictStatus.RESOLVED
                            if selected_id is not None
                            else ConflictStatus.INVALIDATED
                        ),
                        confirmed=True,
                        reviewed_by=contribution.contribution_id,
                        selected_contribution_id=selected_id,
                        resolution_reason="explicit correction superseded the open conflict",
                        evidence_refs=tuple(
                            dict.fromkeys(
                                conflict.evidence_refs + contribution.evidence_refs
                            )
                        ),
                    )
                )
            result = replace(result, conflicts=frozenset(conflicts))
        return result

    def _review_conflict(
        self,
        state: SubjectState,
        contribution: CognitiveContribution,
        decided_at: datetime,
    ) -> tuple[SubjectState, StateDecisionStatus, str]:
        try:
            review = ConflictReview.from_state_value(contribution.value)
        except ContractValidationError as error:
            return state, StateDecisionStatus.REJECTED, str(error)
        candidates = {
            change.contribution.contribution_id: change.contribution
            for change in state.changes
            if change.contribution.target_field == contribution.target_field
            and change.contribution.operation is ContributionOperation.SET
            and change.status is not StateDecisionStatus.REJECTED
        }
        if any(item not in candidates for item in review.candidate_contribution_ids):
            return (
                state,
                StateDecisionStatus.REJECTED,
                "conflict candidate is unknown or rejected",
            )
        proposed = ConflictRecord(
            contribution.target_field,
            tuple(sorted(review.candidate_contribution_ids)),
            review.reason,
        )
        previous = next(
            (
                item
                for item in state.conflicts
                if item.conflict_id == proposed.conflict_id
            ),
            None,
        )
        if previous is None and review.status is not ConflictStatus.OPEN:
            return (
                state,
                StateDecisionStatus.REJECTED,
                "conflict must be opened before review",
            )
        if previous is not None and previous.status is not ConflictStatus.OPEN:
            return (
                state,
                StateDecisionStatus.REJECTED,
                "closed conflict cannot be silently reopened",
            )
        requires_confirmation = review.requires_confirmation or (
            previous is not None and previous.requires_confirmation
        )
        if (
            review.status is not ConflictStatus.OPEN
            and requires_confirmation
            and not contribution.explicitly_confirmed
        ):
            return (
                state,
                StateDecisionStatus.PENDING,
                "conflict review requires explicit confirmation",
            )
        evidence = tuple(
            dict.fromkeys(
                (previous.evidence_refs if previous is not None else ())
                + tuple(
                    ref
                    for item in review.candidate_contribution_ids
                    for ref in candidates[item].evidence_refs
                )
                + contribution.evidence_refs
            )
        )
        record = replace(
            previous or proposed,
            status=review.status,
            evidence_refs=evidence,
            requires_confirmation=requires_confirmation,
            confirmed=contribution.explicitly_confirmed,
            resolution_reason=(
                review.reason if review.status is not ConflictStatus.OPEN else None
            ),
            reviewed_by=contribution.contribution_id,
            selected_contribution_id=review.selected_contribution_id,
        )
        entries = dict(state.entries)
        if review.status is ConflictStatus.RESOLVED:
            selected = candidates[review.selected_contribution_id]
            selection = replace(
                selected,
                target_version=state.version,
                explicitly_confirmed=contribution.explicitly_confirmed,
            )
            rejection = self._preliminary_decision(state, selection, decided_at)
            if rejection is not None:
                return state, rejection[0], rejection[1]
            entries[contribution.target_field] = StateAtom(
                value=selected.value,
                cognition_type=selected.cognition_type,
                confidence=selected.confidence,
                scope=selected.scope,
                evidence_refs=evidence,
                contribution_ids=(
                    selected.contribution_id,
                    contribution.contribution_id,
                ),
                created_at=contribution.created_at,
                valid_from=selected.valid_from,
                expires_at=selected.expires_at,
            )
        conflicts = frozenset(
            item for item in state.conflicts if item.conflict_id != record.conflict_id
        ) | {record}
        return (
            replace(state, entries=entries, conflicts=conflicts),
            StateDecisionStatus.ACCEPTED,
            "conflict review applied",
        )

    @staticmethod
    def _preliminary_decision(
        state: SubjectState,
        contribution: CognitiveContribution,
        decided_at: datetime,
    ) -> tuple[StateDecisionStatus, str] | None:
        if contribution.target_version != state.version:
            return StateDecisionStatus.REJECTED, STALE_VERSION_REASON
        if contribution.valid_from > decided_at:
            return StateDecisionStatus.PENDING, NOT_YET_VALID_REASON
        if (
            contribution.expires_at is not None
            and contribution.expires_at <= decided_at
        ):
            return StateDecisionStatus.REJECTED, EXPIRED_REASON
        if contribution.operation is ContributionOperation.REVIEW_CONFLICT:
            return None
        if not contribution.target_field.startswith(PROTECTED_FIELD_PREFIXES):
            return None
        if contribution.confidence < PROTECTED_FIELD_MIN_CONFIDENCE:
            return StateDecisionStatus.REJECTED, LOW_CONFIDENCE_REASON

        existing = state.entries.get(contribution.target_field)
        authorized = contribution.explicitly_confirmed or any(
            evidence.source_kind is EvidenceSourceKind.SYSTEM_PRIOR
            for evidence in contribution.evidence_refs
        )
        if (
            state.subject_kind is SubjectKind.MIND
            and existing is None
            and not authorized
        ):
            return StateDecisionStatus.PENDING, CONFIRMATION_REQUIRED_REASON
        if (
            existing is not None
            and contribution.value != existing.value
            and not authorized
        ):
            return StateDecisionStatus.PENDING, CONFIRMATION_REQUIRED_REASON
        return None

    @staticmethod
    def _record_decision(
        state: SubjectState,
        contribution: CognitiveContribution,
        status: StateDecisionStatus,
        reason: str,
        decided_at: datetime,
        conflicts: frozenset[ConflictRecord],
    ) -> SubjectState:
        entries = dict(state.entries)
        applied_ids = state.applied_contribution_ids
        if status is StateDecisionStatus.ACCEPTED:
            applied_ids = applied_ids | {contribution.contribution_id}
        if (
            status is StateDecisionStatus.ACCEPTED
            and contribution.operation is ContributionOperation.SET
        ):
            existing = entries.get(contribution.target_field)
            existing_evidence = (
                existing.evidence_refs if existing is not None else ()
            )
            existing_ids = existing.contribution_ids if existing is not None else ()
            entries[contribution.target_field] = StateAtom(
                value=contribution.value,
                cognition_type=contribution.cognition_type,
                confidence=contribution.confidence,
                scope=contribution.scope,
                evidence_refs=tuple(
                    dict.fromkeys(existing_evidence + contribution.evidence_refs)
                ),
                contribution_ids=tuple(
                    dict.fromkeys(existing_ids + (contribution.contribution_id,))
                ),
                created_at=contribution.created_at,
                valid_from=contribution.valid_from,
                expires_at=contribution.expires_at,
            )
            applied_ids = applied_ids | {contribution.contribution_id}

        new_version = state.version + 1
        change = StateChangeRecord(
            contribution=contribution,
            status=status,
            reason=reason,
            old_version=state.version,
            new_version=new_version,
            decided_at=decided_at,
        )
        return SubjectState(
            subject_id=state.subject_id,
            version=new_version,
            entries=entries,
            applied_contribution_ids=applied_ids,
            conflicts=conflicts,
            changes=state.changes + (change,),
            mind_id=state.mind_id,
            subject_kind=state.subject_kind,
        )

    @staticmethod
    def _require_matching_scope(
        state: SubjectState,
        contribution: CognitiveContribution,
    ) -> None:
        if contribution.target.mind != state.subject_scope.mind:
            raise ScopeMismatchError(
                "contribution target belongs to a different mind"
            )
        if contribution.target != state.subject_scope:
            raise SubjectMismatchError(
                "contribution target subject does not match state subject"
            )
