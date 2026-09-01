from datetime import datetime
from uuid import UUID

from self_cognition.core.contributions import CognitiveContribution
from self_cognition.core.errors import ScopeMismatchError, SubjectMismatchError
from self_cognition.core.state import (
    ConflictRecord,
    StateAtom,
    StateChangeRecord,
    StateDecisionStatus,
    SubjectState,
)


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
        for contribution in pending:
            decision = self._preliminary_decision(
                state,
                contribution,
                decided_at,
            )
            if decision is None:
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
            status, reason = decisions[contribution.contribution_id]
            result = self._record_decision(
                result,
                contribution,
                status,
                reason,
                decided_at,
                final_conflicts,
            )
        return result

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
        if not contribution.target_field.startswith(PROTECTED_FIELD_PREFIXES):
            return None
        if contribution.confidence < PROTECTED_FIELD_MIN_CONFIDENCE:
            return StateDecisionStatus.REJECTED, LOW_CONFIDENCE_REASON

        existing = state.entries.get(contribution.target_field)
        if (
            existing is not None
            and contribution.value != existing.value
            and not contribution.explicitly_confirmed
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
