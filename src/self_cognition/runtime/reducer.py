from self_cognition.core.contributions import Contribution
from self_cognition.core.errors import SubjectMismatchError
from self_cognition.core.state import ConflictRecord, StateEntry, SubjectState


CONFLICT_REASON = "different values for the same field in one batch"
PROTECTED_FIELD_PREFIXES = ("identity.", "values.")
PROTECTED_FIELD_MIN_CONFIDENCE = 0.9


class StateReducer:
    def apply(
        self,
        state: SubjectState,
        contribution: Contribution,
    ) -> SubjectState:
        if contribution.target_subject_id != state.subject_id:
            raise SubjectMismatchError(
                "contribution target subject does not match state subject"
            )
        if contribution.contribution_id in state.applied_contribution_ids:
            return state

        entries = dict(state.entries)
        existing_entry = entries.get(contribution.target_field)
        if self._reject_protected_update(existing_entry, contribution):
            return state
        existing_evidence_ids = (
            existing_entry.evidence_event_ids if existing_entry is not None else ()
        )
        existing_contribution_ids = (
            existing_entry.contribution_ids if existing_entry is not None else ()
        )
        entries[contribution.target_field] = StateEntry(
            value=contribution.value,
            confidence=contribution.confidence,
            evidence_event_ids=tuple(
                dict.fromkeys(
                    existing_evidence_ids + contribution.evidence_event_ids
                )
            ),
            contribution_ids=tuple(
                dict.fromkeys(
                    existing_contribution_ids + (contribution.contribution_id,)
                )
            ),
        )

        return SubjectState(
            subject_id=state.subject_id,
            version=state.version + 1,
            entries=entries,
            applied_contribution_ids=(
                state.applied_contribution_ids | {contribution.contribution_id}
            ),
            conflicts=state.conflicts,
        )

    @staticmethod
    def _reject_protected_update(
        existing_entry: StateEntry | None,
        contribution: Contribution,
    ) -> bool:
        if not contribution.target_field.startswith(PROTECTED_FIELD_PREFIXES):
            return False
        if contribution.confidence < PROTECTED_FIELD_MIN_CONFIDENCE:
            return True
        return (
            existing_entry is not None
            and contribution.value != existing_entry.value
            and not contribution.explicitly_confirmed
        )

    def apply_many(
        self,
        state: SubjectState,
        contributions: tuple[Contribution, ...],
    ) -> SubjectState:
        unique_contributions: dict[object, Contribution] = {}
        for contribution in contributions:
            if contribution.target_subject_id != state.subject_id:
                raise SubjectMismatchError(
                    "contribution target subject does not match state subject"
                )
            unique_contributions.setdefault(
                contribution.contribution_id,
                contribution,
            )

        pending = sorted(
            (
                contribution
                for contribution_id, contribution in unique_contributions.items()
                if contribution_id not in state.applied_contribution_ids
                and not self._reject_protected_update(
                    state.entries.get(contribution.target_field),
                    contribution,
                )
            ),
            key=lambda contribution: (
                contribution.target_field,
                contribution.contribution_id.int,
            ),
        )
        if not pending:
            return state

        by_field: dict[str, list[Contribution]] = {}
        for contribution in pending:
            by_field.setdefault(contribution.target_field, []).append(contribution)

        accepted: list[Contribution] = []
        conflicts = set(state.conflicts)
        for target_field in sorted(by_field):
            candidates = by_field[target_field]
            first_value = candidates[0].value
            if any(candidate.value != first_value for candidate in candidates[1:]):
                conflicts.add(
                    ConflictRecord(
                        target_field=target_field,
                        candidate_contribution_ids=tuple(
                            candidate.contribution_id for candidate in candidates
                        ),
                        reason=CONFLICT_REASON,
                    )
                )
                continue
            accepted.extend(candidates)

        result = state
        for contribution in accepted:
            result = self.apply(result, contribution)

        new_conflicts = frozenset(conflicts)
        if new_conflicts == result.conflicts:
            return result

        return SubjectState(
            subject_id=result.subject_id,
            version=result.version + 1,
            entries=result.entries,
            applied_contribution_ids=result.applied_contribution_ids,
            conflicts=new_conflicts,
        )
