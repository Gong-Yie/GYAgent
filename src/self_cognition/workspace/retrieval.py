from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from self_cognition.core.indexes import WorkspaceIndex, text_terms
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.memories import MemoryLifecycleStatus, MemoryRecord, MemoryType
from self_cognition.core.protocols import MemoryRepository
from self_cognition.core.state import StateAtom, SubjectState
from self_cognition.core.workspace import (
    _closed_legacy_conflict,
    conflict_candidates,
    state_content,
    RetrievalCandidate,
    RetrievalQuery,
    RetrievalResult,
    RetrievalSource,
    WorkspaceRunInfo,
    estimate_tokens,
    evidence_quality,
)
from self_cognition.memory.behavior import MemoryRecallView, MemoryRetrievalService


class HybridWorkspaceRetriever:
    """Select state, memory, conflict and run candidates without owning facts."""

    def __init__(self, memory_repository: MemoryRepository) -> None:
        self._memory_repository = memory_repository
        self._memory_retrieval = MemoryRetrievalService(memory_repository)

    def retrieve(
        self,
        query: RetrievalQuery,
        state: SubjectState,
        *,
        as_of: datetime,
        index: WorkspaceIndex | None = None,
        run_info: WorkspaceRunInfo | None = None,
    ) -> RetrievalResult:
        records = tuple(
            record
            for record in self._memory_repository.read_by_subject(query.subject)
            if record.lifecycle_status is MemoryLifecycleStatus.ACTIVE
            and (record.expires_at is None or record.expires_at > as_of)
        )
        if index is None:
            active_index = None
            index_status = "authoritative_scan"
        elif index.is_compatible(state, records):
            active_index = index
            index_status = "used"
        else:
            active_index = None
            index_status = "fallback_incompatible"

        indexed_refs = self._indexed_refs(query, active_index)
        candidates = [
            *self._state_candidates(query, state, as_of, indexed_refs),
            *self._memory_candidates(
                query,
                records,
                as_of,
                indexed_refs,
                frozenset(state.entries),
            ),
            *conflict_candidates(query, state),
        ]
        if run_info is not None:
            candidates.append(self._run_candidate(run_info))
        unique = _deduplicate_candidates(tuple(candidates))
        return RetrievalResult(
            candidates=_with_diversity(unique),
            index_status=index_status,
        )

    @staticmethod
    def _indexed_refs(
        query: RetrievalQuery,
        index: WorkspaceIndex | None,
    ) -> frozenset[str]:
        if index is None:
            return frozenset()
        return frozenset(
            (
                *index.references_for_text(query.task),
                *index.references_for_time(query.time_from, query.time_to),
            )
        )

    @staticmethod
    def _state_candidates(
        query: RetrievalQuery,
        state: SubjectState,
        as_of: datetime,
        indexed_refs: frozenset[str],
    ) -> tuple[RetrievalCandidate, ...]:
        result: list[RetrievalCandidate] = []
        for field_name, atom in sorted(state.entries.items()):
            memory_type = _field_memory_type(field_name)
            if query.memory_types and memory_type not in query.memory_types:
                continue
            reference = f"state:{field_name}"
            relevance = _relevance(
                query,
                field_name,
                atom.value,
                reference in indexed_refs,
            )
            if relevance == 0.0 or not _in_time_range(query, atom.created_at):
                continue
            if _closed_legacy_conflict(field_name, state):
                continue
            content = state_content(field_name, atom, query, as_of)
            if content is None:
                continue
            risk = 1.0 if field_name.startswith("metacognition.") else 0.0
            result.append(
                _candidate_from_state(
                    field_name,
                    atom,
                    content,
                    state.version,
                    relevance,
                    risk,
                )
            )
        return tuple(result)

    def _memory_candidates(
        self,
        query: RetrievalQuery,
        records: tuple[MemoryRecord, ...],
        as_of: datetime,
        indexed_refs: frozenset[str],
        current_state_fields: frozenset[str],
    ) -> tuple[RetrievalCandidate, ...]:
        views = self._memory_retrieval.retrieve(
            query.subject,
            query.memory_cues,
            now=as_of,
        )
        records_by_id = {record.memory_id: record for record in records}
        result: list[RetrievalCandidate] = []
        for view in views:
            if (
                query.memory_types
                and view.record.memory_type not in query.memory_types
            ):
                continue
            record = records_by_id.get(view.record.memory_id)
            if record is None or not _in_time_range(query, record.created_at):
                continue
            target_field = (
                record.sources[-1].target_field
                if record.sources
                else f"memory.{record.memory_type.value}"
            )
            if target_field in current_state_fields:
                continue
            reference = f"memory:{record.memory_id}"
            task_relevance = _relevance(
                query,
                target_field,
                record.content,
                reference in indexed_refs,
            )
            if task_relevance == 0.0 and not view.matched_cues:
                continue
            relevance = max(task_relevance, view.score)
            result.append(_candidate_from_memory(view, target_field, relevance))
        return tuple(result)

    @staticmethod
    def _run_candidate(run_info: WorkspaceRunInfo) -> RetrievalCandidate:
        content = {
            "run_id": str(run_info.run_id),
            "correlation_id": str(run_info.correlation_id),
            "deadline": run_info.deadline.isoformat(),
            "cancelled": run_info.cancelled,
        }
        return RetrievalCandidate(
            candidate_id=f"run:{run_info.run_id}",
            source=RetrievalSource.RUN,
            source_ref=f"run:{run_info.run_id}",
            target_field="run.current",
            content=content,
            evidence_refs=(),
            confidence=1.0,
            state_version=None,
            relevance=1.0,
            evidence_quality=1.0,
            risk=1.0 if run_info.cancelled else 0.5,
            diversity=1.0,
            task_relevance=1.0,
            estimated_tokens=estimate_tokens(content),
            reason="current run information is required for this call",
        )


def _candidate_from_state(
    field_name: str,
    atom: StateAtom,
    content: object,
    state_version: int,
    relevance: float,
    risk: float,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        candidate_id=f"state:{field_name}",
        source=RetrievalSource.STATE,
        source_ref=f"state:{field_name}",
        target_field=field_name,
        content=content,
        evidence_refs=atom.evidence_refs,
        confidence=atom.confidence,
        state_version=state_version,
        relevance=relevance,
        evidence_quality=evidence_quality(atom.evidence_refs, atom.confidence),
        risk=risk,
        diversity=1.0,
        task_relevance=relevance,
        estimated_tokens=estimate_tokens(content),
        reason="authoritative state matched task fields, time, or full text",
    )


def _candidate_from_memory(
    view: MemoryRecallView,
    target_field: str,
    relevance: float,
) -> RetrievalCandidate:
    record = view.record
    return RetrievalCandidate(
        candidate_id=f"memory:{record.memory_id}",
        source=RetrievalSource.MEMORY,
        source_ref=f"memory:{record.memory_id}:v{record.version}",
        target_field=target_field,
        content=record.content,
        evidence_refs=record.evidence_refs,
        confidence=record.confidence,
        state_version=(
            record.sources[-1].new_state_version if record.sources else None
        ),
        relevance=relevance,
        evidence_quality=evidence_quality(record.evidence_refs, record.confidence),
        risk=1.0 - view.score,
        diversity=1.0,
        task_relevance=relevance,
        estimated_tokens=estimate_tokens(record.content),
        reason=f"active memory matched task; {view.explanation}",
    )


def _field_memory_type(field_name: str) -> MemoryType | None:
    if field_name.startswith("episodic."):
        return MemoryType.EPISODIC
    if field_name.startswith("procedural."):
        return MemoryType.PROCEDURAL
    if field_name.startswith(
        ("profile.", "preferences.", "identity.", "values.", "semantic.")
    ):
        return MemoryType.SEMANTIC
    if field_name.startswith("relationships."):
        return MemoryType.RELATIONSHIP
    if field_name.startswith("narrative."):
        return MemoryType.NARRATIVE
    return None


def _relevance(
    query: RetrievalQuery,
    target_field: str,
    content: object,
    indexed_match: bool,
) -> float:
    if any(_matches_pattern(target_field, pattern) for pattern in query.field_patterns):
        return 1.0
    query_terms = text_terms(query.task)
    if not query_terms:
        return 0.0
    overlap = query_terms & text_terms(target_field, content)
    if overlap:
        return min(1.0, len(overlap) / len(query_terms) + 0.5)
    if query.time_from is not None or query.time_to is not None:
        return 0.5
    return 0.5 if indexed_match else 0.0


def _matches_pattern(field_name: str, pattern: str) -> bool:
    return (
        field_name.startswith(pattern[:-1])
        if pattern.endswith("*")
        else field_name == pattern
    )


def _in_time_range(query: RetrievalQuery, occurred_at: datetime) -> bool:
    return (
        (query.time_from is None or occurred_at >= query.time_from)
        and (query.time_to is None or occurred_at <= query.time_to)
    )


def _with_diversity(
    candidates: tuple[RetrievalCandidate, ...],
) -> tuple[RetrievalCandidate, ...]:
    counts = {
        source: sum(candidate.source is source for candidate in candidates)
        for source in RetrievalSource
    }
    return tuple(
        replace(candidate, diversity=1.0 / counts[candidate.source])
        for candidate in candidates
    )


def _deduplicate_candidates(
    candidates: tuple[RetrievalCandidate, ...],
) -> tuple[RetrievalCandidate, ...]:
    result: list[RetrievalCandidate] = []
    seen_ids: set[str] = set()
    seen_facts: set[tuple[str, str, tuple]] = set()
    for candidate in candidates:
        fact = (
            candidate.target_field,
            repr(candidate.content),
            tuple(reference.evidence_id for reference in candidate.evidence_refs),
        )
        if candidate.candidate_id in seen_ids or fact in seen_facts:
            continue
        seen_ids.add(candidate.candidate_id)
        seen_facts.add(fact)
        result.append(candidate)
    return tuple(result)
