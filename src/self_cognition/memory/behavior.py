from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from self_cognition.core.evidence import EvidenceRef, EvidenceSourceKind
from self_cognition.core.ids import consolidated_memory_id
from self_cognition.core.memories import (
    MemoryConsolidationStatus,
    MemoryCues,
    MemoryLifecycleStatus,
    MemoryRecord,
    MemorySourceRef,
    MemoryType,
)
from self_cognition.core.protocols import MemoryRepository
from self_cognition.core.scopes import SubjectScope


@dataclass(frozen=True, slots=True)
class MemoryBehaviorPolicy:
    policy_version: str = "1"
    minimum_consolidation_confidence: float = 0.7
    minimum_consolidation_sources: int = 3
    minimum_consolidation_interval_days: float = 1.0
    reliable_threshold: float = 0.65
    uncertain_threshold: float = 0.35
    decay_period_days: float = 30.0
    access_reinforcement: float = 0.05
    spaced_reinforcement: float = 0.1
    cue_bonus: float = 0.15
    interference_penalty: float = 0.3

    def __post_init__(self) -> None:
        if not self.policy_version.strip():
            raise ValueError("policy_version must not be blank")
        if self.minimum_consolidation_sources < 2:
            raise ValueError("minimum consolidation sources must be at least two")
        if self.minimum_consolidation_interval_days <= 0 or self.decay_period_days <= 0:
            raise ValueError("memory time periods must be positive")
        for name in (
            "minimum_consolidation_confidence",
            "reliable_threshold",
            "uncertain_threshold",
            "access_reinforcement",
            "spaced_reinforcement",
            "cue_bonus",
            "interference_penalty",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if self.uncertain_threshold >= self.reliable_threshold:
            raise ValueError("uncertain threshold must be below reliable threshold")


@dataclass(frozen=True, slots=True)
class MemoryInterference:
    memory_id: UUID
    overlap: float
    reason: str


@dataclass(frozen=True, slots=True)
class MemoryRecallView:
    record: MemoryRecord
    score: float
    effective_stability: float
    matched_cues: tuple[str, ...]
    interference: tuple[MemoryInterference, ...]
    confidence_label: str
    explanation: str


def derive_recall_view(
    record: MemoryRecord,
    query: MemoryCues,
    accesses: tuple[datetime, ...],
    now: datetime,
    candidates: tuple[MemoryRecord, ...] = (),
    policy: MemoryBehaviorPolicy = MemoryBehaviorPolicy(),
) -> MemoryRecallView:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must include timezone information")
    ordered_accesses = tuple(sorted(value for value in accesses if value <= now))
    effective_stability = _effective_stability(record, ordered_accesses, policy)
    last_used_at = ordered_accesses[-1] if ordered_accesses else record.created_at
    age_days = max(0.0, (now - last_used_at).total_seconds() / 86400)
    decay = math.exp(-age_days / max(policy.decay_period_days * effective_stability, 0.1))
    matched_cues = _matched_cues(record.cues, query)
    query_size = len(_cue_values(query))
    cue_bonus = policy.cue_bonus * len(matched_cues) / query_size if query_size else 0.0
    interference = tuple(
        explanation
        for candidate in candidates
        if candidate.memory_id != record.memory_id
        for explanation in _interference(record, candidate, policy)
    )
    penalty = sum(item.overlap for item in interference)
    recall_strength = record.retrievability * decay + cue_bonus - penalty
    score = _clamp(
        recall_strength * record.confidence * (0.5 + 0.5 * record.salience)
    )
    label = (
        "RELIABLE"
        if score >= policy.reliable_threshold
        else "UNCERTAIN"
        if score >= policy.uncertain_threshold
        else "UNAVAILABLE"
    )
    explanation = (
        f"decay={decay:.3f}; stability={effective_stability:.3f}; "
        f"cue_bonus={cue_bonus:.3f}; interference_penalty={penalty:.3f}"
    )
    return MemoryRecallView(
        record=record,
        score=score,
        effective_stability=effective_stability,
        matched_cues=matched_cues,
        interference=interference,
        confidence_label=label,
        explanation=explanation,
    )


class MemoryRetrievalService:
    def __init__(
        self,
        repository: MemoryRepository,
        policy: MemoryBehaviorPolicy = MemoryBehaviorPolicy(),
    ) -> None:
        self._repository = repository
        self._policy = policy

    def retrieve(
        self,
        subject: SubjectScope,
        query: MemoryCues,
        *,
        now: datetime,
    ) -> tuple[MemoryRecallView, ...]:
        records = tuple(
            record
            for record in self._repository.read_by_subject(subject)
            if record.lifecycle_status is MemoryLifecycleStatus.ACTIVE
        )
        views = tuple(
            derive_recall_view(
                record,
                query,
                tuple(
                    access.accessed_at
                    for access in self._repository.read_access_history(
                        subject,
                        record.memory_id,
                    )
                ),
                now,
                records,
                self._policy,
            )
            for record in records
        )
        return tuple(sorted(views, key=lambda view: (-view.score, view.record.memory_id.int)))


class MemoryConsolidationService:
    def __init__(
        self,
        repository: MemoryRepository,
        policy: MemoryBehaviorPolicy = MemoryBehaviorPolicy(),
    ) -> None:
        self._repository = repository
        self._policy = policy

    def consolidate(
        self,
        subject: SubjectScope,
        *,
        now: datetime,
    ) -> tuple[MemoryRecord, ...]:
        records = tuple(
            record
            for record in self._repository.read_by_subject(subject)
            if record.lifecycle_status is MemoryLifecycleStatus.ACTIVE
            and record.consolidation_status is MemoryConsolidationStatus.RAW
            and record.memory_type in {MemoryType.EPISODIC, MemoryType.PROCEDURAL}
        )
        groups: dict[tuple[object, ...], list[MemoryRecord]] = {}
        for record in records:
            key = (
                record.memory_type,
                _canonical_content(record.content),
                record.scope,
                _stable_cue_key(record.cues),
            )
            groups.setdefault(key, []).append(record)
        consolidated: list[MemoryRecord] = []
        existing = self._repository.read_by_subject(subject)
        for group in groups.values():
            if not self._eligible(group):
                continue
            source_ids = tuple(record.memory_id for record in group)
            new_id = consolidated_memory_id(source_ids, self._policy.policy_version)
            if any(record.memory_id == new_id for record in existing):
                continue
            first = sorted(group, key=lambda record: record.created_at)[0]
            result = MemoryRecord(
                memory_id=new_id,
                memory_type=(
                    MemoryType.SEMANTIC
                    if first.memory_type is MemoryType.EPISODIC
                    else MemoryType.PROCEDURAL
                ),
                subject=subject,
                scope=first.scope,
                content=first.content,
                evidence_refs=_unique_evidence(group),
                confidence=min(record.confidence for record in group),
                salience=max(record.salience for record in group),
                stability=min(1.0, max(record.stability for record in group) + 0.15),
                retrievability=1.0,
                version=1,
                lifecycle_status=MemoryLifecycleStatus.ACTIVE,
                created_at=now,
                source_module="memory.consolidator",
                source_module_version=self._policy.policy_version,
                sources=_unique_sources(group),
                cues=_merge_cues(group),
                consolidation_status=MemoryConsolidationStatus.CONSOLIDATED,
            )
            self._repository.save(result, expected_version=0)
            consolidated.append(result)
        return tuple(consolidated)

    def _eligible(self, group: list[MemoryRecord]) -> bool:
        if len(group) < self._policy.minimum_consolidation_sources:
            return False
        if any(record.confidence < self._policy.minimum_consolidation_confidence for record in group):
            return False
        contribution_ids = {
            source.contribution_id for record in group for source in record.sources
        }
        if len(contribution_ids) < self._policy.minimum_consolidation_sources:
            return False
        source_kind = (
            EvidenceSourceKind.EVENT
            if group[0].memory_type is MemoryType.EPISODIC
            else EvidenceSourceKind.TOOL_RESULT
        )
        evidence = {
            item.evidence_id: item
            for record in group
            for item in record.evidence_refs
            if item.source_kind is source_kind and item.observed_at is not None
        }
        if len(evidence) < self._policy.minimum_consolidation_sources:
            return False
        times = [item.observed_at for item in evidence.values()]
        span_days = (max(times) - min(times)).total_seconds() / 86400
        return span_days >= self._policy.minimum_consolidation_interval_days


def _effective_stability(
    record: MemoryRecord,
    accesses: tuple[datetime, ...],
    policy: MemoryBehaviorPolicy,
) -> float:
    previous = record.created_at
    value = record.stability
    for accessed_at in accesses:
        gap_days = max(0.0, (accessed_at - previous).total_seconds() / 86400)
        value += policy.access_reinforcement
        if gap_days >= policy.minimum_consolidation_interval_days:
            value += policy.spaced_reinforcement
        previous = accessed_at
    return _clamp(value)


def _interference(
    record: MemoryRecord,
    candidate: MemoryRecord,
    policy: MemoryBehaviorPolicy,
) -> tuple[MemoryInterference, ...]:
    if record.memory_type is not candidate.memory_type:
        return ()
    if _canonical_content(record.content) == _canonical_content(candidate.content):
        return ()
    left = set(_cue_values(record.cues))
    right = set(_cue_values(candidate.cues))
    if not left or not right:
        return ()
    overlap = len(left & right) / len(left | right)
    if overlap == 0:
        return ()
    return (
        MemoryInterference(
            memory_id=candidate.memory_id,
            overlap=overlap * policy.interference_penalty,
            reason=f"same type with {overlap:.3f} cue overlap",
        ),
    )


def _matched_cues(record: MemoryCues, query: MemoryCues) -> tuple[str, ...]:
    return tuple(sorted(set(_cue_values(record)) & set(_cue_values(query))))


def _cue_values(cues: MemoryCues) -> tuple[str, ...]:
    return tuple(
        f"{name}:{value.casefold()}"
        for name, values in (
            ("person", cues.people),
            ("topic", cues.topics),
            ("time", cues.time_keys),
            ("relationship", cues.relationships),
            ("task", cues.tasks),
        )
        for value in values
    )


def _stable_cue_key(cues: MemoryCues) -> tuple[tuple[str, ...], ...]:
    return (
        cues.people,
        cues.topics,
        cues.relationships,
        cues.tasks,
    )


def _merge_cues(records: list[MemoryRecord]) -> MemoryCues:
    return MemoryCues(
        **{
            name: tuple(
                sorted(
                    {
                        value
                        for record in records
                        for value in getattr(record.cues, name)
                    }
                )
            )
            for name in ("people", "topics", "time_keys", "relationships", "tasks")
        }
    )


def _unique_evidence(records: list[MemoryRecord]) -> tuple[EvidenceRef, ...]:
    return tuple(
        dict.fromkeys(
            evidence
            for record in records
            for evidence in record.evidence_refs
        )
    )


def _unique_sources(records: list[MemoryRecord]) -> tuple[MemorySourceRef, ...]:
    return tuple(dict.fromkeys(source for record in records for source in record.sources))


def _canonical_content(content: object) -> str:
    return json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
