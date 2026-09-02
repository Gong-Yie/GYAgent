from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol, runtime_checkable
from uuid import UUID

from self_cognition.core.affect import decay_assessment
from self_cognition.core.errors import ContractValidationError
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.indexes import WorkspaceIndex
from self_cognition.core.memories import MemoryCues
from self_cognition.core.scopes import SubjectScope
from self_cognition.core.state import SubjectState
from self_cognition.core.time import Clock, SYSTEM_CLOCK


QUESTION_FIELDS = {
    "我喜欢什么时候学习？": (
        "metacognition.conflicts.preferences.study_time",
        "metacognition.uncertainties.preferences.study_time",
        "preferences.study_time",
    ),
    "我经历过什么？": ("episodic.experience.*",),
    "我和小明是什么关系？": ("relationships.小明.role",),
    "我和小红是什么关系？": ("relationships.小红.role",),
    "我的角色是什么？": ("identity.role",),
    "我最重视什么？": ("values.principle",),
    "我对这次考试感觉怎么样？": ("affect.current.exam",),
    "我对这个项目感觉怎么样？": ("affect.current.project",),
    "我的项目经历如何发展？": ("narrative.chapter.*",),
}
NARRATIVE_STAGE_ORDER = {
    "启动": 0,
    "进行": 1,
    "转折": 2,
    "完成": 3,
}
WORKSPACE_SCHEMA_VERSION = 1


class RetrievalSource(str, Enum):
    STATE = "state"
    MEMORY = "memory"
    CONFLICT = "conflict"
    RUN = "run"


@dataclass(frozen=True, slots=True)
class RetrievalBudget:
    max_tokens: int = 1024
    max_items: int = 16

    def __post_init__(self) -> None:
        for name in ("max_tokens", "max_items"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ContractValidationError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class WorkspaceFixedContext:
    identity: tuple[str, ...] = ()
    current_goal: str = ""
    safety_rules: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("identity", "safety_rules"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ContractValidationError(f"{name} must contain text values")
        if not isinstance(self.current_goal, str):
            raise ContractValidationError("current_goal must be text")


@dataclass(frozen=True, slots=True)
class WorkspaceRunInfo:
    run_id: UUID
    correlation_id: UUID
    deadline: datetime
    cancelled: bool

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, UUID) or not isinstance(
            self.correlation_id,
            UUID,
        ):
            raise ContractValidationError("run identifiers must be UUID values")
        if self.deadline.tzinfo is None or self.deadline.utcoffset() is None:
            raise ContractValidationError(
                "run deadline must include timezone information"
            )
        if not isinstance(self.cancelled, bool):
            raise ContractValidationError("cancelled must be a boolean")


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    subject: SubjectScope
    task: str
    purpose: str = "dialogue"
    field_patterns: tuple[str, ...] = ()
    memory_cues: MemoryCues = MemoryCues()
    budget: RetrievalBudget = RetrievalBudget()
    fixed_context: WorkspaceFixedContext = WorkspaceFixedContext()
    time_from: datetime | None = None
    time_to: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.subject, SubjectScope):
            raise ContractValidationError("query subject must be a SubjectScope")
        for name in ("task", "purpose"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ContractValidationError(f"query {name} must not be blank")
        if any(not value.strip() for value in self.field_patterns):
            raise ContractValidationError("field patterns must not be blank")
        for name in ("time_from", "time_to"):
            value = getattr(self, name)
            if value is not None and (
                value.tzinfo is None or value.utcoffset() is None
            ):
                raise ContractValidationError(
                    f"query {name} must include timezone information"
                )
        if (
            self.time_from is not None
            and self.time_to is not None
            and self.time_from > self.time_to
        ):
            raise ContractValidationError("query time range is reversed")

    @classmethod
    def for_question(
        cls,
        question: str,
        subject: SubjectScope,
        *,
        budget: RetrievalBudget = RetrievalBudget(),
        fixed_context: WorkspaceFixedContext | None = None,
    ) -> RetrievalQuery:
        patterns = QUESTION_FIELDS.get(question, ())
        topics = tuple(pattern.rstrip(".*") for pattern in patterns)
        return cls(
            subject=subject,
            task=question,
            field_patterns=patterns,
            memory_cues=MemoryCues(topics=topics),
            budget=budget,
            fixed_context=(
                fixed_context
                if fixed_context is not None
                else WorkspaceFixedContext(current_goal=question)
            ),
        )


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    candidate_id: str
    source: RetrievalSource
    source_ref: str
    target_field: str
    content: object
    evidence_refs: tuple[EvidenceRef, ...]
    confidence: float
    state_version: int | None
    relevance: float
    evidence_quality: float
    risk: float
    diversity: float
    task_relevance: float
    estimated_tokens: int
    reason: str

    @property
    def score(self) -> float:
        return (
            0.25 * self.relevance
            + 0.25 * self.evidence_quality
            + 0.2 * self.risk
            + 0.1 * self.diversity
            + 0.2 * self.task_relevance
        )


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    candidates: tuple[RetrievalCandidate, ...]
    index_status: str


@runtime_checkable
class Retriever(Protocol):
    def retrieve(
        self,
        query: RetrievalQuery,
        state: SubjectState,
        *,
        as_of: datetime,
        index: WorkspaceIndex | None = None,
        run_info: WorkspaceRunInfo | None = None,
    ) -> RetrievalResult: ...


@dataclass(frozen=True, slots=True)
class WorkspaceItem:
    target_field: str
    content: object
    evidence_refs: tuple[EvidenceRef, ...]
    confidence: float
    selection_reason: str
    source: RetrievalSource = RetrievalSource.STATE
    source_ref: str = ""
    score: float = 1.0
    estimated_tokens: int = 0


@dataclass(frozen=True, slots=True)
class RetrievalDecision:
    candidate_id: str
    source: RetrievalSource
    source_ref: str
    selected: bool
    score: float
    estimated_tokens: int
    reason: str


@dataclass(frozen=True, slots=True)
class WorkspacePacket:
    subject_id: str
    state_version: int
    items: tuple[WorkspaceItem, ...]
    task_context: str = ""
    fixed_context: WorkspaceFixedContext = WorkspaceFixedContext()
    budget: RetrievalBudget = RetrievalBudget()
    used_tokens: int = 0
    decisions: tuple[RetrievalDecision, ...] = ()
    index_status: str = "authoritative_scan"
    run_info: WorkspaceRunInfo | None = None
    workspace_version: int = WORKSPACE_SCHEMA_VERSION

    @property
    def evidence_refs(self) -> tuple[EvidenceRef, ...]:
        seen: set[UUID] = set()
        result: list[EvidenceRef] = []
        for item in self.items:
            for evidence in item.evidence_refs:
                if evidence.evidence_id in seen:
                    continue
                seen.add(evidence.evidence_id)
                result.append(evidence)
        return tuple(result)

    def items_from(self, source: RetrievalSource) -> tuple[WorkspaceItem, ...]:
        return tuple(item for item in self.items if item.source is source)


# Existing dialogue callers keep the old name while consuming the richer packet.
Workspace = WorkspacePacket


class WorkspaceBuilder:
    def __init__(
        self,
        retriever: Retriever | None = None,
        index: WorkspaceIndex | None = None,
        clock: Clock = SYSTEM_CLOCK,
    ) -> None:
        self._retriever = retriever
        self._index = index
        self._clock = clock

    def build(
        self,
        question: str,
        state: SubjectState,
        *,
        as_of: datetime | None = None,
        index: WorkspaceIndex | None = None,
        query: RetrievalQuery | None = None,
        budget: RetrievalBudget = RetrievalBudget(),
        fixed_context: WorkspaceFixedContext | None = None,
        run_info: WorkspaceRunInfo | None = None,
    ) -> WorkspacePacket:
        evaluation_time = as_of or self._clock.now()
        active_query = query or RetrievalQuery.for_question(
            question,
            state.subject_scope,
            budget=budget,
            fixed_context=fixed_context,
        )
        if active_query.subject != state.subject_scope:
            raise ContractValidationError("query and state subjects do not match")

        active_index = index or self._index
        if self._retriever is None:
            result = _retrieve_state(active_query, state, evaluation_time, active_index)
        else:
            result = self._retriever.retrieve(
                active_query,
                state,
                as_of=evaluation_time,
                index=active_index,
                run_info=run_info,
            )
        return _select(active_query, state, result, run_info)


def estimate_tokens(value: object) -> int:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return max(1, (len(serialized.encode("utf-8")) + 3) // 4)


def evidence_quality(
    evidence_refs: tuple[EvidenceRef, ...],
    fallback: float,
) -> float:
    reliabilities = tuple(
        evidence.reliability
        for evidence in evidence_refs
        if evidence.reliability is not None
    )
    return sum(reliabilities) / len(reliabilities) if reliabilities else fallback


def _select(
    query: RetrievalQuery,
    state: SubjectState,
    result: RetrievalResult,
    run_info: WorkspaceRunInfo | None,
) -> WorkspacePacket:
    fixed_tokens = estimate_tokens(
        {
            "task": query.task,
            "identity": query.fixed_context.identity,
            "current_goal": query.fixed_context.current_goal,
            "safety_rules": query.fixed_context.safety_rules,
            "run_info": run_info,
        }
    )
    if fixed_tokens > query.budget.max_tokens:
        raise ContractValidationError("fixed workspace context exceeds token budget")

    used_tokens = fixed_tokens
    items: list[WorkspaceItem] = []
    decisions: list[RetrievalDecision] = []
    ordered = sorted(
        result.candidates,
        key=lambda candidate: (-candidate.score, candidate.candidate_id),
    )
    for candidate in ordered:
        if len(items) >= query.budget.max_items:
            excluded_reason = "excluded: item budget exhausted"
        elif used_tokens + candidate.estimated_tokens > query.budget.max_tokens:
            excluded_reason = "excluded: token budget exhausted"
        else:
            excluded_reason = ""
            used_tokens += candidate.estimated_tokens
            items.append(
                WorkspaceItem(
                    target_field=candidate.target_field,
                    content=candidate.content,
                    evidence_refs=candidate.evidence_refs,
                    confidence=candidate.confidence,
                    selection_reason=(
                        f"{candidate.reason}; score={candidate.score:.3f}"
                    ),
                    source=candidate.source,
                    source_ref=candidate.source_ref,
                    score=candidate.score,
                    estimated_tokens=candidate.estimated_tokens,
                )
            )
        decisions.append(
            RetrievalDecision(
                candidate_id=candidate.candidate_id,
                source=candidate.source,
                source_ref=candidate.source_ref,
                selected=not excluded_reason,
                score=candidate.score,
                estimated_tokens=candidate.estimated_tokens,
                reason=excluded_reason or "selected within item and token budgets",
            )
        )

    if query.task == "我的项目经历如何发展？":
        items.sort(key=_narrative_order)

    return WorkspacePacket(
        subject_id=state.subject_id,
        state_version=state.version,
        items=tuple(items),
        task_context=query.task,
        fixed_context=query.fixed_context,
        budget=query.budget,
        used_tokens=used_tokens,
        decisions=tuple(decisions),
        index_status=result.index_status,
        run_info=run_info,
    )


def _retrieve_state(
    query: RetrievalQuery,
    state: SubjectState,
    as_of: datetime,
    index: WorkspaceIndex | None,
) -> RetrievalResult:
    active_index = index if index is not None and index.is_compatible(state) else None
    candidates: list[RetrievalCandidate] = []
    for pattern in query.field_patterns:
        if pattern.endswith(".*"):
            prefix = pattern[:-1]
            fields = (
                active_index.fields_for_prefix(
                    prefix,
                    chronological=query.task == "我的项目经历如何发展？",
                )
                if active_index is not None
                else tuple(
                    sorted(
                        field
                        for field in state.entries
                        if field.startswith(prefix)
                    )
                )
            )
        else:
            fields = (pattern,)
        for field_name in fields:
            entry = state.entries.get(field_name)
            if entry is None:
                continue
            content = entry.value
            if field_name.startswith("affect.current."):
                content = decay_assessment(content, as_of)
                if content is None:
                    continue
            risk = 1.0 if field_name.startswith("metacognition.") else 0.0
            candidates.append(
                RetrievalCandidate(
                    candidate_id=f"state:{field_name}",
                    source=RetrievalSource.STATE,
                    source_ref=field_name,
                    target_field=field_name,
                    content=content,
                    evidence_refs=entry.evidence_refs,
                    confidence=entry.confidence,
                    state_version=state.version,
                    relevance=1.0,
                    evidence_quality=evidence_quality(
                        entry.evidence_refs,
                        entry.confidence,
                    ),
                    risk=risk,
                    diversity=1.0,
                    task_relevance=1.0,
                    estimated_tokens=estimate_tokens(content),
                    reason="question maps to this state field",
                )
            )
    unique = {candidate.candidate_id: candidate for candidate in candidates}
    return RetrievalResult(
        candidates=tuple(unique.values()),
        index_status="used" if active_index is not None else "authoritative_scan",
    )


def _narrative_order(item: WorkspaceItem) -> tuple[object, ...]:
    content = item.content
    return (
        NARRATIVE_STAGE_ORDER.get(
            content.get("stage", "") if isinstance(content, dict) else "",
            99,
        ),
        str(content.get("occurred_at", "")) if isinstance(content, dict) else "",
        item.target_field,
    )
