from dataclasses import dataclass
from datetime import datetime
from self_cognition.core.affect import decay_assessment
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.indexes import WorkspaceIndex
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


@dataclass(frozen=True, slots=True)
class WorkspaceItem:
    target_field: str
    content: object
    evidence_refs: tuple[EvidenceRef, ...]
    confidence: float
    selection_reason: str


@dataclass(frozen=True, slots=True)
class Workspace:
    subject_id: str
    state_version: int
    items: tuple[WorkspaceItem, ...]


class WorkspaceBuilder:
    def __init__(
        self,
        index: WorkspaceIndex | None = None,
        clock: Clock = SYSTEM_CLOCK,
    ) -> None:
        self._index = index
        self._clock = clock

    def build(
        self,
        question: str,
        state: SubjectState,
        *,
        as_of: datetime | None = None,
        index: WorkspaceIndex | None = None,
    ) -> Workspace:
        evaluation_time = as_of or self._clock.now()
        active_index = index or self._index
        if active_index is not None and not active_index.is_compatible(state):
            active_index = None
        items: list[WorkspaceItem] = []
        for mapped_field in QUESTION_FIELDS.get(question, ()):
            if mapped_field.endswith(".*"):
                prefix = mapped_field[:-1]
                target_fields = (
                    active_index.fields_for_prefix(
                        prefix,
                        chronological=question == "我的项目经历如何发展？",
                    )
                    if active_index is not None
                    else sorted(
                        field
                        for field in state.entries
                        if field.startswith(prefix)
                    )
                )
            else:
                target_fields = [mapped_field]
            for target_field in target_fields:
                entry = state.entries.get(target_field)
                if entry is None:
                    continue
                content = entry.value
                if target_field.startswith("affect.current."):
                    content = decay_assessment(content, evaluation_time)
                    if content is None:
                        continue
                items.append(
                    WorkspaceItem(
                        target_field=target_field,
                        content=content,
                        evidence_refs=entry.evidence_refs,
                        confidence=entry.confidence,
                        selection_reason="question maps to this state field",
                    )
                )

        if question == "我的项目经历如何发展？":
            items.sort(
                key=lambda item: (
                    NARRATIVE_STAGE_ORDER.get(
                        item.content.get("stage", "")
                        if isinstance(item.content, dict)
                        else "",
                        99,
                    ),
                    str(item.content.get("occurred_at", ""))
                    if isinstance(item.content, dict)
                    else "",
                    item.target_field,
                )
            )

        return Workspace(
            subject_id=state.subject_id,
            state_version=state.version,
            items=tuple(items),
        )
