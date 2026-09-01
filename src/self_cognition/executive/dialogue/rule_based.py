from dataclasses import dataclass
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.workspace import Workspace


STUDY_TIME_QUESTION = "我喜欢什么时候学习？"
STUDY_TIME_FIELD = "preferences.study_time"
STUDY_TIME_UNCERTAINTY_FIELD = (
    "metacognition.uncertainties.preferences.study_time"
)
STUDY_TIME_CONFLICT_FIELD = "metacognition.conflicts.preferences.study_time"
EXPERIENCE_QUESTION = "我经历过什么？"
EXPERIENCE_FIELD_PREFIX = "episodic.experience."
RELATIONSHIP_QUESTIONS = {
    "我和小明是什么关系？": ("relationships.小明.role", "小明"),
    "我和小红是什么关系？": ("relationships.小红.role", "小红"),
}
IDENTITY_ROLE_QUESTION = "我的角色是什么？"
IDENTITY_ROLE_FIELD = "identity.role"
VALUE_PRINCIPLE_QUESTION = "我最重视什么？"
VALUE_PRINCIPLE_FIELD = "values.principle"
AFFECT_QUESTIONS = {
    "我对这次考试感觉怎么样？": "affect.current.exam",
    "我对这个项目感觉怎么样？": "affect.current.project",
}
NARRATIVE_QUESTION = "我的项目经历如何发展？"
NARRATIVE_FIELD_PREFIX = "narrative.chapter."


@dataclass(frozen=True, slots=True)
class DialogueResponse:
    text: str
    evidence_refs: tuple[EvidenceRef, ...]


class RuleBasedDialogueModel:
    def respond(self, question: str, workspace: Workspace) -> DialogueResponse:
        if question == STUDY_TIME_QUESTION:
            by_field = {item.target_field: item for item in workspace.items}
            conflict = by_field.get(STUDY_TIME_CONFLICT_FIELD)
            if conflict is not None:
                return DialogueResponse(
                    text=(
                        "你的学习时间偏好存在冲突："
                        f"同时提到了{conflict.content}。"
                    ),
                    evidence_refs=conflict.evidence_refs,
                )
            uncertainty = by_field.get(STUDY_TIME_UNCERTAINTY_FIELD)
            if uncertainty is not None:
                return DialogueResponse(
                    text="你还不确定更喜欢早上还是晚上学习。",
                    evidence_refs=uncertainty.evidence_refs,
                )
            for item in workspace.items:
                if item.target_field == STUDY_TIME_FIELD:
                    return DialogueResponse(
                        text=f"你喜欢{item.content}学习。",
                        evidence_refs=item.evidence_refs,
                    )
            return DialogueResponse(
                text="我还不知道你喜欢什么时候学习。",
                evidence_refs=(),
            )

        if question == EXPERIENCE_QUESTION:
            items = [
                item
                for item in workspace.items
                if item.target_field.startswith(EXPERIENCE_FIELD_PREFIX)
            ]
            if not items:
                return DialogueResponse(
                    text="我还没有记住具体经历。",
                    evidence_refs=(),
                )
            return DialogueResponse(
                text="我记得：" + "；".join(str(item.content) for item in items),
                evidence_refs=tuple(
                    evidence_ref
                    for item in items
                    for evidence_ref in item.evidence_refs
                ),
            )

        relationship = RELATIONSHIP_QUESTIONS.get(question)
        if relationship is not None:
            target_field, related_subject = relationship
            for item in workspace.items:
                if item.target_field == target_field:
                    return DialogueResponse(
                        text=f"{related_subject}是你的{item.content}。",
                        evidence_refs=item.evidence_refs,
                    )
            return DialogueResponse(
                text=f"我还不知道你和{related_subject}是什么关系。",
                evidence_refs=(),
            )

        if question == IDENTITY_ROLE_QUESTION:
            return self._answer_single_field(
                workspace,
                IDENTITY_ROLE_FIELD,
                known_template="你的角色是{}。",
                unknown_text="我还不知道你的角色。",
            )

        if question == VALUE_PRINCIPLE_QUESTION:
            return self._answer_single_field(
                workspace,
                VALUE_PRINCIPLE_FIELD,
                known_template="你最重视{}。",
                unknown_text="我还不知道你最重视什么。",
            )

        affect_field = AFFECT_QUESTIONS.get(question)
        if affect_field is not None:
            for item in workspace.items:
                if item.target_field != affect_field:
                    continue
                content = item.content
                if not isinstance(content, dict):
                    break
                return DialogueResponse(
                    text=(
                        f"你对{content['target']}感到{content['emotion']}，"
                        "当前强度约为"
                        f"{float(content['current_intensity']):.2f}。"
                    ),
                    evidence_refs=item.evidence_refs,
                )
            return DialogueResponse(
                text="我没有足够强的当前情感评估。",
                evidence_refs=(),
            )

        if question == NARRATIVE_QUESTION:
            chapters = [
                item
                for item in workspace.items
                if item.target_field.startswith(NARRATIVE_FIELD_PREFIX)
                and isinstance(item.content, dict)
            ]
            if not chapters:
                return DialogueResponse(
                    text="我还没有形成项目叙事。",
                evidence_refs=(),
                )
            summaries = [str(item.content["summary"]) for item in chapters]
            if len(summaries) == 1:
                narrative_text = summaries[0]
            else:
                narrative_text = (
                    "先是"
                    + summaries[0]
                    + "，后来"
                    + "，接着".join(summaries[1:])
                )
            return DialogueResponse(
                text=f"你的项目叙事是：{narrative_text}。",
                evidence_refs=tuple(
                    evidence_ref
                    for item in chapters
                    for evidence_ref in item.evidence_refs
                ),
            )

        return DialogueResponse(
            text="我还不知道这个问题的答案。",
            evidence_refs=(),
        )

    @staticmethod
    def _answer_single_field(
        workspace: Workspace,
        target_field: str,
        *,
        known_template: str,
        unknown_text: str,
    ) -> DialogueResponse:
        for item in workspace.items:
            if item.target_field == target_field:
                return DialogueResponse(
                    text=known_template.format(item.content),
                    evidence_refs=item.evidence_refs,
                )
        return DialogueResponse(text=unknown_text, evidence_refs=())
