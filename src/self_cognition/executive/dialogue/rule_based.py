from dataclasses import dataclass
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.identity import (
    CapabilityExecutionStatus,
    CapabilityRecord,
    GoalRecord,
    GoalStatus,
    LimitationRecord,
    LimitationStatus,
)
from self_cognition.core.workspace import Workspace, WorkspaceItem
from self_cognition.core.workspace import RetrievalSource
from self_cognition.core.metacognition import MetacognitiveAssessment

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
SELF_IDENTITY_QUESTION = "你是谁？"
SELF_CAPABILITY_QUESTION = "你能做什么？"
SELF_LIMITATION_QUESTION = "你不能做什么？"
SELF_GOAL_QUESTION = "你当前的目标是什么？"


@dataclass(frozen=True, slots=True)
class DialogueResponse:
    text: str
    evidence_refs: tuple[EvidenceRef, ...]


class RuleBasedDialogueModel:
    def respond(self, question: str, workspace: Workspace) -> DialogueResponse:
        assessments = tuple(
            item
            for item in workspace.items
            if item.target_field.startswith("metacognition.assessments.")
        )
        if assessments:
            return DialogueResponse(
                "；".join(_describe_assessment(item) for item in assessments),
                _evidence_from(assessments),
            )
        conflicts = workspace.items_from(RetrievalSource.CONFLICT)
        if conflicts and not any(
            item.target_field == STUDY_TIME_CONFLICT_FIELD for item in workspace.items
        ):
            return DialogueResponse(
                "当前仍有未解决的冲突，不能给出确定结论："
                + "；".join(str(item.content["reason"]) for item in conflicts),
                _evidence_from(conflicts),
            )
        response = self._respond(question, workspace)
        used = tuple(
            item
            for item in workspace.items
            if any(ref in response.evidence_refs for ref in item.evidence_refs)
        )
        if used and any(item.confidence < 1.0 for item in used):
            confidence = min(item.confidence for item in used)
            return DialogueResponse(
                f"以下判断尚不确定（评估置信度 {confidence:.2f}，不是正确概率）：{response.text}",
                response.evidence_refs,
            )
        return response

    def _respond(self, question: str, workspace: Workspace) -> DialogueResponse:
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
                text="我记得："
                + "；".join(
                    str(item.content.get("text", item.content))
                    if isinstance(item.content, dict)
                    else str(item.content)
                    for item in items
                ),
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

        if question == SELF_IDENTITY_QUESTION:
            identity = tuple(
                item
                for item in workspace.items
                if item.target_field.startswith(("identity.", "values."))
            )
            if not identity:
                return DialogueResponse("我还没有足够证据说明自己是谁。", ())
            return DialogueResponse(
                "我的自我认知包括："
                + "；".join(str(item.content) for item in identity)
                + "。",
                _evidence_from(identity),
            )

        if question == SELF_CAPABILITY_QUESTION:
            capability_items = tuple(
                item
                for item in workspace.items
                if item.target_field.startswith("capabilities.")
            )
            capabilities = tuple(
                (item, CapabilityRecord.from_state_value(item.content))
                for item in capability_items
            )
            available = tuple(
                (item, capability)
                for item, capability in capabilities
                if capability.available
            )
            if not available:
                return DialogueResponse("我当前没有已登记且获准的能力。", ())
            descriptions = tuple(
                capability.name
                + (
                    "（已有成功执行证据）"
                    if capability.verified
                    else "（尚无成功执行证据）"
                )
                for _, capability in available
            )
            return DialogueResponse(
                "我当前可用的能力包括：" + "；".join(descriptions) + "。",
                _evidence_from(tuple(item for item, _ in available)),
            )

        if question == SELF_LIMITATION_QUESTION:
            limitation_items = tuple(
                item
                for item in workspace.items
                if item.target_field.startswith("limitations.")
            )
            limitations = tuple(
                (item, LimitationRecord.from_state_value(item.content))
                for item in limitation_items
            )
            active = tuple(
                (item, limitation)
                for item, limitation in limitations
                if limitation.status is LimitationStatus.ACTIVE
            )
            capability_items = tuple(
                item
                for item in workspace.items
                if item.target_field.startswith("capabilities.")
            )
            unavailable = tuple(
                (item, capability)
                for item in capability_items
                for capability in (CapabilityRecord.from_state_value(item.content),)
                if not capability.available
                or capability.execution_status is CapabilityExecutionStatus.FAILED
            )
            if not active and not unavailable:
                return DialogueResponse("我目前没有已记录的能力限制。", ())
            descriptions = tuple(
                f"{limitation.description}（{limitation.reason}）"
                for _, limitation in active
            ) + tuple(
                f"{capability.name}（{capability.reason}）"
                for _, capability in unavailable
            )
            evidence_items = tuple(item for item, _ in active + unavailable)
            return DialogueResponse(
                "我当前的限制包括：" + "；".join(descriptions) + "。",
                _evidence_from(evidence_items),
            )

        if question == SELF_GOAL_QUESTION:
            goal_items = tuple(
                item
                for item in workspace.items
                if item.target_field.startswith("goals.")
            )
            goals = tuple(
                (item, GoalRecord.from_state_value(item.content))
                for item in goal_items
            )
            active = tuple(
                (item, goal)
                for item, goal in goals
                if goal.status is GoalStatus.ACTIVE
            )
            if not active:
                return DialogueResponse("我当前没有有证据支持的活动目标。", ())
            return DialogueResponse(
                "我当前的目标是："
                + "；".join(goal.description for _, goal in active)
                + "。",
                _evidence_from(tuple(item for item, _ in active)),
            )

        affect_field = AFFECT_QUESTIONS.get(question)
        if affect_field is not None:
            for item in workspace.items:
                if item.target_field != affect_field:
                    continue
                content = item.content
                if not isinstance(content, dict):
                    break
                if "goal_ids" in content:
                    return _affect_response(item)
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

        for item in workspace.items:
            if (
                item.target_field.startswith("affect.current.")
                and isinstance(item.content, dict)
                and "goal_ids" in item.content
            ):
                return _affect_response(item)
        return DialogueResponse(text="我还不知道这个问题的答案。", evidence_refs=())

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


def _describe_assessment(item: WorkspaceItem) -> str:
    assessment = MetacognitiveAssessment.from_state_value(item.content)
    statuses = {
        "known": "有相关认知记录",
        "unknown": "仍不知道",
        "conflict": "存在冲突",
        "expired": "认知已过期",
    }
    bases = {"direct": "直接证据", "inference": "推断", "hypothesis": "假设"}
    causes = {
        "permission": "权限",
        "environment": "环境",
        "input": "输入",
        "model": "模型",
        "strategy": "策略",
        "unknown": "原因未知",
    }
    actions = {
        "ask": "询问",
        "search": "搜索",
        "retry": "重试",
        "change_strategy": "换策略",
        "stop": "停止",
    }
    text = (
        f"关于{assessment.target}：{statuses[assessment.status.value]}，"
        f"依据类型为{bases[assessment.basis.value]}；{assessment.explanation}。"
        f"评估置信度 {item.confidence:.2f}（不是统计校准的正确概率）。"
    )
    if assessment.failure_cause is not None:
        text += f"失败归因：{causes[assessment.failure_cause.value]}。"
    if assessment.suggestions:
        text += (
            "建议"
            + "、".join(actions[action.value] for action in assessment.suggestions)
            + "；尚未执行。"
        )
    return text


def _affect_response(item: WorkspaceItem) -> DialogueResponse:
    content = item.content
    return DialogueResponse(
        f"对{content['target']}的计算性情感评估为{content['emotion']}，"
        f"当前强度约 {content['current_intensity']:.2f}；"
        "这是可衰减的评估状态，不代表真实感受，也不替代价值判断。",
        item.evidence_refs,
    )


def _evidence_from(items: tuple[WorkspaceItem, ...]) -> tuple[EvidenceRef, ...]:
    seen = set()
    evidence: list[EvidenceRef] = []
    for item in items:
        for ref in item.evidence_refs:
            if ref.evidence_id in seen:
                continue
            seen.add(ref.evidence_id)
            evidence.append(ref)
    return tuple(evidence)
