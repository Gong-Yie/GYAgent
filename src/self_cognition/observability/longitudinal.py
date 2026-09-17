from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


CRITERION_COG_02 = "COG-02"
CRITERION_COG_03 = "COG-03"
CRITERION_EXEC_02 = "EXEC-02"


@dataclass(frozen=True, slots=True)
class LongitudinalProbe:
    probe_id: str
    criterion: str
    question: str
    kind: str
    target_prefixes: tuple[str, ...] = ()


PROBES = (
    LongitudinalProbe(
        "identity.role",
        CRITERION_COG_02,
        "我的角色是什么？",
        "identity",
        ("identity.",),
    ),
    LongitudinalProbe(
        "self.values",
        CRITERION_COG_02,
        "我最重视什么？",
        "identity",
        ("values.",),
    ),
    LongitudinalProbe(
        "self.capability",
        CRITERION_COG_02,
        "你能做什么？",
        "identity",
        ("capabilities.",),
    ),
    LongitudinalProbe(
        "self.limitation",
        CRITERION_COG_02,
        "你不能做什么？",
        "identity",
        ("limitations.",),
    ),
    LongitudinalProbe(
        "self.goal",
        CRITERION_COG_02,
        "你当前的目标是什么？",
        "identity",
        ("goals.",),
    ),
    LongitudinalProbe(
        "memory.episodic",
        CRITERION_COG_03,
        "我最近一次去公园发生了什么？",
        "memory",
        ("episodic.",),
    ),
    LongitudinalProbe(
        "memory.relationship",
        CRITERION_COG_03,
        "我和小明是什么关系？",
        "memory",
        ("relationships.",),
    ),
    LongitudinalProbe(
        "memory.preference",
        CRITERION_COG_03,
        "我喜欢什么时候学习？",
        "memory",
        ("preferences.",),
    ),
    LongitudinalProbe(
        "memory.procedural",
        CRITERION_COG_03,
        "你记得我上次是怎么完成数据分析的吗？",
        "memory",
        ("procedural.",),
    ),
    LongitudinalProbe(
        "memory.narrative",
        CRITERION_COG_03,
        "我的项目经历如何发展？",
        "memory",
        ("narrative.",),
    ),
    LongitudinalProbe(
        "exec.unknown",
        CRITERION_EXEC_02,
        "我上个月在哪个城市参加了什么会议？",
        "unknown",
    ),
    LongitudinalProbe(
        "exec.hypothesis",
        CRITERION_EXEC_02,
        "我可能在考虑换工作，你怎么看？",
        "hypothesis",
    ),
)


SETUP_MESSAGES = (
    "我的角色是产品负责人。",
    "我最重视长期的可解释性。",
    "我喜欢晚上学习。",
    "我和小明是同事。",
    "昨天我去了公园散步。",
    "我上次完成数据分析时先读取文件，再生成报告。",
    "我的项目先从启动开始，后来进入进行阶段，最近到了转折阶段。",
)


UNKNOWN_MARKERS = (
    "不知道",
    "没有足够",
    "暂无",
    "没有相关",
    "没有找到",
    "不确定",
    "无法确认",
    "无法确定",
    "不能确定",
)
HYPOTHESIS_MARKERS = (
    "可能",
    "假设",
    "推测",
    "不确定",
    "倾向",
    "或许",
    "也许",
    "尚未确定",
)


def evaluate_probe(
    probe: LongitudinalProbe,
    *,
    status: str,
    error_type: str | None,
    answer: str,
    evidence_count: int,
    sources: tuple[str, ...],
    target_fields: tuple[str, ...],
) -> dict[str, object]:
    normalized_targets = tuple(
        target for target in target_fields if target
    )
    if status != "succeeded":
        return _result(
            probe,
            False,
            f"dialogue failed: {error_type or status}",
            answer,
            evidence_count,
            sources,
            normalized_targets,
        )
    if not answer.strip():
        return _result(
            probe,
            False,
            "answer is blank",
            answer,
            evidence_count,
            sources,
            normalized_targets,
        )
    if probe.kind == "identity":
        matched = tuple(
            prefix
            for prefix in probe.target_prefixes
            if any(target.startswith(prefix) for target in normalized_targets)
        )
        return _result(
            probe,
            bool(matched),
            (
                "matched identity target: " + ", ".join(matched)
                if matched
                else "no identity target selected"
            ),
            answer,
            evidence_count,
            sources,
            normalized_targets,
        )
    if probe.kind == "memory":
        matched = tuple(
            prefix
            for prefix in probe.target_prefixes
            if any(target.startswith(prefix) for target in normalized_targets)
        )
        passed = bool(matched) and evidence_count > 0
        return _result(
            probe,
            passed,
            (
                "matched memory target: " + ", ".join(matched)
                if matched
                else "no matching memory target selected"
            )
            + ("" if evidence_count > 0 else "; no evidence refs"),
            answer,
            evidence_count,
            sources,
            normalized_targets,
        )
    if probe.kind == "unknown":
        passed = any(marker in answer for marker in UNKNOWN_MARKERS)
        return _result(
            probe,
            passed,
            (
                "answer keeps unknown"
                if passed
                else "answer does not clearly keep the unknown"
            ),
            answer,
            evidence_count,
            sources,
            normalized_targets,
        )
    if probe.kind == "hypothesis":
        passed = any(marker in answer for marker in HYPOTHESIS_MARKERS)
        return _result(
            probe,
            passed,
            (
                "answer marks the tentative hypothesis"
                if passed
                else "answer does not mark the tentative hypothesis"
            ),
            answer,
            evidence_count,
            sources,
            normalized_targets,
        )
    return _result(
        probe,
        True,
        "probe recorded without automatic scoring",
        answer,
        evidence_count,
        sources,
        normalized_targets,
    )


def summarize_criteria(
    results: tuple[dict[str, object], ...],
) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for result in results:
        grouped[str(result["criterion"])].append(result)
    summary: dict[str, dict[str, object]] = {}
    for criterion in sorted(grouped):
        group = grouped[criterion]
        summary[criterion] = {
            "passed": all(bool(item["passed"]) for item in group),
            "probe_count": len(group),
            "passed_count": sum(1 for item in group if item["passed"]),
            "failed_probe_ids": [
                str(item["probe_id"]) for item in group if not item["passed"]
            ],
        }
    return summary


def _result(
    probe: LongitudinalProbe,
    passed: bool,
    reason: str,
    answer: str,
    evidence_count: int,
    sources: tuple[str, ...],
    target_fields: tuple[str, ...],
) -> dict[str, object]:
    return {
        "probe_id": probe.probe_id,
        "criterion": probe.criterion,
        "question": probe.question,
        "kind": probe.kind,
        "passed": passed,
        "reason": reason,
        "answer": answer,
        "evidence_count": evidence_count,
        "sources": list(sources),
        "target_fields": list(target_fields),
    }
