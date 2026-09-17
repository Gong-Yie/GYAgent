from __future__ import annotations

from self_cognition.observability.longitudinal import (
    CRITERION_COG_02,
    CRITERION_COG_03,
    CRITERION_EXEC_02,
    PROBES,
    evaluate_probe,
    summarize_criteria,
)


def _probe(probe_id: str):
    return next(item for item in PROBES if item.probe_id == probe_id)


def test_identity_probe_requires_selected_identity_target() -> None:
    passed = evaluate_probe(
        _probe("identity.role"),
        status="succeeded",
        error_type=None,
        answer="你的角色是产品负责人。",
        evidence_count=1,
        sources=("RetrievalSource.STATE",),
        target_fields=("identity.role",),
    )
    failed = evaluate_probe(
        _probe("identity.role"),
        status="succeeded",
        error_type=None,
        answer="我不知道。",
        evidence_count=0,
        sources=(),
        target_fields=(),
    )

    assert passed["passed"] is True
    assert failed["passed"] is False


def test_memory_probe_requires_matching_target_and_evidence() -> None:
    passed = evaluate_probe(
        _probe("memory.episodic"),
        status="succeeded",
        error_type=None,
        answer="你在公园散步。",
        evidence_count=1,
        sources=("RetrievalSource.MEMORY",),
        target_fields=("episodic.experience.park",),
    )
    failed = evaluate_probe(
        _probe("memory.episodic"),
        status="succeeded",
        error_type=None,
        answer="你在公园散步。",
        evidence_count=0,
        sources=(),
        target_fields=("episodic.experience.park",),
    )

    assert passed["passed"] is True
    assert failed["passed"] is False


def test_unknown_and_hypothesis_probes_use_epistemic_markers() -> None:
    unknown = evaluate_probe(
        _probe("exec.unknown"),
        status="succeeded",
        error_type=None,
        answer="我没有足够信息确认上个月会议。",
        evidence_count=0,
        sources=(),
        target_fields=(),
    )
    hypothesis = evaluate_probe(
        _probe("exec.hypothesis"),
        status="succeeded",
        error_type=None,
        answer="这只是推测，你可能有换工作的倾向。",
        evidence_count=1,
        sources=("RetrievalSource.STATE",),
        target_fields=("metacognition.assessments.user_goal",),
    )
    overly_certain = evaluate_probe(
        _probe("exec.hypothesis"),
        status="succeeded",
        error_type=None,
        answer="你要换工作。",
        evidence_count=1,
        sources=("RetrievalSource.STATE",),
        target_fields=(),
    )

    assert unknown["passed"] is True
    assert hypothesis["passed"] is True
    assert overly_certain["passed"] is False


def test_summarize_criteria_groups_probe_results() -> None:
    results = (
        evaluate_probe(
            _probe("identity.role"),
            status="succeeded",
            error_type=None,
            answer="产品负责人。",
            evidence_count=1,
            sources=(),
            target_fields=("identity.role",),
        ),
        evaluate_probe(
            _probe("self.values"),
            status="failed",
            error_type="GroundingRejected",
            answer="",
            evidence_count=0,
            sources=(),
            target_fields=(),
        ),
        evaluate_probe(
            _probe("exec.unknown"),
            status="succeeded",
            error_type=None,
            answer="我不知道。",
            evidence_count=0,
            sources=(),
            target_fields=(),
        ),
    )

    summary = summarize_criteria(results)

    assert summary[CRITERION_COG_02]["passed"] is False
    assert summary[CRITERION_COG_02]["failed_probe_ids"] == ["self.values"]
    assert summary[CRITERION_EXEC_02]["passed"] is True
    assert CRITERION_COG_03 not in summary
