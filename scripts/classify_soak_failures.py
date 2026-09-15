from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from uuid import UUID


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from self_cognition.core.affect import AffectAssessment
from self_cognition.core.errors import ContractValidationError, ModelOutputError
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.events import CognitionModuleResultPayload
from self_cognition.core.metacognition import (
    ConflictReview,
    EvidenceBasis,
    KnowledgeStatus,
    MetacognitiveAssessment,
)
from self_cognition.infrastructure.llm.openai_responses import (
    OpenAIResponsesCognitionModel,
)
from self_cognition.infrastructure.persistence.file_event_store import (
    FileEventStore,
)


def _structural_category(message: str) -> str:
    checks = (
        ("must contain only candidates", "top_level_shape"),
        ("candidates must be an array", "candidates_not_array"),
        ("candidate must be an object", "candidate_not_object"),
        ("candidate fields are invalid", "candidate_fields"),
        ("confidence must be a number", "candidate_confidence"),
        ("operation or cognition_type is invalid", "candidate_operation_or_type"),
        ("evidence_ids", "candidate_evidence_ids"),
        ("must be a string", "candidate_string_field"),
    )
    for needle, category in checks:
        if needle in message:
            return category
    return "other_structure"


def _evidence_issue(
    candidate: dict[str, Any],
    source_event: Any,
) -> tuple[str, str] | None:
    evidence_ids = candidate.get("evidence_ids")
    if not isinstance(evidence_ids, list) or not all(
        isinstance(item, str) for item in evidence_ids
    ):
        return ("candidate_evidence_ids", "evidence_ids is not a string array")
    for item in evidence_ids:
        try:
            UUID(item)
        except ValueError:
            return ("invalid_evidence_id", item)
    if source_event is not None and str(source_event.event_id) not in evidence_ids:
        return (
            "missing_source_event_evidence",
            str(source_event.event_id),
        )
    return None


def _classify_semantic(
    module_id: str,
    payload: dict[str, Any],
    source_event: Any,
) -> tuple[str, str]:
    candidates = payload["candidates"]
    if module_id == "affect.affect_extractor":
        for candidate in candidates:
            evidence_issue = _evidence_issue(candidate, source_event)
            if evidence_issue is not None:
                return evidence_issue
            target_field = str(candidate.get("target_field", ""))
            if not target_field.startswith("affect.current."):
                return (
                    "affect_target_outside_ownership",
                    f"target_field={target_field}",
                )
            try:
                assessment = AffectAssessment.from_state_value(
                    candidate.get("value")
                )
            except ContractValidationError as error:
                return ("affect_value_contract", str(error))
            if (
                source_event is not None
                and assessment.assessed_at != source_event.occurred_at
            ):
                return (
                    "affect_time_mismatch",
                    (
                        f"assessed_at={assessment.assessed_at.isoformat()};"
                        f"event_occurred_at={source_event.occurred_at.isoformat()}"
                    ),
                )
        return ("affect_semantic_validation_other", "structure passed, engine rejected")

    if module_id == "metacognition.conflict_extractor":
        for candidate in candidates:
            evidence_issue = _evidence_issue(candidate, source_event)
            if evidence_issue is not None:
                return evidence_issue
            operation = str(candidate.get("operation", ""))
            target_field = str(candidate.get("target_field", ""))
            value = candidate.get("value")
            if operation == "review_conflict":
                try:
                    ConflictReview.from_state_value(value)
                except ContractValidationError as error:
                    return ("metacognition_conflict_review_contract", str(error))
                continue
            if not target_field.startswith("metacognition.assessments."):
                return (
                    "metacognition_target_outside_ownership",
                    f"target_field={target_field}",
                )
            try:
                assessment = MetacognitiveAssessment.from_state_value(value)
            except ContractValidationError as error:
                return ("metacognition_assessment_contract", str(error))
            expected_type = (
                "unknown"
                if assessment.status is KnowledgeStatus.UNKNOWN
                else "inference"
            )
            if (
                assessment.basis is EvidenceBasis.DIRECT
                and assessment.status is KnowledgeStatus.KNOWN
            ):
                expected_type = "fact"
            declared_type = str(candidate.get("cognition_type", ""))
            if declared_type != expected_type:
                return (
                    "metacognition_cognition_type_mismatch",
                    f"declared={declared_type};expected={expected_type}",
                )
        return (
            "metacognition_semantic_validation_other",
            "structure passed, engine rejected",
        )

    return ("semantic_validation_other", "structure passed, engine rejected")


def _classify_failure(
    module_id: str,
    error_type: str | None,
    raw_output: str | None,
    source_event: Any,
    response_event: Any,
) -> tuple[str, str]:
    if error_type == "ModelTimeoutError" or not raw_output:
        return ("provider_timeout_or_empty_output", "")
    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError as error:
        return ("invalid_json", str(error))
    try:
        OpenAIResponsesCognitionModel._parse_result(
            "classification-only",
            payload,
            EvidenceRef.for_event(response_event),
        )
    except ModelOutputError as error:
        return (_structural_category(str(error)), str(error))
    except Exception as error:
        return ("classifier_error", type(error).__name__)
    if not isinstance(payload, dict):
        return ("top_level_shape", "payload is not an object")
    return _classify_semantic(module_id, payload, source_event)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify real soak model failures")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sample-limit", type=int, default=10)
    args = parser.parse_args(argv)

    run_root = args.run_dir.resolve()
    data_dir = run_root / "data"
    store = FileEventStore(
        data_dir / "events" / "events.jsonl",
        data_dir / "deletions" / "event_tombstones.jsonl",
    )
    events = list(store.read_all())
    by_id = {event.event_id: event for event in events}
    records: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    module_counts: Counter[str] = Counter()
    module_categories: dict[str, Counter[str]] = defaultdict(Counter)

    for event in events:
        payload = event.payload
        if not isinstance(payload, CognitionModuleResultPayload):
            continue
        if payload.status != "failed":
            continue
        module_counts[payload.module_id] += 1
        source_event = by_id.get(getattr(event, "causation_id", None))
        response_events = [
            by_id[item]
            for item in payload.response_event_ids
            if item in by_id
        ]
        if not response_events:
            category, detail = ("provider_timeout_or_no_response", "")
            records.append(
                {
                    "module_id": payload.module_id,
                    "error_type": payload.error_type,
                    "category": category,
                    "detail": detail,
                    "source_event_id": str(source_event.event_id) if source_event else None,
                    "response_event_ids": [],
                    "raw_output": None,
                }
            )
            category_counts[category] += 1
            module_categories[payload.module_id][category] += 1
            continue
        for response_event in response_events:
            raw_output = getattr(response_event.payload, "raw_output", None)
            category, detail = _classify_failure(
                payload.module_id,
                payload.error_type,
                raw_output,
                source_event,
                response_event,
            )
            category_counts[category] += 1
            module_categories[payload.module_id][category] += 1
            records.append(
                {
                    "module_id": payload.module_id,
                    "error_type": payload.error_type,
                    "category": category,
                    "detail": detail,
                    "source_event_id": str(source_event.event_id) if source_event else None,
                    "response_event_ids": [str(item) for item in payload.response_event_ids],
                    "response_event_id": str(response_event.event_id),
                    "raw_output": raw_output,
                }
            )

    report = {
        "run_dir": str(run_root),
        "module_failure_counts": dict(module_counts),
        "category_counts": dict(category_counts),
        "module_categories": {
            module_id: dict(counter)
            for module_id, counter in module_categories.items()
        },
        "failures": records[: args.sample_limit],
        "total_failures": len(records),
    }
    output = args.output or (run_root / "failure_classification.json")
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
