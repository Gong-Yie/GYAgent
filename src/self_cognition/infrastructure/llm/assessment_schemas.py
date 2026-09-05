from copy import deepcopy
from typing import Any

from self_cognition.core.metacognition import (
    ConflictStatus,
    EvidenceBasis,
    FailureCause,
    KnowledgeStatus,
    SuggestedAction,
)


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


METACOGNITION_VALUE = _object(
    {
        "target": {"type": "string"},
        "status": {"type": "string", "enum": [item.value for item in KnowledgeStatus]},
        "basis": {"type": "string", "enum": [item.value for item in EvidenceBasis]},
        "explanation": {"type": "string"},
        "failure_cause": {
            "type": ["string", "null"],
            "enum": [None, *(item.value for item in FailureCause)],
        },
        "suggestions": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [item.value for item in SuggestedAction],
            },
        },
    }
)

CONFLICT_VALUE = _object(
    {
        "candidate_contribution_ids": {"type": "array", "items": {"type": "string"}},
        "status": {"type": "string", "enum": [item.value for item in ConflictStatus]},
        "reason": {"type": "string"},
        "requires_confirmation": {"type": "boolean"},
        "selected_contribution_id": {"type": ["string", "null"]},
    }
)

AFFECT_VALUE = _object(
    {
        "target": {"type": "string"},
        "goal_ids": {"type": "array", "items": {"type": "string"}},
        "emotion": {"type": "string"},
        "valence": {
            "type": "string",
            "enum": ["positive", "negative", "neutral", "mixed"],
        },
        "scope": {"type": "string"},
        "initial_intensity": {"type": "number", "minimum": 0, "maximum": 1},
        "assessed_at": {"type": "string"},
        "half_life_seconds": {"type": "number"},
        "active_threshold": {"type": "number"},
    }
)

ASSESSMENT_INSTRUCTIONS = {
    "metacognition": (
        "Assess epistemic status, evidence basis, expiry, contradictions and failures. "
        "Use set on metacognition.assessments.<stable-topic> only. UNKNOWN maps to "
        "cognition_type unknown; KNOWN + DIRECT maps to fact; otherwise use inference. "
        "Distinguish historical changes from unresolved contradictions. Failure cause "
        "must remain unknown without sufficient evidence. Suggestions are advisory only. "
        "To review a conflict, use review_conflict on its original target_field and "
        "candidate_contribution_ids supplied in context, never invented IDs. Open first; "
        "resolved selects a candidate; invalidated selects none. Use inference for reviews. "
        "You cannot claim human confirmation or change identity, values, or execute actions."
    ),
    "affect": (
        "Produce computational affect appraisals using set and cognition_type affect, "
        "only on affect.current.<stable-topic>. Preserve target, goal associations, scope, "
        "valence and uncertainty. goal_ids must reference supplied goals or be empty. "
        "Use the supplied assessment_time exactly. Default half_life_seconds=3600 and "
        "active_threshold=0.1. Do not claim real feelings or change values or permissions. "
        "For user.message assess the user's stated emotion, not the Agent's. For "
        "cognition.assessment_requested assess the Agent's response to the source event."
    ),
}


def assessment_schema(base: dict[str, Any], kind: str) -> dict[str, Any]:
    schema = deepcopy(base)
    properties = schema["properties"]["candidates"]["items"]["properties"]
    if kind == "metacognition":
        properties["operation"]["enum"] = ["set", "review_conflict"]
        properties["value"] = {"anyOf": [METACOGNITION_VALUE, CONFLICT_VALUE]}
    elif kind == "affect":
        properties["value"] = AFFECT_VALUE
        properties["cognition_type"]["enum"] = ["affect"]
    else:
        raise ValueError("unsupported assessment kind")
    return schema
