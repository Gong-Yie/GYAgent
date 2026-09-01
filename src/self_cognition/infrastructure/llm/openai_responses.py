import json
from typing import Any

from self_cognition.core.contributions import (
    CognitionType,
    ContributionOperation,
)
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.errors import ModelOutputError, ModelTimeoutError
from self_cognition.core.events import EventEnvelope
from self_cognition.core.model_outputs import (
    ContributionCandidate,
    ModelExtractionResult,
)
from self_cognition.runtime.run_context import RunContext


OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "target_field": {"type": "string"},
                    "operation": {"type": "string", "enum": ["set"]},
                    "cognition_type": {
                        "type": "string",
                        "enum": [item.value for item in CognitionType],
                    },
                    "value": {"type": ["string", "number", "boolean", "null"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence_ids": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "target_field",
                    "operation",
                    "cognition_type",
                    "value",
                    "confidence",
                    "evidence_ids",
                ],
            },
        }
    },
    "required": ["candidates"],
}


class OpenAIResponsesCognitionModel:
    def __init__(
        self,
        client: Any,
        model: str,
        *,
        timeout_seconds: float = 30.0,
        max_output_tokens: int = 512,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be blank")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        self._client = client
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._max_output_tokens = max_output_tokens

    @classmethod
    def from_api_key(
        cls,
        api_key: str,
        model: str,
        *,
        timeout_seconds: float = 30.0,
        max_output_tokens: int = 512,
    ) -> "OpenAIResponsesCognitionModel":
        from openai import OpenAI

        return cls(
            OpenAI(api_key=api_key, max_retries=0),
            model,
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
        )

    def extract(
        self,
        event: EventEnvelope,
        context: RunContext,
    ) -> ModelExtractionResult:
        timeout = min(self._timeout_seconds, self._remaining_seconds(context))
        if timeout <= 0:
            raise ModelTimeoutError("run deadline reached before model call")

        try:
            response = self._client.responses.create(
                model=self._model,
                instructions=(
                    "Extract only explicit user cognition facts. Return no candidate "
                    "when unsupported. Classify every candidate with cognition_type. "
                    "Every candidate must cite the supplied event ID."
                ),
                input=(
                    f"event_id={event.event_id}\n"
                    "subject_id="
                    f"{event.subject.subject.subject_id}\n"
                    f"event_type={event.event_type}\n"
                    f"content={event.payload.text}"
                ),
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "cognition_candidates",
                        "strict": True,
                        "schema": OUTPUT_SCHEMA,
                    }
                },
                max_output_tokens=self._max_output_tokens,
                store=False,
                timeout=timeout,
            )
        except Exception as error:
            if type(error).__name__ in {"APITimeoutError", "TimeoutError"}:
                raise ModelTimeoutError("cognition model timed out") from error
            raise

        response_id = getattr(response, "id", None)
        output_text = getattr(response, "output_text", None)
        if not isinstance(response_id, str) or not response_id.strip():
            raise ModelOutputError("model response is missing an ID")
        if not isinstance(output_text, str) or not output_text.strip():
            raise ModelOutputError("model response is missing structured output")

        response_event = EventEnvelope.model_response(
            event,
            model=self._model,
            response_id=response_id,
            raw_output=output_text,
            clock=context.clock,
            run_id=context.run_id,
            correlation_id=context.correlation_id,
        )
        context.emit_event(response_event)

        try:
            payload = json.loads(output_text)
        except json.JSONDecodeError as error:
            raise ModelOutputError("model output is not valid JSON") from error
        return self._parse_result(
            response_id,
            payload,
            EvidenceRef.for_event(response_event),
        )

    @staticmethod
    def _parse_result(
        response_id: str,
        payload: object,
        response_evidence: EvidenceRef,
    ) -> ModelExtractionResult:
        if not isinstance(payload, dict) or set(payload) != {"candidates"}:
            raise ModelOutputError("model output must contain only candidates")
        raw_candidates = payload["candidates"]
        if not isinstance(raw_candidates, list):
            raise ModelOutputError("model candidates must be an array")

        candidates: list[ContributionCandidate] = []
        expected_fields = {
            "target_field",
            "operation",
            "cognition_type",
            "value",
            "confidence",
            "evidence_ids",
        }
        for raw_candidate in raw_candidates:
            if not isinstance(raw_candidate, dict):
                raise ModelOutputError("model candidate must be an object")
            if set(raw_candidate) != expected_fields:
                raise ModelOutputError("model candidate fields are invalid")
            evidence_ids = raw_candidate["evidence_ids"]
            if not isinstance(evidence_ids, list) or not all(
                isinstance(item, str) for item in evidence_ids
            ):
                raise ModelOutputError(
                    "candidate evidence_ids must be a string array"
                )
            confidence = raw_candidate["confidence"]
            if not isinstance(confidence, (int, float)) or isinstance(
                confidence, bool
            ):
                raise ModelOutputError("candidate confidence must be a number")
            try:
                operation = ContributionOperation(
                    _require_string(
                        raw_candidate["operation"],
                        "candidate operation",
                    )
                )
                cognition_type = CognitionType(
                    _require_string(
                        raw_candidate["cognition_type"],
                        "candidate cognition_type",
                    )
                )
            except ValueError as error:
                raise ModelOutputError(
                    "candidate operation or cognition_type is invalid"
                ) from error
            candidates.append(
                ContributionCandidate(
                    target_field=_require_string(
                        raw_candidate["target_field"],
                        "candidate target_field",
                    ),
                    operation=operation,
                    cognition_type=cognition_type,
                    value=raw_candidate["value"],
                    confidence=float(confidence),
                    evidence_ids=tuple(evidence_ids),
                )
            )
        return ModelExtractionResult(
            response_id=response_id,
            candidates=tuple(candidates),
            response_evidence=response_evidence,
        )

    @staticmethod
    def _remaining_seconds(context: RunContext) -> float:
        return (context.deadline - context.clock.now()).total_seconds()


def _require_string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise ModelOutputError(f"{path} must be a string")
    return value
