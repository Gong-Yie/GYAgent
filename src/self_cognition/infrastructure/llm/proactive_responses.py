from __future__ import annotations

import json
from typing import Any

from self_cognition.core.errors import ModelOutputError, ModelTimeoutError
from self_cognition.core.events import EventEnvelope
from self_cognition.core.proactivity import MotiveProposal, ProactiveIntention
from self_cognition.core.workspace import WorkspacePacket, workspace_model_context
from self_cognition.runtime.run_context import RunContext


PROACTIVE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "should_form": {"type": "boolean"},
        "kind": {"type": "string"},
        "description": {"type": "string"},
        "expected_behavior": {"type": "string"},
        "strength": {"type": "number", "minimum": 0, "maximum": 1},
        "priority": {"type": "integer", "minimum": 0},
        "valid_for_seconds": {"type": "integer", "minimum": 1},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "should_form",
        "kind",
        "description",
        "expected_behavior",
        "strength",
        "priority",
        "valid_for_seconds",
        "evidence_ids",
    ],
}


PROACTIVE_EXPRESSION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
}


class OpenAIResponsesProactivityModel:
    def __init__(
        self,
        client: Any,
        model: str,
        *,
        timeout_seconds: float = 30.0,
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        if not model.strip() or timeout_seconds <= 0 or max_output_tokens < 1:
            raise ValueError("invalid proactivity model configuration")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        self._client = client
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature

    @classmethod
    def from_api_key(
        cls,
        api_key: str,
        model: str,
        *,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> "OpenAIResponsesProactivityModel":
        from openai import OpenAI

        return cls(
            OpenAI(api_key=api_key, base_url=base_url, max_retries=0),
            model,
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )

    def express(
        self,
        intention: ProactiveIntention,
        workspace: WorkspacePacket,
        context: RunContext,
    ) -> str:
        remaining = (context.deadline - context.clock.now()).total_seconds()
        timeout = min(self._timeout_seconds, remaining)
        if timeout <= 0:
            raise ModelTimeoutError("proactivity expression deadline reached")
        response = self._client.responses.create(
            model=self._model,
            instructions=(
                "Write one short, natural message that expresses the supplied "
                "proactive intention to the target user. Use only the bounded "
                "workspace. Do not execute tools, claim results, invent memories, "
                "or disclose secrets. Respect the relationship and disclosure "
                "intent. Return only a JSON object with one text field."
            ),
            input=json.dumps(
                {
                    "intention": intention.to_state_value(),
                    "motive_kind": intention.motive.kind,
                    "motive_description": intention.motive.description,
                    "workspace": workspace_model_context(workspace),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "proactive_expression",
                    "schema": PROACTIVE_EXPRESSION_SCHEMA,
                    "strict": True,
                }
            },
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
            timeout=timeout,
        )
        context.record_model_usage(response)
        output = getattr(response, "output_text", None)
        if not isinstance(output, str) or not output.strip():
            raise ModelOutputError("proactivity expression returned no output")
        try:
            values = json.loads(output)
            text = str(values["text"]).strip()
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ModelOutputError("proactivity expression output is invalid") from error
        if not text:
            raise ModelOutputError("proactivity expression text is blank")
        return text

    def propose(
        self,
        event: EventEnvelope,
        workspace: WorkspacePacket,
        context: RunContext,
    ) -> MotiveProposal:
        remaining = (context.deadline - context.clock.now()).total_seconds()
        timeout = min(self._timeout_seconds, remaining)
        if timeout <= 0:
            raise ModelTimeoutError("proactivity deadline reached")
        response = self._client.responses.create(
            model=self._model,
            instructions=(
                "Evaluate whether the Agent should form one proactive motive from the "
                "current event and bounded workspace. Consider goals, relationships, "
                "emotion, commitments, consequences, and user benefit. Do not form a "
                "motive for ordinary small talk or unsupported assumptions. A motive "
                "must lead to one concrete behavior, cite supplied evidence IDs, and "
                "never execute a tool or disclose information. Return only the JSON "
                "object matching the schema."
            ),
            input=json.dumps(
                {
                    "event_id": str(event.event_id),
                    "event_type": event.event_type,
                    "event": getattr(event.payload, "text", ""),
                    "workspace": workspace_model_context(workspace),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            text={"format": {"type": "json_schema", "name": "proactive_motive", "schema": PROACTIVE_SCHEMA, "strict": True}},
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
            timeout=timeout,
        )
        context.record_model_usage(response)
        output = getattr(response, "output_text", None)
        if not isinstance(output, str) or not output.strip():
            raise ModelOutputError("proactivity model returned no output")
        try:
            values = json.loads(output)
            return MotiveProposal(
                bool(values["should_form"]),
                str(values["kind"]),
                str(values["description"]),
                str(values["expected_behavior"]),
                float(values["strength"]),
                int(values["priority"]),
                int(values["valid_for_seconds"]),
                tuple(str(item) for item in values["evidence_ids"]),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ModelOutputError("proactivity model output is invalid") from error
