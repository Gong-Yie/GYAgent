from __future__ import annotations

import json
import mimetypes
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import UUID

from self_cognition.bootstrap import ApplicationContainer, build_container
from self_cognition.core.actions import ActionDecisionPayload
from self_cognition.core.dialogue import DialogueRequest, draft_to_dict
from self_cognition.core.events import EventEnvelope
from self_cognition.core.identity import GoalPriority
from self_cognition.core.plans import GoalPlanningRequest, PlanBudget, GoalPlannedPayload, PlanRevisedPayload, plan_to_dict
from self_cognition.core.scopes import SubjectScope
from self_cognition.core.time import SYSTEM_CLOCK
from self_cognition.infrastructure.persistence.serialization import memory_to_dict, state_to_dict
from self_cognition.core.runs import run_to_dict
from self_cognition.core.ids import new_correlation_id, new_event_id, new_run_id
from self_cognition.runtime.run_context import RunContext


def create_app(container: ApplicationContainer | None = None, *, static_directory: str | Path | None = None) -> type[BaseHTTPRequestHandler]:
    """Return a request handler bound to one application container."""
    dependencies = container or build_container()
    static_root = Path(static_directory) if static_directory is not None else Path("webui")

    class Handler(BaseHTTPRequestHandler):
        server_version = "SelfCognition/1"

        def do_GET(self) -> None:  # noqa: N802
            self._dispatch("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._dispatch("POST")

        def log_message(self, format: str, *args: object) -> None:
            return

        def _dispatch(self, method: str) -> None:
            try:
                parsed = urlparse(self.path)
                if method == "GET" and _serve_static(self, static_root, parsed.path):
                    return
                payload = self._json_body() if method == "POST" else {}
                result = _handle(dependencies, method, parsed.path, parse_qs(parsed.query), payload)
                _write_json(self, 200, result)
            except LookupError as error:
                _write_json(self, 404, {"status": "failed", "error_code": "not_found", "error": str(error)})
            except (ValueError, TypeError) as error:
                _write_json(self, 400, {"status": "failed", "error_code": "invalid_request", "error": str(error)})
            except Exception as error:
                _write_json(self, 500, {"status": "failed", "error_code": "internal_error", "error_type": type(error).__name__})

        def _json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 1_000_000:
                raise ValueError("request body is too large")
            raw = self.rfile.read(length) if length else b"{}"
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("request body must be a JSON object")
            return value

    return Handler


def create_server(container: ApplicationContainer | None = None, *, host: str = "127.0.0.1", port: int = 0, static_directory: str | Path | None = None) -> ThreadingHTTPServer:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("HTTP interface only accepts loopback hosts")
    return ThreadingHTTPServer((host, port), create_app(container, static_directory=static_directory))


def _serve_static(handler: BaseHTTPRequestHandler, root: Path, request_path: str) -> bool:
    if request_path == "/":
        request_path = "/index.html"
    candidate = (root / request_path.lstrip("/")).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return False
    if not candidate.is_file():
        return False
    body = candidate.read_bytes()
    handler.send_response(200)
    handler.send_header("Content-Type", mimetypes.guess_type(candidate.name)[0] or "application/octet-stream")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)
    return True


def _handle(container: ApplicationContainer, method: str, path: str, query: dict[str, list[str]], body: dict[str, Any]) -> dict[str, object]:
    if method == "GET" and path == "/health":
        return container.health.check().as_dict()
    if method == "GET" and path in {"/metrics", "/usage"}:
        snapshot = container.metrics.snapshot()
        return {"counters": dict(snapshot.counters), "gauges": dict(snapshot.gauges), "timings": {key: list(value) for key, value in snapshot.timings.items()}}
    if method == "GET" and path == "/traces":
        return {"spans": [_span_to_dict(span) for span in container.traces.spans()]}
    subject = _subject(body or {key: values[-1] for key, values in query.items()})
    if method == "POST" and path == "/chat":
        text = _text(body, "message")
        context = _context()
        event = EventEnvelope.user_message(subject, text, run_id=context.run_id, correlation_id=context.correlation_id)
        result = container.converse.converse(DialogueRequest(event), context)
        return _converse(result)
    if method == "GET" and path == "/memories":
        return {"memories": [memory_to_dict(item) for item in container.memory_repository.read_by_subject(subject)]}
    if method == "GET" and path == "/goals":
        events = container.event_store.read_by_subject(SubjectScope.for_mind(subject.mind.mind_id))
        plans = [event.payload.plan for event in events if isinstance(event.payload, (GoalPlannedPayload, PlanRevisedPayload))]
        return {"plans": [plan_to_dict(item) for item in plans]}
    if method == "POST" and path == "/goals":
        owner = SubjectScope.for_mind(subject.mind.mind_id)
        request = GoalPlanningRequest(
            new_event_id(), owner, subject, _text(body, "goal_id"), _text(body, "description"), GoalPriority(str(body.get("priority", "normal"))), tuple(body.get("completion_conditions", ())), PlanBudget(int(body.get("max_steps", 16)), int(body.get("max_tool_steps", 8))),
        )
        result = container.pursue_goal.create(request, _context())
        return {"status": result.status.value, "run_id": str(result.run_id), "goal": _jsonable(result.goal), "plan": plan_to_dict(result.plan) if result.plan else None, "error_type": result.error_type}
    if method == "GET" and path.startswith("/goals/") and path.endswith("/progress"):
        progress = container.pursue_goal.progress(SubjectScope.for_mind(subject.mind.mind_id), UUID(path.split("/")[2]))
        return _jsonable(progress)
    if method == "POST" and path.startswith("/goals/"):
        plan_id = UUID(path.split("/")[2])
        action = path.split("/")[-1]
        context = _context()
        if action == "pause":
            result = container.pursue_goal.pause(SubjectScope.for_mind(subject.mind.mind_id), plan_id, context, actor=subject)
        elif action == "resume":
            result = container.pursue_goal.continue_goal(SubjectScope.for_mind(subject.mind.mind_id), plan_id, context, actor=subject)
        elif action == "cancel":
            result = container.pursue_goal.cancel(SubjectScope.for_mind(subject.mind.mind_id), plan_id, context, actor=subject)
        elif action == "complete":
            result = container.pursue_goal.complete(SubjectScope.for_mind(subject.mind.mind_id), plan_id, context, actor=subject)
        else:
            raise LookupError("goal route does not exist")
        return {"status": result.status.value, "run_id": str(result.run_id), "goal": _jsonable(result.goal), "progress": _jsonable(result.progress), "error_type": result.error_type}
    if method == "GET" and path == "/approvals":
        events = container.event_store.read_by_subject(SubjectScope.for_mind(subject.mind.mind_id))
        return {"approvals": [_jsonable(event.payload) for event in events if isinstance(event.payload, ActionDecisionPayload)]}
    if method == "POST" and path.startswith("/approvals/") and path.endswith("/approve"):
        result = container.action.approve_and_execute(UUID(path.split("/")[2]), subject, _context())
        return {"status": result.status.value, "run_id": str(result.run_id), "action": _jsonable(result.request), "decision": _jsonable(result.decision), "result": _jsonable(result.result), "error_type": result.error_type, "reused": result.reused}
    if method == "GET" and path == "/relationships":
        return {"entries": _state_entries(container, subject, "relationship.")}
    if method == "GET" and path == "/conflicts":
        return {"entries": _state_entries(container, subject, "conflict")}
    if method == "POST" and path == "/memories/correct":
        result = container.user_control.correct(subject, target_field=_text(body, "target_field"), cognition_type=str(body.get("cognition_type", "preference")), value=body.get("value"), context=_context())
        return {"status": result.status.value, "run_id": str(result.run_id), "new_version": result.new_version, "error_type": result.error_type}
    if method == "POST" and path == "/export":
        result = container.user_control.export(subject)
        return {"export_id": str(result.export_id), "path": str(result.path), "counts": result.counts}
    if method == "POST" and path == "/forget/dry-run":
        from self_cognition.core.deletions import DeletionSelector
        plan = container.forget.dry_run(DeletionSelector(subject, delete_subject=True), now=SYSTEM_CLOCK.now())
        return _deletion(plan)
    if method == "POST" and path == "/forget":
        from self_cognition.core.deletions import DeletionSelector
        plan = container.forget.dry_run(DeletionSelector(subject, delete_subject=True), now=SYSTEM_CLOCK.now())
        return _deletion(container.forget.execute(plan, now=SYSTEM_CLOCK.now()))
    if method == "GET" and path == "/replay":
        return state_to_dict(container.replay.replay(subject))
    if method == "GET" and path == "/self-model":
        state = container.state_repository.load(SubjectScope.for_mind(subject.mind.mind_id))
        return state_to_dict(state) if state is not None else {"version": 0, "entries": {}}
    if method == "GET" and path == "/settings":
        return {"controls": _jsonable(container.user_control.controls(subject))}
    if method == "GET" and path == "/runs":
        return {"runs": [run_to_dict(item) for item in container.run_repository.read_by_subject(subject)]}
    if method == "POST" and path.startswith("/runs/") and path.endswith("/cancel"):
        run_id = UUID(path.split("/")[2])
        record = container.run_lifecycle.request_cancel(run_id)
        if record.subject != subject:
            raise LookupError("run does not belong to subject")
        return run_to_dict(record)
    if method == "GET" and path.startswith("/runs/"):
        record = container.run_repository.get(UUID(path.split("/")[-1]))
        if record is None or record.subject != subject:
            raise LookupError("run does not exist")
        return run_to_dict(record)
    raise LookupError("route does not exist")


def _state_entries(container: ApplicationContainer, subject: SubjectScope, prefix: str) -> dict[str, object]:
    state = container.state_repository.load(subject)
    if state is None:
        state = container.state_repository.load(SubjectScope.for_mind(subject.mind.mind_id))
    if state is None:
        return {}
    return {key: _jsonable(atom.value) for key, atom in state.entries.items() if key.startswith(prefix) or prefix == "conflict" and "conflict" in key}


def _subject(data: dict[str, Any]) -> SubjectScope:
    mind_id = str(data.get("mind_id", "default-mind"))
    subject_id = str(data.get("subject_id", "user-1"))
    kind = str(data.get("subject_kind", "user"))
    if kind != "user":
        raise ValueError("local HTTP subject_kind must be user")
    return SubjectScope.legacy_user(subject_id) if mind_id == "default-mind" else SubjectScope(SubjectScope.for_mind(mind_id).mind, SubjectScope.legacy_user(subject_id).subject)


def _context() -> Any:
    from self_cognition.core.ids import new_correlation_id, new_run_id
    from self_cognition.runtime.run_context import RunContext
    return RunContext(new_run_id(), new_correlation_id(), SYSTEM_CLOCK.now() + timedelta(seconds=30))


def _text(data: dict[str, Any], name: str) -> str:
    value = data.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value


def _converse(result: Any) -> dict[str, object]:
    payload = {"status": result.status.value, "run_id": str(result.run_id), "correlation_id": str(result.correlation_id), "old_version": result.old_version, "new_version": result.new_version, "state_changed": result.state_changed, "event_saved": result.event_saved, "error_type": result.error_type}
    if result.response is not None:
        payload.update({"response": result.response.text, "response_event_id": str(result.response_event_id), "answer": draft_to_dict(result.response), "evidence_refs": [_jsonable(item) for item in result.evidence_refs]})
    return payload


def _deletion(plan: Any) -> dict[str, object]:
    return {"plan_id": str(plan.plan_id), "status": plan.status.value, "event_count": len(plan.event_ids), "memory_count": len(plan.memory_ids)}


def _span_to_dict(span: Any) -> dict[str, object]:
    return {"name": span.name, "span_id": str(span.span_id), "trace_id": str(span.trace_id), "status": span.status, "duration_seconds": span.duration_seconds, "attributes": dict(span.attributes)}


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "value") and not isinstance(value, (str, int, float, bool)):
        return value.value
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return {field: _jsonable(getattr(value, field)) for field in value.__dataclass_fields__}
    if isinstance(value, UUID):
        return str(value)
    return value


def _write_json(handler: BaseHTTPRequestHandler, status: int, payload: object) -> None:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=_jsonable).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)
