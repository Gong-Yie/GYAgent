import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Thread
from urllib.request import Request, urlopen
from uuid import uuid4

import pytest

from self_cognition.bootstrap import build_container
from self_cognition.core.actions import ActionDecision, ActionDecisionPayload, ActionDecisionStatus, ActionRequest
from self_cognition.core.events import EventEnvelope, EventSource
from self_cognition.core.scopes import DataScope, DisclosureScope, SubjectScope
from self_cognition.core.actions import ActionResultStatus
from self_cognition.tools.executor import FileReadToolExecutor, ToolExecutionPolicy
from self_cognition.core.ids import new_correlation_id, new_run_id
from self_cognition.runtime.run_context import RunContext
from self_cognition.core.runs import RunKind
from self_cognition.interfaces.http.server import _handle
from self_cognition.interfaces.http import create_server
from self_cognition.executive.dialogue.rule_based import RuleBasedDialogueModel
from self_cognition.workers.cognition import CognitionWorker
from self_cognition.workers.scheduler import SchedulerWorker


def test_http_loopback_chat_memory_and_health(tmp_path):
    container = build_container(tmp_path / "data")
    server = create_server(container, static_directory=Path("webui"))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        request = Request(
            base + "/chat",
            data=json.dumps({"subject_id": "user-1", "message": "我喜欢晚上学习"}).encode(),
            headers={"content-type": "application/json"},
        )
        with urlopen(request) as response:
            chat = json.loads(response.read())
        with urlopen(base + "/memories?subject_id=user-1") as response:
            memories = json.loads(response.read())
        with urlopen(base + "/health") as response:
            health = json.loads(response.read())
        with urlopen(base + "/") as response:
            page = response.read().decode("utf-8")
        assert chat["status"] == "succeeded"
        assert memories["memories"]
        assert "components" in health
        assert "自我认知控制台" in page
    finally:
        server.shutdown()
        server.server_close()


def test_http_chat_preserves_conversation_scope(tmp_path):
    app = build_container(tmp_path / "data", dialogue_model=RuleBasedDialogueModel())
    result = _handle(
        app,
        "POST",
        "/chat",
        {},
        {"subject_id": "user-1", "message": "你好", "conversation_id": "chat-1"},
    )

    assert result["status"] == "succeeded"
    events = app.event_store.read_by_subject(SubjectScope.legacy_user("user-1"))
    message = next(event for event in events if event.event_type == "user.message")
    assert message.scope.conversation.conversation_id == "chat-1"


def test_http_rejects_non_loopback_host():
    with pytest.raises(ValueError, match="loopback"):
        create_server(host="0.0.0.0")


def test_workers_run_and_stop():
    calls: list[str] = []
    cognition = CognitionWorker(lambda: calls.append("cognition"), poll_interval_seconds=0.01)
    scheduler = SchedulerWorker((lambda: calls.append("scheduled"),), interval_seconds=0.01)
    cognition.start()
    scheduler.start()
    cognition.stop(timeout=1)
    scheduler.stop(timeout=1)
    assert "cognition" in calls
    assert "scheduled" in calls


def test_confirmation_is_audited_and_executes_once(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "note.txt").write_text("ok", encoding="utf-8")
    executor = FileReadToolExecutor(ToolExecutionPolicy((allowed,)))
    app = build_container(tmp_path / "data", tool_executor=executor)
    mind = SubjectScope.for_mind("default-mind")
    user = SubjectScope.legacy_user("user-1")
    now = datetime.now(timezone.utc)
    request = ActionRequest(uuid4(), uuid4(), mind, uuid4(), 1, "read", "file.read", {"path": "note.txt"}, (), "approval-once", now)
    decision = ActionDecision(uuid4(), request.action_id, ActionDecisionStatus.CONFIRMATION_REQUIRED, "needs user confirmation", ("user request",), "local", ("file read",), (), now, now + timedelta(minutes=1), request.action_id, confirmation_prompt="Confirm?")
    app.event_store.append(EventEnvelope(uuid4(), "action.decided", None, mind, ActionDecisionPayload(request, decision), now, now, EventSource.SYSTEM, DataScope(mind, DisclosureScope.MIND), causation_id=request.proposal_request_id))
    context = RunContext(new_run_id(), new_correlation_id(), now + timedelta(minutes=1))
    result = app.action.approve_and_execute(request.action_id, user, context)
    assert result.result is not None and result.result.status is ActionResultStatus.SUCCEEDED
    assert any(record.target_id == str(request.action_id) for record in app.governance.read_audit(mind))

@pytest.mark.parametrize(
    ("mind_id", "subject_id", "allowed"),
    (("default-mind", "user-1", True), ("default-mind", "user-2", False), ("mind-b", "user-1", False)),
)
def test_http_cancel_checks_subject_before_side_effects(tmp_path, mind_id, subject_id, allowed):
    app = build_container(tmp_path / "data")
    owner = SubjectScope.legacy_user("user-1")
    context = RunContext(new_run_id(), new_correlation_id(), datetime.now(timezone.utc) + timedelta(minutes=1))
    app.run_lifecycle.begin(context, RunKind.ACTION, owner)
    original = app.run_repository.get(context.run_id)

    if allowed:
        result = _handle(app, "POST", f"/runs/{context.run_id}/cancel", {}, {"mind_id": mind_id, "subject_id": subject_id})
        assert result["cancel_requested"] is True
    else:
        with pytest.raises(LookupError, match="subject"):
            _handle(app, "POST", f"/runs/{context.run_id}/cancel", {}, {"mind_id": mind_id, "subject_id": subject_id})

    stored = app.run_repository.get(context.run_id)
    assert stored.cancel_requested is allowed
    assert context.is_cancelled is allowed
    if not allowed:
        assert stored == original
