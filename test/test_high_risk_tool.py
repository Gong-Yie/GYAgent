from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from self_cognition.bootstrap import build_container
from self_cognition.core.actions import (
    ActionDecision,
    ActionDecisionPayload,
    ActionDecisionStatus,
    ActionRequest,
    ActionResultPayload,
    ActionResultStatus,
    decision_draft_from_json,
)
from self_cognition.core.events import EventEnvelope, EventSource
from self_cognition.core.ids import new_correlation_id, new_run_id
from self_cognition.core.scopes import DataScope, DisclosureScope, SubjectScope
from self_cognition.core.workspace import WorkspacePacket
from self_cognition.executive.action.fake import RuleActionModel
from self_cognition.runtime.run_context import RunContext
from self_cognition.tools.executor import (
    ToolExecutionPolicy,
    WorkspaceWriteFileExecutor,
)


def _now() -> datetime:
    return datetime(2026, 9, 16, 15, tzinfo=timezone.utc)


def _request(
    executor: WorkspaceWriteFileExecutor,
    *,
    arguments: dict[str, object],
    now: datetime,
) -> ActionRequest:
    descriptor = executor.descriptor
    return ActionRequest(
        uuid4(),
        uuid4(),
        SubjectScope.for_mind("default-mind"),
        uuid4(),
        1,
        "write-step",
        executor.tool_id,
        arguments,
        descriptor.expected_side_effects,
        "write-once",
        now,
    )


def _context(now: datetime) -> RunContext:
    return RunContext(
        new_run_id(),
        new_correlation_id(),
        now + timedelta(minutes=5),
        clock=_FixedClock(now),
    )


class _FixedClock:
    def __init__(self, value: datetime) -> None:
        self._value = value

    def now(self) -> datetime:
        return self._value


def _append_confirmation(
    app,
    request: ActionRequest,
    now: datetime,
) -> None:
    decision = ActionDecision(
        uuid4(),
        request.action_id,
        ActionDecisionStatus.CONFIRMATION_REQUIRED,
        "the action writes a file and is irreversible",
        ("user request",),
        "local workspace",
        ("overwrites a workspace file",),
        (),
        now,
        now + timedelta(minutes=1),
        request.action_id,
        confirmation_prompt="Confirm the file write?",
    )
    app.event_store.append(
        EventEnvelope(
            uuid4(),
            "action.decided",
            None,
            request.owner,
            ActionDecisionPayload(request, decision),
            now,
            now,
            EventSource.SYSTEM,
            DataScope(request.owner, DisclosureScope.MIND),
            causation_id=request.proposal_request_id,
        )
    )


def test_write_file_is_irreversible_and_requires_confirmation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    executor = WorkspaceWriteFileExecutor(ToolExecutionPolicy((root,)))
    now = _now()
    request = _request(
        executor,
        arguments={"path": "note.txt", "content": "hello"},
        now=now,
    )

    output = RuleActionModel().decide(
        request,
        WorkspacePacket("default-mind", 0, ()),
        _context(now),
    )
    draft = decision_draft_from_json(output.raw_output)

    assert draft.status is ActionDecisionStatus.CONFIRMATION_REQUIRED
    assert executor.descriptor.expected_side_effects[0].reversible is False


def test_high_risk_write_requires_approval_and_executes_once(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    executor = WorkspaceWriteFileExecutor(ToolExecutionPolicy((root,)))
    app = build_container(tmp_path / "data", tool_executor=executor)
    user = SubjectScope.legacy_user("user-1")
    now = _now()
    request = _request(
        executor,
        arguments={"path": "note.txt", "content": "hello"},
        now=now,
    )
    _append_confirmation(app, request, now)

    first = app.action.approve_and_execute(
        request.action_id,
        user,
        _context(now),
    )
    second = app.action.approve_and_execute(
        request.action_id,
        user,
        _context(now),
    )

    assert first.result is not None
    assert first.result.status is ActionResultStatus.SUCCEEDED
    assert (root / "note.txt").read_text(encoding="utf-8") == "hello"
    assert second.reused is True
    assert any(
        record.target_id == str(request.action_id)
        for record in app.governance.read_audit(request.owner)
    )


def test_high_risk_write_failure_records_standard_failed_result(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    executor = WorkspaceWriteFileExecutor(ToolExecutionPolicy((root,)))
    app = build_container(tmp_path / "data", tool_executor=executor)
    user = SubjectScope.legacy_user("user-1")
    now = _now()
    request = _request(
        executor,
        arguments={"path": "../escape.txt", "content": "nope"},
        now=now,
    )
    _append_confirmation(app, request, now)

    result = app.action.approve_and_execute(
        request.action_id,
        user,
        _context(now),
    )

    assert result.result is not None
    assert result.result.status is ActionResultStatus.FAILED
    assert result.result.error_type == "PathOutsideWorkspace"
    events = app.event_store.read_by_subject(request.owner)
    assert any(
        isinstance(event.payload, ActionResultPayload)
        and event.payload.result.status is ActionResultStatus.FAILED
        for event in events
    )

def test_default_container_registers_high_risk_write_tool(
    tmp_path: Path,
) -> None:
    app = build_container(tmp_path / "data")
    try:
        capability_ids = {
            item.capability_id
            for item in app.capability_registry.registrations()
        }
        assert "workspace.write_file" in capability_ids
    finally:
        app.lifecycle.stop()
