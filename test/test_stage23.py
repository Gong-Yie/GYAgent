from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from self_cognition.bootstrap import build_container
from self_cognition.core.actions import (
    ActionDecision,
    ActionDecisionPayload,
    ActionDecisionStatus,
    ActionRequest,
    ActionResultStatus,
    ExpectedSideEffect,
)
from self_cognition.core.events import EventEnvelope, EventSource
from self_cognition.core.errors import ContractValidationError
from self_cognition.core.scopes import DataScope, DisclosureScope, SubjectScope
from self_cognition.runtime.run_context import RunContext
import self_cognition.tools.executor as executor_module
from self_cognition.tools.executor import FileReadToolExecutor, ToolExecutionPolicy


NOW = datetime(2026, 9, 8, 10, tzinfo=timezone.utc)
MIND = SubjectScope.for_mind("mind-23")


@dataclass
class FixedClock:
    value: datetime = NOW

    def now(self) -> datetime:
        return self.value


def context(*, deadline: datetime = NOW + timedelta(minutes=5), cancelled: bool = False):
    return RunContext(uuid4(), uuid4(), deadline, cancelled=cancelled, clock=FixedClock())


def action() -> tuple[ActionRequest, ActionDecision]:
    request = ActionRequest(
        uuid4(),
        uuid4(),
        MIND,
        uuid4(),
        1,
        "read-file",
        "file.read",
        {"path": "notes.txt"},
        (),
        "idempotent-file-read",
        NOW,
    )
    return request, ActionDecision(
        uuid4(),
        request.action_id,
        ActionDecisionStatus.ALLOWED,
        "read the requested file",
        ("complete the active task",),
        "current user request",
        ("bounded read",),
        (),
        NOW,
        NOW + timedelta(minutes=30),
        request.action_id,
    )


def append_decision(app, request: ActionRequest, decision: ActionDecision) -> None:
    app.event_store.append(
        EventEnvelope(
            uuid4(),
            "action.decided",
            None,
            MIND,
            ActionDecisionPayload(request, decision),
            NOW,
            NOW,
            EventSource.SYSTEM,
            DataScope(MIND, DisclosureScope.MIND),
            causation_id=request.proposal_request_id,
        )
    )


def test_file_read_is_scoped_and_action_result_is_idempotent(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "notes.txt").write_text("hello", encoding="utf-8")
    executor = FileReadToolExecutor(
        ToolExecutionPolicy((allowed,), temp_root=tmp_path / "sandboxes")
    )
    app = build_container(tmp_path / "data", tool_executor=executor)
    assert [item.capability_id for item in app.capability_registry.registrations()] == [
        "file.read"
    ]
    request, decision = action()
    append_decision(app, request, decision)

    first = app.action.execute(request, decision, context())
    second = app.action.execute(request, decision, context())
    cancelled_retry = app.action.execute(
        request,
        decision,
        context(cancelled=True),
    )

    assert first.result is not None
    assert first.result.status is ActionResultStatus.SUCCEEDED
    assert first.result.output == {
        "path": "notes.txt",
        "content": "hello",
        "bytes_read": 5,
    }
    assert second.reused is True
    assert cancelled_retry.reused is True
    assert [
        event.event_type
        for event in app.event_store.read_by_subject(MIND)
        if event.event_type == "action.result"
    ] == ["action.result"]
    assert not list((tmp_path / "sandboxes").iterdir())


@pytest.mark.parametrize(
    ("path", "expected_error"),
    (("../outside.txt", "ContractValidationError"), ("large.txt", "OutputLimitExceeded")),
)
def test_file_read_rejects_scope_escape_and_output_over_limit(
    tmp_path: Path,
    path: str,
    expected_error: str,
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (tmp_path / "outside.txt").write_text("outside", encoding="utf-8")
    (allowed / "large.txt").write_text("0123456789", encoding="utf-8")
    executor = FileReadToolExecutor(
        ToolExecutionPolicy((allowed,), max_output_bytes=5)
    )
    request, decision = action()
    request = ActionRequest(
        request.action_id,
        request.proposal_request_id,
        request.owner,
        request.plan_id,
        request.plan_version,
        request.step_id,
        request.tool_id,
        {"path": path},
        request.expected_side_effects,
        request.idempotency_key,
        request.requested_at,
    )

    result = executor.execute(request, decision, context())

    assert result.status is ActionResultStatus.FAILED
    assert result.error_type == expected_error


def test_file_read_records_cancelled_and_timed_out_results(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "notes.txt").write_text("hello", encoding="utf-8")
    executor = FileReadToolExecutor(ToolExecutionPolicy((allowed,)))

    request, decision = action()
    cancelled = executor.execute(request, decision, context(cancelled=True))
    timed_out = executor.execute(
        request,
        decision,
        context(deadline=NOW),
    )

    assert cancelled.status is ActionResultStatus.CANCELLED
    assert timed_out.status is ActionResultStatus.TIMED_OUT
    assert timed_out.error_type == "ToolTimeout"


def test_file_read_rejects_expired_decisions_and_declared_side_effects(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    executor = FileReadToolExecutor(ToolExecutionPolicy((allowed,)))
    request, decision = action()

    expired = replace(
        decision,
        decided_at=NOW - timedelta(minutes=2),
        valid_until=NOW - timedelta(seconds=1),
    )
    with pytest.raises(ContractValidationError, match="expired"):
        executor.execute(request, expired, context())

    side_effect_request = ActionRequest(
        request.action_id,
        request.proposal_request_id,
        request.owner,
        request.plan_id,
        request.plan_version,
        request.step_id,
        request.tool_id,
        request.arguments,
        (ExpectedSideEffect("read", "declared read", True),),
        request.idempotency_key,
        request.requested_at,
    )
    with pytest.raises(ContractValidationError, match="side effects"):
        executor.execute(side_effect_request, decision, context())


def test_file_read_records_sandbox_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "notes.txt").write_text("hello", encoding="utf-8")
    executor = FileReadToolExecutor(ToolExecutionPolicy((allowed,)))
    request, decision = action()

    def fail_cleanup(path: Path) -> None:
        raise OSError("cleanup failed")

    monkeypatch.setattr(executor_module.shutil, "rmtree", fail_cleanup)
    result = executor.execute(request, decision, context())

    assert result.status is ActionResultStatus.SUCCEEDED
    assert result.output["cleanup_error"] == "OSError"
