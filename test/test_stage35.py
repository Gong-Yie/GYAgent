from pathlib import Path
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest

from self_cognition.core.errors import ContractValidationError
from self_cognition.core.actions import ActionDecisionStatus
from self_cognition.core.scopes import SubjectScope
from self_cognition.tools.executor import (
    ToolExecutionPolicy,
    ToolRouterExecutor,
    WorkspaceShellExecutor,
)
from self_cognition.bootstrap import build_container


def test_tool_timeout_defaults_and_caps(tmp_path: Path) -> None:
    policy = ToolExecutionPolicy((tmp_path,))
    assert policy.timeout_for({}) == 300.0
    assert policy.timeout_for({"timeout": 10}) == 10.0
    with pytest.raises(ContractValidationError):
        policy.timeout_for({"timeout": 601})


def test_shell_blocks_destructive_commands(tmp_path: Path) -> None:
    executor = WorkspaceShellExecutor(ToolExecutionPolicy((tmp_path,)))
    action_id = uuid4()
    action = SimpleNamespace(tool_id="workspace.shell", action_id=action_id, arguments={"command": "del file.txt"})
    decision = SimpleNamespace(status=ActionDecisionStatus.ALLOWED, one_time_scope=action_id)
    with pytest.raises(ContractValidationError):
        executor.execute(action, decision, SimpleNamespace(cancelled=False))


def test_router_rejects_unknown_tool() -> None:
    router = ToolRouterExecutor({})
    with pytest.raises(ContractValidationError):
        router.execute(SimpleNamespace(tool_id="missing"), SimpleNamespace(), SimpleNamespace())


def test_shell_returns_cancelled_before_side_effect(tmp_path: Path) -> None:
    action_id = uuid4()
    owner = SubjectScope.for_mind("test-mind")
    action = SimpleNamespace(tool_id="workspace.shell", action_id=action_id, owner=owner, arguments={"command": "echo ok"})
    decision = SimpleNamespace(status=ActionDecisionStatus.ALLOWED, one_time_scope=action_id)
    context = SimpleNamespace(cancelled=True, clock=SimpleNamespace(now=lambda: __import__("datetime").datetime.now(__import__("datetime").timezone.utc)))
    result = WorkspaceShellExecutor(ToolExecutionPolicy((tmp_path,))).execute(action, decision, context)
    assert result.status.value == "cancelled"


def test_shell_enforces_actual_timeout(tmp_path: Path) -> None:
    action_id = uuid4()
    owner = SubjectScope.for_mind("timeout-mind")
    action = SimpleNamespace(
        tool_id="workspace.shell",
        action_id=action_id,
        owner=owner,
        arguments={"command": f'"{sys.executable}" -c "import time; time.sleep(2)"', "timeout": 0.1},
    )
    decision = SimpleNamespace(status=ActionDecisionStatus.ALLOWED, one_time_scope=action_id)
    now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
    context = SimpleNamespace(cancelled=False, clock=SimpleNamespace(now=lambda: now))
    result = WorkspaceShellExecutor(ToolExecutionPolicy((tmp_path,))).execute(action, decision, context)
    assert result.status.value == "timed_out"


def test_default_container_registers_workspace_tools(tmp_path: Path) -> None:
    app = build_container(tmp_path / "data")
    try:
        assert {item.capability_id for item in app.capability_registry.registrations()} >= {
            "workspace.shell",
            "workspace.web_search",
        }
    finally:
        app.lifecycle.stop()
