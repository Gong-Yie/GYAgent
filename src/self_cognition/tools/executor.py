from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from urllib.error import URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from datetime import datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Mapping, Protocol
from uuid import UUID, uuid5

from self_cognition.core.actions import (
    ActionDecision,
    ActionDecisionStatus,
    ActionRequest,
    ActionResult,
    ActionResultStatus,
    ExpectedSideEffect,
    ToolDescriptor,
)
from self_cognition.core.errors import ContractValidationError
from self_cognition.core.identity import CapabilityKind, CapabilityPermission
from self_cognition.infrastructure.temp_paths import create_private_temp_dir
from self_cognition.runtime.run_context import RunContext
from self_cognition.tools.registry import CapabilityRegistration


@dataclass(frozen=True, slots=True)
class ToolExecutionPolicy:
    allowed_roots: tuple[Path, ...]
    network_enabled: bool = False
    timeout_seconds: float = 300.0
    max_processes: int = 1
    max_output_bytes: int = 64 * 1024
    temp_root: Path | None = None
    environment_allowlist: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.allowed_roots:
            raise ContractValidationError("tool execution requires an allowed root")
        if any(not isinstance(root, Path) for root in self.allowed_roots):
            raise ContractValidationError("allowed roots must be pathlib paths")
        if self.timeout_seconds <= 0:
            raise ContractValidationError("tool timeout must be positive")
        if self.timeout_seconds > 600.0:
            raise ContractValidationError("tool timeout must not exceed 600 seconds")
        if self.max_processes < 1:
            raise ContractValidationError("max_processes must be positive")
        if self.max_output_bytes < 1:
            raise ContractValidationError("max_output_bytes must be positive")
        if any(
            not isinstance(name, str) or not name.strip()
            for name in self.environment_allowlist
        ):
            raise ContractValidationError("environment allowlist contains invalid names")

    @property
    def resolved_roots(self) -> tuple[Path, ...]:
        return tuple(root.resolve() for root in self.allowed_roots)

    def timeout_for(self, arguments: Mapping[str, object]) -> float:
        value = arguments.get("timeout", self.timeout_seconds)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractValidationError("tool timeout must be numeric")
        if value <= 0 or value > 600.0:
            raise ContractValidationError(
                "tool timeout must be between 0 and 600 seconds"
            )
        return min(float(value), self.timeout_seconds)


class RunSandbox:
    def __init__(self, policy: ToolExecutionPolicy, run_id: UUID) -> None:
        self._policy = policy
        self._run_id = run_id
        self.path: Path | None = None
        self.cleanup_error: str | None = None

    def __enter__(self) -> "RunSandbox":
        self.path = create_private_temp_dir(
            prefix=f"run-{self._run_id}-",
            directory=self._policy.temp_root,
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if self.path is None:
            return False
        try:
            shutil.rmtree(self.path)
        except OSError as error:
            self.cleanup_error = type(error).__name__
        return False


class ToolExecutor(Protocol):
    def execute(
        self,
        action: ActionRequest,
        decision: ActionDecision,
        context: RunContext,
    ) -> ActionResult: ...


@dataclass(slots=True)
class ToolRouterExecutor:
    executors: Mapping[str, ToolExecutor]

    def execute(
        self,
        action: ActionRequest,
        decision: ActionDecision,
        context: RunContext,
    ) -> ActionResult:
        executor = self.executors.get(action.tool_id)
        if executor is None:
            raise ContractValidationError(f"no executor is registered for {action.tool_id}")
        return executor.execute(action, decision, context)


@dataclass(slots=True)
class WorkspaceShellExecutor:
    policy: ToolExecutionPolicy
    tool_id: str = "workspace.shell"

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(self.tool_id, "Workspace shell", "Run a bounded command in workspace", {"type":"object","properties":{"command":{"type":"string"},"timeout":{"type":"number"}},"required":["command"],"additionalProperties":False}, {"type":"object"}, ())

    @property
    def registration(self) -> CapabilityRegistration:
        return CapabilityRegistration(self.tool_id, self.descriptor.name, CapabilityKind.TOOL, CapabilityPermission.GRANTED, description=self.descriptor.description, input_schema=dict(self.descriptor.input_schema), output_schema=dict(self.descriptor.output_schema))

    def execute(self, action: ActionRequest, decision: ActionDecision, context: RunContext) -> ActionResult:
        if action.tool_id != self.tool_id or decision.status is not ActionDecisionStatus.ALLOWED:
            raise ContractValidationError("shell action is not allowed")
        if decision.one_time_scope != action.action_id:
            raise ContractValidationError("shell decision scope does not match action")
        if context.cancelled:
            return ActionResult(uuid5(action.action_id, "tool-result"), action.action_id, action.owner, ActionResultStatus.CANCELLED, "shell command cancelled", None, (), context.clock.now())
        command = action.arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ContractValidationError("shell command is required")
        if any(token.lower() in {"del", "erase", "format", "shutdown", "reboot", "sudo", "rm"} for token in command.split()):
            raise ContractValidationError("shell command is blocked")
        try:
            completed = subprocess.run(command, cwd=self.policy.resolved_roots[0], shell=True, capture_output=True, text=True, timeout=self.policy.timeout_for(action.arguments), check=False)
            output = {"returncode": completed.returncode, "stdout": completed.stdout[: self.policy.max_output_bytes], "stderr": completed.stderr[: self.policy.max_output_bytes]}
            return ActionResult(uuid5(action.action_id, "tool-result"), action.action_id, action.owner, ActionResultStatus.SUCCEEDED if completed.returncode == 0 else ActionResultStatus.FAILED, "shell command completed", output, (), context.clock.now(), None if completed.returncode == 0 else "CommandFailed")
        except subprocess.TimeoutExpired:
            return ActionResult(uuid5(action.action_id, "tool-result"), action.action_id, action.owner, ActionResultStatus.TIMED_OUT, "shell command timed out", None, (), context.clock.now(), "ToolTimeout")


@dataclass(slots=True)
class WorkspaceWebSearchExecutor:
    policy: ToolExecutionPolicy
    tool_id: str = "workspace.web_search"

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(self.tool_id, "Web search", "Search the public web", {"type":"object","properties":{"query":{"type":"string"},"timeout":{"type":"number"}},"required":["query"],"additionalProperties":False}, {"type":"object"}, ())

    @property
    def registration(self) -> CapabilityRegistration:
        return CapabilityRegistration(self.tool_id, self.descriptor.name, CapabilityKind.TOOL, CapabilityPermission.GRANTED, description=self.descriptor.description, input_schema=dict(self.descriptor.input_schema), output_schema=dict(self.descriptor.output_schema))

    def execute(self, action: ActionRequest, decision: ActionDecision, context: RunContext) -> ActionResult:
        if action.tool_id != self.tool_id or decision.status is not ActionDecisionStatus.ALLOWED:
            raise ContractValidationError("web search action is not allowed")
        if decision.one_time_scope != action.action_id:
            raise ContractValidationError("search decision scope does not match action")
        if context.cancelled:
            return ActionResult(uuid5(action.action_id, "tool-result"), action.action_id, action.owner, ActionResultStatus.CANCELLED, "web search cancelled", None, (), context.clock.now())
        query = action.arguments.get("query")
        if not self.policy.network_enabled or not isinstance(query, str) or not query.strip():
            raise ContractValidationError("web search is disabled or query is invalid")
        try:
            request = Request("https://www.baidu.com/s?wd=" + quote_plus(query), headers={"User-Agent": "self-cognition-agent/1.0"})
            with urlopen(request, timeout=self.policy.timeout_for(action.arguments)) as response:
                content = response.read(self.policy.max_output_bytes).decode("utf-8", errors="replace")
            return ActionResult(uuid5(action.action_id, "tool-result"), action.action_id, action.owner, ActionResultStatus.SUCCEEDED, "web search completed", {"query": query, "content": content}, (), context.clock.now())
        except (TimeoutError, URLError) as error:
            if isinstance(error, URLError) and not isinstance(error.reason, TimeoutError):
                return ActionResult(uuid5(action.action_id, "tool-result"), action.action_id, action.owner, ActionResultStatus.FAILED, "web search failed", None, (), context.clock.now(), type(error.reason).__name__)
            return ActionResult(uuid5(action.action_id, "tool-result"), action.action_id, action.owner, ActionResultStatus.TIMED_OUT, "web search timed out", None, (), context.clock.now(), "ToolTimeout")


@dataclass(slots=True)
class FileReadToolExecutor:
    policy: ToolExecutionPolicy
    tool_id: str = "file.read"
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            self.tool_id,
            "Read file",
            "Read UTF-8 text from an explicitly allowed root.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "timeout": {"type": "number"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "bytes_read": {"type": "integer"},
                },
                "required": ["path", "content", "bytes_read"],
                "additionalProperties": False,
            },
            (),
        )

    @property
    def registration(self) -> CapabilityRegistration:
        descriptor = self.descriptor
        return CapabilityRegistration(
            descriptor.tool_id,
            descriptor.name,
            CapabilityKind.TOOL,
            CapabilityPermission.GRANTED,
            description=descriptor.description,
            input_schema=dict(descriptor.input_schema),
            output_schema=dict(descriptor.output_schema),
            expected_side_effects=descriptor.expected_side_effects,
        )

    def execute(
        self,
        action: ActionRequest,
        decision: ActionDecision,
        context: RunContext,
    ) -> ActionResult:
        with self._lock:
            self._validate_request(action, decision, context)
            execution_deadline = min(
                context.deadline,
                context.clock.now()
                + timedelta(seconds=self.policy.timeout_for(action.arguments)),
            )
            result: ActionResult
            with RunSandbox(self.policy, context.run_id) as sandbox:
                try:
                    path = self._resolve_path(action.arguments)
                    self._ensure_active(context, execution_deadline)
                    if path.stat().st_size > self.policy.max_output_bytes:
                        result = self._failure(
                            action,
                            context,
                            "OutputLimitExceeded",
                            f"file output exceeds {self.policy.max_output_bytes} bytes",
                        )
                    else:
                        content = path.read_text(encoding="utf-8")
                        size = len(content.encode("utf-8"))
                        if size > self.policy.max_output_bytes:
                            result = self._failure(
                                action,
                                context,
                                "OutputLimitExceeded",
                                f"file output exceeds {self.policy.max_output_bytes} bytes",
                            )
                        else:
                            self._ensure_active(context, execution_deadline)
                            result = ActionResult(
                                uuid5(action.action_id, "tool-result"),
                                action.action_id,
                                action.owner,
                                ActionResultStatus.SUCCEEDED,
                                "file read completed",
                                {
                                    "path": action.arguments["path"],
                                    "content": content,
                                    "bytes_read": size,
                                },
                                (),
                                context.clock.now(),
                            )
                except _ExecutionCancelled:
                    result = ActionResult(
                        uuid5(action.action_id, "tool-result"),
                        action.action_id,
                        action.owner,
                        ActionResultStatus.CANCELLED,
                        "file read cancelled",
                        None,
                        (),
                        context.clock.now(),
                    )
                except _ExecutionTimedOut:
                    result = ActionResult(
                        uuid5(action.action_id, "tool-result"),
                        action.action_id,
                        action.owner,
                        ActionResultStatus.TIMED_OUT,
                        "file read timed out",
                        None,
                        (),
                        context.clock.now(),
                        "ToolTimeout",
                    )
                except ContractValidationError as error:
                    result = self._failure(
                        action,
                        context,
                        type(error).__name__,
                        str(error),
                    )
                except UnicodeDecodeError:
                    result = self._failure(
                        action,
                        context,
                        "InvalidUtf8",
                        "file is not valid UTF-8 text",
                    )
                except FileNotFoundError:
                    result = self._failure(
                        action,
                        context,
                        "FileNotFound",
                        "file does not exist",
                    )
                except IsADirectoryError:
                    result = self._failure(
                        action,
                        context,
                        "NotAFile",
                        "path is not a file",
                    )
                except OSError as error:
                    result = self._failure(
                        action,
                        context,
                        type(error).__name__,
                        "file read failed",
                    )
            if sandbox.cleanup_error is not None:
                result = self._with_cleanup_error(result, sandbox.cleanup_error)
            return result

    def _resolve_path(self, arguments: Mapping[str, object]) -> Path:
        if set(arguments) - {"path", "timeout"} or not isinstance(
            arguments.get("path"), str
        ):
            raise ContractValidationError("file.read arguments must contain only path")
        relative = Path(arguments["path"])
        if relative.is_absolute() or not str(relative).strip():
            raise ContractValidationError("file.read path must be relative")
        for root in self.policy.resolved_roots:
            candidate = (root / relative).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                continue
            return candidate
        raise ContractValidationError("file.read path is outside allowed roots")

    def _validate_request(
        self,
        action: ActionRequest,
        decision: ActionDecision,
        context: RunContext,
    ) -> None:
        if action.tool_id != self.tool_id:
            raise ContractValidationError("file executor received another tool")
        if decision.action_id != action.action_id:
            raise ContractValidationError("action decision does not match action")
        if decision.status is not ActionDecisionStatus.ALLOWED:
            raise ContractValidationError("only an allowed action may execute")
        if decision.one_time_scope != action.action_id:
            raise ContractValidationError("action decision scope does not match action")
        if action.expected_side_effects:
            raise ContractValidationError("file.read cannot declare side effects")
        if context.clock.now() > decision.valid_until:
            raise ContractValidationError("action decision has expired")

    def _ensure_active(
        self,
        context: RunContext,
        execution_deadline: datetime | None = None,
    ) -> None:
        if context.cancelled:
            raise _ExecutionCancelled
        if context.clock.now() >= min(
            context.deadline,
            execution_deadline or context.deadline,
        ):
            raise _ExecutionTimedOut

    @staticmethod
    def _failure(
        action: ActionRequest,
        context: RunContext,
        error_type: str,
        summary: str,
    ) -> ActionResult:
        return ActionResult(
            uuid5(action.action_id, "tool-result"),
            action.action_id,
            action.owner,
            ActionResultStatus.FAILED,
            summary,
            None,
            (),
            context.clock.now(),
            error_type,
        )

    @staticmethod
    def _with_cleanup_error(result: ActionResult, error_type: str) -> ActionResult:
        output = result.output
        if isinstance(output, dict):
            output = {**output, "cleanup_error": error_type}
        else:
            output = {"cleanup_error": error_type, "result": output}
        return ActionResult(
            result.result_id,
            result.action_id,
            result.owner,
            result.status,
            result.summary,
            output,
            result.actual_side_effects,
            result.recorded_at,
            result.error_type,
        )


class _ExecutionCancelled(Exception):
    pass


class _ExecutionTimedOut(Exception):
    pass
