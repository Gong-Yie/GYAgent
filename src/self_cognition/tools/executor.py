from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
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
from self_cognition.runtime.run_context import RunContext
from self_cognition.tools.registry import CapabilityRegistration


@dataclass(frozen=True, slots=True)
class ToolExecutionPolicy:
    allowed_roots: tuple[Path, ...]
    network_enabled: bool = False
    timeout_seconds: float = 30.0
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


class RunSandbox:
    def __init__(self, policy: ToolExecutionPolicy, run_id: UUID) -> None:
        self._policy = policy
        self._run_id = run_id
        self.path: Path | None = None
        self.cleanup_error: str | None = None

    def __enter__(self) -> "RunSandbox":
        temp_root = self._policy.temp_root
        if temp_root is not None:
            temp_root.mkdir(parents=True, exist_ok=True)
        self.path = Path(
            tempfile.mkdtemp(
                prefix=f"run-{self._run_id}-",
                dir=str(temp_root) if temp_root is not None else None,
            )
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
                "properties": {"path": {"type": "string"}},
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
                context.clock.now() + timedelta(seconds=self.policy.timeout_seconds),
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
        if set(arguments) != {"path"} or not isinstance(arguments["path"], str):
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
