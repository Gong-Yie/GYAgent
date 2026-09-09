"""Compatibility entry point for the stateless workspace builder."""

from self_cognition.core.workspace import (
    WorkspaceBuilder,
    WorkspaceFixedContext,
    WorkspacePacket,
    WorkspaceRunInfo,
    estimate_tokens,
    workspace_model_context,
)

__all__ = [
    "WorkspaceBuilder",
    "WorkspaceFixedContext",
    "WorkspacePacket",
    "WorkspaceRunInfo",
    "estimate_tokens",
    "workspace_model_context",
]
