"""Shared tool governance decisions for visibility and execution policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.core.tool_selector import (
    get_capability_group_for_tool,
    get_capability_groups_for_mode,
    get_tools_for_mode,
    get_visible_modes_for_tool,
)


@dataclass(slots=True)
class ToolPolicy:
    """Product-facing policy for a runtime tool."""

    tool_name: str
    visible: bool
    visible_modes: list[str]
    requires_approval: bool
    sandbox: str
    audit_level: str
    background_safe: bool
    capability_group: str | None
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ToolPolicyEngine:
    """Resolve visibility and execution policy for a tool in a runtime context."""

    WRITE_LIKE_TOOLS = {
        "write",
        "multi_edit",
        "notebook_edit",
        "lsp_rename",
        "memory_put",
        "memory_delete",
        "db_write_query",
        "run",
    }
    INTERACTIVE_TOOLS = {"ask_user"}

    def describe_tool(
        self,
        tool_name: str,
        *,
        mode: str | None = None,
        source: str = "built_in",
        sensitive: bool = False,
        metadata: dict[str, Any] | None = None,
        active_mcp_extensions: list[str] | None = None,
    ) -> ToolPolicy:
        metadata = metadata or {}
        capability_group = metadata.get("capability_group") or get_capability_group_for_tool(tool_name)
        visible_modes = self._visible_modes(tool_name, source, capability_group)
        visible = self._is_visible(
            tool_name,
            mode=mode,
            source=source,
            capability_group=capability_group,
            metadata=metadata,
            active_mcp_extensions=active_mcp_extensions,
        )
        sandbox = self._sandbox_class(tool_name, source, capability_group)
        requires_approval = sensitive or sandbox in {"workspace_write", "workspace_exec"}
        audit_level = self._audit_level(tool_name, source, requires_approval)
        background_safe = self._background_safe(tool_name, requires_approval)

        return ToolPolicy(
            tool_name=tool_name,
            visible=visible,
            visible_modes=visible_modes,
            requires_approval=requires_approval,
            sandbox=sandbox,
            audit_level=audit_level,
            background_safe=background_safe,
            capability_group=capability_group,
            source=source,
        )

    def _is_visible(
        self,
        tool_name: str,
        *,
        mode: str | None,
        source: str,
        capability_group: str | None,
        metadata: dict[str, Any],
        active_mcp_extensions: list[str] | None,
    ) -> bool:
        if mode is None:
            return True

        if tool_name in get_tools_for_mode(mode):
            return True

        if capability_group is not None and capability_group in get_capability_groups_for_mode(mode):
            return True

        active_extensions = active_mcp_extensions or []
        if source == "mcp_extension":
            extension_group = metadata.get("extension_group")
            if extension_group is None:
                return mode == "build"
            return bool(extension_group and extension_group in active_extensions)

        if source == "mcp_remote":
            mcp_server = metadata.get("mcp_server")
            return bool(mcp_server and mcp_server in active_extensions)

        return False

    def _visible_modes(
        self,
        tool_name: str,
        source: str,
        capability_group: str | None,
    ) -> list[str]:
        if source in {"mcp_extension", "mcp_remote"}:
            return ["plan", "build"]
        if capability_group is None:
            return []
        visible_modes = get_visible_modes_for_tool(tool_name)
        if visible_modes:
            return visible_modes
        return [
            mode
            for mode in ("plan", "build")
            if capability_group in get_capability_groups_for_mode(mode)
        ]

    def _sandbox_class(
        self,
        tool_name: str,
        source: str,
        capability_group: str | None,
    ) -> str:
        if source == "mcp_remote":
            return "mcp_remote"
        if source == "mcp_extension":
            return "extension"
        if tool_name == "run":
            return "workspace_exec"
        if capability_group == "edit" or tool_name in self.WRITE_LIKE_TOOLS:
            return "workspace_write"
        if capability_group == "research":
            return "network_read"
        return "workspace_read"

    def _audit_level(
        self,
        tool_name: str,
        source: str,
        requires_approval: bool,
    ) -> str:
        if source in {"mcp_remote", "mcp_extension"}:
            return "full"
        if requires_approval or tool_name in self.WRITE_LIKE_TOOLS:
            return "full"
        return "basic"

    def _background_safe(
        self,
        tool_name: str,
        requires_approval: bool,
    ) -> bool:
        if tool_name in self.INTERACTIVE_TOOLS:
            return False
        return not requires_approval


_default_policy_engine: ToolPolicyEngine | None = None


def get_default_tool_policy_engine() -> ToolPolicyEngine:
    """Return the shared runtime tool policy engine."""
    global _default_policy_engine
    if _default_policy_engine is None:
        _default_policy_engine = ToolPolicyEngine()
    return _default_policy_engine
