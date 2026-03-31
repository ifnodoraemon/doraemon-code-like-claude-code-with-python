"""Compatibility shim for built-in tools and optional MCP-style extensions.

The default runtime path uses only the in-process built-in registry. Optional
extension tools such as browser/database are kept separate so they do not
pollute the default `plan` / `build` mainline.
"""

from pathlib import Path

from src.core.config.config import load_config
from src.core.config.schema import MCPServerConfig
from src.core.tool_selector import get_tools_for_mode
from src.host.mcp_runtime import RemoteMCPServerConfig, build_remote_mcp_registry
from src.host.tools import ToolRegistry, get_default_registry, get_extension_registry

EXTENSION_GROUPS: dict[str, list[str]] = {
    "browser": [
        "browse_page",
        "take_screenshot",
        "browser_click",
        "browser_fill",
        "browser_evaluate",
        "browser_wait",
        "browser_pdf",
        "browser_get_html",
        "browser_close_page",
        "browser_list_pages",
    ],
    "database": [
        "db_read_query",
        "db_write_query",
        "db_list_tables",
        "db_describe_table",
    ],
}


def _load_raw_config(config_path: Path | None = None) -> dict:
    """Load config data without validation for runtime extension resolution."""
    override_path = str(config_path) if config_path else None
    return load_config(override_path=override_path, validate=False)


def get_enabled_mcp_extensions(config_path: Path | None = None) -> list[str]:
    """Return configured extension groups from project config."""
    try:
        config = _load_raw_config(config_path)
    except Exception:
        return []

    raw = config.get("mcp_extensions") or []
    if not isinstance(raw, list):
        return []

    normalized: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        name = item.strip()
        if not name or name in normalized:
            continue
        normalized.append(name)
    return normalized


def get_enabled_mcp_servers(config_path: Path | None = None) -> list[RemoteMCPServerConfig]:
    """Return enabled remote MCP servers from project config."""
    try:
        config = _load_raw_config(config_path)
    except Exception:
        return []

    raw = config.get("mcp_servers") or []
    if not isinstance(raw, list):
        return []

    servers: list[RemoteMCPServerConfig] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        parsed = MCPServerConfig.model_validate(item)
        if not parsed.enabled:
            continue
        servers.append(
            RemoteMCPServerConfig(
                name=parsed.name,
                transport=parsed.transport,
                url=parsed.url,
                command=parsed.command,
                args=parsed.args,
                env=parsed.env,
                cwd=parsed.cwd,
                headers=parsed.headers,
                timeout_seconds=parsed.timeout_seconds,
                tool_prefix=parsed.tool_prefix,
                enabled=parsed.enabled,
            )
        )
    return servers


def resolve_extension_tools(extension_names: list[str] | None) -> list[str]:
    """Resolve extension group names into concrete tool names."""
    resolved: list[str] = []
    for name in extension_names or []:
        tool_names = EXTENSION_GROUPS.get(name, [name])
        for tool_name in tool_names:
            if tool_name not in resolved:
                resolved.append(tool_name)
    return resolved


async def create_tool_registry(
    config_path: Path | None = None,
    *,
    mode: str | None = None,
    extension_tools: list[str] | None = None,
) -> ToolRegistry:
    """Return the runtime tool registry.

    Args:
        config_path: Kept for backward compatibility; ignored.
        mode: Optional agent mode used to select a minimal tool set.
        extension_tools: Optional browser/db extension tools to attach.
    """
    enabled_extensions = get_enabled_mcp_extensions(config_path)
    enabled_remote_servers = get_enabled_mcp_servers(config_path)
    resolved_extension_tools = resolve_extension_tools(
        extension_tools if extension_tools is not None else enabled_extensions
    )
    tool_names = get_tools_for_mode(mode) if mode else None
    registry = get_default_registry(tool_names)

    merged = ToolRegistry()
    for tool_name, definition in registry._tools.items():
        merged._tools[tool_name] = definition

    active_mcp_extensions: list[str] = []

    if resolved_extension_tools:
        extension_registry = get_extension_registry(resolved_extension_tools)
        for tool_name in extension_registry.get_tool_names():
            if tool_name in merged.get_tool_names():
                continue
            definition = extension_registry._tools[tool_name]
            merged._tools[tool_name] = definition
        for name in extension_tools if extension_tools is not None else enabled_extensions:
            if name not in active_mcp_extensions:
                active_mcp_extensions.append(name)

    if enabled_remote_servers:
        remote_registry, active_remote_servers, remote_clients = await build_remote_mcp_registry(
            enabled_remote_servers
        )
        for tool_name in remote_registry.get_tool_names():
            if tool_name in merged.get_tool_names():
                raise RuntimeError(
                    f"MCP tool name collision: {tool_name}. "
                    "Use tool_prefix in mcp_servers configuration."
                )
            merged._tools[tool_name] = remote_registry._tools[tool_name]
        for name in active_remote_servers:
            if name not in active_mcp_extensions:
                active_mcp_extensions.append(name)
        merged._mcp_clients = remote_clients
        merged._mcp_server_errors = getattr(remote_registry, "_mcp_server_errors", {})

    merged._active_mcp_extensions = active_mcp_extensions
    return merged


async def create_unified_registry(
    config_path: Path | None = None,
    *,
    mode: str | None = None,
    extension_tools: list[str] | None = None,
) -> ToolRegistry:
    """Backward-compatible alias for older callers."""
    return await create_tool_registry(config_path, mode=mode, extension_tools=extension_tools)
