"""
MCP Integration - Unified Tool Registry with MCP Support

Simplifies MCP integration by treating MCP tools the same as built-in tools.
"""

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from src.core.mcp_client import MCPClient, MCPConnection, MCPServerConfig, MCPTransport

logger = logging.getLogger(__name__)


class MCPToolRegistry:
    """
    Unified registry combining built-in tools with MCP server tools.

    Usage:
        from src.host.mcp_registry import MCPToolRegistry

        # Create and initialize
        registry = MCPToolRegistry()
        await registry.load_from_config(config_path)
        await registry.connect_all()

        # Call any tool (built-in or MCP)
        result = await registry.call_tool("read", {"path": "file.py"})
    """

    def __init__(self, built_in_registry=None):
        """Initialize with optional built-in registry."""
        self._built_in = built_in_registry
        self._mcp_client = MCPClient()
        self._mcp_tools: dict[str, str] = {}  # tool_name -> server_name
        self._tool_schemas: dict[str, dict] = {}  # tool_name -> schema
        self._sensitive_tools: set[str] = set()

    def load_from_config(self, path: Path) -> int:
        """
        Load MCP server configurations from JSON file.

        Returns:
            Number of servers configured
        """
        if not path.exists():
            logger.debug(f"Config file not found: {path}")
            return 0

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            servers = data.get("mcpServers", {})

            for name, server_data in servers.items():
                args = self._expand_env_vars(server_data.get("args", []))
                env = {
                    k: self._expand_env_var(str(v)) for k, v in server_data.get("env", {}).items()
                }

                transport = MCPTransport.STDIO
                if server_data.get("type") == "http" or server_data.get("url"):
                    transport = MCPTransport.HTTP

                config = MCPServerConfig(
                    name=name,
                    command=server_data.get("command", ""),
                    args=args,
                    env=env,
                    transport=transport,
                    url=server_data.get("url"),
                    timeout=server_data.get("timeout", 30.0),
                )
                self._mcp_client.add_server(config)

            logger.info(f"Loaded {len(servers)} MCP server configs")
            return len(servers)

        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}")
            return 0

    def _expand_env_vars(self, values: list[str]) -> list[str]:
        """Expand environment variables in a list."""
        return [self._expand_env_var(v) for v in values]

    def _expand_env_var(self, value: str) -> str:
        """Expand ${VAR} and ${VAR:-default} in a string."""

        def replace_var(match):
            var_expr = match.group(1)
            if ":-" in var_expr:
                var_name, default = var_expr.split(":-", 1)
                return os.environ.get(var_name, default)
            return os.environ.get(var_expr, "")

        return re.sub(r"\$\{([^}]+)\}", replace_var, value)

    async def connect_all(self) -> dict[str, bool]:
        """Connect to all configured MCP servers."""
        results = await self._mcp_client.connect_all()

        for server_name, success in results.items():
            if success:
                await self._discover_tools(server_name)

        return results

    async def _discover_tools(self, server_name: str) -> int:
        """Discover tools from a connected server."""
        connection = self._mcp_client._connections.get(server_name)
        if not connection:
            return 0

        try:
            tools = await connection.list_tools()
            for tool in tools:
                self._mcp_tools[tool.name] = server_name
                self._tool_schemas[tool.name] = {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                    "server": server_name,
                }

                sensitive_keywords = ["delete", "write", "execute", "run", "remove"]
                if any(kw in tool.name.lower() for kw in sensitive_keywords):
                    self._sensitive_tools.add(tool.name)

            logger.info(f"Discovered {len(tools)} tools from {server_name}")
            return len(tools)

        except Exception as e:
            logger.error(f"Failed to discover tools from {server_name}: {e}")
            return 0

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """
        Call a tool by name.

        Automatically routes to MCP server or built-in registry.
        """
        if name in self._mcp_tools:
            return await self._call_mcp_tool(name, arguments)

        if self._built_in:
            return await self._built_in.call_tool(name, arguments)

        raise ValueError(f"Tool not found: {name}")

    async def _call_mcp_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call an MCP tool."""
        server_name = self._mcp_tools.get(name)
        if not server_name:
            return f"Error: MCP tool {name} not found"

        try:
            result = await self._mcp_client.call_tool(server_name, name, arguments)
            return str(result) if result else "Success"
        except Exception as e:
            logger.error(f"MCP tool {name} failed: {e}")
            return f"Error: {e}"

    def get_tool_names(self) -> list[str]:
        """Get all tool names (built-in + MCP)."""
        names = set(self._mcp_tools.keys())
        if self._built_in:
            names.update(self._built_in.get_tool_names())
        return list(names)

    def get_genai_tools(self, tool_names: list[str] | None = None) -> list:
        """Get all tools as GenAI FunctionDeclarations."""
        from google.genai import types

        declarations = []

        if self._built_in:
            declarations.extend(self._built_in.get_genai_tools(tool_names))

        for name, schema in self._tool_schemas.items():
            if tool_names and name not in tool_names:
                continue

            declarations.append(
                types.FunctionDeclaration(
                    name=schema["name"],
                    description=schema["description"],
                    parameters=schema["parameters"],
                )
            )

        return declarations

    def is_sensitive(self, name: str) -> bool:
        """Check if a tool is sensitive."""
        if name in self._sensitive_tools:
            return True
        if self._built_in:
            return self._built_in.is_sensitive(name)
        return False

    def get_sensitive_tools(self) -> list[str]:
        """Get all sensitive tools."""
        tools = list(self._sensitive_tools)
        if self._built_in:
            tools.extend(self._built_in.get_sensitive_tools())
        return tools

    def mark_sensitive(self, name: str, sensitive: bool = True) -> None:
        """Mark a tool as sensitive."""
        if sensitive:
            self._sensitive_tools.add(name)
        else:
            self._sensitive_tools.discard(name)

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        await self._mcp_client.disconnect_all()

    def get_summary(self) -> dict[str, Any]:
        """Get registry summary."""
        return {
            "mcp_tools": len(self._mcp_tools),
            "mcp_servers": list(self._mcp_client._servers.keys()),
            "connected_servers": self._mcp_client.get_connected_servers(),
            "built_in_tools": len(self._built_in.get_tool_names()) if self._built_in else 0,
        }


async def create_unified_registry(config_path: Path | None = None) -> MCPToolRegistry:
    """
    Create a unified registry with built-in and MCP tools.

    Args:
        config_path: Path to MCP config file (defaults to .agent/config.json)

    Returns:
        Initialized MCPToolRegistry
    """
    from src.host.tools import get_default_registry

    built_in = get_default_registry()
    registry = MCPToolRegistry(built_in_registry=built_in)

    if config_path is None:
        config_path = Path.cwd() / ".agent" / "config.json"

    if config_path.exists():
        registry.load_from_config(config_path)
        await registry.connect_all()

    return registry
