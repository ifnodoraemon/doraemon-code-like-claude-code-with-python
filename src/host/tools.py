"""
Simplified Tool Registry - Direct Function Calls

This module provides a simplified alternative to MCP server architecture.
Instead of spawning subprocess and communicating via JSON-RPC, it directly
imports and calls tool functions.

Benefits:
- Zero subprocess overhead (~10ms savings per call)
- Simpler debugging (direct stack traces)
- Easier testing (just call the function)
- ~80% less code complexity

Usage:
    from src.host.tools import ToolRegistry, get_default_registry

    registry = get_default_registry()

    # Get tools for GenAI
    genai_tools = registry.get_genai_tools()

    # Call a tool
    result = await registry.call_tool("read_file", {"path": "main.py"})
"""

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from google.genai import types

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """Definition of a tool."""

    name: str
    description: str
    function: Callable
    parameters: dict[str, Any]
    sensitive: bool = False  # Requires HITL approval


class ToolRegistry:
    """
    Simple tool registry with direct function calls.

    Example:
        registry = ToolRegistry()
        registry.register(read_file, sensitive=False)
        registry.register(write_file, sensitive=True)

        result = await registry.call_tool("read_file", {"path": "main.py"})
    """

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        func: Callable,
        name: str | None = None,
        description: str | None = None,
        sensitive: bool = False,
    ) -> None:
        """
        Register a tool function.

        Args:
            func: The tool function
            name: Tool name (defaults to function name)
            description: Tool description (defaults to docstring)
            sensitive: Whether this tool requires HITL approval
        """
        tool_name = name or func.__name__
        tool_desc = description or (func.__doc__ or "").strip()

        # Extract parameters from function signature
        parameters = self._extract_parameters(func)

        self._tools[tool_name] = ToolDefinition(
            name=tool_name,
            description=tool_desc,
            function=func,
            parameters=parameters,
            sensitive=sensitive,
        )

        logger.debug(f"Registered tool: {tool_name}")

    def _extract_parameters(self, func: Callable) -> dict[str, Any]:
        """Extract JSON Schema parameters from function signature."""
        import types
        import typing

        sig = inspect.signature(func)
        properties = {}
        required = []

        type_mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            # Get type annotation
            param_type = param.annotation
            json_type = "string"  # Default

            if param_type != inspect.Parameter.empty:
                # Handle Python 3.10+ Union types (int | None)
                if isinstance(param_type, types.UnionType):
                    args = getattr(param_type, "__args__", ())
                    for arg in args:
                        if arg is not type(None):
                            json_type = type_mapping.get(arg, "string")
                            break
                # Handle typing.Union and typing.Optional
                elif getattr(param_type, "__origin__", None) is typing.Union:
                    args = getattr(param_type, "__args__", ())
                    for arg in args:
                        if arg is not type(None):
                            json_type = type_mapping.get(arg, "string")
                            break
                else:
                    json_type = type_mapping.get(param_type, "string")

            properties[param_name] = {"type": json_type}

            # Check if required (no default value)
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def get_genai_tools(self) -> list[types.FunctionDeclaration]:
        """Get tools as GenAI FunctionDeclarations."""
        declarations = []

        for tool in self._tools.values():
            decl = types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
            )
            declarations.append(decl)

        return declarations

    def get_tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    def get_sensitive_tools(self) -> list[str]:
        """Get list of sensitive tool names."""
        return [name for name, tool in self._tools.items() if tool.sensitive]

    def is_sensitive(self, tool_name: str) -> bool:
        """Check if a tool is sensitive."""
        tool = self._tools.get(tool_name)
        return tool.sensitive if tool else False

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """
        Call a tool by name.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool result as string

        Raises:
            ValueError: If tool not found
        """
        tool = self._tools.get(name)
        if not tool:
            available = ", ".join(self._tools.keys())
            raise ValueError(f"Tool '{name}' not found. Available: {available}")

        try:
            # Call the function
            result = tool.function(**arguments)

            # Handle async functions
            if inspect.iscoroutine(result):
                result = await result

            return str(result) if result is not None else "Success"

        except Exception as e:
            logger.error(f"Tool {name} failed: {e}")
            return f"Error: {e}"


# ========================================
# Default Registry with All Tools
# ========================================

_default_registry: ToolRegistry | None = None


def get_default_registry() -> ToolRegistry:
    """
    Get the default tool registry with all standard tools registered.

    This is a singleton that lazily initializes on first access.
    """
    global _default_registry

    if _default_registry is None:
        _default_registry = _create_default_registry()

    return _default_registry


def _create_default_registry() -> ToolRegistry:
    """Create and populate the default tool registry."""
    registry = ToolRegistry()

    # Import tool functions directly from servers
    # These are the actual implementations, bypassing MCP

    try:
        # File Reading Tools
        from src.servers.fs_read import (
            find_symbol,
            glob_files,
            grep_search,
            list_directory,
            read_file,
            read_file_outline,
        )

        registry.register(read_file, sensitive=False)
        registry.register(read_file_outline, sensitive=False)
        registry.register(list_directory, sensitive=False)
        registry.register(glob_files, sensitive=False)
        registry.register(grep_search, sensitive=False)
        registry.register(find_symbol, sensitive=False)
    except ImportError as e:
        logger.warning(f"Failed to import fs_read tools: {e}")

    try:
        # File Writing Tools
        from src.servers.fs_write import write_file

        registry.register(write_file, sensitive=True)
    except ImportError as e:
        logger.warning(f"Failed to import fs_write tools: {e}")

    try:
        # File Editing Tools
        from src.servers.fs_edit import edit_file

        registry.register(edit_file, sensitive=True)
    except ImportError as e:
        logger.warning(f"Failed to import fs_edit tools: {e}")

    try:
        # Computer/Execution Tools
        from src.servers.computer import execute_python, install_package

        registry.register(execute_python, sensitive=True)
        registry.register(install_package, sensitive=True)
    except ImportError as e:
        logger.warning(f"Failed to import computer tools: {e}")

    try:
        # Memory Tools
        from src.servers.memory import save_note, search_notes

        registry.register(save_note, sensitive=True)
        registry.register(search_notes, sensitive=False)
    except ImportError as e:
        logger.warning(f"Failed to import memory tools: {e}")

    try:
        # Web Tools
        from src.servers.web import fetch_page, search_internet

        registry.register(fetch_page, name="fetch_url", sensitive=False)
        registry.register(search_internet, name="web_search", sensitive=False)
    except ImportError as e:
        logger.warning(f"Failed to import web tools: {e}")

    try:
        # Task Tools
        from src.servers.task import add_task, list_tasks, update_task_status

        registry.register(add_task, name="task_create", sensitive=False)
        registry.register(list_tasks, name="task_list", sensitive=False)
        registry.register(update_task_status, name="task_update_status", sensitive=False)
    except ImportError as e:
        logger.warning(f"Failed to import task tools: {e}")

    try:
        # Shell Tools
        from src.servers.shell import execute_command, execute_command_background

        registry.register(execute_command, name="shell_execute", sensitive=True)
        registry.register(execute_command_background, name="shell_background", sensitive=True)
    except ImportError as e:
        logger.warning(f"Failed to import shell tools: {e}")

    try:
        # Git Tools
        from src.servers.git import git_add, git_commit, git_diff, git_log, git_status

        registry.register(git_status, sensitive=False)
        registry.register(git_diff, sensitive=False)
        registry.register(git_log, sensitive=False)
        registry.register(git_add, sensitive=False)
        registry.register(git_commit, sensitive=True)
    except ImportError as e:
        logger.warning(f"Failed to import git tools: {e}")

    logger.info(f"Tool registry initialized with {len(registry.get_tool_names())} tools")
    return registry


# ========================================
# Convenience Functions
# ========================================


async def call_tool(name: str, arguments: dict[str, Any]) -> str:
    """Call a tool from the default registry."""
    return await get_default_registry().call_tool(name, arguments)


def get_genai_tools() -> list[types.FunctionDeclaration]:
    """Get all tools as GenAI FunctionDeclarations."""
    return get_default_registry().get_genai_tools()


def is_sensitive_tool(name: str) -> bool:
    """Check if a tool requires HITL approval."""
    return get_default_registry().is_sensitive(name)
