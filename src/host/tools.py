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

import asyncio
import inspect
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from google.genai import types

from src.core.config import load_config

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """Definition of a tool."""

    name: str
    description: str
    function: Callable
    parameters: dict[str, Any]
    sensitive: bool = False  # Requires HITL approval
    timeout: float = 60.0    # Default timeout in seconds


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
        timeout: float = 60.0,
    ) -> None:
        """
        Register a tool function.

        Args:
            func: The tool function
            name: Tool name (defaults to function name)
            description: Tool description (defaults to docstring)
            sensitive: Whether this tool requires HITL approval
            timeout: Timeout in seconds (default 60.0)
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
            timeout=timeout,
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

    def get_genai_tools(
        self, tool_names: list[str] | None = None
    ) -> list[types.FunctionDeclaration]:
        """
        Get tools as GenAI FunctionDeclarations.

        Args:
            tool_names: Optional list of tool names to include.
                       If None, returns all tools.
        """
        declarations = []

        for tool in self._tools.values():
            # 如果指定了工具名称列表，只返回列表中的工具
            if tool_names is not None and tool.name not in tool_names:
                continue

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
            if inspect.iscoroutinefunction(tool.function):
                # Async function - use wait_for
                # Async function - use wait_for
                try:
                    result = await asyncio.wait_for(tool.function(**arguments), timeout=tool.timeout)
                except asyncio.TimeoutError:
                    error_msg = f"Tool '{name}' timed out after {tool.timeout} seconds."
                    logger.error(error_msg)
                    return error_msg
            else:
                # Sync function - run directly (cannot timeout easily without threads/processes)
                # For critical safety, we might want to run this in a thread executor with timeout,
                # but for now we'll just run it. Most heavy tools should be async.
                result = tool.function(**arguments)

            return str(result) if result is not None else "Success"

        except Exception as e:
            logger.error(f"Tool {name} failed: {e}", exc_info=True)
            # Return error message for graceful handling in chat loop
            return f"Error executing {name}: {type(e).__name__}: {e}"


# ========================================
# Default Registry with All Tools
# ========================================

_default_registry: ToolRegistry | None = None
_registry_lock = threading.Lock()


def get_default_registry() -> ToolRegistry:
    """
    Get the default tool registry with all standard tools registered.

    This is a singleton that lazily initializes on first access.
    Thread-safe using double-check locking pattern.
    """
    global _default_registry

    if _default_registry is None:
        with _registry_lock:
            # Double-check locking to prevent race conditions
            if _default_registry is None:
                _default_registry = _create_default_registry()

    return _default_registry


def _create_default_registry() -> ToolRegistry:
    """Create and populate the default tool registry."""
    registry = ToolRegistry()

    # Load configuration
    try:
        config = load_config(validate=False)
        tool_timeouts = config.get("tool_timeouts", {})
    except Exception as e:
        logger.warning(f"Failed to load config for tool timeouts: {e}")
        tool_timeouts = {}

    def _get_timeout(tool_name: str, default: float) -> float:
        """Get timeout from config or default."""
        return float(tool_timeouts.get(tool_name, default))

    # Import tool functions directly from servers
    # These are the actual implementations, bypassing MCP

    # Collect failed tool imports
    failed_tools: list[tuple[str, str]] = []

    try:
        # Filesystem Tools (Unified)
        from src.servers.filesystem import (
            edit_file,
            find_symbol,
            glob_files,
            grep_search,
            list_directory,
            read_file,
            read_file_outline,
            write_file,
        )

        registry.register(read_file, sensitive=False, timeout=_get_timeout("read_file", 60.0))
        registry.register(read_file_outline, sensitive=False, timeout=_get_timeout("read_file_outline", 60.0))
        registry.register(list_directory, sensitive=False, timeout=_get_timeout("list_directory", 60.0))
        registry.register(glob_files, sensitive=False, timeout=_get_timeout("glob_files", 60.0))
        registry.register(grep_search, sensitive=False, timeout=_get_timeout("grep_search", 120.0))  # Grep can be slow
        registry.register(find_symbol, sensitive=False, timeout=_get_timeout("find_symbol", 60.0))

        registry.register(write_file, sensitive=True, timeout=_get_timeout("write_file", 60.0))
        registry.register(edit_file, sensitive=True, timeout=_get_timeout("edit_file", 120.0))  # Edits might involve processing
    except ImportError as e:
        logger.error(f"Failed to import filesystem tools: {e}")
        failed_tools.append(("filesystem", str(e)))

    try:
        # Computer/Execution Tools
        from src.servers.computer import execute_python, install_package

        registry.register(execute_python, sensitive=True, timeout=_get_timeout("execute_python", 300.0))  # Allow 5 mins for scripts
        registry.register(install_package, sensitive=True, timeout=_get_timeout("install_package", 300.0))  # Installations take time
    except ImportError as e:
        logger.warning(f"Failed to import computer tools: {e}")

    try:
        # Memory Tools
        from src.servers.memory import save_note, search_notes

        registry.register(save_note, sensitive=True, timeout=_get_timeout("save_note", 60.0))
        registry.register(search_notes, sensitive=False, timeout=_get_timeout("search_notes", 60.0))
    except ImportError as e:
        logger.warning(f"Failed to import memory tools: {e}")

    try:
        # Web Tools
        from src.servers.web import fetch_page, search_internet

        registry.register(fetch_page, name="fetch_url", sensitive=False, timeout=_get_timeout("fetch_url", 30.0))  # Web should be fast
        registry.register(search_internet, name="web_search", sensitive=False, timeout=_get_timeout("web_search", 30.0))
    except ImportError as e:
        logger.warning(f"Failed to import web tools: {e}")

    try:
        # Task Tools
        from src.servers.task import add_task, list_tasks, update_task_status

        registry.register(add_task, name="task_create", sensitive=False, timeout=_get_timeout("task_create", 60.0))
        registry.register(list_tasks, name="task_list", sensitive=False, timeout=_get_timeout("task_list", 60.0))
        registry.register(update_task_status, name="task_update_status", sensitive=False, timeout=_get_timeout("task_update_status", 60.0))
    except ImportError as e:
        logger.warning(f"Failed to import task tools: {e}")

    try:
        # Shell Tools
        from src.servers.shell import execute_command, execute_command_background

        registry.register(execute_command, name="shell_execute", sensitive=True, timeout=_get_timeout("shell_execute", 300.0))  # Allow 5 mins
        registry.register(execute_command_background, name="shell_background", sensitive=True, timeout=_get_timeout("shell_background", 60.0))
    except ImportError as e:
        logger.error(f"Failed to import shell tools: {e}")
        failed_tools.append(("shell", str(e)))

    try:
        # Git Tools
        from src.servers.git import git_add, git_commit, git_diff, git_log, git_status

        registry.register(git_status, sensitive=False, timeout=_get_timeout("git_status", 60.0))
        registry.register(git_diff, sensitive=False, timeout=_get_timeout("git_diff", 60.0))
        registry.register(git_log, sensitive=False, timeout=_get_timeout("git_log", 60.0))
        registry.register(git_add, sensitive=False, timeout=_get_timeout("git_add", 60.0))
        registry.register(git_commit, sensitive=True, timeout=_get_timeout("git_commit", 60.0))
    except ImportError as e:
        logger.warning(f"Failed to import git tools: {e}")

    try:
        # LSP Tools
        from src.servers.lsp import (
            lsp_completions,
            lsp_definition,
            lsp_diagnostics,
            lsp_hover,
            lsp_references,
            lsp_rename,
        )

        registry.register(lsp_diagnostics, sensitive=False, timeout=_get_timeout("lsp_diagnostics", 120.0))
        registry.register(lsp_completions, sensitive=False, timeout=_get_timeout("lsp_completions", 30.0))
        registry.register(lsp_hover, sensitive=False, timeout=_get_timeout("lsp_hover", 30.0))
        registry.register(lsp_references, sensitive=False, timeout=_get_timeout("lsp_references", 60.0))
        registry.register(lsp_rename, sensitive=True, timeout=_get_timeout("lsp_rename", 60.0))
        registry.register(lsp_definition, sensitive=False, timeout=_get_timeout("lsp_definition", 30.0))
    except ImportError as e:
        logger.warning(f"Failed to import LSP tools: {e}")

    try:
        # Linting Tools
        from src.servers.lint import (
            check_security,
            code_complexity,
            format_python_ruff,
            get_lint_summary,
            lint_all,
            lint_javascript_eslint,
            lint_python_ruff,
            typecheck_python_mypy,
        )

        registry.register(lint_python_ruff, sensitive=False, timeout=_get_timeout("lint_python_ruff", 60.0))
        registry.register(format_python_ruff, sensitive=True, timeout=_get_timeout("format_python_ruff", 60.0))
        registry.register(typecheck_python_mypy, sensitive=False, timeout=_get_timeout("typecheck_python_mypy", 120.0))
        registry.register(lint_javascript_eslint, sensitive=False, timeout=_get_timeout("lint_javascript_eslint", 60.0))
        registry.register(lint_all, sensitive=False, timeout=_get_timeout("lint_all", 180.0))
        registry.register(code_complexity, sensitive=False, timeout=_get_timeout("code_complexity", 60.0))
        registry.register(check_security, sensitive=False, timeout=_get_timeout("check_security", 60.0))
        registry.register(get_lint_summary, sensitive=False, timeout=_get_timeout("get_lint_summary", 60.0))
    except ImportError as e:
        logger.warning(f"Failed to import lint tools: {e}")

    try:
        # Semantic Search Tools
        from src.servers.semantic_search import index_codebase, semantic_search

        registry.register(semantic_search, sensitive=False, timeout=_get_timeout("semantic_search", 120.0))
        registry.register(index_codebase, sensitive=False, timeout=_get_timeout("index_codebase", 300.0))
    except ImportError as e:
        logger.warning(f"Failed to import semantic search tools: {e}")

    try:
        # System Tools
        from src.servers.system import switch_mode

        registry.register(switch_mode, sensitive=True, timeout=_get_timeout("switch_mode", 10.0))
    except ImportError as e:
        logger.warning(f"Failed to import system tools: {e}")

    logger.info(f"Tool registry initialized with {len(registry.get_tool_names())} tools")

    try:
        # Browser Tools
        from src.servers.browser import browse_page, take_screenshot

        registry.register(browse_page, name="browse_page", sensitive=False, timeout=_get_timeout("browse_page", 60.0))
        registry.register(take_screenshot, name="take_screenshot", sensitive=False, timeout=_get_timeout("take_screenshot", 60.0))

        # GitHub Tools
        from src.servers.github import github_create_issue, github_list_issues

        registry.register(github_list_issues, name="github_list_issues", sensitive=False, timeout=_get_timeout("github_list_issues", 30.0))
        registry.register(github_create_issue, name="github_create_issue", sensitive=True, timeout=_get_timeout("github_create_issue", 60.0))

        # Database Tools
        from src.servers.database import (
            db_describe_table,
            db_list_tables,
            db_read_query,
            db_write_query,
        )

        registry.register(db_read_query, name="db_read_query", sensitive=False, timeout=_get_timeout("db_read_query", 60.0))
        registry.register(db_write_query, name="db_write_query", sensitive=True, timeout=_get_timeout("db_write_query", 60.0))
        registry.register(db_list_tables, name="db_list_tables", sensitive=False, timeout=_get_timeout("db_list_tables", 30.0))
        registry.register(db_describe_table, name="db_describe_table", sensitive=False, timeout=_get_timeout("db_describe_table", 30.0))

    except ImportError as e:
        logger.warning(f"Failed to import browser/github/database tools: {e}")

    # Report failed tool imports
    if failed_tools:
        error_msg = "Failed to load tools:\n" + "\n".join(
            f"  - {name}: {error}" for name, error in failed_tools
        )
        # Critical tools: filesystem and shell are essential
        critical_tools = {"filesystem", "shell"}
        if any(name in critical_tools for name, _ in failed_tools):
            from src.core.errors import ConfigurationError
            raise ConfigurationError(error_msg)
        else:
            logger.warning(error_msg)

    return registry


# ========================================
# Convenience Functions
# ========================================


async def call_tool(name: str, arguments: dict[str, Any]) -> str:
    """Call a tool from the default registry."""
    return await get_default_registry().call_tool(name, arguments)


def get_genai_tools(tool_names: list[str] | None = None) -> list[types.FunctionDeclaration]:
    """
    Get tools as GenAI FunctionDeclarations.

    Args:
        tool_names: Optional list of tool names to include.
                   If None, returns all tools.
    """
    return get_default_registry().get_genai_tools(tool_names)


def is_sensitive_tool(name: str) -> bool:
    """Check if a tool requires HITL approval."""
    return get_default_registry().is_sensitive(name)
