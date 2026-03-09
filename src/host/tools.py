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
    timeout: float = 60.0  # Default timeout in seconds


class ToolRegistry:
    """
    Simple tool registry with direct function calls.

    Example:
        registry = ToolRegistry()
        registry.register(read_file, sensitive=False)
        registry.register(write_file, sensitive=True)

        result = await registry.call_tool("read_file", {"path": "main.py"})
    """

    MAX_RESULT_LENGTH = 30000  # Max chars in tool result (matches Claude Code)

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

    def get_sensitive_tools(self) -> set[str]:
        """Get set of sensitive tool names."""
        return {name for name, tool in self._tools.items() if tool.sensitive}

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
                    result = await asyncio.wait_for(
                        tool.function(**arguments), timeout=tool.timeout
                    )
                except asyncio.TimeoutError:
                    error_msg = f"Tool '{name}' timed out after {tool.timeout} seconds."
                    logger.error(error_msg)
                    return error_msg
            else:
                # Sync function - run directly (cannot timeout easily without threads/processes)
                # For critical safety, we might want to run this in a thread executor with timeout,
                # but for now we'll just run it. Most heavy tools should be async.
                result = tool.function(**arguments)

            result_str = str(result) if result is not None else "Success"

            # Truncate oversized results to prevent context overflow
            if len(result_str) > self.MAX_RESULT_LENGTH:
                truncated = result_str[: self.MAX_RESULT_LENGTH]
                result_str = (
                    f"{truncated}\n\n... [Output truncated: {len(result_str):,} chars, "
                    f"showing first {self.MAX_RESULT_LENGTH:,}]"
                )

            return result_str

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


@dataclass
class ToolSpec:
    """Declarative specification for a tool to register."""

    module: str  # e.g. "src.servers.filesystem_unified"
    func_name: str  # function name to import
    name: str | None = None  # registered name (defaults to func_name)
    sensitive: bool = False
    timeout: float = 60.0
    critical: bool = False  # raise error if import fails


# fmt: off
TOOL_SPECS: list[ToolSpec] = [
    # ── Filesystem (critical) ─────────────────────────────────────────
    ToolSpec("src.servers.filesystem_unified", "read",          sensitive=False, timeout=60.0,  critical=True),
    ToolSpec("src.servers.filesystem_unified", "write",         sensitive=True,  timeout=120.0, critical=True),
    ToolSpec("src.servers.filesystem_unified", "search",        sensitive=False, timeout=120.0, critical=True),
    ToolSpec("src.servers.filesystem_unified", "notebook_read", sensitive=False, timeout=60.0,  critical=True),
    ToolSpec("src.servers.filesystem_unified", "notebook_edit", sensitive=True,  timeout=60.0,  critical=True),
    ToolSpec("src.servers.filesystem_unified", "multi_edit",    sensitive=True,  timeout=120.0, critical=True),

    # ── Run (unified) ────────────────────────────────────────────────
    ToolSpec("src.servers.run_unified", "run", sensitive=True, timeout=300.0),

    # ── Computer (legacy) ────────────────────────────────────────────
    ToolSpec("src.servers.computer", "execute_python",  sensitive=True, timeout=300.0),
    ToolSpec("src.servers.computer", "install_package", sensitive=True, timeout=300.0),

    # ── Memory ───────────────────────────────────────────────────────
    ToolSpec("src.servers.memory", "note",         sensitive=True,  timeout=60.0),
    ToolSpec("src.servers.memory", "save_note",    sensitive=True,  timeout=60.0),
    ToolSpec("src.servers.memory", "search_notes", sensitive=False, timeout=60.0),

    # ── Web ──────────────────────────────────────────────────────────
    ToolSpec("src.servers.web", "fetch_page",      name="fetch_url",  sensitive=False, timeout=30.0),
    ToolSpec("src.servers.web", "search_internet", name="web_search", sensitive=False, timeout=30.0),

    # ── Task ─────────────────────────────────────────────────────────
    ToolSpec("src.servers.task", "task",               sensitive=False, timeout=60.0),
    ToolSpec("src.servers.task", "add_task",            name="task_create",        sensitive=False, timeout=60.0),
    ToolSpec("src.servers.task", "list_tasks",          name="task_list",          sensitive=False, timeout=60.0),
    ToolSpec("src.servers.task", "update_task_status",  name="task_update_status", sensitive=False, timeout=60.0),

    # ── Shell (critical) ─────────────────────────────────────────────
    ToolSpec("src.servers.shell", "execute_command",            name="shell_execute",    sensitive=True, timeout=300.0, critical=True),
    ToolSpec("src.servers.shell", "execute_command_background", name="shell_background", sensitive=True, timeout=60.0,  critical=True),

    # ── Git ──────────────────────────────────────────────────────────
    ToolSpec("src.servers.git", "git",        sensitive=True,  timeout=60.0),
    ToolSpec("src.servers.git", "git_status", sensitive=False, timeout=60.0),
    ToolSpec("src.servers.git", "git_diff",   sensitive=False, timeout=60.0),
    ToolSpec("src.servers.git", "git_log",    sensitive=False, timeout=60.0),
    ToolSpec("src.servers.git", "git_add",    sensitive=False, timeout=60.0),
    ToolSpec("src.servers.git", "git_commit", sensitive=True,  timeout=60.0),

    # ── LSP ──────────────────────────────────────────────────────────
    ToolSpec("src.servers.lsp", "lsp",             sensitive=False, timeout=120.0),
    ToolSpec("src.servers.lsp", "lsp_diagnostics", sensitive=False, timeout=120.0),
    ToolSpec("src.servers.lsp", "lsp_completions", sensitive=False, timeout=30.0),
    ToolSpec("src.servers.lsp", "lsp_hover",       sensitive=False, timeout=30.0),
    ToolSpec("src.servers.lsp", "lsp_references",  sensitive=False, timeout=60.0),
    ToolSpec("src.servers.lsp", "lsp_rename",      sensitive=True,  timeout=60.0),
    ToolSpec("src.servers.lsp", "lsp_definition",  sensitive=False, timeout=30.0),

    # ── Lint ─────────────────────────────────────────────────────────
    ToolSpec("src.servers.lint", "lint",                    sensitive=False, timeout=180.0),
    ToolSpec("src.servers.lint", "lint_python_ruff",        sensitive=False, timeout=60.0),
    ToolSpec("src.servers.lint", "format_python_ruff",      sensitive=True,  timeout=60.0),
    ToolSpec("src.servers.lint", "typecheck_python_mypy",   sensitive=False, timeout=120.0),
    ToolSpec("src.servers.lint", "lint_javascript_eslint",  sensitive=False, timeout=60.0),
    ToolSpec("src.servers.lint", "lint_all",                sensitive=False, timeout=180.0),
    ToolSpec("src.servers.lint", "code_complexity",         sensitive=False, timeout=60.0),
    ToolSpec("src.servers.lint", "check_security",          sensitive=False, timeout=60.0),
    ToolSpec("src.servers.lint", "get_lint_summary",        sensitive=False, timeout=60.0),

    # ── Semantic Search ──────────────────────────────────────────────
    ToolSpec("src.servers.semantic_search", "semantic_search", sensitive=False, timeout=120.0),
    ToolSpec("src.servers.semantic_search", "index_codebase",  sensitive=False, timeout=300.0),

    # ── Misc ─────────────────────────────────────────────────────────
    ToolSpec("src.servers.ask_user", "ask_user",      sensitive=False, timeout=300.0),
    ToolSpec("src.servers.system",   "switch_mode",   sensitive=True,  timeout=10.0),

    # ── Spec ─────────────────────────────────────────────────────────
    ToolSpec("src.servers.spec", "spec_update_task", sensitive=False, timeout=10.0),
    ToolSpec("src.servers.spec", "spec_check_item",  sensitive=False, timeout=10.0),
    ToolSpec("src.servers.spec", "spec_progress",    sensitive=False, timeout=10.0),

    # ── Browser ──────────────────────────────────────────────────────
    ToolSpec("src.servers.browser", "browse_page",     sensitive=False, timeout=60.0),
    ToolSpec("src.servers.browser", "take_screenshot", sensitive=False, timeout=60.0),

    # ── GitHub ───────────────────────────────────────────────────────
    ToolSpec("src.servers.github", "github_list_issues",  sensitive=False, timeout=30.0),
    ToolSpec("src.servers.github", "github_create_issue", sensitive=True,  timeout=60.0),

    # ── Database ─────────────────────────────────────────────────────
    ToolSpec("src.servers.database", "db_read_query",     sensitive=False, timeout=60.0),
    ToolSpec("src.servers.database", "db_write_query",    sensitive=True,  timeout=60.0),
    ToolSpec("src.servers.database", "db_list_tables",    sensitive=False, timeout=30.0),
    ToolSpec("src.servers.database", "db_describe_table", sensitive=False, timeout=30.0),
]
# fmt: on


def _create_default_registry() -> ToolRegistry:
    """Create and populate the default tool registry."""
    import importlib

    registry = ToolRegistry()

    # Load configuration for custom timeouts
    try:
        config = load_config(validate=False)
        tool_timeouts = config.get("tool_timeouts", {})
    except Exception as e:
        logger.warning(f"Failed to load config for tool timeouts: {e}")
        tool_timeouts = {}

    failed_critical: list[tuple[str, str]] = []

    for spec in TOOL_SPECS:
        tool_name = spec.name or spec.func_name
        timeout = float(tool_timeouts.get(tool_name, spec.timeout))
        try:
            module = importlib.import_module(spec.module)
            func = getattr(module, spec.func_name)
            registry.register(
                func,
                name=spec.name,
                sensitive=spec.sensitive,
                timeout=timeout,
            )
        except (ImportError, AttributeError) as e:
            if spec.critical:
                logger.error(f"Failed to import critical tool {tool_name}: {e}")
                failed_critical.append((spec.module, str(e)))
            else:
                logger.warning(f"Failed to import tool {tool_name}: {e}")

    logger.info(f"Tool registry initialized with {len(registry.get_tool_names())} tools")

    # Critical tools must be available
    if failed_critical:
        error_msg = "Failed to load critical tools:\n" + "\n".join(
            f"  - {mod}: {err}" for mod, err in failed_critical
        )
        from src.core.errors import ConfigurationError

        raise ConfigurationError(error_msg)

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
