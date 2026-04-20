"""
Simplified Tool Registry - Direct Function Calls with Lazy Loading

This module provides the in-process tool registry used by the runtime.
Instead of spawning subprocesses and communicating over a sidecar protocol,
it directly imports and calls tool functions.

Benefits:
- Zero subprocess overhead (~10ms savings per call)
- Simpler debugging (direct stack traces)
- Easier testing (just call the function)
- ~80% less code complexity
- Lazy loading for faster startup

Usage:
    from src.host.tools import ToolRegistry, get_default_registry

    registry = get_default_registry()

    # Get tools for GenAI
    genai_tools = registry.get_genai_tools()

    # Call a tool
    result = await registry.call_tool("read", {"path": "main.py", "mode": "file"})
"""

import asyncio
import importlib
import inspect
import logging
import threading
import time
import types as py_types
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Union, get_args, get_origin

from google.genai import types
from src.core.config.config import load_config
from src.core.tool_policy import get_default_tool_policy_engine
from src.core.tool_selector import get_capability_group_for_tool

logger = logging.getLogger(__name__)


class LazyToolFunction:
    """
    Lazy loader for tool functions.

    Only imports the module when the function is first called.
    This significantly reduces startup time.
    """

    def __init__(self, module: str, func_name: str):
        self._module = module
        self._func_name = func_name
        self._loaded_func: Callable | None = None
        self._load_error: str | None = None

    def _load(self) -> Callable:
        """Load the actual function on first access."""
        if self._loaded_func is not None:
            return self._loaded_func

        if self._load_error:
            raise ImportError(self._load_error)

        try:
            mod = importlib.import_module(self._module)
            self._loaded_func = getattr(mod, self._func_name)
            return self._loaded_func
        except (ImportError, AttributeError) as e:
            self._load_error = f"Failed to load {self._module}.{self._func_name}: {e}"
            raise ImportError(self._load_error) from e

    def __call__(self, *args, **kwargs):
        """Forward call to the actual function."""
        func = self._load()
        return func(*args, **kwargs)

    async def __acall__(self, *args, **kwargs):
        """Forward async call to the actual function."""
        func = self._load()
        return await func(*args, **kwargs)

    @property
    def __name__(self) -> str:
        return self._func_name

    @property
    def __doc__(self) -> str:
        try:
            func = self._load()
            return func.__doc__ or ""
        except ImportError:
            return f"Tool from {self._module}.{self._func_name}"

    @property
    def __signature__(self) -> inspect.Signature:
        func = self._load()
        return inspect.signature(func)


@dataclass
class ToolDefinition:
    """Definition of a tool."""

    name: str
    description: str
    function: Callable
    parameters: dict[str, Any]
    sensitive: bool = False  # Requires HITL approval
    timeout: float = 60.0  # Default timeout in seconds
    source: str = "built_in"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolAuditEntry:
    """Audit record for tool governance and execution decisions."""

    timestamp: float
    tool_name: str
    action: str
    allowed: bool
    mode: str | None
    source: str
    background: bool = False
    requires_approval: bool = False
    sandbox: str | None = None
    audit_level: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "tool_name": self.tool_name,
            "action": self.action,
            "allowed": self.allowed,
            "mode": self.mode,
            "source": self.source,
            "background": self.background,
            "requires_approval": self.requires_approval,
            "sandbox": self.sandbox,
            "audit_level": self.audit_level,
            "error": self.error,
        }


class ToolRegistry:
    """
    Simple tool registry with direct function calls.

    Example:
        registry = ToolRegistry()
        registry.register(read, sensitive=False)
        registry.register(write, sensitive=True)

        result = await registry.call_tool("read", {"path": "main.py", "mode": "file"})
    """

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._audit_log: list[ToolAuditEntry] = []
        self._max_audit_entries = 1000

    def register(
        self,
        func: Callable,
        name: str | None = None,
        description: str | None = None,
        sensitive: bool = False,
        timeout: float = 60.0,
        source: str = "built_in",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Register a tool function.

        Args:
            func: The tool function
            name: Tool name (defaults to function name)
            description: Tool description (defaults to docstring)
            sensitive: Whether this tool requires HITL approval
            timeout: Timeout in seconds (default 60.0)
            source: Tool source classification
            metadata: Additional registry metadata
        """
        tool_name = name or func.__name__
        tool_desc = description or (func.__doc__ or "").strip()

        # Extract parameters from function signature
        if isinstance(func, LazyToolFunction) and func._loaded_func is None:
            parameters = {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": True,
            }
        else:
            parameters = self._extract_parameters(func)

        self._tools[tool_name] = ToolDefinition(
            name=tool_name,
            description=tool_desc,
            function=func,
            parameters=parameters,
            sensitive=sensitive,
            timeout=timeout,
            source=source,
            metadata=metadata or {},
        )

        logger.debug("Registered tool: %s", tool_name)

    def _extract_parameters(self, func: Callable) -> dict[str, Any]:
        """Extract JSON Schema parameters from function signature."""
        try:
            sig = inspect.signature(func)
        except (ImportError, ValueError, TypeError) as e:
            logger.warning("Falling back to generic tool schema for %s: %s", func, e)
            return {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": True,
            }

        properties = {}
        required = []
        descriptions = self._extract_param_descriptions(func.__doc__ or "")

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            schema = self._annotation_to_schema(param.annotation)
            if description := descriptions.get(param_name):
                schema["description"] = description
            if param.default != inspect.Parameter.empty and param.default is not None:
                if isinstance(param.default, (str, int, float, bool, list, dict)):
                    schema["default"] = param.default

            properties[param_name] = schema

            # Check if required (no default value)
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def _annotation_to_schema(self, annotation: Any) -> dict[str, Any]:
        """Convert a Python annotation into JSON Schema."""
        if annotation == inspect.Parameter.empty:
            return {"type": "string"}

        primitive_types = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
        }

        origin = get_origin(annotation)
        args = get_args(annotation)

        if origin is None and isinstance(annotation, py_types.UnionType):
            origin = py_types.UnionType
            args = getattr(annotation, "__args__", ())

        if origin is not None:
            origin_name = getattr(origin, "__name__", "")

            if origin_name == "Literal":
                values = [value for value in args if value is not None]
                value_type = "string"
                for value in values:
                    inferred = primitive_types.get(type(value))
                    if inferred:
                        value_type = inferred
                        break

                schema = {"type": value_type}
                if values:
                    schema["enum"] = list(values)
                return schema

            if origin in (list, tuple, set):
                item_schema = self._annotation_to_schema(args[0]) if args else {"type": "string"}
                return {"type": "array", "items": item_schema}

            if origin is dict:
                value_schema = self._annotation_to_schema(args[1]) if len(args) > 1 else {}
                return {"type": "object", "additionalProperties": value_schema}

            if origin in (py_types.UnionType, Union):
                non_none_args = [arg for arg in args if arg is not type(None)]
                if len(non_none_args) == 1:
                    return self._annotation_to_schema(non_none_args[0])
                return {"anyOf": [self._annotation_to_schema(arg) for arg in non_none_args]}

        if annotation in primitive_types:
            return {"type": primitive_types[annotation]}
        if annotation in (list, tuple, set):
            return {"type": "array", "items": {"type": "string"}}
        if annotation is dict:
            return {"type": "object"}

        return {"type": "string"}

    def _extract_param_descriptions(self, docstring: str) -> dict[str, str]:
        """Extract Google-style parameter descriptions from a docstring."""
        if not docstring:
            return {}

        descriptions: dict[str, str] = {}
        current_param: str | None = None
        in_args_block = False

        for raw_line in inspect.cleandoc(docstring).splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()

            if stripped in {"Args:", "Arguments:"}:
                in_args_block = True
                current_param = None
                continue

            if not in_args_block:
                continue

            if not stripped:
                current_param = None
                continue

            if not line.startswith((" ", "\t")) and stripped.endswith(":"):
                break

            if ":" in stripped and not stripped.startswith(("-", "*")):
                name, description = stripped.split(":", 1)
                if name.replace("_", "").isalnum():
                    current_param = name.strip()
                    descriptions[current_param] = description.strip()
                    continue

            if current_param and stripped:
                descriptions[current_param] = f"{descriptions[current_param]} {stripped}".strip()

        return descriptions

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

            json_schema = types.JSONSchema.model_validate(tool.parameters or {"type": "object"})
            decl = types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=types.Schema.from_json_schema(
                    json_schema=json_schema,
                    api_option="GEMINI_API",
                    raise_error_on_unsupported_field=False,
                ),
            )
            declarations.append(decl)

        return declarations

    def get_tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    def get_sensitive_tools(self) -> list[str]:
        """Get sensitive tool names in registration order."""
        return [name for name, tool in self._tools.items() if tool.sensitive]

    def is_sensitive(self, tool_name: str) -> bool:
        """Check if a tool is sensitive."""
        tool = self._tools.get(tool_name)
        return tool.sensitive if tool else False

    def get_tool_policy(
        self,
        tool_name: str,
        *,
        mode: str | None = None,
        active_mcp_extensions: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Get product-facing governance metadata for a tool."""
        tool = self._tools.get(tool_name)
        if tool is None:
            return None

        policy = get_default_tool_policy_engine().describe_tool(
            tool_name=tool.name,
            mode=mode,
            source=tool.source,
            sensitive=tool.sensitive,
            metadata=tool.metadata,
            active_mcp_extensions=active_mcp_extensions,
        )
        return policy.to_dict()

    def get_tool_policies(
        self,
        *,
        mode: str | None = None,
        active_mcp_extensions: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Get governance metadata for every tool in the registry."""
        return {
            tool_name: policy
            for tool_name in self.get_tool_names()
            if (policy := self.get_tool_policy(
                tool_name,
                mode=mode,
                active_mcp_extensions=active_mcp_extensions,
            ))
            is not None
        }

    def check_tool_execution(
        self,
        tool_name: str,
        *,
        mode: str | None = None,
        active_mcp_extensions: list[str] | None = None,
        background: bool = False,
    ) -> tuple[bool, str | None, dict[str, Any] | None]:
        """Check whether a tool can run in the current runtime context."""
        policy = self.get_tool_policy(
            tool_name,
            mode=mode,
            active_mcp_extensions=active_mcp_extensions,
        )
        if policy is None:
            reason = f"Tool '{tool_name}' not found."
            self._record_audit(
                ToolAuditEntry(
                    timestamp=time.time(),
                    tool_name=tool_name,
                    action="policy_check",
                    allowed=False,
                    mode=mode,
                    source="unknown",
                    background=background,
                    error=reason,
                )
            )
            return False, reason, None

        reason = None
        allowed = True
        if not policy["visible"]:
            allowed = False
            if mode:
                reason = f"Tool '{tool_name}' is not available in {mode} mode."
            else:
                reason = f"Tool '{tool_name}' is not available in this runtime."
        elif background and not policy["background_safe"]:
            allowed = False
            reason = f"Tool '{tool_name}' cannot run as a background task."

        self._record_audit(
            ToolAuditEntry(
                timestamp=time.time(),
                tool_name=tool_name,
                action="policy_check",
                allowed=allowed,
                mode=mode,
                source=policy["source"],
                background=background,
                requires_approval=policy["requires_approval"],
                sandbox=policy["sandbox"],
                audit_level=policy["audit_level"],
                error=None if allowed else reason,
            )
        )
        return allowed, reason, policy

    def record_tool_execution(
        self,
        tool_name: str,
        *,
        action: str,
        mode: str | None = None,
        active_mcp_extensions: list[str] | None = None,
        background: bool = False,
        allowed: bool = True,
        error: str | None = None,
    ) -> None:
        """Record a tool execution lifecycle event."""
        policy = self.get_tool_policy(
            tool_name,
            mode=mode,
            active_mcp_extensions=active_mcp_extensions,
        )
        source = policy["source"] if policy is not None else "unknown"
        requires_approval = policy["requires_approval"] if policy is not None else False
        sandbox = policy["sandbox"] if policy is not None else None
        audit_level = policy["audit_level"] if policy is not None else None
        self._record_audit(
            ToolAuditEntry(
                timestamp=time.time(),
                tool_name=tool_name,
                action=action,
                allowed=allowed,
                mode=mode,
                source=source,
                background=background,
                requires_approval=requires_approval,
                sandbox=sandbox,
                audit_level=audit_level,
                error=error,
            )
        )

    def get_audit_log(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent tool audit entries."""
        return [entry.to_dict() for entry in self._audit_log[-limit:]]

    def clear_audit_log(self) -> None:
        """Clear runtime tool audit entries."""
        self._audit_log.clear()

    def _record_audit(self, entry: ToolAuditEntry) -> None:
        self._audit_log.append(entry)
        if len(self._audit_log) > self._max_audit_entries:
            self._audit_log = self._audit_log[-self._max_audit_entries // 2 :]

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
        from src.core.tool_cache import ToolCache, should_cache_tool

        cache = ToolCache.get_instance()

        if should_cache_tool(name):
            cached = cache.get(name, arguments)
            if cached is not None:
                return cached

        tool = self._tools.get(name)
        if not tool:
            available = ", ".join(self._tools.keys())
            raise ValueError(f"Tool '{name}' not found. Available: {available}")

        try:
            func = tool.function

            # Handle LazyToolFunction
            if isinstance(func, LazyToolFunction):
                loaded_func = func._load()
                is_async = inspect.iscoroutinefunction(loaded_func)
            else:
                is_async = inspect.iscoroutinefunction(func)

            if is_async:
                try:
                    result = await asyncio.wait_for(func(**arguments), timeout=tool.timeout)
                except asyncio.TimeoutError:
                    error_msg = f"Tool '{name}' timed out after {tool.timeout} seconds."
                    logger.error(error_msg)
                    return error_msg
            else:
                result = func(**arguments)

            result_str = str(result) if result is not None else "Success"

            if should_cache_tool(name) and not result_str.startswith("Error"):
                cache.set(name, arguments, result_str)

            return result_str

        except Exception as e:
            logger.error("Tool %s failed: %s", name, e, exc_info=True)
            return f"Error executing {name}: {type(e).__name__}: {e}"


# ========================================
# Default Registry with All Tools
# ========================================

_default_registry: ToolRegistry | None = None
_registry_cache: dict[tuple[str, ...], ToolRegistry] = {}
_default_extension_registry: ToolRegistry | None = None
_extension_registry_cache: dict[tuple[str, ...], ToolRegistry] = {}
_registry_lock = threading.Lock()


def get_default_registry(tool_names: list[str] | None = None) -> ToolRegistry:
    """
    Get the default built-in tool registry.

    This is a singleton cache keyed by the allowed tool set.
    Thread-safe using double-check locking pattern.
    """
    global _default_registry

    if tool_names is None:
        if _default_registry is None:
            with _registry_lock:
                # Double-check locking to prevent race conditions
                if _default_registry is None:
                    _default_registry = _create_registry(BUILTIN_TOOL_SPECS, source="built_in")
        return _default_registry

    cache_key = tuple(tool_names)
    if cache_key in _registry_cache:
        return _registry_cache[cache_key]

    with _registry_lock:
        if cache_key not in _registry_cache:
            _registry_cache[cache_key] = _create_registry(
                BUILTIN_TOOL_SPECS,
                set(tool_names),
                source="built_in",
            )

    return _registry_cache[cache_key]


def get_extension_registry(tool_names: list[str] | None = None) -> ToolRegistry:
    """Get the optional extension tool registry."""
    global _default_extension_registry

    if tool_names is None:
        if _default_extension_registry is None:
            with _registry_lock:
                if _default_extension_registry is None:
                    _default_extension_registry = _create_registry(
                        EXTENSION_TOOL_SPECS,
                        source="mcp_extension",
                    )
        return _default_extension_registry

    cache_key = tuple(tool_names)
    if cache_key in _extension_registry_cache:
        return _extension_registry_cache[cache_key]

    with _registry_lock:
        if cache_key not in _extension_registry_cache:
            _extension_registry_cache[cache_key] = _create_registry(
                EXTENSION_TOOL_SPECS,
                set(tool_names),
                source="mcp_extension",
            )

    return _extension_registry_cache[cache_key]


@dataclass
class ToolSpec:
    """Declarative specification for a tool to register."""

    module: str  # e.g. "src.servers.filesystem"
    func_name: str  # function name to import
    name: str | None = None  # registered name (defaults to func_name)
    sensitive: bool = False
    timeout: float = 60.0
    critical: bool = False  # raise error if import fails


# fmt: off
BUILTIN_TOOL_SPECS: list[ToolSpec] = [
    # ── Filesystem (critical) ─────────────────────────────────────────
    ToolSpec("src.servers.filesystem", "read",          sensitive=False, timeout=60.0,  critical=True),
    ToolSpec("src.servers.filesystem", "write",         sensitive=True,  timeout=120.0, critical=True),
    ToolSpec("src.servers.filesystem", "search",        sensitive=False, timeout=120.0, critical=True),
    ToolSpec("src.servers.filesystem", "notebook_read", sensitive=False, timeout=60.0,  critical=True),
    ToolSpec("src.servers.filesystem", "notebook_edit", sensitive=True,  timeout=60.0,  critical=True),
    ToolSpec("src.servers.filesystem", "multi_edit",    sensitive=True,  timeout=120.0, critical=True),

    # ── Run (unified) ────────────────────────────────────────────────
    ToolSpec("src.servers.run", "run", sensitive=True, timeout=300.0),

    # ── Memory ───────────────────────────────────────────────────────
    ToolSpec("src.servers.memory", "memory_get",    sensitive=False, timeout=60.0),
    ToolSpec("src.servers.memory", "memory_put",    sensitive=True,  timeout=60.0),
    ToolSpec("src.servers.memory", "memory_search", sensitive=False, timeout=60.0),
    ToolSpec("src.servers.memory", "memory_list",   sensitive=False, timeout=30.0),
    ToolSpec("src.servers.memory", "memory_delete", sensitive=True,  timeout=30.0),

    # ── Web ──────────────────────────────────────────────────────────
    ToolSpec("src.servers.web", "web_fetch",  sensitive=False, timeout=30.0),
    ToolSpec("src.servers.web", "web_search", sensitive=False, timeout=30.0),

    # ── Task ─────────────────────────────────────────────────────────
    ToolSpec("src.servers.task", "task", sensitive=False, timeout=60.0),
    ToolSpec("src.servers.lsp", "lsp_diagnostics", sensitive=False, timeout=120.0),
    ToolSpec("src.servers.lsp", "lsp_completions", sensitive=False, timeout=30.0),
    ToolSpec("src.servers.lsp", "lsp_hover",       sensitive=False, timeout=30.0),
    ToolSpec("src.servers.lsp", "lsp_references",  sensitive=False, timeout=60.0),
    ToolSpec("src.servers.lsp", "lsp_rename",      sensitive=True,  timeout=60.0),
    ToolSpec("src.servers.lsp", "lsp_definition",  sensitive=False, timeout=30.0),

    # ── Misc ─────────────────────────────────────────────────────────
    ToolSpec("src.servers.ask_user", "ask_user",      sensitive=False, timeout=300.0),
]

EXTENSION_TOOL_SPECS: list[ToolSpec] = [
    # ── Browser ──────────────────────────────────────────────────────
    ToolSpec("src.servers.browser", "browse_page",        sensitive=False, timeout=60.0),
    ToolSpec("src.servers.browser", "take_screenshot",    sensitive=False, timeout=60.0),
    ToolSpec("src.servers.browser", "browser_click",      sensitive=False, timeout=30.0),
    ToolSpec("src.servers.browser", "browser_fill",       sensitive=False, timeout=30.0),
    ToolSpec("src.servers.browser", "browser_evaluate",   sensitive=False, timeout=30.0),
    ToolSpec("src.servers.browser", "browser_wait",       sensitive=False, timeout=60.0),
    ToolSpec("src.servers.browser", "browser_pdf",        sensitive=False, timeout=30.0),
    ToolSpec("src.servers.browser", "browser_get_html",   sensitive=False, timeout=30.0),
    ToolSpec("src.servers.browser", "browser_close_page", sensitive=False, timeout=10.0),
    ToolSpec("src.servers.browser", "browser_list_pages", sensitive=False, timeout=10.0),

    # ── Database ─────────────────────────────────────────────────────
    ToolSpec("src.servers.database", "db_read_query",     sensitive=False, timeout=60.0),
    ToolSpec("src.servers.database", "db_write_query",    sensitive=True,  timeout=60.0),
    ToolSpec("src.servers.database", "db_list_tables",    sensitive=False, timeout=30.0),
    ToolSpec("src.servers.database", "db_describe_table", sensitive=False, timeout=30.0),
]
# fmt: on


def _create_registry(
    specs: list[ToolSpec],
    allowed_tool_names: set[str] | None = None,
    *,
    source: str = "built_in",
) -> ToolRegistry:
    """Create and populate a tool registry with lazy loading."""
    registry = ToolRegistry()

    # Load configuration for custom timeouts
    try:
        config = load_config(validate=False)
        tool_timeouts = config.get("tool_timeouts", {})
    except Exception as e:
        logger.warning("Failed to load config for tool timeouts: %s", e)
        tool_timeouts = {}

    failed_critical: list[tuple[str, str]] = []

    for spec in specs:
        tool_name = spec.name or spec.func_name
        if allowed_tool_names is not None and tool_name not in allowed_tool_names:
            continue
        timeout = float(tool_timeouts.get(tool_name, spec.timeout))

        # Use lazy loading proxy instead of importing immediately
        lazy_func = LazyToolFunction(spec.module, spec.func_name)

        # For critical tools, verify import works at startup
        if spec.critical:
            try:
                lazy_func._load()
            except ImportError as e:
                logger.error("Failed to import critical tool %s: %s", tool_name, e)
                failed_critical.append((spec.module, str(e)))
                continue

        registry.register(
            lazy_func,
            name=spec.name,
            sensitive=spec.sensitive,
            timeout=timeout,
            source=source,
            metadata={
                "capability_group": get_capability_group_for_tool(tool_name),
            },
        )

    logger.info(
        f"Tool registry initialized with {len(registry.get_tool_names())} tools (lazy loading)"
    )

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
    return get_default_registry(tool_names).get_genai_tools(tool_names)


def is_sensitive_tool(name: str) -> bool:
    """Check if a tool requires HITL approval."""
    return get_default_registry().is_sensitive(name)
