"""Tests for simplified tool registry."""

from typing import Literal

import pytest

from src.core.tool_selector import get_tools_for_mode
from src.host.tools import ToolRegistry, get_default_registry, get_extension_registry


class TestToolRegistry:
    """Test the ToolRegistry class."""

    def test_register_function(self):
        """Test registering a simple function."""
        registry = ToolRegistry()

        def my_tool(path: str, count: int = 10) -> str:
            """A test tool."""
            return f"path={path}, count={count}"

        registry.register(my_tool)

        assert "my_tool" in registry.get_tool_names()
        assert not registry.is_sensitive("my_tool")

    def test_register_with_custom_name(self):
        """Test registering with a custom name."""
        registry = ToolRegistry()

        def internal_name(x: str) -> str:
            """Internal function."""
            return x

        registry.register(internal_name, name="public_name")

        assert "public_name" in registry.get_tool_names()
        assert "internal_name" not in registry.get_tool_names()

    def test_register_sensitive_tool(self):
        """Test registering a sensitive tool."""
        registry = ToolRegistry()

        def dangerous_tool(cmd: str) -> str:
            """Execute something dangerous."""
            return cmd

        registry.register(dangerous_tool, sensitive=True)

        assert registry.is_sensitive("dangerous_tool")
        assert "dangerous_tool" in registry.get_sensitive_tools()

    def test_get_genai_tools(self):
        """Test generating GenAI FunctionDeclarations."""
        registry = ToolRegistry()

        def test_tool(name: str, value: int = 0) -> str:
            """A test tool with description."""
            return f"{name}={value}"

        registry.register(test_tool)
        tools = registry.get_genai_tools()

        assert len(tools) == 1
        assert tools[0].name == "test_tool"
        assert "A test tool with description" in tools[0].description


class TestTypeExtraction:
    """Test parameter type extraction."""

    def test_basic_types(self):
        """Test extraction of basic types."""
        registry = ToolRegistry()

        def typed_func(s: str, i: int, f: float, b: bool) -> str:
            """Function with typed params."""
            return ""

        registry.register(typed_func)
        tools = registry.get_genai_tools()

        props = tools[0].parameters.properties
        assert props["s"].type.value == "STRING"
        assert props["i"].type.value == "INTEGER"
        assert props["f"].type.value == "NUMBER"
        assert props["b"].type.value == "BOOLEAN"

    def test_optional_type(self):
        """Test extraction of optional types (int | None)."""
        registry = ToolRegistry()

        def optional_func(required: str, optional: int | None = None) -> str:
            """Function with optional param."""
            return ""

        registry.register(optional_func)
        tools = registry.get_genai_tools()

        props = tools[0].parameters.properties
        assert props["optional"].type.value == "INTEGER"

    def test_required_params(self):
        """Test required parameter detection."""
        registry = ToolRegistry()

        def mixed_func(required: str, optional: int = 10) -> str:
            """Function with mixed params."""
            return ""

        registry.register(mixed_func)
        tools = registry.get_genai_tools()

        required = tools[0].parameters.required
        assert "required" in required
        assert "optional" not in required

    def test_literal_enum_extraction(self):
        """Test extraction of Literal enum values and docstring descriptions."""
        registry = ToolRegistry()

        def literal_func(mode: Literal["read", "write"], path: str) -> str:
            """Tool with enum mode.

            Args:
                mode: Execution mode.
                path: Target path.
            """
            return ""

        registry.register(literal_func)
        tools = registry.get_genai_tools()

        props = tools[0].parameters.properties
        assert props["mode"].type.value == "STRING"
        assert props["mode"].enum == ["read", "write"]
        assert "Execution mode." in props["mode"].description


class TestToolCalling:
    """Test tool calling functionality."""

    @pytest.mark.asyncio
    async def test_call_tool_sync(self):
        """Test calling a synchronous tool."""
        registry = ToolRegistry()

        def sync_tool(x: int, y: int) -> str:
            return str(x + y)

        registry.register(sync_tool)
        result = await registry.call_tool("sync_tool", {"x": 1, "y": 2})

        assert result == "3"

    @pytest.mark.asyncio
    async def test_call_tool_async(self):
        """Test calling an asynchronous tool."""
        registry = ToolRegistry()

        async def async_tool(name: str) -> str:
            return f"Hello, {name}!"

        registry.register(async_tool)
        result = await registry.call_tool("async_tool", {"name": "World"})

        assert result == "Hello, World!"

    @pytest.mark.asyncio
    async def test_call_unknown_tool(self):
        """Test calling an unknown tool raises error."""
        registry = ToolRegistry()

        with pytest.raises(ValueError, match="not found"):
            await registry.call_tool("nonexistent", {})

    @pytest.mark.asyncio
    async def test_call_tool_with_error(self):
        """Test tool error handling."""
        registry = ToolRegistry()

        def failing_tool() -> str:
            raise RuntimeError("Tool failed!")

        registry.register(failing_tool)
        result = await registry.call_tool("failing_tool", {})

        assert "Error:" in result
        assert "Tool failed!" in result


class TestDefaultRegistry:
    """Test the default registry singleton."""

    def test_default_registry_has_tools(self):
        """Test that default registry loads tools."""
        registry = get_default_registry()
        tools = registry.get_tool_names()

        assert len(tools) > 0
        assert "read" in tools
        assert "write" in tools
        assert "search" in tools

    def test_default_registry_sensitive_tools(self):
        """Test that sensitive tools are marked."""
        registry = get_default_registry()
        sensitive = registry.get_sensitive_tools()

        assert "write" in sensitive
        assert "read" not in sensitive

    def test_default_registry_singleton(self):
        """Test that get_default_registry returns same instance."""
        registry1 = get_default_registry()
        registry2 = get_default_registry()

        assert registry1 is registry2

    def test_default_registry_excludes_extension_tools(self):
        """Default built-in registry should not include browser/db extensions."""
        registry = get_default_registry()
        tools = registry.get_tool_names()

        assert "browser_click" not in tools
        assert "db_read_query" not in tools

    def test_filtered_registry_only_loads_requested_tools(self):
        """Filtered registry should only expose the requested tool set."""
        registry = get_default_registry(["read", "search", "ask_user", "write", "run"])

        assert set(registry.get_tool_names()) == {"read", "search", "ask_user", "write", "run"}

    def test_filtered_registry_is_cached_by_tool_set(self):
        """Filtered registries should be cached per tool set."""
        registry1 = get_default_registry(["read", "search", "ask_user"])
        registry2 = get_default_registry(["read", "search", "ask_user"])

        assert registry1 is registry2

    def test_plan_mode_registry_matches_capability_groups(self):
        """Plan mode should expose only the plan tool surface."""
        plan_tools = get_tools_for_mode("plan")
        registry = get_default_registry(plan_tools)

        assert set(registry.get_tool_names()) == set(plan_tools)
        assert "write" not in registry.get_tool_names()
        assert "run" not in registry.get_tool_names()
        assert "memory_put" in registry.get_tool_names()
        assert "web_search" in registry.get_tool_names()
        assert "task" in registry.get_tool_names()
        assert "update_user_persona" not in registry.get_tool_names()

    def test_build_mode_registry_matches_capability_groups(self):
        """Build mode should expose the coding mainline tool surface."""
        build_tools = get_tools_for_mode("build")
        registry = get_default_registry(build_tools)

        assert set(registry.get_tool_names()) == set(build_tools)
        assert len(build_tools) == 12
        assert "write" in registry.get_tool_names()
        assert "run" in registry.get_tool_names()
        assert "memory_put" in registry.get_tool_names()
        assert "web_fetch" in registry.get_tool_names()
        assert "task" in registry.get_tool_names()
        assert "update_user_persona" not in registry.get_tool_names()

    def test_extension_registry_only_loads_extension_tools(self):
        """Extension registry should keep browser/db tools outside the mainline."""
        registry = get_extension_registry(["browser_click", "db_read_query"])

        assert set(registry.get_tool_names()) == {"browser_click", "db_read_query"}
        assert "read" not in registry.get_tool_names()
