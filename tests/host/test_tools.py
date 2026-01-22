"""Tests for simplified tool registry."""

import pytest

from src.host.tools import ToolRegistry, get_default_registry


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
        assert props["s"].type.name == "STRING"
        assert props["i"].type.name == "INTEGER"
        assert props["f"].type.name == "NUMBER"
        assert props["b"].type.name == "BOOLEAN"

    def test_optional_type(self):
        """Test extraction of optional types (int | None)."""
        registry = ToolRegistry()

        def optional_func(required: str, optional: int | None = None) -> str:
            """Function with optional param."""
            return ""

        registry.register(optional_func)
        tools = registry.get_genai_tools()

        props = tools[0].parameters.properties
        # Optional int should still be INTEGER
        assert props["optional"].type.name == "INTEGER"

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

        # Should have at least some basic tools
        assert len(tools) > 0
        assert "read_file" in tools
        assert "list_directory" in tools

    def test_default_registry_sensitive_tools(self):
        """Test that sensitive tools are marked."""
        registry = get_default_registry()
        sensitive = registry.get_sensitive_tools()

        # write_file should be sensitive
        assert "write_file" in sensitive

        # read_file should not be sensitive
        assert "read_file" not in sensitive

    def test_default_registry_singleton(self):
        """Test that get_default_registry returns same instance."""
        registry1 = get_default_registry()
        registry2 = get_default_registry()

        assert registry1 is registry2
