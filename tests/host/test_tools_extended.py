"""Tests for tools.py extended functionality"""

import pytest

from src.host.tools import ToolRegistry, get_default_registry


class TestToolRegistryExtended:
    """Extended tests for ToolRegistry."""

    def test_registry_initialization(self):
        """Test ToolRegistry initialization."""
        registry = ToolRegistry()
        assert registry is not None

    def test_register_tool(self):
        """Test registering a tool."""
        registry = ToolRegistry()

        def test_tool(arg: str) -> str:
            return f"Result: {arg}"

        registry.register(test_tool, sensitive=False)
        assert "test_tool" in registry.get_tool_names()

    def test_get_tool_names(self):
        """Test getting tool names."""
        registry = ToolRegistry()
        names = registry.get_tool_names()
        assert isinstance(names, list)

    def test_get_tool(self):
        """Test getting a specific tool."""
        registry = ToolRegistry()

        def my_tool() -> str:
            return "test"

        registry.register(my_tool, sensitive=False)
        # Check tool is registered
        assert "my_tool" in registry.get_tool_names()
        # Check we can get GenAI tools
        tools = registry.get_genai_tools(["my_tool"])
        assert len(tools) > 0

    @pytest.mark.asyncio
    async def test_call_tool(self):
        """Test calling a tool."""
        registry = ToolRegistry()

        def echo_tool(message: str) -> str:
            return message

        registry.register(echo_tool, sensitive=False)
        result = await registry.call_tool("echo_tool", {"message": "hello"})
        assert result == "hello"

    def test_get_genai_tools(self):
        """Test getting GenAI format tools."""
        registry = ToolRegistry()

        def sample_tool(x: int) -> int:
            return x * 2

        registry.register(sample_tool, sensitive=False)
        tools = registry.get_genai_tools()
        assert isinstance(tools, list)

    def test_sensitive_tool_marking(self):
        """Test that sensitive tools are marked."""
        registry = ToolRegistry()

        def dangerous_tool() -> str:
            return "danger"

        registry.register(dangerous_tool, sensitive=True)
        # Check tool is marked as sensitive
        assert registry.is_sensitive("dangerous_tool") is True
        # Check it's in sensitive tools list
        assert "dangerous_tool" in registry.get_sensitive_tools()


class TestDefaultRegistry:
    """Tests for default registry."""

    def test_get_default_registry(self):
        """Test getting default registry."""
        registry = get_default_registry()
        assert registry is not None

    def test_default_registry_has_tools(self):
        """Test that default registry has tools."""
        registry = get_default_registry()
        names = registry.get_tool_names()
        assert len(names) > 0

    def test_default_registry_is_singleton(self):
        """Test that default registry is singleton."""
        registry1 = get_default_registry()
        registry2 = get_default_registry()
        assert registry1 is registry2
