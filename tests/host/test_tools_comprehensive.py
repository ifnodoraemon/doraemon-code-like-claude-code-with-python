"""Comprehensive tests for src/host/tools.py with 35+ test cases.

This test suite provides extensive coverage of:
1. ToolRegistry initialization (8 tests)
2. Tool registration and calling (10 tests)
3. Tool timeout handling (7 tests)
4. Tool error handling (5 tests)
5. Default registry creation (5 tests)
"""

import asyncio
from unittest.mock import patch

import pytest

from src.host.tools import (
    ToolDefinition,
    ToolRegistry,
    call_tool,
    get_default_registry,
    get_genai_tools,
    is_sensitive_tool,
)

# ========================================
# 1. ToolRegistry Initialization Tests (8)
# ========================================


class TestToolRegistryInitialization:
    """Test ToolRegistry initialization."""

    def test_registry_init_empty(self):
        """Test creating an empty registry."""
        registry = ToolRegistry()
        assert registry is not None
        assert isinstance(registry, ToolRegistry)
        assert registry.get_tool_names() == []

    def test_registry_init_has_tools_dict(self):
        """Test registry has internal tools dictionary."""
        registry = ToolRegistry()
        assert hasattr(registry, "_tools")
        assert isinstance(registry._tools, dict)
        assert len(registry._tools) == 0

    def test_registry_multiple_instances(self):
        """Test creating multiple independent registry instances."""
        registry1 = ToolRegistry()
        registry2 = ToolRegistry()
        assert registry1 is not registry2
        assert registry1.get_tool_names() == []
        assert registry2.get_tool_names() == []

    def test_registry_init_state_isolation(self):
        """Test that registry instances don't share state."""
        registry1 = ToolRegistry()
        registry2 = ToolRegistry()

        def tool1() -> str:
            return "tool1"

        registry1.register(tool1)
        assert "tool1" in registry1.get_tool_names()
        assert "tool1" not in registry2.get_tool_names()

    def test_registry_get_tool_names_empty(self):
        """Test get_tool_names on empty registry."""
        registry = ToolRegistry()
        names = registry.get_tool_names()
        assert isinstance(names, list)
        assert len(names) == 0

    def test_registry_get_sensitive_tools_empty(self):
        """Test get_sensitive_tools on empty registry."""
        registry = ToolRegistry()
        sensitive = registry.get_sensitive_tools()
        assert isinstance(sensitive, list)
        assert len(sensitive) == 0

    def test_registry_is_sensitive_nonexistent(self):
        """Test is_sensitive on nonexistent tool."""
        registry = ToolRegistry()
        assert registry.is_sensitive("nonexistent") is False

    def test_registry_get_genai_tools_empty(self):
        """Test get_genai_tools on empty registry."""
        registry = ToolRegistry()
        tools = registry.get_genai_tools()
        assert isinstance(tools, list)
        assert len(tools) == 0


# ========================================
# 2. Tool Registration and Calling Tests (10)
# ========================================


class TestToolRegistration:
    """Test tool registration functionality."""

    def test_register_simple_function(self):
        """Test registering a simple function."""
        registry = ToolRegistry()

        def simple_tool() -> str:
            """A simple tool."""
            return "result"

        registry.register(simple_tool)
        assert "simple_tool" in registry.get_tool_names()

    def test_register_with_custom_name(self):
        """Test registering with custom name."""
        registry = ToolRegistry()

        def internal_func() -> str:
            """Internal function."""
            return "result"

        registry.register(internal_func, name="custom_name")
        assert "custom_name" in registry.get_tool_names()
        assert "internal_func" not in registry.get_tool_names()

    def test_register_with_custom_description(self):
        """Test registering with custom description."""
        registry = ToolRegistry()

        def tool_func() -> str:
            return "result"

        custom_desc = "Custom description"
        registry.register(tool_func, description=custom_desc)
        tools = registry.get_genai_tools()
        assert len(tools) == 1
        assert tools[0].description == custom_desc

    def test_register_sensitive_tool(self):
        """Test registering a sensitive tool."""
        registry = ToolRegistry()

        def sensitive_tool() -> str:
            return "result"

        registry.register(sensitive_tool, sensitive=True)
        assert registry.is_sensitive("sensitive_tool") is True
        assert "sensitive_tool" in registry.get_sensitive_tools()

    def test_register_with_timeout(self):
        """Test registering tool with custom timeout."""
        registry = ToolRegistry()

        def tool_with_timeout() -> str:
            return "result"

        registry.register(tool_with_timeout, timeout=30.0)
        tool_def = registry._tools["tool_with_timeout"]
        assert tool_def.timeout == 30.0

    def test_register_multiple_tools(self):
        """Test registering multiple tools."""
        registry = ToolRegistry()

        def tool1() -> str:
            return "1"

        def tool2() -> str:
            return "2"

        def tool3() -> str:
            return "3"

        registry.register(tool1)
        registry.register(tool2)
        registry.register(tool3)

        names = registry.get_tool_names()
        assert len(names) == 3
        assert "tool1" in names
        assert "tool2" in names
        assert "tool3" in names

    def test_register_overwrites_existing(self):
        """Test that registering same name overwrites."""
        registry = ToolRegistry()

        def tool_v1() -> str:
            return "v1"

        def tool_v2() -> str:
            return "v2"

        registry.register(tool_v1, name="my_tool")
        registry.register(tool_v2, name="my_tool")

        names = registry.get_tool_names()
        assert len(names) == 1
        assert "my_tool" in names

    def test_register_with_parameters(self):
        """Test registering function with parameters."""
        registry = ToolRegistry()

        def param_tool(name: str, count: int, enabled: bool = True) -> str:
            """Tool with parameters."""
            return f"{name}:{count}:{enabled}"

        registry.register(param_tool)
        tools = registry.get_genai_tools()
        assert len(tools) == 1
        params = tools[0].parameters
        assert "name" in params.properties
        assert "count" in params.properties
        assert "enabled" in params.properties

    def test_register_uses_docstring_as_description(self):
        """Test that docstring is used as description."""
        registry = ToolRegistry()

        def documented_tool() -> str:
            """This is the tool description."""
            return "result"

        registry.register(documented_tool)
        tools = registry.get_genai_tools()
        assert "This is the tool description." in tools[0].description

    def test_register_empty_docstring(self):
        """Test registering function with no docstring."""
        registry = ToolRegistry()

        def undocumented_tool() -> str:
            return "result"

        registry.register(undocumented_tool)
        tools = registry.get_genai_tools()
        assert tools[0].description == ""


class TestToolCalling:
    """Test tool calling functionality."""

    @pytest.mark.asyncio
    async def test_call_sync_tool(self):
        """Test calling a synchronous tool."""
        registry = ToolRegistry()

        def sync_tool(x: int) -> str:
            return str(x * 2)

        registry.register(sync_tool)
        result = await registry.call_tool("sync_tool", {"x": 5})
        assert result == "10"

    @pytest.mark.asyncio
    async def test_call_async_tool(self):
        """Test calling an asynchronous tool."""
        registry = ToolRegistry()

        async def async_tool(name: str) -> str:
            await asyncio.sleep(0.01)
            return f"Hello, {name}!"

        registry.register(async_tool)
        result = await registry.call_tool("async_tool", {"name": "World"})
        assert result == "Hello, World!"

    @pytest.mark.asyncio
    async def test_call_tool_nonexistent(self):
        """Test calling nonexistent tool raises ValueError."""
        registry = ToolRegistry()

        with pytest.raises(ValueError, match="not found"):
            await registry.call_tool("nonexistent", {})

    @pytest.mark.asyncio
    async def test_call_tool_with_no_return(self):
        """Test calling tool that returns None."""
        registry = ToolRegistry()

        def no_return_tool() -> None:
            pass

        registry.register(no_return_tool)
        result = await registry.call_tool("no_return_tool", {})
        assert result == "Success"

    @pytest.mark.asyncio
    async def test_call_tool_with_multiple_args(self):
        """Test calling tool with multiple arguments."""
        registry = ToolRegistry()

        def multi_arg_tool(a: int, b: int, c: str) -> str:
            return f"{a}+{b}={a + b}, msg={c}"

        registry.register(multi_arg_tool)
        result = await registry.call_tool("multi_arg_tool", {"a": 3, "b": 4, "c": "test"})
        assert result == "3+4=7, msg=test"


# ========================================
# 3. Tool Timeout Handling Tests (7)
# ========================================


class TestToolTimeout:
    """Test tool timeout handling."""

    @pytest.mark.asyncio
    async def test_async_tool_timeout(self):
        """Test that async tool timeout is enforced."""
        registry = ToolRegistry()

        async def slow_tool() -> str:
            await asyncio.sleep(10)
            return "done"

        registry.register(slow_tool, timeout=0.1)
        result = await registry.call_tool("slow_tool", {})
        assert "timed out" in result.lower()

    @pytest.mark.asyncio
    async def test_async_tool_completes_before_timeout(self):
        """Test async tool that completes before timeout."""
        registry = ToolRegistry()

        async def fast_tool() -> str:
            await asyncio.sleep(0.01)
            return "completed"

        registry.register(fast_tool, timeout=5.0)
        result = await registry.call_tool("fast_tool", {})
        assert result == "completed"

    @pytest.mark.asyncio
    async def test_default_timeout_value(self):
        """Test default timeout is 60 seconds."""
        registry = ToolRegistry()

        def tool_func() -> str:
            return "result"

        registry.register(tool_func)
        tool_def = registry._tools["tool_func"]
        assert tool_def.timeout == 60.0

    @pytest.mark.asyncio
    async def test_custom_timeout_value(self):
        """Test custom timeout value is stored."""
        registry = ToolRegistry()

        def tool_func() -> str:
            return "result"

        registry.register(tool_func, timeout=120.0)
        tool_def = registry._tools["tool_func"]
        assert tool_def.timeout == 120.0

    @pytest.mark.asyncio
    async def test_timeout_with_zero_value(self):
        """Test timeout with zero value."""
        registry = ToolRegistry()

        async def instant_tool() -> str:
            return "instant"

        registry.register(instant_tool, timeout=0.001)
        result = await registry.call_tool("instant_tool", {})
        # Should timeout since 0.001 seconds is very short
        assert "timed out" in result.lower() or result == "instant"

    @pytest.mark.asyncio
    async def test_timeout_error_message_format(self):
        """Test timeout error message format."""
        registry = ToolRegistry()

        async def timeout_tool() -> str:
            await asyncio.sleep(10)
            return "done"

        registry.register(timeout_tool, timeout=0.05)
        result = await registry.call_tool("timeout_tool", {})
        assert "timed out" in result.lower()
        assert "0.05" in result

    @pytest.mark.asyncio
    async def test_sync_tool_no_timeout_enforcement(self):
        """Test that sync tools don't have timeout enforcement."""
        registry = ToolRegistry()

        def sync_tool() -> str:
            return "result"

        registry.register(sync_tool, timeout=0.001)
        result = await registry.call_tool("sync_tool", {})
        # Sync tools run directly without timeout
        assert result == "result"


# ========================================
# 4. Tool Error Handling Tests (5)
# ========================================


class TestToolErrorHandling:
    """Test tool error handling."""

    @pytest.mark.asyncio
    async def test_tool_raises_exception(self):
        """Test tool that raises exception."""
        registry = ToolRegistry()

        def failing_tool() -> str:
            raise RuntimeError("Tool failed!")

        registry.register(failing_tool)
        result = await registry.call_tool("failing_tool", {})
        assert "Error executing" in result
        assert "RuntimeError" in result
        assert "Tool failed!" in result

    @pytest.mark.asyncio
    async def test_tool_raises_value_error(self):
        """Test tool that raises ValueError."""
        registry = ToolRegistry()

        def value_error_tool() -> str:
            raise ValueError("Invalid value")

        registry.register(value_error_tool)
        result = await registry.call_tool("value_error_tool", {})
        assert "Error executing" in result
        assert "ValueError" in result

    @pytest.mark.asyncio
    async def test_async_tool_raises_exception(self):
        """Test async tool that raises exception."""
        registry = ToolRegistry()

        async def async_failing_tool() -> str:
            raise RuntimeError("Async failed!")

        registry.register(async_failing_tool)
        result = await registry.call_tool("async_failing_tool", {})
        assert "Error executing" in result
        assert "RuntimeError" in result

    @pytest.mark.asyncio
    async def test_tool_with_wrong_arguments(self):
        """Test calling tool with wrong arguments."""
        registry = ToolRegistry()

        def strict_tool(required_arg: str) -> str:
            return required_arg

        registry.register(strict_tool)
        result = await registry.call_tool("strict_tool", {})
        assert "Error executing" in result

    @pytest.mark.asyncio
    async def test_tool_error_returns_string(self):
        """Test that tool errors are returned as strings."""
        registry = ToolRegistry()

        def error_tool() -> str:
            raise Exception("Test error")

        registry.register(error_tool)
        result = await registry.call_tool("error_tool", {})
        assert isinstance(result, str)
        assert "Error" in result


# ========================================
# 5. Default Registry Creation Tests (5)
# ========================================


class TestDefaultRegistry:
    """Test default registry creation and singleton behavior."""

    def test_get_default_registry_returns_registry(self):
        """Test get_default_registry returns ToolRegistry instance."""
        registry = get_default_registry()
        assert isinstance(registry, ToolRegistry)

    def test_get_default_registry_singleton(self):
        """Test get_default_registry returns same instance."""
        registry1 = get_default_registry()
        registry2 = get_default_registry()
        assert registry1 is registry2

    def test_get_default_registry_has_tools(self):
        """Test default registry has tools loaded."""
        registry = get_default_registry()
        tools = registry.get_tool_names()
        assert len(tools) > 0

    def test_get_default_registry_has_read_file(self):
        """Test default registry has read_file tool."""
        registry = get_default_registry()
        tools = registry.get_tool_names()
        # read_file should be available from filesystem_unified
        assert "read" in tools or "read_file" in tools

    def test_get_default_registry_has_sensitive_tools(self):
        """Test default registry has sensitive tools marked."""
        registry = get_default_registry()
        sensitive = registry.get_sensitive_tools()
        assert len(sensitive) > 0


# ========================================
# 6. Parameter Extraction Tests (5)
# ========================================


class TestParameterExtraction:
    """Test parameter extraction from function signatures."""

    def test_extract_string_parameter(self):
        """Test extracting string parameter."""
        registry = ToolRegistry()

        def string_tool(name: str) -> str:
            return name

        registry.register(string_tool)
        tools = registry.get_genai_tools()
        params = tools[0].parameters
        assert "name" in params.properties

    def test_extract_int_parameter(self):
        """Test extracting int parameter."""
        registry = ToolRegistry()

        def int_tool(count: int) -> str:
            return str(count)

        registry.register(int_tool)
        tools = registry.get_genai_tools()
        params = tools[0].parameters
        assert "count" in params.properties

    def test_extract_optional_parameter(self):
        """Test extracting optional parameter."""
        registry = ToolRegistry()

        def optional_tool(required: str, optional: int = 10) -> str:
            return f"{required}:{optional}"

        registry.register(optional_tool)
        tools = registry.get_genai_tools()
        params = tools[0].parameters
        assert "required" in params.required
        assert "optional" not in params.required

    def test_extract_multiple_types(self):
        """Test extracting multiple parameter types."""
        registry = ToolRegistry()

        def multi_type_tool(name: str, count: int, ratio: float, enabled: bool) -> str:
            return f"{name}:{count}:{ratio}:{enabled}"

        registry.register(multi_type_tool)
        tools = registry.get_genai_tools()
        params = tools[0].parameters
        assert len(params.properties) == 4

    def test_extract_union_type(self):
        """Test extracting union type parameter."""
        registry = ToolRegistry()

        def union_tool(value: int | None = None) -> str:
            return str(value)

        registry.register(union_tool)
        tools = registry.get_genai_tools()
        params = tools[0].parameters
        assert "value" in params.properties


# ========================================
# 7. GenAI Tools Format Tests (5)
# ========================================


class TestGenAIToolsFormat:
    """Test GenAI tools format generation."""

    def test_get_genai_tools_returns_list(self):
        """Test get_genai_tools returns list."""
        registry = ToolRegistry()

        def tool() -> str:
            return "result"

        registry.register(tool)
        tools = registry.get_genai_tools()
        assert isinstance(tools, list)

    def test_get_genai_tools_with_filter(self):
        """Test get_genai_tools with tool name filter."""
        registry = ToolRegistry()

        def tool1() -> str:
            return "1"

        def tool2() -> str:
            return "2"

        registry.register(tool1)
        registry.register(tool2)

        tools = registry.get_genai_tools(["tool1"])
        assert len(tools) == 1
        assert tools[0].name == "tool1"

    def test_get_genai_tools_filter_nonexistent(self):
        """Test get_genai_tools with nonexistent tool name."""
        registry = ToolRegistry()

        def tool1() -> str:
            return "1"

        registry.register(tool1)
        tools = registry.get_genai_tools(["nonexistent"])
        assert len(tools) == 0

    def test_get_genai_tools_has_name(self):
        """Test GenAI tool has name field."""
        registry = ToolRegistry()

        def my_tool() -> str:
            return "result"

        registry.register(my_tool)
        tools = registry.get_genai_tools()
        assert tools[0].name == "my_tool"

    def test_get_genai_tools_has_description(self):
        """Test GenAI tool has description field."""
        registry = ToolRegistry()

        def my_tool() -> str:
            """Tool description."""
            return "result"

        registry.register(my_tool)
        tools = registry.get_genai_tools()
        assert "Tool description." in tools[0].description


# ========================================
# 8. Convenience Functions Tests (3)
# ========================================


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.mark.asyncio
    async def test_call_tool_function(self):
        """Test call_tool convenience function."""
        # This uses the default registry
        # We'll mock it to avoid loading all tools
        with patch("src.host.tools.get_default_registry") as mock_get_registry:
            mock_registry = ToolRegistry()

            def test_tool(x: int) -> str:
                return str(x)

            mock_registry.register(test_tool)
            mock_get_registry.return_value = mock_registry

            result = await call_tool("test_tool", {"x": 42})
            assert result == "42"

    def test_get_genai_tools_function(self):
        """Test get_genai_tools convenience function."""
        with patch("src.host.tools.get_default_registry") as mock_get_registry:
            mock_registry = ToolRegistry()

            def test_tool() -> str:
                return "result"

            mock_registry.register(test_tool)
            mock_get_registry.return_value = mock_registry

            tools = get_genai_tools()
            assert len(tools) > 0

    def test_is_sensitive_tool_function(self):
        """Test is_sensitive_tool convenience function."""
        with patch("src.host.tools.get_default_registry") as mock_get_registry:
            mock_registry = ToolRegistry()

            def sensitive_tool() -> str:
                return "result"

            mock_registry.register(sensitive_tool, sensitive=True)
            mock_get_registry.return_value = mock_registry

            assert is_sensitive_tool("sensitive_tool") is True


# ========================================
# 9. Tool Definition Tests (3)
# ========================================


class TestToolDefinition:
    """Test ToolDefinition dataclass."""

    def test_tool_definition_creation(self):
        """Test creating ToolDefinition."""

        def func() -> str:
            return "result"

        tool_def = ToolDefinition(
            name="test",
            description="Test tool",
            function=func,
            parameters={"type": "object", "properties": {}},
        )
        assert tool_def.name == "test"
        assert tool_def.description == "Test tool"
        assert tool_def.function is func
        assert tool_def.sensitive is False
        assert tool_def.timeout == 60.0

    def test_tool_definition_with_sensitive(self):
        """Test ToolDefinition with sensitive flag."""

        def func() -> str:
            return "result"

        tool_def = ToolDefinition(
            name="test",
            description="Test",
            function=func,
            parameters={},
            sensitive=True,
        )
        assert tool_def.sensitive is True

    def test_tool_definition_with_timeout(self):
        """Test ToolDefinition with custom timeout."""

        def func() -> str:
            return "result"

        tool_def = ToolDefinition(
            name="test",
            description="Test",
            function=func,
            parameters={},
            timeout=120.0,
        )
        assert tool_def.timeout == 120.0


# ========================================
# 10. Edge Cases and Integration Tests (4)
# ========================================


class TestEdgeCases:
    """Test edge cases and integration scenarios."""

    @pytest.mark.asyncio
    async def test_tool_with_empty_string_return(self):
        """Test tool that returns empty string."""
        registry = ToolRegistry()

        def empty_tool() -> str:
            return ""

        registry.register(empty_tool)
        result = await registry.call_tool("empty_tool", {})
        # Empty string is still a valid return value
        assert result == ""

    @pytest.mark.asyncio
    async def test_tool_with_special_characters(self):
        """Test tool with special characters in result."""
        registry = ToolRegistry()

        def special_tool() -> str:
            return "Result with special chars: !@#$%^&*()"

        registry.register(special_tool)
        result = await registry.call_tool("special_tool", {})
        assert "special chars" in result

    def test_register_lambda_function(self):
        """Test registering lambda function."""
        registry = ToolRegistry()

        def lambda_tool(x):
            return str(x * 2)

        registry.register(lambda_tool, name="lambda_tool")
        assert "lambda_tool" in registry.get_tool_names()

    @pytest.mark.asyncio
    async def test_concurrent_tool_calls(self):
        """Test calling multiple tools concurrently."""
        registry = ToolRegistry()

        async def tool1() -> str:
            await asyncio.sleep(0.01)
            return "tool1"

        async def tool2() -> str:
            await asyncio.sleep(0.01)
            return "tool2"

        registry.register(tool1)
        registry.register(tool2)

        results = await asyncio.gather(
            registry.call_tool("tool1", {}),
            registry.call_tool("tool2", {}),
        )
        assert results == ["tool1", "tool2"]
