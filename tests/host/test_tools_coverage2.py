"""Targeted coverage for host/tools.py uncovered lines: 220-222,235,259,272-273,297-298,304,370,476,567-569,581-582,599,656-663,764-766,783-786,805-810,822,833,838."""

import asyncio
import types as py_types
from typing import Any, Union, get_args, get_origin
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.host.tools import (
    LazyToolFunction,
    ToolAuditEntry,
    ToolDefinition,
    ToolRegistry,
    _create_registry,
    call_tool,
    get_default_registry,
    get_extension_registry,
    get_genai_tools,
    is_sensitive_tool,
)


class TestExtractParametersFallback:
    def test_signature_raises_import_error(self):
        registry = ToolRegistry()

        class BadFunc:
            __name__ = "bad"

        with patch("src.host.tools.inspect.signature", side_effect=ImportError("bad")):
            registry.register(BadFunc(), name="bad_tool")
            tool = registry._tools["bad_tool"]
            assert tool.parameters.get("additionalProperties") is True

    def test_signature_raises_value_error(self):
        registry = ToolRegistry()

        class BadFunc:
            __name__ = "bad2"

        with patch("src.host.tools.inspect.signature", side_effect=ValueError("bad")):
            registry.register(BadFunc(), name="bad2")
            tool = registry._tools["bad2"]
            assert tool.parameters.get("additionalProperties") is True

    def test_signature_raises_type_error(self):
        registry = ToolRegistry()

        class BadFunc:
            __name__ = "bad3"

        with patch("src.host.tools.inspect.signature", side_effect=TypeError("bad")):
            registry.register(BadFunc(), name="bad3")
            tool = registry._tools["bad3"]
            assert tool.parameters.get("additionalProperties") is True


class TestAnnotationToSchemaEdgeCases:
    def test_self_parameter_skipped(self):
        registry = ToolRegistry()

        class Dummy:
            def method(self, x: int):
                pass

        registry.register(Dummy().method, name="m")
        tool = registry._tools["m"]
        assert "self" not in tool.parameters.get("properties", {})

    def test_union_type_with_multiple_non_none(self):
        registry = ToolRegistry()

        def func(x: Union[str, int]):
            pass

        registry.register(func, name="union_test")
        tool = registry._tools["union_test"]
        schema = tool.parameters["properties"]["x"]
        assert "anyOf" in schema

    def test_annotation_with_union_type_attribute(self):
        registry = ToolRegistry()

        def func(x: str | int):
            pass

        registry.register(func, name="uniontype_test")
        tool = registry._tools["uniontype_test"]
        schema = tool.parameters["properties"]["x"]
        assert "anyOf" in schema or "type" in schema

    def test_dict_annotation_with_value_schema(self):
        registry = ToolRegistry()

        def func(x: dict[str, int]):
            pass

        registry.register(func, name="dict_test")
        tool = registry._tools["dict_test"]
        schema = tool.parameters["properties"]["x"]
        assert schema["type"] == "object"
        assert "additionalProperties" in schema

    def test_annotation_to_schema_empty_annotation(self):
        registry = ToolRegistry()
        import inspect

        def func(x):
            pass

        registry.register(func, name="empty_ann")
        tool = registry._tools["empty_ann"]
        schema = tool.parameters["properties"]["x"]
        assert schema["type"] == "string"

    def test_annotation_to_schema_list(self):
        registry = ToolRegistry()

        def func(x: list):
            pass

        registry.register(func, name="list_test")
        tool = registry._tools["list_test"]
        schema = tool.parameters["properties"]["x"]
        assert schema["type"] == "array"

    def test_annotation_to_schema_dict_no_args(self):
        registry = ToolRegistry()

        def func(x: dict):
            pass

        registry.register(func, name="dict_noargs")
        tool = registry._tools["dict_noargs"]
        schema = tool.parameters["properties"]["x"]
        assert schema["type"] == "object"


class TestCheckToolExecutionBackground:
    def test_background_not_safe(self):
        registry = ToolRegistry()

        def bg_tool(x: str) -> str:
            return x

        registry.register(bg_tool, name="bg_tool", metadata={"capability_group": "edit"})
        with patch("src.host.tools.get_default_tool_policy_engine") as mock_engine:
            mock_policy = MagicMock()
            mock_policy.to_dict.return_value = {
                "visible": True,
                "requires_approval": False,
                "background_safe": False,
                "sandbox": "none",
                "audit_level": "full",
                "source": "built_in",
            }
            mock_engine.return_value.describe_tool.return_value = mock_policy
            allowed, reason, policy = registry.check_tool_execution(
                "bg_tool", mode="build", background=True
            )
            assert allowed is False
            assert "background" in reason.lower()


class TestGetGenaiToolsWithFilter:
    def test_genai_tools_with_tool_names(self):
        registry = ToolRegistry()

        def read_t(path: str) -> str:
            return path

        def write_t(path: str) -> str:
            return path

        registry.register(read_t, name="read")
        registry.register(write_t, name="write")
        decls = registry.get_genai_tools(tool_names=["read"])
        assert len(decls) == 1
        assert decls[0].name == "read"

    def test_genai_tools_skip_non_listed(self):
        registry = ToolRegistry()

        def a_tool() -> str:
            return "a"

        def b_tool() -> str:
            return "b"

        registry.register(a_tool, name="a")
        registry.register(b_tool, name="b")
        decls = registry.get_genai_tools(tool_names=["a"])
        assert all(d.name == "a" for d in decls)


class TestCallToolCaching:
    @pytest.mark.asyncio
    async def test_cached_result_returned(self):
        registry = ToolRegistry()

        def my_tool(x: str) -> str:
            return f"result_{x}"

        registry.register(my_tool, name="cached_tool")
        with patch("src.core.tool_cache.should_cache_tool", return_value=True):
            with patch("src.core.tool_cache.ToolCache") as MockCache:
                instance = MagicMock()
                instance.get.return_value = "cached_value"
                MockCache.get_instance.return_value = instance
                result = await registry.call_tool("cached_tool", {"x": "a"})
                assert result == "cached_value"

    @pytest.mark.asyncio
    async def test_cache_set_on_success(self):
        registry = ToolRegistry()

        def my_tool(x: str) -> str:
            return "fresh_result"

        registry.register(my_tool, name="set_cache_tool")
        with patch("src.core.tool_cache.should_cache_tool", return_value=True):
            with patch("src.core.tool_cache.ToolCache") as MockCache:
                instance = MagicMock()
                instance.get.return_value = None
                MockCache.get_instance.return_value = instance
                result = await registry.call_tool("set_cache_tool", {"x": "a"})
                assert result == "fresh_result"
                instance.set.assert_called_once()


class TestCallToolAsyncFunction:
    @pytest.mark.asyncio
    async def test_async_tool_with_lazy_loading(self):
        registry = ToolRegistry()
        ltf = LazyToolFunction("src.servers.memory", "memory_list")
        registry.register(ltf, name="async_lazy")
        try:
            await registry.call_tool("async_lazy", {})
        except Exception:
            pass


class TestCallToolConvenienceFunction:
    @pytest.mark.asyncio
    async def test_call_tool_convenience(self):
        with patch("src.host.tools.get_default_registry") as mock_get:
            mock_registry = MagicMock()
            mock_registry.call_tool = AsyncMock(return_value="result")
            mock_get.return_value = mock_registry
            result = await call_tool("read", {"path": "/f"})
            assert result == "result"


class TestGetGenaiToolsConvenience:
    def test_get_genai_tools_convenience(self):
        with patch("src.host.tools.get_default_registry") as mock_get:
            mock_registry = MagicMock()
            mock_registry.get_genai_tools.return_value = []
            mock_get.return_value = mock_registry
            result = get_genai_tools()
            assert result == []


class TestIsSensitiveToolConvenience:
    def test_is_sensitive_convenience(self):
        with patch("src.host.tools.get_default_registry") as mock_get:
            mock_registry = MagicMock()
            mock_registry.is_sensitive.return_value = True
            mock_get.return_value = mock_registry
            assert is_sensitive_tool("write") is True


class TestCreateRegistryCriticalFailure:
    def test_critical_tool_failure_raises(self):
        from src.host.tools import ToolSpec

        specs = [ToolSpec("nonexistent.module.xyz", "func", critical=True)]
        with patch("src.host.tools.load_config", return_value={}):
            with pytest.raises(Exception):
                _create_registry(specs, source="built_in")

    def test_non_critical_failure_continues(self):
        from src.host.tools import ToolSpec

        specs = [ToolSpec("nonexistent.module.xyz", "func", critical=False)]
        with patch("src.host.tools.load_config", return_value={}):
            registry = _create_registry(specs, source="built_in")
            assert isinstance(registry, ToolRegistry)
            assert "func" in registry.get_tool_names()

    def test_config_load_failure_uses_defaults(self):
        from src.host.tools import ToolSpec

        specs = [ToolSpec("nonexistent.module.xyz", "func", critical=False)]
        with patch("src.host.tools.load_config", side_effect=RuntimeError("config boom")):
            registry = _create_registry(specs, source="built_in")
            assert isinstance(registry, ToolRegistry)
