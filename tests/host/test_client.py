"""Tests for src/host/client.py"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.host.client import InProcessToolClient, _InProcessSession, _ListToolsResult, _ToolInfo


class TestToolInfo:
    def test_fields(self):
        info = _ToolInfo(name="read", description="Read files", inputSchema={"type": "object"})
        assert info.name == "read"
        assert info.description == "Read files"
        assert info.inputSchema == {"type": "object"}


class TestListToolsResult:
    def test_tools_list(self):
        tools = [_ToolInfo(name="a", description="", inputSchema={})]
        result = _ListToolsResult(tools=tools)
        assert len(result.tools) == 1
        assert result.tools[0].name == "a"


class TestInProcessSession:
    @pytest.mark.asyncio
    async def test_list_tools(self):
        mock_def = MagicMock()
        mock_def.name = "read"
        mock_def.description = "Read a file"
        mock_def.parameters = {"type": "object"}

        registry = MagicMock()
        registry.get_tool_names.return_value = ["read"]
        registry._tools = {"read": mock_def}

        session = _InProcessSession(registry)
        result = await session.list_tools()
        assert isinstance(result, _ListToolsResult)
        assert len(result.tools) == 1
        assert result.tools[0].name == "read"
        assert result.tools[0].description == "Read a file"

    @pytest.mark.asyncio
    async def test_list_tools_empty(self):
        registry = MagicMock()
        registry.get_tool_names.return_value = []
        registry._tools = {}

        session = _InProcessSession(registry)
        result = await session.list_tools()
        assert result.tools == []


class TestInProcessToolClient:
    def test_init_defaults(self):
        client = InProcessToolClient()
        assert client.tracer is None
        assert client.registry is None
        assert client.sessions == {}
        assert client._runtime is None

    def test_init_with_tracer(self):
        tracer = MagicMock()
        client = InProcessToolClient(tracer=tracer)
        assert client.tracer is tracer

    @pytest.mark.asyncio
    async def test_connect_to_config(self):
        mock_runtime = MagicMock()
        mock_registry = MagicMock()
        mock_runtime.registry = mock_registry
        mock_runtime.aclose = AsyncMock()

        with patch("src.host.client.bootstrap_runtime", new_callable=AsyncMock, return_value=mock_runtime):
            client = InProcessToolClient()
            config = {"mode": "build", "project": "test_proj"}
            await client.connect_to_config(config)
            assert client.registry is mock_registry
            assert "default" in client.sessions
            assert client._runtime is mock_runtime

    @pytest.mark.asyncio
    async def test_connect_to_config_none(self):
        mock_runtime = MagicMock()
        mock_registry = MagicMock()
        mock_runtime.registry = mock_registry
        mock_runtime.aclose = AsyncMock()

        with patch("src.host.client.bootstrap_runtime", new_callable=AsyncMock, return_value=mock_runtime) as mock_bs:
            client = InProcessToolClient()
            await client.connect_to_config(None)
            mock_bs.assert_called_once_with(
                mode="build",
                project="default",
                extension_tools=None,
                create_model_client=False,
            )

    @pytest.mark.asyncio
    async def test_call_tool_not_connected(self):
        client = InProcessToolClient()
        with pytest.raises(RuntimeError, match="Client not connected"):
            await client.call_tool("read", {"path": "test.py"})

    @pytest.mark.asyncio
    async def test_call_tool_success(self):
        registry = MagicMock()
        registry.call_tool = AsyncMock(return_value="file contents")
        client = InProcessToolClient()
        client.registry = registry

        result = await client.call_tool("read", {"path": "test.py"})
        assert result == "file contents"
        registry.call_tool.assert_called_once_with("read", {"path": "test.py"})

    @pytest.mark.asyncio
    async def test_call_tool_with_tracer(self):
        tracer = MagicMock()
        registry = MagicMock()
        registry.call_tool = AsyncMock(return_value="result")
        client = InProcessToolClient(tracer=tracer)
        client.registry = registry

        await client.call_tool("read", {"path": "x.py"})
        tracer.log.assert_any_call("tool_call", "read", {"path": "x.py"})
        tracer.log.assert_any_call("tool_result", "read", "result")

    @pytest.mark.asyncio
    async def test_cleanup(self):
        runtime = MagicMock()
        runtime.aclose = AsyncMock()
        client = InProcessToolClient()
        client._runtime = runtime
        client.sessions = {"default": MagicMock()}

        await client.cleanup()
        runtime.aclose.assert_called_once()
        assert client.sessions == {}

    @pytest.mark.asyncio
    async def test_cleanup_no_runtime(self):
        client = InProcessToolClient()
        client.sessions = {"default": MagicMock()}
        await client.cleanup()
        assert client.sessions == {}
