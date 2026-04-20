"""Tests for src/gateway/client.py"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.gateway.client import GatewayClient, GatewayConfig


class TestGatewayConfig:
    def test_defaults(self):
        cfg = GatewayConfig()
        assert cfg.base_url == "http://localhost:8000"
        assert cfg.api_key is None
        assert cfg.timeout == 120.0

    def test_custom(self):
        cfg = GatewayConfig(base_url="http://remote:9000", api_key="sk-123", timeout=60.0)
        assert cfg.base_url == "http://remote:9000"
        assert cfg.api_key == "sk-123"
        assert cfg.timeout == 60.0

    def test_from_env_defaults(self, monkeypatch):
        monkeypatch.delenv("AGENT_GATEWAY_URL", raising=False)
        monkeypatch.delenv("AGENT_API_KEY", raising=False)
        monkeypatch.delenv("AGENT_GATEWAY_TIMEOUT", raising=False)
        cfg = GatewayConfig.from_env()
        assert cfg.base_url == "http://localhost:8000"
        assert cfg.api_key is None

    def test_from_env_custom(self, monkeypatch):
        monkeypatch.setenv("AGENT_GATEWAY_URL", "http://custom:5000")
        monkeypatch.setenv("AGENT_API_KEY", "key-abc")
        monkeypatch.setenv("AGENT_GATEWAY_TIMEOUT", "30")
        cfg = GatewayConfig.from_env()
        assert cfg.base_url == "http://custom:5000"
        assert cfg.api_key == "key-abc"
        assert cfg.timeout == 30.0


class TestGatewayClient:
    def test_init_default(self):
        client = GatewayClient()
        assert client._client is None

    def test_init_custom_config(self):
        cfg = GatewayConfig(base_url="http://test:1234")
        client = GatewayClient(config=cfg)
        assert client.config.base_url == "http://test:1234"

    @pytest.mark.asyncio
    async def test_connect(self):
        client = GatewayClient(config=GatewayConfig(base_url="http://localhost:9999"))
        await client.connect()
        assert client._client is not None
        await client.close()

    @pytest.mark.asyncio
    async def test_connect_with_api_key(self):
        client = GatewayClient(config=GatewayConfig(api_key="sk-test"))
        await client.connect()
        assert "Authorization" in client._client.headers
        assert client._client.headers["Authorization"] == "Bearer sk-test"
        await client.close()

    @pytest.mark.asyncio
    async def test_close(self):
        client = GatewayClient()
        await client.connect()
        await client.close()
        assert client._client is not None  # client object still exists but is closed

    @pytest.mark.asyncio
    async def test_close_without_connect(self):
        client = GatewayClient()
        await client.close()

    @pytest.mark.asyncio
    async def test_chat_not_connected_raises(self):
        client = GatewayClient()
        with pytest.raises(RuntimeError, match="not connected"):
            await client.chat("model", [])

    @pytest.mark.asyncio
    async def test_list_models_not_connected_raises(self):
        client = GatewayClient()
        with pytest.raises(RuntimeError, match="not connected"):
            await client.list_models()

    @pytest.mark.asyncio
    async def test_health_not_connected_raises(self):
        client = GatewayClient()
        with pytest.raises(RuntimeError, match="not connected"):
            await client.health()

    @pytest.mark.asyncio
    async def test_chat_stream_not_connected_raises(self):
        client = GatewayClient()
        with pytest.raises(RuntimeError, match="not connected"):
            async for _ in client.chat_stream("model", []):
                pass

    @pytest.mark.asyncio
    async def test_context_manager(self):
        cfg = GatewayConfig(base_url="http://localhost:9999")
        async with GatewayClient(config=cfg) as client:
            assert client._client is not None

    @pytest.mark.asyncio
    async def test_chat_sends_request(self):
        client = GatewayClient(config=GatewayConfig(base_url="http://localhost:9999"))
        await client.connect()

        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"choices": []})
        mock_response.raise_for_status = MagicMock()

        mock_post = AsyncMock(return_value=mock_response)
        client._client.post = mock_post

        result = await client.chat(
            "gpt-4", [{"role": "user", "content": "hi"}], temperature=0.5, max_tokens=100
        )
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["json"]["model"] == "gpt-4"
        assert call_kwargs[1]["json"]["temperature"] == 0.5
        assert call_kwargs[1]["json"]["max_tokens"] == 100

        await client.close()

    @pytest.mark.asyncio
    async def test_chat_with_tools(self):
        client = GatewayClient(config=GatewayConfig(base_url="http://localhost:9999"))
        await client.connect()

        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"choices": []})
        mock_response.raise_for_status = MagicMock()
        client._client.post = AsyncMock(return_value=mock_response)

        tools = [{"type": "function", "function": {"name": "read"}}]
        await client.chat("model", [], tools=tools)
        call_kwargs = client._client.post.call_args
        assert "tools" in call_kwargs[1]["json"]

        await client.close()


class TestCreateClient:
    @pytest.mark.asyncio
    async def test_create_client(self, monkeypatch):
        monkeypatch.delenv("AGENT_GATEWAY_URL", raising=False)
        monkeypatch.delenv("AGENT_API_KEY", raising=False)
        monkeypatch.delenv("AGENT_GATEWAY_TIMEOUT", raising=False)
        client = await __import__("src.gateway.client", fromlist=["create_client"]).create_client()
        assert isinstance(client, GatewayClient)
        assert client._client is not None
        await client.close()


class TestGatewayClientStreamAndHealth:
    @pytest.mark.asyncio
    async def test_chat_stream_lines(self):
        client = GatewayClient(config=GatewayConfig(base_url="http://localhost:9999"))
        await client.connect()

        async def aiter_lines():
            yield 'data: {"choices": []}'
            yield "data: [DONE]"
            yield "data: not-json"
            yield ""

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = aiter_lines

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)
        client._client.stream = MagicMock(return_value=mock_stream_ctx)

        chunks = []
        async for chunk in client.chat_stream("model", []):
            chunks.append(chunk)
        assert len(chunks) == 1
        assert chunks[0] == {"choices": []}
        await client.close()

    @pytest.mark.asyncio
    async def test_list_models_success(self):
        client = GatewayClient(config=GatewayConfig(base_url="http://localhost:9999"))
        await client.connect()

        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value={"data": [{"id": "gpt-4"}]})
        mock_response.raise_for_status = MagicMock()
        client._client.get = AsyncMock(return_value=mock_response)

        result = await client.list_models()
        assert result == [{"id": "gpt-4"}]
        await client.close()

    @pytest.mark.asyncio
    async def test_health_success(self):
        client = GatewayClient(config=GatewayConfig(base_url="http://localhost:9999"))
        await client.connect()

        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value={"status": "ok"})
        mock_response.raise_for_status = MagicMock()
        client._client.get = AsyncMock(return_value=mock_response)

        result = await client.health()
        assert result == {"status": "ok"}
        await client.close()

    @pytest.mark.asyncio
    async def test_chat_stream_not_connected_raises(self):
        client = GatewayClient()
        with pytest.raises(RuntimeError, match="not connected"):
            async for _ in client.chat_stream("model", []):
                pass
