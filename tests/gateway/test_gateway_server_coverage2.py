"""Additional coverage tests for gateway.server - middleware, anthropic route, streaming errors, providers endpoint."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.gateway.schema import ChatMessage, ChatResponse, ToolCall, Usage


class TestMiddlewareChain:
    def test_content_length_too_large(self, monkeypatch):
        from src.gateway.server import app
        from fastapi.testclient import TestClient
        monkeypatch.setattr("src.gateway.server.GATEWAY_API_KEY", None)
        monkeypatch.setattr("src.gateway.server.GATEWAY_ALLOW_NO_KEY", True)
        monkeypatch.setattr("src.gateway.server.MAX_REQUEST_BODY_BYTES", 100)
        client = TestClient(app, raise_server_exceptions=False)
        big_body = json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": "x" * 200}]})
        resp = client.post(
            "/v1/chat/completions",
            content=big_body,
            headers={"Content-Type": "application/json", "Content-Length": str(len(big_body))},
        )
        assert resp.status_code == 413

    def test_invalid_content_length(self, monkeypatch):
        from src.gateway.server import app
        from fastapi.testclient import TestClient
        monkeypatch.setattr("src.gateway.server.GATEWAY_API_KEY", None)
        monkeypatch.setattr("src.gateway.server.GATEWAY_ALLOW_NO_KEY", True)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/v1/chat/completions",
            content="{}",
            headers={"Content-Type": "application/json", "Content-Length": "not-a-number"},
        )
        assert resp.status_code == 400

    def test_chunked_transfer_oversized(self, monkeypatch):
        from src.gateway.server import app
        from fastapi.testclient import TestClient
        monkeypatch.setattr("src.gateway.server.GATEWAY_API_KEY", None)
        monkeypatch.setattr("src.gateway.server.GATEWAY_ALLOW_NO_KEY", True)
        monkeypatch.setattr("src.gateway.server.MAX_REQUEST_BODY_BYTES", 50)
        client = TestClient(app, raise_server_exceptions=False)
        big_body = json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": "x" * 200}]})
        resp = client.post(
            "/v1/chat/completions",
            content=big_body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 413


class TestAnthropicRouteWithToolCalls:
    def test_anthropic_messages_with_tools_non_streaming(self, monkeypatch):
        from src.gateway.server import app, _build_chat_request_from_anthropic
        from fastapi.testclient import TestClient
        monkeypatch.setattr("src.gateway.server.GATEWAY_API_KEY", None)
        monkeypatch.setattr("src.gateway.server.GATEWAY_ALLOW_NO_KEY", True)
        chat_response = ChatResponse(
            id="r1",
            model="claude-3",
            choices=[
                {
                    "index": 0,
                    "message": ChatMessage(
                        role="assistant",
                        content="hello",
                        tool_calls=[ToolCall(id="tc1", name="fn", arguments={"a": 1})],
                    ),
                    "finish_reason": "tool_calls",
                }
            ],
            usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )
        with patch("src.gateway.server.ModelRouter") as MockRouter:
            mock_router = MagicMock()
            mock_router.chat = AsyncMock(return_value=chat_response)
            app.state.router = mock_router
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/v1/messages",
                json={
                    "model": "claude-3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": [{"name": "fn", "description": "test", "input_schema": {}}],
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert any(c["type"] == "tool_use" for c in data["content"])


class TestStreamingError:
    @pytest.mark.asyncio
    async def test_stream_response_error(self):
        from src.gateway.server import stream_response, ChatRequest

        class FakeRouter:
            async def chat_stream(self, request, **kw):
                raise RuntimeError("stream broke")
                yield

        req = ChatRequest(model="x", messages=[])
        chunks = []
        async for chunk in stream_response(req, FakeRouter()):
            chunks.append(chunk)
        assert any("error" in c for c in chunks)
        assert any("[DONE]" in c for c in chunks)


class TestProvidersEndpoint:
    def test_providers_with_router(self, monkeypatch):
        from src.gateway.server import app
        from fastapi.testclient import TestClient
        monkeypatch.setattr("src.gateway.server.GATEWAY_API_KEY", None)
        monkeypatch.setattr("src.gateway.server.GATEWAY_ALLOW_NO_KEY", True)
        mock_router = MagicMock()
        mock_router.get_providers.return_value = [{"name": "openai", "enabled": True}]
        app.state.router = mock_router
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/v1/providers")
        assert resp.status_code == 200
        assert resp.json()["providers"] == [{"name": "openai", "enabled": True}]

    def test_providers_no_router(self, monkeypatch):
        from src.gateway.server import app
        from fastapi.testclient import TestClient
        monkeypatch.setattr("src.gateway.server.GATEWAY_API_KEY", None)
        monkeypatch.setattr("src.gateway.server.GATEWAY_ALLOW_NO_KEY", True)
        app.state.router = None
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/v1/providers")
        assert resp.status_code == 503


class TestHealthEndpoint:
    def test_health_no_router(self, monkeypatch):
        from src.gateway.server import app
        from fastapi.testclient import TestClient
        monkeypatch.setattr("src.gateway.server.GATEWAY_API_KEY", None)
        monkeypatch.setattr("src.gateway.server.GATEWAY_ALLOW_NO_KEY", True)
        app.state.router = None
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "initializing"

    def test_health_with_router(self, monkeypatch):
        from src.gateway.server import app
        from fastapi.testclient import TestClient
        monkeypatch.setattr("src.gateway.server.GATEWAY_API_KEY", None)
        monkeypatch.setattr("src.gateway.server.GATEWAY_ALLOW_NO_KEY", True)
        mock_router = MagicMock()
        mock_router.health_check = AsyncMock(return_value={"openai": "healthy"})
        app.state.router = mock_router
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"


class TestAnthropicConversion:
    def test_convert_with_dict_choice(self):
        from src.gateway.server import _convert_chat_response_to_anthropic
        response = ChatResponse(
            id="r1",
            model="m",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi", "tool_calls": None},
                    "finish_reason": "stop",
                }
            ],
            usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )
        payload = _convert_chat_response_to_anthropic(response)
        assert payload["content"][0]["type"] == "text"
        assert payload["content"][0]["text"] == "hi"

    def test_convert_tool_calls_as_dicts(self):
        from src.gateway.server import _convert_chat_response_to_anthropic
        response = ChatResponse(
            id="r1",
            model="m",
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"id": "tc1", "function": {"name": "fn", "arguments": {"x": 1}}}],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )
        payload = _convert_chat_response_to_anthropic(response)
        assert payload["content"][0]["type"] == "tool_use"

    def test_anthropic_stop_reason_length(self):
        from src.gateway.server import _anthropic_stop_reason
        assert _anthropic_stop_reason("length") == "max_tokens"

    def test_anthropic_stop_reason_default(self):
        from src.gateway.server import _anthropic_stop_reason
        assert _anthropic_stop_reason("stop") == "end_turn"
