"""Targeted coverage tests for gateway.server."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.gateway.schema import ChatMessage, ChatResponse, ToolCall, Usage


class TestAnthropicMessagesRoute:
    @pytest.mark.asyncio
    async def test_anthropic_messages_unauthenticated(self):
        from src.gateway.server import app
        from fastapi.testclient import TestClient

        with patch("src.gateway.server.GATEWAY_API_KEY", "secret"):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/v1/messages",
                json={"model": "claude-3", "messages": []},
            )
            assert resp.status_code == 401


class TestProvidersRoute:
    @pytest.mark.asyncio
    async def test_providers_unauthenticated(self):
        from src.gateway.server import app
        from fastapi.testclient import TestClient

        with patch("src.gateway.server.GATEWAY_API_KEY", "secret"):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/v1/providers")
            assert resp.status_code == 401


class TestStreamAnthropicResponseContentBlock:
    @pytest.mark.asyncio
    async def test_stream_anthropic_with_content_and_tools(self):
        from src.gateway.server import stream_anthropic_response, ChatRequest

        class FakeStreamChunk:
            def __init__(self, **kw):
                self.id = kw.get("id", "msg_1")
                self.model = kw.get("model", "m")
                self.delta_content = kw.get("delta_content")
                self.delta_tool_calls = kw.get("delta_tool_calls")
                self.finish_reason = kw.get("finish_reason")
                self.usage = kw.get("usage", Usage())

        class FakeRouter:
            async def chat_stream(self, request, **kw):
                yield FakeStreamChunk(delta_content="Hello ")
                yield FakeStreamChunk(delta_content="World")
                tc = ToolCall(id="tc_1", name="lookup", arguments={"k": "v"})
                yield FakeStreamChunk(delta_tool_calls=[tc])
                yield FakeStreamChunk(finish_reason="stop", usage=Usage(prompt_tokens=10, completion_tokens=5))

        req = ChatRequest(model="x", messages=[])
        chunks = []
        async for chunk in stream_anthropic_response(req, FakeRouter()):
            chunks.append(chunk)
        assert len(chunks) >= 3
        assert any("message_start" in c for c in chunks)
        assert any("content_block_delta" in c for c in chunks)
        assert any("message_stop" in c for c in chunks)


class TestStreamResponseNormal:
    @pytest.mark.asyncio
    async def test_stream_normal_chunk(self):
        from src.gateway.server import stream_response, ChatRequest

        class FakeChunk:
            def to_dict(self):
                return {"choices": [{"delta": {"content": "ok"}}]}

        class FakeRouter:
            async def chat_stream(self, request, **kw):
                yield FakeChunk()

        req = ChatRequest(model="x", messages=[])
        chunks = []
        async for chunk in stream_response(req, FakeRouter()):
            chunks.append(chunk)
        assert any("ok" in c for c in chunks)
        assert any("[DONE]" in c for c in chunks)


class TestBuildAnthropicRequestEdgeCases:
    def test_user_with_only_tool_result_blocks_no_extra_message(self):
        from src.gateway.server import (
            AnthropicMessage,
            AnthropicMessagesRequest,
            _build_chat_request_from_anthropic,
        )

        request = AnthropicMessagesRequest(
            model="claude-3",
            messages=[
                AnthropicMessage(
                    role="user",
                    content=[
                        {"type": "tool_result", "tool_use_id": "t1", "content": "result"},
                    ],
                )
            ],
        )
        chat_req = _build_chat_request_from_anthropic(request)
        tool_msgs = [m for m in chat_req.messages if m.role == "tool"]
        assert len(tool_msgs) == 1

    def test_assistant_with_text_and_tool_use(self):
        from src.gateway.server import (
            AnthropicMessage,
            AnthropicMessagesRequest,
            _build_chat_request_from_anthropic,
        )

        request = AnthropicMessagesRequest(
            model="claude-3",
            messages=[
                AnthropicMessage(
                    role="assistant",
                    content=[
                        {"type": "text", "text": "calling tool"},
                        {"type": "tool_use", "id": "tu1", "name": "fn", "input": {}},
                    ],
                )
            ],
        )
        chat_req = _build_chat_request_from_anthropic(request)
        assert chat_req.messages[0].content == "calling tool"
        assert chat_req.messages[0].tool_calls is not None


class TestConvertChatResponseToAnthropicToolCallObject:
    def test_tool_call_as_object_not_dict(self):
        from src.gateway.server import _convert_chat_response_to_anthropic

        tc = ToolCall(id="tc1", name="fn", arguments={"a": 1})
        response = ChatResponse(
            id="r1",
            model="m",
            choices=[
                {
                    "index": 0,
                    "message": ChatMessage(role="assistant", content=None, tool_calls=[tc]),
                    "finish_reason": "tool_calls",
                }
            ],
            usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )
        payload = _convert_chat_response_to_anthropic(response)
        assert payload["content"][0]["type"] == "tool_use"
        assert payload["content"][0]["id"] == "tc1"
        assert payload["content"][0]["name"] == "fn"


class TestVerifyApiKeyEdge:
    def test_no_key_with_no_key_allowed(self, monkeypatch):
        from src.gateway.server import verify_api_key
        monkeypatch.setattr("src.gateway.server.GATEWAY_API_KEY", None)
        monkeypatch.setattr("src.gateway.server.GATEWAY_ALLOW_NO_KEY", True)
        assert verify_api_key(None) is True

    def test_key_without_bearer_prefix(self, monkeypatch):
        from src.gateway.server import verify_api_key
        monkeypatch.setattr("src.gateway.server.GATEWAY_API_KEY", "mykey")
        assert verify_api_key("mykey") is True
