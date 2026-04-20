"""Targeted coverage for gateway/server.py uncovered lines: 56-64,115,388-389,474-479,515-523,549-569,586,591,599,743-749,753."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.gateway.server import (
    _anthropic_stop_reason,
    _build_chat_request_from_anthropic,
    _build_chat_request_from_openai,
    _convert_chat_response_to_anthropic,
    _parse_tool_call_arguments,
    app,
    verify_api_key,
    load_config,
)


class TestLifespan:
    @pytest.mark.asyncio
    async def test_lifespan_initializes_router(self):
        from src.gateway.server import lifespan
        mock_app = MagicMock()
        mock_app.state = MagicMock()
        with patch("src.gateway.server.ModelRouter") as MockRouter:
            mock_router = MagicMock()
            mock_router.initialize = AsyncMock()
            mock_router.close = AsyncMock()
            MockRouter.return_value = mock_router
            async with lifespan(mock_app):
                assert mock_app.state.router is not None
            mock_router.close.assert_called_once()
            assert mock_app.state.router is None


class TestMiddlewareLargeBody:
    def test_chunked_transfer_large_body(self):
        client = TestClient(app)
        with patch("src.gateway.server.verify_api_key", return_value=True):
            resp = client.post(
                "/v1/chat/completions",
                content="x" * (11 * 1024 * 1024),
                headers={"content-type": "application/json"},
            )
            assert resp.status_code == 413


class TestVerifyApiKey:
    def test_no_key_configured_allowed(self, monkeypatch):
        import src.gateway.server as srv
        monkeypatch.setattr(srv, "GATEWAY_API_KEY", None)
        monkeypatch.setattr(srv, "GATEWAY_ALLOW_NO_KEY", True)
        assert verify_api_key(None) is True

    def test_no_key_configured_not_allowed(self, monkeypatch):
        import src.gateway.server as srv
        monkeypatch.setattr(srv, "GATEWAY_API_KEY", None)
        monkeypatch.setattr(srv, "GATEWAY_ALLOW_NO_KEY", False)
        assert verify_api_key(None) is False

    def test_bearer_prefix(self, monkeypatch):
        import src.gateway.server as srv
        monkeypatch.setattr(srv, "GATEWAY_API_KEY", "mykey")
        assert verify_api_key("Bearer mykey") is True

    def test_raw_key(self, monkeypatch):
        import src.gateway.server as srv
        monkeypatch.setattr(srv, "GATEWAY_API_KEY", "mykey")
        assert verify_api_key("mykey") is True

    def test_wrong_key(self, monkeypatch):
        import src.gateway.server as srv
        monkeypatch.setattr(srv, "GATEWAY_API_KEY", "mykey")
        assert verify_api_key("wrong") is False


class TestParseToolCallArguments:
    def test_dict_passthrough(self):
        assert _parse_tool_call_arguments({"a": 1}) == {"a": 1}

    def test_none_returns_empty(self):
        assert _parse_tool_call_arguments(None) == {}

    def test_empty_returns_empty(self):
        assert _parse_tool_call_arguments("") == {}

    def test_valid_json(self):
        assert _parse_tool_call_arguments('{"a":1}') == {"a": 1}

    def test_invalid_json_returns_empty(self):
        assert _parse_tool_call_arguments("not json") == {}


class TestAnthropicStopReason:
    def test_tool_calls(self):
        assert _anthropic_stop_reason("tool_calls") == "tool_use"

    def test_length(self):
        assert _anthropic_stop_reason("length") == "max_tokens"

    def test_stop(self):
        assert _anthropic_stop_reason("stop") == "end_turn"

    def test_none(self):
        assert _anthropic_stop_reason(None) == "end_turn"


class TestConvertChatResponseToAnthropic:
    def test_with_dict_choice(self):
        from src.gateway.schema import ChatResponse, Usage

        resp = ChatResponse(
            id="test",
            model="gpt-4",
            choices=[{"message": {"content": "hi", "tool_calls": []}, "finish_reason": "stop"}],
            usage=Usage(prompt_tokens=10, completion_tokens=5),
        )
        result = _convert_chat_response_to_anthropic(resp)
        assert result["role"] == "assistant"
        assert result["stop_reason"] == "end_turn"

    def test_with_object_choice(self):
        from src.gateway.schema import ChatResponse, Choice, ChatMessage, Usage

        msg = ChatMessage(role="assistant", content="hello")
        resp = ChatResponse(
            id="test2",
            model="gpt-4",
            choices=[
                Choice(
                    message=msg,
                    finish_reason="stop",
                    index=0,
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=5),
        )
        result = _convert_chat_response_to_anthropic(resp)
        assert result["role"] == "assistant"

    def test_with_tool_calls_as_objects(self):
        from src.gateway.schema import ChatResponse, Choice, ChatMessage, ToolCall, Usage

        tc = ToolCall(id="c1", name="read", arguments={"path": "/f"})
        msg = ChatMessage(role="assistant", content="using tool", tool_calls=[tc])
        resp = ChatResponse(
            id="test3",
            model="gpt-4",
            choices=[
                Choice(
                    message=msg,
                    finish_reason="tool_calls",
                    index=0,
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=5),
        )
        result = _convert_chat_response_to_anthropic(resp)
        assert result["stop_reason"] == "tool_use"
        tool_uses = [c for c in result["content"] if c["type"] == "tool_use"]
        assert len(tool_uses) == 1


class TestBuildChatRequestFromAnthropic:
    def test_with_system_and_tool_result(self):
        from src.gateway.server import AnthropicMessage, AnthropicMessagesRequest, AnthropicToolDefinition

        req = AnthropicMessagesRequest(
            model="claude-3",
            messages=[
                AnthropicMessage(role="user", content="hi"),
                AnthropicMessage(
                    role="assistant",
                    content=[
                        {"type": "text", "text": "thinking"},
                        {"type": "tool_use", "id": "c1", "name": "read", "input": {"path": "/f"}},
                    ],
                ),
                AnthropicMessage(
                    role="user",
                    content=[
                        {"type": "tool_result", "tool_use_id": "c1", "content": "file content"},
                    ],
                ),
            ],
            system="You are helpful",
            tools=[AnthropicToolDefinition(name="read", description="read file")],
        )
        chat_req = _build_chat_request_from_anthropic(req)
        assert chat_req.messages[0].role == "system"
        assert len(chat_req.tools) == 1


class TestModelEndpoints:
    def test_list_models_no_auth(self, monkeypatch):
        monkeypatch.setenv("AGENT_API_KEY", "secret")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/v1/models")
        assert resp.status_code == 401

    def test_get_model_no_auth(self, monkeypatch):
        monkeypatch.setenv("AGENT_API_KEY", "secret")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/v1/models/gpt-4")
        assert resp.status_code == 401

    def test_providers_no_auth(self, monkeypatch):
        monkeypatch.setenv("AGENT_API_KEY", "secret")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/v1/providers")
        assert resp.status_code == 401


class TestMainFunction:
    def test_main_function_exists(self):
        from src.gateway.server import main
        assert callable(main)
