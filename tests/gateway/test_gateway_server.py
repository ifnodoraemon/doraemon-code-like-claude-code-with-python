"""Tests for the Gateway server."""

import pytest

# Note: These tests require fastapi and httpx
pytest.importorskip("fastapi")
pytest.importorskip("httpx")


from src.gateway.schema import (
    ChatMessage as Message,
)
from src.gateway.schema import (
    ChatRequest,
    ChatResponse,
    ModelInfo,
    ToolCall,
    Usage,
)


class TestGatewaySchema:
    """Tests for Gateway schema models."""

    def test_message_user(self):
        """Test user message creation."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_message_assistant_with_tool_calls(self):
        """Test assistant message with tool calls."""
        msg = Message(
            role="assistant",
            content=None,
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path": "/test"}'},
                }
            ],
        )
        assert msg.role == "assistant"
        assert len(msg.tool_calls) == 1

    def test_chat_request(self):
        """Test chat request creation."""
        request = ChatRequest(
            model="gemini-2.5-flash",
            messages=[
                Message(role="user", content="Hello"),
            ],
        )
        assert request.model == "gemini-2.5-flash"
        assert len(request.messages) == 1

    def test_chat_request_with_tools(self):
        """Test chat request with tools."""
        request = ChatRequest(
            model="gpt-4",
            messages=[Message(role="user", content="Read file")],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "description": "Read a file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                        },
                    },
                }
            ],
        )
        assert len(request.tools) == 1

    def test_chat_response(self):
        """Test chat response creation."""
        response = ChatResponse(
            id="resp_123",
            model="gemini-2.5-flash",
            choices=[
                {
                    "index": 0,
                    "message": Message(role="assistant", content="Hello!"),
                    "finish_reason": "stop",
                }
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        assert response.id == "resp_123"
        assert len(response.choices) == 1
        assert response.usage.total_tokens == 15

    def test_model_info(self):
        """Test model info creation."""
        info = ModelInfo(
            id="gemini-2.5-flash",
            name="Gemini 2.5 Flash",
            provider="google",
            context_window=1000000,
        )
        assert info.id == "gemini-2.5-flash"
        assert info.provider == "google"


class TestGatewayRouter:
    """Tests for Gateway router."""

    def test_provider_detection_google(self):
        """Test Google provider detection patterns."""
        from src.gateway.router import ModelRouter

        # Test PROVIDER_PATTERNS directly since get_provider is internal
        patterns = ModelRouter.PROVIDER_PATTERNS
        model = "gemini-2.5-flash"
        provider = None
        for p, pats in patterns.items():
            for pat in pats:
                if model.startswith(pat):
                    provider = p
                    break
        assert provider == "google"

    def test_provider_detection_openai(self):
        """Test OpenAI provider detection patterns."""
        from src.gateway.router import ModelRouter

        patterns = ModelRouter.PROVIDER_PATTERNS
        model = "gpt-4"
        provider = None
        for p, pats in patterns.items():
            for pat in pats:
                if model.startswith(pat):
                    provider = p
                    break
        assert provider == "openai"

    def test_provider_detection_anthropic(self):
        """Test Anthropic provider detection patterns."""
        from src.gateway.router import ModelRouter

        patterns = ModelRouter.PROVIDER_PATTERNS
        model = "claude-3-opus"
        provider = None
        for p, pats in patterns.items():
            for pat in pats:
                if model.startswith(pat):
                    provider = p
                    break
        assert provider == "anthropic"

    def test_list_models_requires_initialization(self):
        """Test that list_models requires initialized adapters."""
        from src.gateway.router import ModelRouter

        # Router needs config and initialization to list models
        router = ModelRouter(config={})
        models = router.list_models()

        # Without initialization, no models should be returned
        assert models == []


class TestGatewayServerConversions:
    """Tests for protocol conversion helpers."""

    def test_openai_tool_call_arguments_are_parsed(self):
        """OpenAI tool call arguments should be parsed from the arguments field."""
        from src.gateway.server import (
            ChatCompletionMessage,
            ChatCompletionRequest,
            ToolCallFunction,
            ToolCallInfo,
            _build_chat_request_from_openai,
        )

        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[
                ChatCompletionMessage(
                    role="assistant",
                    tool_calls=[
                        ToolCallInfo(
                            id="call_1",
                            function=ToolCallFunction(
                                name="read",
                                arguments='{"path":"/tmp/demo.txt"}',
                            ),
                        )
                    ],
                )
            ],
        )

        chat_request = _build_chat_request_from_openai(request)
        tool_call = chat_request.messages[0].tool_calls[0]

        assert tool_call.name == "read"
        assert tool_call.arguments == {"path": "/tmp/demo.txt"}

    def test_build_chat_request_from_anthropic(self):
        """Anthropic request bodies should map into the unified schema."""
        from src.gateway.server import AnthropicMessagesRequest, _build_chat_request_from_anthropic

        request = AnthropicMessagesRequest(
            model="minimax-m2.5",
            system="You are concise.",
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello"}],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "calling tool"},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "lookup",
                            "input": {"city": "Shanghai"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "Sunny",
                        }
                    ],
                },
            ],
            tools=[
                {
                    "name": "lookup",
                    "description": "Lookup weather",
                    "input_schema": {"type": "object"},
                }
            ],
            max_tokens=128,
        )

        chat_request = _build_chat_request_from_anthropic(request)

        assert chat_request.messages[0].role == "system"
        assert chat_request.messages[1].content == "hello"
        assert chat_request.messages[2].tool_calls[0].name == "lookup"
        assert chat_request.messages[3].role == "tool"
        assert chat_request.messages[3].tool_call_id == "toolu_1"
        assert chat_request.tools[0].name == "lookup"

    def test_convert_chat_response_to_anthropic(self):
        """Unified responses should map back to Anthropic format."""
        from src.gateway.server import _convert_chat_response_to_anthropic

        response = ChatResponse(
            id="msg_123",
            model="minimax-m2.5",
            choices=[
                {
                    "index": 0,
                    "message": Message(
                        role="assistant",
                        content="pong",
                        tool_calls=[
                            ToolCall(
                                id="toolu_1",
                                name="lookup",
                                arguments={"city": "Shanghai"},
                            )
                        ],
                    ),
                    "finish_reason": "tool_calls",
                }
            ],
            usage=Usage(prompt_tokens=11, completion_tokens=7, total_tokens=18),
        )

        payload = _convert_chat_response_to_anthropic(response)

        assert payload["type"] == "message"
        assert payload["content"][0]["text"] == "pong"
        assert payload["content"][1]["type"] == "tool_use"
        assert payload["content"][1]["input"] == {"city": "Shanghai"}
        assert payload["stop_reason"] == "tool_use"
        assert payload["usage"] == {"input_tokens": 11, "output_tokens": 7}


class TestUsage:
    """Tests for Usage model."""

    def test_usage_calculation(self):
        """Test usage token calculation."""
        usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150

    def test_usage_default(self):
        """Test usage with defaults."""
        usage = Usage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0


class TestVerifyApiKey:
    """Tests for API key verification."""

    def test_no_key_configured_allow_no_key(self, monkeypatch):
        from src.gateway.server import verify_api_key

        monkeypatch.setattr("src.gateway.server.GATEWAY_API_KEY", None)
        monkeypatch.setattr("src.gateway.server.GATEWAY_ALLOW_NO_KEY", True)
        assert verify_api_key(None) is True

    def test_no_key_configured_deny(self, monkeypatch):
        from src.gateway.server import verify_api_key

        monkeypatch.setattr("src.gateway.server.GATEWAY_API_KEY", None)
        monkeypatch.setattr("src.gateway.server.GATEWAY_ALLOW_NO_KEY", False)
        assert verify_api_key(None) is False

    def test_valid_key(self, monkeypatch):
        from src.gateway.server import verify_api_key

        monkeypatch.setattr("src.gateway.server.GATEWAY_API_KEY", "secret123")
        assert verify_api_key("secret123") is True

    def test_bearer_prefix(self, monkeypatch):
        from src.gateway.server import verify_api_key

        monkeypatch.setattr("src.gateway.server.GATEWAY_API_KEY", "secret123")
        assert verify_api_key("Bearer secret123") is True

    def test_invalid_key(self, monkeypatch):
        from src.gateway.server import verify_api_key

        monkeypatch.setattr("src.gateway.server.GATEWAY_API_KEY", "secret123")
        assert verify_api_key("wrong") is False

    def test_no_authorization_with_key_set(self, monkeypatch):
        from src.gateway.server import verify_api_key

        monkeypatch.setattr("src.gateway.server.GATEWAY_API_KEY", "secret123")
        assert verify_api_key(None) is False


class TestLoadConfig:
    def test_load_config(self, monkeypatch):
        from src.gateway.server import load_config

        monkeypatch.setenv("GOOGLE_API_KEY", "g-key")
        monkeypatch.setenv("OPENAI_API_KEY", "o-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "a-key")
        config = load_config()
        assert config["google"]["enabled"] is True
        assert config["openai"]["api_key_env"] == "OPENAI_API_KEY"
        assert config["anthropic"]["enabled"] is True


class TestParseToolCallArguments:
    def test_dict_passthrough(self):
        from src.gateway.server import _parse_tool_call_arguments

        assert _parse_tool_call_arguments({"a": 1}) == {"a": 1}

    def test_none_returns_empty(self):
        from src.gateway.server import _parse_tool_call_arguments

        assert _parse_tool_call_arguments(None) == {}

    def test_empty_string(self):
        from src.gateway.server import _parse_tool_call_arguments

        assert _parse_tool_call_arguments("") == {}

    def test_valid_json(self):
        from src.gateway.server import _parse_tool_call_arguments

        assert _parse_tool_call_arguments('{"x": 1}') == {"x": 1}

    def test_invalid_json(self):
        from src.gateway.server import _parse_tool_call_arguments

        assert _parse_tool_call_arguments("not-json") == {}


class TestAnthropicStopReason:
    def test_tool_calls(self):
        from src.gateway.server import _anthropic_stop_reason

        assert _anthropic_stop_reason("tool_calls") == "tool_use"

    def test_length(self):
        from src.gateway.server import _anthropic_stop_reason

        assert _anthropic_stop_reason("length") == "max_tokens"

    def test_stop(self):
        from src.gateway.server import _anthropic_stop_reason

        assert _anthropic_stop_reason("stop") == "end_turn"

    def test_none(self):
        from src.gateway.server import _anthropic_stop_reason

        assert _anthropic_stop_reason(None) == "end_turn"


class TestFormatSse:
    def test_dict_payload(self):
        from src.gateway.server import _format_sse

        result = _format_sse("message_start", {"type": "message_start"})
        assert "event: message_start" in result
        assert "data:" in result

    def test_string_payload(self):
        from src.gateway.server import _format_sse

        result = _format_sse("ping", "ok")
        assert "event: ping" in result
        assert "data: ok" in result


class TestBuildOpenaiMessages:
    def test_simple_user_message(self):
        from src.gateway.server import (
            ChatCompletionMessage,
            ChatCompletionRequest,
            _build_openai_messages,
        )

        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[ChatCompletionMessage(role="user", content="hi")],
        )
        msgs = _build_openai_messages(request)
        assert len(msgs) == 1
        assert msgs[0].role == "user"
        assert msgs[0].content == "hi"

    def test_message_with_tool_calls(self):
        from src.gateway.server import (
            ChatCompletionMessage,
            ChatCompletionRequest,
            ToolCallFunction,
            ToolCallInfo,
            _build_openai_messages,
        )

        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[
                ChatCompletionMessage(
                    role="assistant",
                    tool_calls=[
                        ToolCallInfo(
                            id="c1",
                            function=ToolCallFunction(
                                name="fn1", arguments={"x": 1}
                            ),
                        )
                    ],
                )
            ],
        )
        msgs = _build_openai_messages(request)
        assert msgs[0].tool_calls is not None
        assert msgs[0].tool_calls[0].name == "fn1"
        assert msgs[0].tool_calls[0].arguments == {"x": 1}

    def test_message_with_tool_response(self):
        from src.gateway.server import (
            ChatCompletionMessage,
            ChatCompletionRequest,
            _build_openai_messages,
        )

        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[
                ChatCompletionMessage(
                    role="tool", content="result", tool_call_id="c1"
                )
            ],
        )
        msgs = _build_openai_messages(request)
        assert msgs[0].role == "tool"
        assert msgs[0].tool_call_id == "c1"


class TestBuildOpenaiTools:
    def test_none_tools(self):
        from src.gateway.server import _build_openai_tools

        assert _build_openai_tools(None) is None

    def test_empty_tools(self):
        from src.gateway.server import _build_openai_tools

        assert _build_openai_tools([]) is None

    def test_with_tools(self):
        from src.gateway.server import (
            ChatCompletionTool,
            FunctionDefinition,
            _build_openai_tools,
        )

        tools = [
            ChatCompletionTool(
                function=FunctionDefinition(
                    name="fn1", description="desc", parameters={"type": "object"}
                )
            )
        ]
        result = _build_openai_tools(tools)
        assert len(result) == 1
        assert result[0].name == "fn1"


class TestBuildChatRequestFromOpenai:
    def test_full_request(self):
        from src.gateway.server import (
            ChatCompletionMessage,
            ChatCompletionRequest,
            ChatCompletionTool,
            FunctionDefinition,
            _build_chat_request_from_openai,
        )

        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[ChatCompletionMessage(role="user", content="hi")],
            tools=[
                ChatCompletionTool(
                    function=FunctionDefinition(name="fn", parameters={})
                )
            ],
            temperature=0.5,
            max_tokens=100,
            stream=True,
            stop=["\n"],
            top_p=0.9,
            presence_penalty=0.1,
            frequency_penalty=0.2,
        )
        chat_req = _build_chat_request_from_openai(request)
        assert chat_req.model == "gpt-4"
        assert chat_req.temperature == 0.5
        assert chat_req.max_tokens == 100
        assert chat_req.stream is True
        assert chat_req.stop == ["\n"]
        assert chat_req.top_p == 0.9
        assert chat_req.presence_penalty == 0.1
        assert chat_req.frequency_penalty == 0.2
        assert chat_req.tools is not None
        assert len(chat_req.tools) == 1


class TestBuildChatRequestFromAnthropic:
    def test_user_with_string_content(self):
        from src.gateway.server import (
            AnthropicMessage,
            AnthropicMessagesRequest,
            _build_chat_request_from_anthropic,
        )

        request = AnthropicMessagesRequest(
            model="claude-3",
            messages=[AnthropicMessage(role="user", content="hello")],
        )
        chat_req = _build_chat_request_from_anthropic(request)
        assert chat_req.messages[0].content == "hello"

    def test_tool_result_with_list_content(self):
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
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [
                                {"type": "text", "text": "part1"},
                                {"type": "text", "text": "part2"},
                            ],
                        }
                    ],
                )
            ],
        )
        chat_req = _build_chat_request_from_anthropic(request)
        tool_msg = chat_req.messages[0]
        assert tool_msg.role == "tool"
        assert tool_msg.content == "part1part2"

    def test_tool_result_with_string_content(self):
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
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "plain string result",
                        }
                    ],
                )
            ],
        )
        chat_req = _build_chat_request_from_anthropic(request)
        tool_msg = chat_req.messages[0]
        assert tool_msg.role == "tool"
        assert tool_msg.content == "plain string result"

    def test_no_system_no_tools(self):
        from src.gateway.server import (
            AnthropicMessagesRequest,
            _build_chat_request_from_anthropic,
        )

        request = AnthropicMessagesRequest(
            model="claude-3",
            messages=[],
            tools=None,
        )
        chat_req = _build_chat_request_from_anthropic(request)
        assert chat_req.tools is None

    def test_user_message_with_only_tool_result_blocks(self):
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
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "result",
                        }
                    ],
                )
            ],
        )
        chat_req = _build_chat_request_from_anthropic(request)
        assert len(chat_req.messages) >= 1

    def test_anthropic_stop_reasons_all(self):
        from src.gateway.server import _anthropic_stop_reason

        assert _anthropic_stop_reason("tool_calls") == "tool_use"
        assert _anthropic_stop_reason("length") == "max_tokens"
        assert _anthropic_stop_reason("stop") == "end_turn"
        assert _anthropic_stop_reason("anything_else") == "end_turn"
        assert _anthropic_stop_reason(None) == "end_turn"


class TestConvertChatResponseToAnthropic:
    def test_with_dict_choice(self):
        from src.gateway.server import _convert_chat_response_to_anthropic

        response = ChatResponse(
            id="r1",
            model="m",
            choices=[
                {
                    "index": 0,
                    "message": {
                        "content": "hi",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {"name": "fn", "arguments": {"a": 1}},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )
        payload = _convert_chat_response_to_anthropic(response)
        assert payload["content"][0]["type"] == "text"
        assert payload["content"][0]["text"] == "hi"
        assert payload["content"][1]["type"] == "tool_use"
        assert payload["content"][1]["name"] == "fn"

    def test_with_no_content(self):
        from src.gateway.server import _convert_chat_response_to_anthropic

        response = ChatResponse(
            id="r1",
            model="m",
            choices=[
                {
                    "index": 0,
                    "message": Message(role="assistant", content=None),
                    "finish_reason": "stop",
                }
            ],
            usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )
        payload = _convert_chat_response_to_anthropic(response)
        assert payload["content"] == []
        assert payload["stop_reason"] == "end_turn"

    def test_with_no_usage(self):
        from src.gateway.server import _convert_chat_response_to_anthropic

        response = ChatResponse(
            id="r1",
            model="m",
            choices=[
                {
                    "index": 0,
                    "message": Message(role="assistant", content="hi"),
                    "finish_reason": "stop",
                }
            ],
            usage=None,
        )
        payload = _convert_chat_response_to_anthropic(response)
        assert payload["usage"]["input_tokens"] == 0
        assert payload["usage"]["output_tokens"] == 0

    def test_tool_call_dict_form(self):
        from src.gateway.server import _convert_chat_response_to_anthropic

        response = ChatResponse(
            id="r1",
            model="m",
            choices=[
                {
                    "index": 0,
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {
                                    "name": "fn",
                                    "arguments": {"x": 1},
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )
        payload = _convert_chat_response_to_anthropic(response)
        assert payload["content"][0]["type"] == "tool_use"
        assert payload["content"][0]["name"] == "fn"
        assert payload["content"][0]["input"] == {"x": 1}


class TestMiddleware:
    def test_limit_request_body_within_limit(self, monkeypatch):
        from src.gateway.server import app

        monkeypatch.setattr("src.gateway.server.MAX_REQUEST_BODY_BYTES", 1000)
        from fastapi.testclient import TestClient

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_limit_request_body_invalid_content_length(self, monkeypatch):
        from src.gateway.server import app

        monkeypatch.setattr("src.gateway.server.MAX_REQUEST_BODY_BYTES", 1000)
        from fastapi.testclient import TestClient

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/v1/chat/completions",
            headers={"Content-Length": "not-a-number"},
            json={"model": "x", "messages": []},
        )
        assert resp.status_code == 400


class TestHealthEndpoint:
    def test_health_no_router(self):
        from src.gateway.server import app
        from fastapi.testclient import TestClient

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] in ("healthy", "initializing")


class TestLimitRequestBodyOversize:
    def test_oversize_body_rejected(self, monkeypatch):
        from src.gateway.server import app
        from fastapi.testclient import TestClient

        monkeypatch.setattr("src.gateway.server.MAX_REQUEST_BODY_BYTES", 10)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/v1/chat/completions",
            content=b"x" * 100,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 413


class TestListModelsEndpoint:
    def test_unauthenticated(self, monkeypatch):
        from src.gateway.server import app
        from fastapi.testclient import TestClient

        monkeypatch.setattr("src.gateway.server.GATEWAY_API_KEY", "secret")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/v1/models")
        assert resp.status_code == 401


class TestGetModelEndpoint:
    def test_unauthenticated(self, monkeypatch):
        from src.gateway.server import app
        from fastapi.testclient import TestClient

        monkeypatch.setattr("src.gateway.server.GATEWAY_API_KEY", "secret")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/v1/models/gpt-4")
        assert resp.status_code == 401


class TestChatCompletionsEndpoint:
    def test_unauthenticated(self, monkeypatch):
        from src.gateway.server import app
        from fastapi.testclient import TestClient

        monkeypatch.setattr("src.gateway.server.GATEWAY_API_KEY", "secret")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": []},
        )
        assert resp.status_code == 401


class TestAnthropicMessagesEndpoint:
    def test_unauthenticated(self, monkeypatch):
        from src.gateway.server import app
        from fastapi.testclient import TestClient

        monkeypatch.setattr("src.gateway.server.GATEWAY_API_KEY", "secret")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/v1/messages",
            json={"model": "claude-3", "messages": []},
        )
        assert resp.status_code == 401


class TestListProvidersEndpoint:
    def test_unauthenticated(self, monkeypatch):
        from src.gateway.server import app
        from fastapi.testclient import TestClient

        monkeypatch.setattr("src.gateway.server.GATEWAY_API_KEY", "secret")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/v1/providers")
        assert resp.status_code == 401


class TestStreamResponseFunction:
    @pytest.mark.asyncio
    async def test_stream_error_handling(self):
        from src.gateway.server import stream_response, ChatRequest

        class ErrorRouter:
            async def chat_stream(self, request, **kw):
                from src.gateway.schema import ErrorResponse
                yield ErrorResponse(code="server_error", error="boom")

        req = ChatRequest(model="x", messages=[])
        chunks = []
        async for chunk in stream_response(req, ErrorRouter()):
            chunks.append(chunk)
        assert len(chunks) >= 1
        assert "error" in chunks[0].lower() or "boom" in chunks[0]

    @pytest.mark.asyncio
    async def test_stream_exception_handling(self):
        from src.gateway.server import stream_response, ChatRequest

        class BrokenRouter:
            async def chat_stream(self, request, **kw):
                raise RuntimeError("stream broke")
                yield

        req = ChatRequest(model="x", messages=[])
        chunks = []
        async for chunk in stream_response(req, BrokenRouter()):
            chunks.append(chunk)
        assert any("error" in c.lower() for c in chunks)


class TestStreamAnthropicResponseFunction:
    @pytest.mark.asyncio
    async def test_error_chunk(self):
        from src.gateway.server import stream_anthropic_response, ChatRequest
        from src.gateway.schema import ErrorResponse

        class ErrorRouter:
            async def chat_stream(self, request, **kw):
                yield ErrorResponse(code="server_error", error="fail")

        req = ChatRequest(model="x", messages=[])
        chunks = []
        async for chunk in stream_anthropic_response(req, ErrorRouter()):
            chunks.append(chunk)
        assert any("error" in c.lower() for c in chunks)
