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
                    "function": {"name": "read_file", "arguments": '{"path": "/test"}'},
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
                        "name": "read_file",
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
        from src.gateway.server import ChatCompletionMessage, ChatCompletionRequest, ToolCallInfo
        from src.gateway.server import ToolCallFunction, _build_chat_request_from_openai

        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[
                ChatCompletionMessage(
                    role="assistant",
                    tool_calls=[
                        ToolCallInfo(
                            id="call_1",
                            function=ToolCallFunction(
                                name="read_file",
                                arguments='{"path":"/tmp/demo.txt"}',
                            ),
                        )
                    ],
                )
            ],
        )

        chat_request = _build_chat_request_from_openai(request)
        tool_call = chat_request.messages[0].tool_calls[0]

        assert tool_call.name == "read_file"
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
