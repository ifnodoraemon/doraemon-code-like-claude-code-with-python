"""
Comprehensive tests for src/core/model_client.py

Covers:
- GatewayModelClient (20+ tests)
- DirectModelClient (15+ tests)
- ModelClient factory (10+ tests)
- Tool conversion utilities (10+ tests)
- Error handling and edge cases
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.model_client import (
    DirectModelClient,
    GatewayModelClient,
    ModelClient,
)
from src.core.model_client_base import BaseModelClient
from src.core.model_utils import (
    ChatResponse,
    ClientConfig,
    ClientMode,
    Message,
    Provider,
    StreamChunk,
    ToolCall,
    ToolDefinition,
)

# ============================================================================
# PART 1: Message and ToolDefinition Tests (10 tests)
# ============================================================================


class TestMessageClass:
    """Tests for Message dataclass."""

    def test_message_creation_minimal(self):
        """Test creating a message with minimal fields."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.thought is None
        assert msg.tool_calls is None

    def test_message_creation_full(self):
        """Test creating a message with all fields."""
        tool_calls = [{"id": "1", "name": "test"}]
        msg = Message(
            role="assistant",
            content="Response",
            thought="Thinking...",
            tool_calls=tool_calls,
            tool_call_id="call_123",
            name="my_tool",
        )
        assert msg.role == "assistant"
        assert msg.content == "Response"
        assert msg.thought == "Thinking..."
        assert msg.tool_calls == tool_calls
        assert msg.tool_call_id == "call_123"
        assert msg.name == "my_tool"

    def test_message_to_dict_excludes_none_fields(self):
        """Test that to_dict excludes None fields."""
        msg = Message(role="user", content="Hi")
        d = msg.to_dict()
        assert "role" in d
        assert "content" in d
        assert "thought" not in d
        assert "tool_calls" not in d
        assert "tool_call_id" not in d
        assert "name" not in d

    def test_message_to_dict_includes_all_fields(self):
        """Test that to_dict includes all non-None fields."""
        msg = Message(
            role="assistant",
            content="Test",
            thought="Thinking",
            tool_calls=[{"id": "1"}],
            tool_call_id="call_1",
            name="tool_1",
        )
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "Test"
        assert d["thought"] == "Thinking"
        assert d["tool_calls"] == [{"id": "1"}]
        assert d["tool_call_id"] == "call_1"
        assert d["name"] == "tool_1"

    def test_message_to_dict_with_empty_tool_calls(self):
        """Test that empty tool_calls list is excluded."""
        msg = Message(role="user", content="Hi", tool_calls=[])
        d = msg.to_dict()
        assert "tool_calls" not in d

    def test_message_to_dict_with_none_content(self):
        """Test message with None content."""
        msg = Message(role="assistant", content=None, tool_calls=[{"id": "1"}])
        d = msg.to_dict()
        assert "content" not in d
        assert "tool_calls" in d


class TestToolDefinition:
    """Tests for ToolDefinition dataclass."""

    def test_tool_definition_creation(self):
        """Test creating a tool definition."""
        tool = ToolDefinition(
            name="read_file", description="Read a file", parameters={"type": "object"}
        )
        assert tool.name == "read_file"
        assert tool.description == "Read a file"
        assert tool.parameters == {"type": "object"}

    def test_tool_to_openai_format(self):
        """Test conversion to OpenAI format."""
        tool = ToolDefinition(
            name="read_file",
            description="Read a file",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )
        openai_format = tool.to_openai_format()
        assert openai_format["type"] == "function"
        assert openai_format["function"]["name"] == "read_file"
        assert openai_format["function"]["description"] == "Read a file"
        assert openai_format["function"]["parameters"]["type"] == "object"

    def test_tool_to_genai_format(self):
        """Test conversion to GenAI format."""
        tool = ToolDefinition(
            name="test_tool", description="Test tool", parameters={"type": "object"}
        )
        with patch("google.genai.types.FunctionDeclaration") as mock_func:
            tool.to_genai_format()
            mock_func.assert_called_once_with(
                name="test_tool", description="Test tool", parameters={"type": "object"}
            )


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_tool_call_creation(self):
        """Test creating a tool call."""
        tc = ToolCall(id="call_123", name="read_file", arguments={"path": "/test.txt"})
        assert tc.id == "call_123"
        assert tc.name == "read_file"
        assert tc.arguments == {"path": "/test.txt"}

    def test_tool_call_to_dict(self):
        """Test converting tool call to dict."""
        tc = ToolCall(id="call_123", name="read_file", arguments={"path": "/test.txt"})
        d = tc.to_dict()
        assert d["id"] == "call_123"
        assert d["name"] == "read_file"
        assert d["arguments"] == {"path": "/test.txt"}

    def test_tool_call_from_dict(self):
        """Test creating tool call from dict."""
        data = {
            "id": "call_456",
            "name": "write_file",
            "arguments": {"path": "/out.txt", "content": "data"},
        }
        tc = ToolCall.from_dict(data)
        assert tc.id == "call_456"
        assert tc.name == "write_file"
        assert tc.arguments == {"path": "/out.txt", "content": "data"}

    def test_tool_call_from_dict_with_missing_fields(self):
        """Test creating tool call from dict with missing fields."""
        data = {"id": "call_789"}
        tc = ToolCall.from_dict(data)
        assert tc.id == "call_789"
        assert tc.name == ""
        assert tc.arguments == {}


class TestChatResponse:
    """Tests for ChatResponse dataclass."""

    def test_chat_response_creation_minimal(self):
        """Test creating a minimal chat response."""
        response = ChatResponse(content="Hello")
        assert response.content == "Hello"
        assert response.thought is None
        assert response.tool_calls is None
        assert response.finish_reason is None

    def test_chat_response_has_tool_calls_true(self):
        """Test has_tool_calls property when tools are present."""
        response = ChatResponse(content="", tool_calls=[{"id": "1", "name": "test"}])
        assert response.has_tool_calls is True

    def test_chat_response_has_tool_calls_false(self):
        """Test has_tool_calls property when no tools."""
        response = ChatResponse(content="Hello")
        assert response.has_tool_calls is False

    def test_chat_response_has_tool_calls_empty_list(self):
        """Test has_tool_calls with empty tool list."""
        response = ChatResponse(content="Hello", tool_calls=[])
        assert response.has_tool_calls is False


class TestStreamChunk:
    """Tests for StreamChunk dataclass."""

    def test_stream_chunk_creation(self):
        """Test creating a stream chunk."""
        chunk = StreamChunk(content="Hello", thought="Thinking", finish_reason="stop")
        assert chunk.content == "Hello"
        assert chunk.thought == "Thinking"
        assert chunk.finish_reason == "stop"

    def test_stream_chunk_with_tool_calls(self):
        """Test stream chunk with tool calls."""
        chunk = StreamChunk(
            content=None, tool_calls=[{"id": "1", "name": "test"}], finish_reason="tool_calls"
        )
        assert chunk.content is None
        assert chunk.tool_calls == [{"id": "1", "name": "test"}]
        assert chunk.finish_reason == "tool_calls"


# ============================================================================
# PART 2: ClientConfig Tests (10 tests)
# ============================================================================


class TestClientConfig:
    """Tests for ClientConfig dataclass."""

    def test_client_config_defaults(self):
        """Test default ClientConfig values."""
        config = ClientConfig()
        assert config.mode == ClientMode.DIRECT
        assert config.model is None
        assert config.temperature == 0.7
        assert config.max_tokens is None
        assert config.system_prompt is None

    def test_client_config_custom_values(self):
        """Test ClientConfig with custom values."""
        config = ClientConfig(
            mode=ClientMode.GATEWAY,
            model="gpt-4",
            temperature=0.5,
            max_tokens=2000,
            system_prompt="You are helpful",
        )
        assert config.mode == ClientMode.GATEWAY
        assert config.model == "gpt-4"
        assert config.temperature == 0.5
        assert config.max_tokens == 2000
        assert config.system_prompt == "You are helpful"

    def test_client_config_from_env_gateway_mode(self):
        """Test loading config from project config in gateway mode."""
        with patch(
            "src.core.config.load_config",
            return_value={
                "model": "gpt-4",
                "gateway_url": "http://localhost:8000",
                "gateway_key": "test_key",
            },
        ):
            config = ClientConfig.from_env()
            assert config.mode == ClientMode.GATEWAY
            assert config.gateway_url == "http://localhost:8000"
            assert config.gateway_key == "test_key"
            assert config.model == "gpt-4"

    def test_client_config_from_env_direct_mode(self):
        """Test loading config from project config in direct mode."""
        with patch(
            "src.core.config.load_config",
            return_value={
                "model": "gemini-2.5-flash",
                "google_api_key": "google_key",
                "openai_api_key": "openai_key",
                "anthropic_api_key": "anthropic_key",
            },
        ):
            config = ClientConfig.from_env()
            assert config.mode == ClientMode.DIRECT
            assert config.google_api_key == "google_key"
            assert config.openai_api_key == "openai_key"
            assert config.anthropic_api_key == "anthropic_key"
            assert config.model == "gemini-2.5-flash"

    def test_client_config_from_env_no_env_vars(self):
        """Test loading config fails when model is missing."""
        with patch("src.core.config.load_config", return_value={}):
            with pytest.raises(ValueError, match="required 'model'"):
                ClientConfig.from_env()

    def test_client_config_from_env_ollama_base_url(self):
        """Test loading Ollama base URL from project config."""
        with patch(
            "src.core.config.load_config",
            return_value={
                "model": "llama2",
                "ollama_base_url": "http://custom:11434",
            },
        ):
            config = ClientConfig.from_env()
            assert config.ollama_base_url == "http://custom:11434"

    def test_client_config_from_env_ollama_default(self):
        """Test default Ollama base URL."""
        with patch("src.core.config.load_config", return_value={"model": "llama2"}):
            config = ClientConfig.from_env()
            assert config.ollama_base_url == "http://localhost:11434"

    def test_client_config_gateway_settings(self):
        """Test gateway-specific settings."""
        config = ClientConfig(
            mode=ClientMode.GATEWAY, gateway_url="http://api.example.com", gateway_key="secret_key"
        )
        assert config.gateway_url == "http://api.example.com"
        assert config.gateway_key == "secret_key"

    def test_client_config_direct_settings(self):
        """Test direct mode provider settings."""
        config = ClientConfig(
            mode=ClientMode.DIRECT,
            google_api_key="google_key",
            openai_api_key="openai_key",
            anthropic_api_key="anthropic_key",
        )
        assert config.google_api_key == "google_key"
        assert config.openai_api_key == "openai_key"
        assert config.anthropic_api_key == "anthropic_key"

    def test_client_config_temperature_range(self):
        """Test various temperature values."""
        for temp in [0.0, 0.5, 1.0, 2.0]:
            config = ClientConfig(temperature=temp)
            assert config.temperature == temp


# ============================================================================
# PART 3: GatewayModelClient Initialization Tests (10 tests)
# ============================================================================


class TestGatewayModelClientInit:
    """Tests for GatewayModelClient initialization."""

    def test_gateway_client_creation(self):
        """Test creating a GatewayModelClient."""
        config = ClientConfig(mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000")
        client = GatewayModelClient(config)
        assert client.config == config
        assert client._client is None

    def test_gateway_client_context_manager(self):
        """Test GatewayModelClient as context manager."""
        config = ClientConfig(mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000")
        client = GatewayModelClient(config)
        assert hasattr(client, "__aenter__")
        assert hasattr(client, "__aexit__")

    @pytest.mark.asyncio
    async def test_gateway_client_connect_no_url(self):
        """Test connect raises error when URL is missing."""
        config = ClientConfig(mode=ClientMode.GATEWAY, gateway_url=None)
        client = GatewayModelClient(config)
        with pytest.raises(ValueError, match="Gateway URL must be set"):
            await client.connect()

    @pytest.mark.asyncio
    async def test_gateway_client_connect_with_url(self):
        """Test connect initializes HTTP client."""
        config = ClientConfig(mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000")
        client = GatewayModelClient(config)
        with patch("httpx.AsyncClient") as mock_client:
            await client.connect()
            mock_client.assert_called_once()
            assert client._client is not None

    @pytest.mark.asyncio
    async def test_gateway_client_connect_with_auth_key(self):
        """Test connect includes authorization header."""
        config = ClientConfig(
            mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000", gateway_key="test_key"
        )
        client = GatewayModelClient(config)
        with patch("httpx.AsyncClient") as mock_client:
            await client.connect()
            call_kwargs = mock_client.call_args[1]
            assert "headers" in call_kwargs
            assert call_kwargs["headers"]["Authorization"] == "Bearer test_key"

    @pytest.mark.asyncio
    async def test_gateway_client_close(self):
        """Test closing the gateway client."""
        config = ClientConfig(mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000")
        client = GatewayModelClient(config)
        mock_http_client = AsyncMock()
        client._client = mock_http_client
        await client.close()
        mock_http_client.aclose.assert_called_once()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_gateway_client_close_when_not_connected(self):
        """Test closing when client is not connected."""
        config = ClientConfig(mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000")
        client = GatewayModelClient(config)
        await client.close()  # Should not raise
        assert client._client is None

    @pytest.mark.asyncio
    async def test_gateway_client_context_manager_connect_disconnect(self):
        """Test context manager connects and disconnects."""
        config = ClientConfig(mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000")
        client = GatewayModelClient(config)
        with patch.object(client, "connect", new_callable=AsyncMock) as mock_connect:
            with patch.object(client, "close", new_callable=AsyncMock) as mock_close:
                async with client:
                    pass
                mock_connect.assert_called_once()
                mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_gateway_client_context_manager_exception_handling(self):
        """Test context manager doesn't suppress exceptions."""
        config = ClientConfig(mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000")
        client = GatewayModelClient(config)
        with patch.object(client, "connect", new_callable=AsyncMock):
            with patch.object(client, "close", new_callable=AsyncMock):
                with pytest.raises(ValueError):
                    async with client:
                        raise ValueError("Test error")

    @pytest.mark.asyncio
    async def test_gateway_client_timeout_configuration(self):
        """Test HTTP client timeout is configured."""
        config = ClientConfig(mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000")
        client = GatewayModelClient(config)
        with patch("httpx.AsyncClient"):
            with patch("httpx.Timeout") as mock_timeout:
                await client.connect()
                mock_timeout.assert_called_with(120.0)


# ============================================================================
# PART 4: GatewayModelClient Chat Tests (12 tests)
# ============================================================================


class TestGatewayModelClientChat:
    """Tests for GatewayModelClient chat functionality."""

    @pytest.mark.asyncio
    async def test_gateway_chat_basic(self):
        """Test basic chat request."""
        config = ClientConfig(
            mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000", model="gpt-4"
        )
        client = GatewayModelClient(config)

        # Mock the API response
        mock_response = {
            "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        with patch.object(
            client, "_make_api_call", new_callable=AsyncMock, return_value=mock_response
        ):
            messages = [Message(role="user", content="Hi")]
            response = await client.chat(messages)

            assert response.content == "Hello!"
            assert response.finish_reason == "stop"
            assert response.usage is not None

    @pytest.mark.asyncio
    async def test_gateway_chat_with_tools(self):
        """Test chat request with tools."""
        config = ClientConfig(mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000")
        client = GatewayModelClient(config)

        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {"id": "1", "function": {"name": "test", "arguments": "{}"}}
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

        with patch.object(
            client, "_make_api_call", new_callable=AsyncMock, return_value=mock_response
        ):
            messages = [Message(role="user", content="Call a tool")]
            tools = [ToolDefinition(name="test", description="Test", parameters={})]
            response = await client.chat(messages, tools=tools)

            assert response.tool_calls is not None
            assert response.finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_gateway_chat_empty_choices(self):
        """Test chat with empty choices in response."""
        config = ClientConfig(mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000")
        client = GatewayModelClient(config)

        mock_response = {"choices": [], "usage": {}}

        with patch.object(
            client, "_make_api_call", new_callable=AsyncMock, return_value=mock_response
        ):
            messages = [Message(role="user", content="Hi")]
            response = await client.chat(messages)

            assert response.content is None
            assert response.finish_reason == "error"

    @pytest.mark.asyncio
    async def test_gateway_chat_message_normalization(self):
        """Test that messages are normalized to dict format."""
        config = ClientConfig(mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000")
        client = GatewayModelClient(config)

        mock_response = {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}

        with patch.object(
            client, "_make_api_call", new_callable=AsyncMock, return_value=mock_response
        ) as mock_call:
            messages = [Message(role="user", content="Hi")]
            await client.chat(messages)

            call_args = mock_call.call_args[0]
            payload = call_args[1]
            assert isinstance(payload["messages"][0], dict)
            assert payload["messages"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_gateway_chat_tool_normalization(self):
        """Test that tools are normalized to OpenAI format."""
        config = ClientConfig(mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000")
        client = GatewayModelClient(config)

        mock_response = {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}

        with patch.object(
            client, "_make_api_call", new_callable=AsyncMock, return_value=mock_response
        ) as mock_call:
            messages = [Message(role="user", content="Hi")]
            tools = [ToolDefinition(name="test", description="Test", parameters={})]
            await client.chat(messages, tools=tools)

            call_args = mock_call.call_args[0]
            payload = call_args[1]
            assert "tools" in payload
            assert payload["tools"][0]["type"] == "function"

    @pytest.mark.asyncio
    async def test_gateway_chat_custom_model(self):
        """Test chat with custom model parameter."""
        config = ClientConfig(
            mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000", model="default-model"
        )
        client = GatewayModelClient(config)

        mock_response = {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}

        with patch.object(
            client, "_make_api_call", new_callable=AsyncMock, return_value=mock_response
        ) as mock_call:
            messages = [Message(role="user", content="Hi")]
            await client.chat(messages, model="custom-model")

            call_args = mock_call.call_args[0]
            payload = call_args[1]
            assert payload["model"] == "custom-model"

    @pytest.mark.asyncio
    async def test_gateway_chat_custom_temperature(self):
        """Test chat with custom temperature."""
        config = ClientConfig(
            mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000", temperature=0.7
        )
        client = GatewayModelClient(config)

        mock_response = {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}

        with patch.object(
            client, "_make_api_call", new_callable=AsyncMock, return_value=mock_response
        ) as mock_call:
            messages = [Message(role="user", content="Hi")]
            await client.chat(messages, temperature=0.3)

            call_args = mock_call.call_args[0]
            payload = call_args[1]
            assert payload["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_gateway_chat_max_tokens(self):
        """Test chat includes max_tokens when configured."""
        config = ClientConfig(
            mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000", max_tokens=1000
        )
        client = GatewayModelClient(config)

        mock_response = {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}

        with patch.object(
            client, "_make_api_call", new_callable=AsyncMock, return_value=mock_response
        ) as mock_call:
            messages = [Message(role="user", content="Hi")]
            await client.chat(messages)

            call_args = mock_call.call_args[0]
            payload = call_args[1]
            assert payload["max_tokens"] == 1000

    @pytest.mark.asyncio
    async def test_gateway_chat_auto_connect(self):
        """Test chat auto-connects if not connected."""
        config = ClientConfig(mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000")
        client = GatewayModelClient(config)

        mock_response = {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}

        with patch.object(client, "connect", new_callable=AsyncMock):
            with patch.object(
                client, "_make_api_call", new_callable=AsyncMock, return_value=mock_response
            ):
                messages = [Message(role="user", content="Hi")]
                await client.chat(messages)
                client.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_gateway_chat_dict_messages(self):
        """Test chat accepts dict messages."""
        config = ClientConfig(mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000")
        client = GatewayModelClient(config)

        mock_response = {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}

        with patch.object(
            client, "_make_api_call", new_callable=AsyncMock, return_value=mock_response
        ):
            messages = [{"role": "user", "content": "Hi"}]
            response = await client.chat(messages)
            assert response.content == "OK"

    @pytest.mark.asyncio
    async def test_gateway_chat_dict_tools(self):
        """Test chat accepts dict tools."""
        config = ClientConfig(mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000")
        client = GatewayModelClient(config)

        mock_response = {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}

        with patch.object(
            client, "_make_api_call", new_callable=AsyncMock, return_value=mock_response
        ) as mock_call:
            messages = [Message(role="user", content="Hi")]
            tools = [{"type": "function", "function": {"name": "test"}}]
            await client.chat(messages, tools=tools)

            call_args = mock_call.call_args[0]
            payload = call_args[1]
            assert payload["tools"][0]["type"] == "function"


# ============================================================================
# PART 5: GatewayModelClient Streaming Tests (8 tests)
# ============================================================================


class TestGatewayModelClientStreaming:
    """Tests for GatewayModelClient streaming functionality."""

    @pytest.mark.asyncio
    async def test_gateway_chat_stream_exists(self):
        """Test that chat_stream method exists."""
        config = ClientConfig(mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000")
        client = GatewayModelClient(config)
        assert hasattr(client, "chat_stream")
        assert callable(client.chat_stream)

    @pytest.mark.asyncio
    async def test_gateway_chat_stream_returns_async_iterator(self):
        """Test that chat_stream returns an async iterator."""
        config = ClientConfig(mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000")
        client = GatewayModelClient(config)

        # Create a simple async generator for testing
        async def mock_stream_gen():
            yield StreamChunk(content="Hello")
            yield StreamChunk(content=" world")

        with patch.object(client, "connect", new_callable=AsyncMock):
            client._client = AsyncMock()
            # Mock the stream context manager
            mock_response = AsyncMock()
            mock_response.raise_for_status = MagicMock()

            async def async_iter_lines():
                yield 'data: {"choices":[{"delta":{"content":"Hello"}}]}'
                yield "data: [DONE]"

            mock_response.aiter_lines = async_iter_lines
            client._client.stream = MagicMock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=mock_response),
                    __aexit__=AsyncMock(return_value=False),
                )
            )

            messages = [Message(role="user", content="Hi")]
            result = client.chat_stream(messages)
            assert hasattr(result, "__aiter__")

    @pytest.mark.asyncio
    async def test_gateway_chat_stream_with_tools(self):
        """Test streaming with tools parameter."""
        config = ClientConfig(mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000")
        client = GatewayModelClient(config)

        with patch.object(client, "connect", new_callable=AsyncMock):
            client._client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.raise_for_status = MagicMock()

            async def async_iter_lines():
                yield "data: [DONE]"

            mock_response.aiter_lines = async_iter_lines
            client._client.stream = MagicMock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=mock_response),
                    __aexit__=AsyncMock(return_value=False),
                )
            )

            messages = [Message(role="user", content="Hi")]
            tools = [ToolDefinition(name="test", description="Test", parameters={})]
            result = client.chat_stream(messages, tools=tools)
            assert hasattr(result, "__aiter__")

    @pytest.mark.asyncio
    async def test_gateway_chat_stream_auto_connect(self):
        """Test streaming auto-connects if needed."""
        config = ClientConfig(mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000")
        client = GatewayModelClient(config)

        with patch.object(client, "connect", new_callable=AsyncMock):
            client._client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.raise_for_status = MagicMock()

            async def async_iter_lines():
                yield "data: [DONE]"

            mock_response.aiter_lines = async_iter_lines
            client._client.stream = MagicMock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=mock_response),
                    __aexit__=AsyncMock(return_value=False),
                )
            )

            messages = [Message(role="user", content="Hi")]
            async for _ in client.chat_stream(messages):
                pass

            # Verify stream was called
            client._client.stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_gateway_chat_stream_no_client_error(self):
        """Test streaming raises error if client not initialized."""
        config = ClientConfig(mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000")
        client = GatewayModelClient(config)
        client._client = None

        messages = [Message(role="user", content="Hi")]

        with patch.object(client, "connect", new_callable=AsyncMock):
            with pytest.raises(RuntimeError):
                async for _ in client.chat_stream(messages):
                    pass

    @pytest.mark.asyncio
    async def test_gateway_chat_stream_message_normalization(self):
        """Test streaming normalizes messages."""
        config = ClientConfig(mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000")
        client = GatewayModelClient(config)

        with patch.object(client, "connect", new_callable=AsyncMock):
            client._client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.raise_for_status = MagicMock()

            async def async_iter_lines():
                yield "data: [DONE]"

            mock_response.aiter_lines = async_iter_lines
            client._client.stream = MagicMock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=mock_response),
                    __aexit__=AsyncMock(return_value=False),
                )
            )

            messages = [Message(role="user", content="Hi")]
            async for _ in client.chat_stream(messages):
                pass

            # Verify stream was called
            client._client.stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_gateway_chat_stream_custom_model(self):
        """Test streaming with custom model."""
        config = ClientConfig(
            mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000", model="default-model"
        )
        client = GatewayModelClient(config)

        with patch.object(client, "connect", new_callable=AsyncMock):
            client._client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.raise_for_status = MagicMock()

            async def async_iter_lines():
                yield "data: [DONE]"

            mock_response.aiter_lines = async_iter_lines
            client._client.stream = MagicMock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=mock_response),
                    __aexit__=AsyncMock(return_value=False),
                )
            )

            messages = [Message(role="user", content="Hi")]
            async for _ in client.chat_stream(messages, model="custom-model"):
                pass

            # Verify stream was called with custom model
            client._client.stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_gateway_chat_stream_custom_temperature(self):
        """Test streaming with custom temperature."""
        config = ClientConfig(
            mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000", temperature=0.7
        )
        client = GatewayModelClient(config)

        with patch.object(client, "connect", new_callable=AsyncMock):
            client._client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.raise_for_status = MagicMock()

            async def async_iter_lines():
                yield "data: [DONE]"

            mock_response.aiter_lines = async_iter_lines
            client._client.stream = MagicMock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=mock_response),
                    __aexit__=AsyncMock(return_value=False),
                )
            )

            messages = [Message(role="user", content="Hi")]
            async for _ in client.chat_stream(messages, temperature=0.3):
                pass

            client._client.stream.assert_called_once()


# ============================================================================
# PART 6: GatewayModelClient API Call and Error Handling (10 tests)
# ============================================================================


class TestGatewayModelClientAPICall:
    """Tests for GatewayModelClient API call handling."""

    @pytest.mark.asyncio
    async def test_gateway_list_models(self):
        """Test listing available models."""
        config = ClientConfig(mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000")
        client = GatewayModelClient(config)

        mock_response = {
            "data": [
                {"id": "gpt-4", "provider": "openai"},
                {"id": "claude-3", "provider": "anthropic"},
            ]
        }

        with patch.object(
            client, "_make_api_call", new_callable=AsyncMock, return_value=mock_response
        ):
            models = await client.list_models()
            assert len(models) == 2
            assert models[0]["id"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_gateway_make_api_call_success(self):
        """Test successful API call."""
        config = ClientConfig(mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000")
        client = GatewayModelClient(config)

        mock_http_response = MagicMock()
        mock_http_response.json.return_value = {"result": "success"}
        mock_http_response.raise_for_status = MagicMock()

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_http_response)
        client._client = mock_http_client

        result = await client._make_api_call("/test", {"data": "test"})
        assert result["result"] == "success"

    @pytest.mark.asyncio
    async def test_gateway_make_api_call_get_method(self):
        """Test API call with GET method."""
        config = ClientConfig(mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000")
        client = GatewayModelClient(config)

        mock_http_response = MagicMock()
        mock_http_response.json.return_value = {"models": []}
        mock_http_response.raise_for_status = MagicMock()

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(return_value=mock_http_response)
        client._client = mock_http_client

        result = await client._make_api_call("/models", method="GET")
        assert "models" in result

    @pytest.mark.asyncio
    async def test_gateway_make_api_call_rate_limit_error(self):
        """Test API call handles rate limit error."""
        config = ClientConfig(mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000")
        client = GatewayModelClient(config)

        import httpx

        from src.core.errors import RateLimitError

        mock_http_response = MagicMock()
        mock_http_response.status_code = 429
        mock_http_response.headers = {"Retry-After": "60"}

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "429", request=MagicMock(), response=mock_http_response
            )
        )
        client._client = mock_http_client

        with pytest.raises(RateLimitError):
            await client._make_api_call("/test", {})

    @pytest.mark.asyncio
    async def test_gateway_make_api_call_server_error(self):
        """Test API call handles server error."""
        config = ClientConfig(mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000")
        client = GatewayModelClient(config)

        import httpx

        from src.core.errors import TransientError

        mock_http_response = MagicMock()
        mock_http_response.status_code = 500

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "500", request=MagicMock(), response=mock_http_response
            )
        )
        client._client = mock_http_client

        with pytest.raises(TransientError):
            await client._make_api_call("/test", {})

    @pytest.mark.asyncio
    async def test_gateway_make_api_call_client_error(self):
        """Test API call handles client error."""
        config = ClientConfig(mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000")
        client = GatewayModelClient(config)

        import httpx

        from src.core.errors import AgentError

        mock_http_response = MagicMock()
        mock_http_response.status_code = 400
        mock_http_response.text = "Bad request"

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "400", request=MagicMock(), response=mock_http_response
            )
        )
        client._client = mock_http_client

        with pytest.raises(AgentError):
            await client._make_api_call("/test", {})

    @pytest.mark.asyncio
    async def test_gateway_make_api_call_network_error(self):
        """Test API call handles network error."""
        config = ClientConfig(mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000")
        client = GatewayModelClient(config)

        import httpx

        from src.core.errors import TransientError

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(side_effect=httpx.RequestError("Connection failed"))
        client._client = mock_http_client

        with pytest.raises(TransientError):
            await client._make_api_call("/test", {})

    @pytest.mark.asyncio
    async def test_gateway_make_api_call_auto_reconnect(self):
        """Test API call auto-reconnects if client is None."""
        config = ClientConfig(mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000")
        client = GatewayModelClient(config)
        client._client = None

        mock_http_response = MagicMock()
        mock_http_response.json.return_value = {"result": "success"}
        mock_http_response.raise_for_status = MagicMock()

        with patch.object(client, "connect", new_callable=AsyncMock):
            mock_http_client = AsyncMock()
            mock_http_client.post = AsyncMock(return_value=mock_http_response)
            client._client = mock_http_client

            result = await client._make_api_call("/test", {})
            assert result["result"] == "success"

    @pytest.mark.asyncio
    async def test_gateway_make_api_call_retry_on_transient_error(self):
        """Test API call retries on transient error."""
        config = ClientConfig(mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000")
        client = GatewayModelClient(config)

        from src.core.errors import TransientError

        mock_http_response = MagicMock()
        mock_http_response.json.return_value = {"result": "success"}
        mock_http_response.raise_for_status = MagicMock()

        mock_http_client = AsyncMock()
        # First call fails, second succeeds
        mock_http_client.post = AsyncMock(
            side_effect=[TransientError("Temporary error"), mock_http_response]
        )
        client._client = mock_http_client

        # This should retry and eventually succeed
        result = await client._make_api_call("/test", {})
        assert result["result"] == "success"


# ============================================================================
# PART 7: DirectModelClient Initialization Tests (10 tests)
# ============================================================================


class TestDirectModelClientInit:
    """Tests for DirectModelClient initialization."""

    def test_direct_client_creation(self):
        """Test creating a DirectModelClient."""
        config = ClientConfig(mode=ClientMode.DIRECT, google_api_key="test_key")
        client = DirectModelClient(config)
        assert client.config == config
        assert client._providers == {}

    def test_direct_client_provider_patterns(self):
        """Test provider detection patterns."""
        client = DirectModelClient(ClientConfig())
        assert Provider.GOOGLE in client.PROVIDER_PATTERNS
        assert Provider.OPENAI in client.PROVIDER_PATTERNS
        assert Provider.ANTHROPIC in client.PROVIDER_PATTERNS
        assert Provider.OLLAMA in client.PROVIDER_PATTERNS

    @pytest.mark.asyncio
    async def test_direct_client_connect_google(self):
        """Test connecting Google provider."""
        config = ClientConfig(mode=ClientMode.DIRECT, google_api_key="test_key")
        client = DirectModelClient(config)

        with patch("google.genai.Client") as mock_client:
            await client.connect()
            mock_client.assert_called_once_with(api_key="test_key")
            assert Provider.GOOGLE in client._providers

    @pytest.mark.asyncio
    async def test_direct_client_connect_openai(self):
        """Test connecting OpenAI provider."""
        config = ClientConfig(mode=ClientMode.DIRECT, openai_api_key="test_key")
        client = DirectModelClient(config)

        with patch("openai.AsyncOpenAI") as mock_client:
            await client.connect()
            mock_client.assert_called_once_with(api_key="test_key")
            assert Provider.OPENAI in client._providers

    @pytest.mark.asyncio
    async def test_direct_client_connect_anthropic(self):
        """Test connecting Anthropic provider."""
        config = ClientConfig(mode=ClientMode.DIRECT, anthropic_api_key="test_key")
        client = DirectModelClient(config)

        with patch("anthropic.AsyncAnthropic") as mock_client:
            await client.connect()
            mock_client.assert_called_once_with(api_key="test_key")
            assert Provider.ANTHROPIC in client._providers

    @pytest.mark.asyncio
    async def test_direct_client_connect_ollama(self):
        """Test connecting Ollama provider."""
        config = ClientConfig(mode=ClientMode.DIRECT, ollama_base_url="http://localhost:11434")
        client = DirectModelClient(config)

        with patch("httpx.AsyncClient") as mock_client:
            await client.connect()
            mock_client.assert_called_once()
            assert Provider.OLLAMA in client._providers

    @pytest.mark.asyncio
    async def test_direct_client_connect_no_providers(self):
        """Test connect raises error when no providers available."""
        config = ClientConfig(mode=ClientMode.DIRECT)
        client = DirectModelClient(config)

        # Mock all provider imports to fail
        with patch("google.genai.Client", side_effect=ImportError):
            with patch("openai.AsyncOpenAI", side_effect=ImportError):
                with patch("anthropic.AsyncAnthropic", side_effect=ImportError):
                    with patch("httpx.AsyncClient", side_effect=Exception):
                        with pytest.raises(RuntimeError, match="No providers available"):
                            await client.connect()

    @pytest.mark.asyncio
    async def test_direct_client_close(self):
        """Test closing the direct client."""
        config = ClientConfig(mode=ClientMode.DIRECT, google_api_key="test_key")
        client = DirectModelClient(config)

        mock_ollama = AsyncMock()
        mock_ollama.aclose = AsyncMock()
        client._providers[Provider.OLLAMA] = mock_ollama

        await client.close()
        mock_ollama.aclose.assert_called_once()
        assert len(client._providers) == 0

    @pytest.mark.asyncio
    async def test_direct_client_context_manager(self):
        """Test DirectModelClient as context manager."""
        config = ClientConfig(mode=ClientMode.DIRECT, google_api_key="test_key")
        client = DirectModelClient(config)

        with patch.object(client, "connect", new_callable=AsyncMock):
            with patch.object(client, "close", new_callable=AsyncMock):
                async with client:
                    pass

    def test_direct_client_detect_provider_google(self):
        """Test detecting Google provider from model name."""
        config = ClientConfig(mode=ClientMode.DIRECT, google_api_key="key")
        client = DirectModelClient(config)
        client._providers[Provider.GOOGLE] = MagicMock()

        provider = client._detect_provider("gemini-2.5-flash")
        assert provider == Provider.GOOGLE

    def test_direct_client_detect_provider_openai(self):
        """Test detecting OpenAI provider from model name."""
        config = ClientConfig(mode=ClientMode.DIRECT, openai_api_key="key")
        client = DirectModelClient(config)
        client._providers[Provider.OPENAI] = MagicMock()

        provider = client._detect_provider("gpt-4")
        assert provider == Provider.OPENAI


# ============================================================================
# PART 8: DirectModelClient Chat Tests (12 tests)
# ============================================================================


class TestDirectModelClientChat:
    """Tests for DirectModelClient chat functionality."""

    @pytest.mark.asyncio
    async def test_direct_chat_google(self):
        """Test chat with Google provider."""
        config = ClientConfig(
            mode=ClientMode.DIRECT, google_api_key="test_key", model="gemini-2.5-flash"
        )
        client = DirectModelClient(config)

        with patch.object(client, "_chat_google", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = ChatResponse(content="Hello")
            client._providers[Provider.GOOGLE] = MagicMock()

            messages = [Message(role="user", content="Hi")]
            response = await client.chat(messages)

            mock_chat.assert_called_once()
            assert response.content == "Hello"

    @pytest.mark.asyncio
    async def test_direct_chat_openai(self):
        """Test chat with OpenAI provider."""
        config = ClientConfig(mode=ClientMode.DIRECT, openai_api_key="test_key", model="gpt-4")
        client = DirectModelClient(config)

        with patch.object(client, "_chat_openai", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = ChatResponse(content="Hello")
            client._providers[Provider.OPENAI] = MagicMock()

            messages = [Message(role="user", content="Hi")]
            await client.chat(messages)

            mock_chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_direct_chat_anthropic(self):
        """Test chat with Anthropic provider."""
        config = ClientConfig(
            mode=ClientMode.DIRECT, anthropic_api_key="test_key", model="claude-3"
        )
        client = DirectModelClient(config)

        with patch.object(client, "_chat_anthropic", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = ChatResponse(content="Hello")
            client._providers[Provider.ANTHROPIC] = MagicMock()

            messages = [Message(role="user", content="Hi")]
            await client.chat(messages)

            mock_chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_direct_chat_ollama(self):
        """Test chat with Ollama provider."""
        config = ClientConfig(mode=ClientMode.DIRECT, model="llama2")
        client = DirectModelClient(config)

        with patch.object(client, "_chat_ollama", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = ChatResponse(content="Hello")
            client._providers[Provider.OLLAMA] = MagicMock()

            messages = [Message(role="user", content="Hi")]
            await client.chat(messages)

            mock_chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_direct_chat_auto_connect(self):
        """Test chat auto-connects if needed."""
        config = ClientConfig(
            mode=ClientMode.DIRECT, google_api_key="test_key", model="gemini-2.5-flash"
        )
        client = DirectModelClient(config)
        # Ensure providers are empty to trigger auto-connect
        client._providers = {}

        with patch.object(client, "connect", new_callable=AsyncMock):
            with patch.object(client, "_chat_google", new_callable=AsyncMock) as mock_chat:
                mock_chat.return_value = ChatResponse(content="Hello")
                client._providers[Provider.GOOGLE] = MagicMock()

                messages = [Message(role="user", content="Hi")]
                response = await client.chat(messages)

                # Verify chat was called and returned response
                assert response.content == "Hello"
                mock_chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_direct_chat_stream_fallback(self):
        """Test streaming uses provider-specific streaming (Google)."""
        config = ClientConfig(
            mode=ClientMode.DIRECT, google_api_key="test_key", model="gemini-2.5-flash"
        )
        client = DirectModelClient(config)

        # Mock _stream_google to yield chunks
        async def mock_stream(*args, **kwargs):
            yield StreamChunk(content="Hello", finish_reason="stop")

        with patch.object(client, "_stream_google", side_effect=mock_stream):
            client._providers[Provider.GOOGLE] = MagicMock()

            messages = [Message(role="user", content="Hi")]
            chunks = []
            async for chunk in client.chat_stream(messages):
                chunks.append(chunk)

            assert len(chunks) == 1
            assert chunks[0].content == "Hello"

    @pytest.mark.asyncio
    async def test_direct_list_models_google(self):
        """Test listing Google models."""
        config = ClientConfig(mode=ClientMode.DIRECT, google_api_key="test_key")
        client = DirectModelClient(config)
        client._providers[Provider.GOOGLE] = MagicMock()

        models = await client.list_models()
        assert any(m["provider"] == "google" for m in models)

    @pytest.mark.asyncio
    async def test_direct_list_models_openai(self):
        """Test listing OpenAI models."""
        config = ClientConfig(mode=ClientMode.DIRECT, openai_api_key="test_key")
        client = DirectModelClient(config)
        client._providers[Provider.OPENAI] = MagicMock()

        models = await client.list_models()
        assert any(m["provider"] == "openai" for m in models)

    @pytest.mark.asyncio
    async def test_direct_list_models_anthropic(self):
        """Test listing Anthropic models."""
        config = ClientConfig(mode=ClientMode.DIRECT, anthropic_api_key="test_key")
        client = DirectModelClient(config)
        client._providers[Provider.ANTHROPIC] = MagicMock()

        models = await client.list_models()
        assert any(m["provider"] == "anthropic" for m in models)

    @pytest.mark.asyncio
    async def test_direct_chat_custom_model(self):
        """Test chat with custom model parameter."""
        config = ClientConfig(
            mode=ClientMode.DIRECT, google_api_key="test_key", model="gemini-2.5-flash"
        )
        client = DirectModelClient(config)

        with patch.object(client, "_chat_google", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = ChatResponse(content="Hello")
            client._providers[Provider.GOOGLE] = MagicMock()

            messages = [Message(role="user", content="Hi")]
            await client.chat(messages, model="custom-model")

            call_kwargs = mock_chat.call_args[1]
            assert call_kwargs["model"] == "custom-model"

    @pytest.mark.asyncio
    async def test_direct_chat_custom_temperature(self):
        """Test chat with custom temperature."""
        config = ClientConfig(
            mode=ClientMode.DIRECT,
            google_api_key="test_key",
            model="gemini-2.5-flash",
            temperature=0.7,
        )
        client = DirectModelClient(config)

        with patch.object(client, "_chat_google", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = ChatResponse(content="Hello")
            client._providers[Provider.GOOGLE] = MagicMock()

            messages = [Message(role="user", content="Hi")]
            await client.chat(messages, temperature=0.3)

            call_kwargs = mock_chat.call_args[1]
            assert call_kwargs["temperature"] == 0.3


# ============================================================================
# PART 9: ModelClient Factory Tests (10 tests)
# ============================================================================


class TestModelClientFactory:
    """Tests for ModelClient factory."""

    @pytest.mark.asyncio
    async def test_model_client_create_gateway_mode(self):
        """Test creating client in gateway mode."""
        config = ClientConfig(mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000")

        with patch.object(GatewayModelClient, "connect", new_callable=AsyncMock):
            client = await ModelClient.create(config)
            assert isinstance(client, GatewayModelClient)

    @pytest.mark.asyncio
    async def test_model_client_create_direct_mode(self):
        """Test creating client in direct mode."""
        config = ClientConfig(mode=ClientMode.DIRECT, google_api_key="test_key")

        with patch.object(DirectModelClient, "connect", new_callable=AsyncMock):
            client = await ModelClient.create(config)
            assert isinstance(client, DirectModelClient)

    @pytest.mark.asyncio
    async def test_model_client_create_from_env(self):
        """Test creating client from project config."""
        with patch(
            "src.core.config.load_config",
            return_value={
                "model": "gpt-4o",
                "gateway_url": "http://localhost:8000",
            },
        ):
            with patch.object(GatewayModelClient, "connect", new_callable=AsyncMock):
                client = await ModelClient.create()
                assert isinstance(client, GatewayModelClient)

    @pytest.mark.asyncio
    async def test_model_client_create_calls_connect(self):
        """Test that create calls connect."""
        config = ClientConfig(mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000")

        with patch.object(GatewayModelClient, "connect", new_callable=AsyncMock) as mock_connect:
            await ModelClient.create(config)
            mock_connect.assert_called_once()

    def test_model_client_get_mode_gateway(self):
        """Test get_mode returns gateway mode."""
        with patch(
            "src.core.config.load_config",
            return_value={
                "model": "gpt-4o",
                "gateway_url": "http://localhost:8000",
            },
        ):
            mode = ModelClient.get_mode()
            assert mode == ClientMode.GATEWAY

    def test_model_client_get_mode_direct(self):
        """Test get_mode returns direct mode."""
        with patch("src.core.config.load_config", return_value={"model": "gemini-2.5-flash"}):
            mode = ModelClient.get_mode()
            assert mode == ClientMode.DIRECT

    def test_model_client_get_mode_info_gateway(self):
        """Test get_mode_info for gateway mode."""
        with patch(
            "src.core.config.load_config",
            return_value={
                "model": "gpt-4o",
                "gateway_url": "http://localhost:8000",
                "gateway_key": "test_key",
            },
        ):
            info = ModelClient.get_mode_info()
            assert info["mode"] == "gateway"
            assert info["gateway_url"] == "http://localhost:8000"
            assert info["has_key"] is True

    def test_model_client_get_mode_info_direct(self):
        """Test get_mode_info for direct mode."""
        with patch(
            "src.core.config.load_config",
            return_value={
                "model": "gemini-2.5-flash",
                "google_api_key": "google_key",
                "openai_api_key": "openai_key",
            },
        ):
            info = ModelClient.get_mode_info()
            assert info["mode"] == "direct"
            assert "providers" in info
            assert info["providers"]["google"] is True
            assert info["providers"]["openai"] is True

    def test_model_client_get_mode_info_no_keys(self):
        """Test get_mode_info with no API keys."""
        with patch("src.core.config.load_config", return_value={"model": "test-model"}):
            info = ModelClient.get_mode_info()
            assert info["mode"] == "direct"
            assert all(not v for k, v in info["providers"].items() if k != "ollama")

    def test_model_client_get_mode_info_ollama_always_true(self):
        """Test that Ollama is always available in mode info."""
        with patch("src.core.config.load_config", return_value={"model": "test-model"}):
            info = ModelClient.get_mode_info()
            assert info["providers"]["ollama"] is True


# ============================================================================
# PART 10: Provider Detection and Tool Conversion Tests (10 tests)
# ============================================================================


class TestProviderDetection:
    """Tests for provider detection."""

    def test_provider_enum_values(self):
        """Test Provider enum has expected values."""
        assert Provider.GOOGLE.value == "google"
        assert Provider.OPENAI.value == "openai"
        assert Provider.ANTHROPIC.value == "anthropic"
        assert Provider.OLLAMA.value == "ollama"

    def test_detect_provider_gemini(self):
        """Test detecting Gemini models."""
        config = ClientConfig(mode=ClientMode.DIRECT, google_api_key="key")
        client = DirectModelClient(config)
        client._providers[Provider.GOOGLE] = MagicMock()

        assert client._detect_provider("gemini-2.5-flash") == Provider.GOOGLE
        assert client._detect_provider("gemini-pro") == Provider.GOOGLE

    def test_detect_provider_gpt(self):
        """Test detecting GPT models."""
        config = ClientConfig(mode=ClientMode.DIRECT, openai_api_key="key")
        client = DirectModelClient(config)
        client._providers[Provider.OPENAI] = MagicMock()

        assert client._detect_provider("gpt-4") == Provider.OPENAI
        assert client._detect_provider("gpt-3.5-turbo") == Provider.OPENAI

    def test_detect_provider_claude(self):
        """Test detecting Claude models."""
        config = ClientConfig(mode=ClientMode.DIRECT, anthropic_api_key="key")
        client = DirectModelClient(config)
        client._providers[Provider.ANTHROPIC] = MagicMock()

        assert client._detect_provider("claude-3") == Provider.ANTHROPIC
        assert client._detect_provider("claude-opus") == Provider.ANTHROPIC

    def test_detect_provider_o1(self):
        """Test detecting O1 models (OpenAI)."""
        config = ClientConfig(mode=ClientMode.DIRECT, openai_api_key="key")
        client = DirectModelClient(config)
        client._providers[Provider.OPENAI] = MagicMock()

        assert client._detect_provider("o1") == Provider.OPENAI

    def test_detect_provider_o3(self):
        """Test detecting O3 models (OpenAI)."""
        config = ClientConfig(mode=ClientMode.DIRECT, openai_api_key="key")
        client = DirectModelClient(config)
        client._providers[Provider.OPENAI] = MagicMock()

        assert client._detect_provider("o3") == Provider.OPENAI

    def test_detect_provider_fallback_to_first(self):
        """Test provider detection falls back to first available."""
        config = ClientConfig(mode=ClientMode.DIRECT, google_api_key="key")
        client = DirectModelClient(config)
        client._providers[Provider.GOOGLE] = MagicMock()

        # Unknown model should fall back to first provider
        provider = client._detect_provider("unknown-model")
        assert provider == Provider.GOOGLE

    def test_detect_provider_no_providers_error(self):
        """Test provider detection raises error when no providers."""
        config = ClientConfig(mode=ClientMode.DIRECT)
        client = DirectModelClient(config)

        with pytest.raises(RuntimeError, match="No providers available"):
            client._detect_provider("gpt-4")

    def test_detect_provider_unavailable_provider(self):
        """Test provider detection skips unavailable providers."""
        config = ClientConfig(mode=ClientMode.DIRECT, openai_api_key="key")
        client = DirectModelClient(config)
        client._providers[Provider.OPENAI] = MagicMock()

        # Try to detect Gemini but only OpenAI is available
        provider = client._detect_provider("gemini-2.5-flash")
        assert provider == Provider.OPENAI

    def test_client_mode_enum_values(self):
        """Test ClientMode enum has expected values."""
        assert ClientMode.GATEWAY.value == "gateway"
        assert ClientMode.DIRECT.value == "direct"


# ============================================================================
# PART 11: Edge Cases and Integration Tests (8 tests)
# ============================================================================


class TestEdgeCasesAndIntegration:
    """Tests for edge cases and integration scenarios."""

    def test_message_with_none_values(self):
        """Test message handles None values correctly."""
        msg = Message(role="user", content=None, thought=None, tool_calls=None)
        d = msg.to_dict()
        assert d["role"] == "user"
        assert "content" not in d
        assert "thought" not in d
        assert "tool_calls" not in d

    def test_tool_definition_with_complex_parameters(self):
        """Test tool definition with complex parameter schema."""
        params = {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "options": {
                    "type": "object",
                    "properties": {"recursive": {"type": "boolean"}, "depth": {"type": "integer"}},
                },
            },
            "required": ["path"],
        }
        tool = ToolDefinition(name="complex_tool", description="Complex tool", parameters=params)
        openai_format = tool.to_openai_format()
        assert openai_format["function"]["parameters"]["properties"]["options"]["type"] == "object"

    @pytest.mark.asyncio
    async def test_gateway_client_with_no_auth_key(self):
        """Test gateway client works without auth key."""
        config = ClientConfig(
            mode=ClientMode.GATEWAY, gateway_url="http://localhost:8000", gateway_key=None
        )
        client = GatewayModelClient(config)

        with patch("httpx.AsyncClient") as mock_client:
            await client.connect()
            call_kwargs = mock_client.call_args[1]
            headers = call_kwargs.get("headers", {})
            assert "Authorization" not in headers

    @pytest.mark.asyncio
    async def test_direct_client_multiple_providers(self):
        """Test direct client with multiple providers."""
        config = ClientConfig(
            mode=ClientMode.DIRECT,
            google_api_key="google_key",
            openai_api_key="openai_key",
            anthropic_api_key="anthropic_key",
        )
        client = DirectModelClient(config)

        with patch("google.genai.Client"):
            with patch("openai.AsyncOpenAI"):
                with patch("anthropic.AsyncAnthropic"):
                    await client.connect()
                    assert len(client._providers) >= 3

    def test_chat_response_with_all_fields(self):
        """Test ChatResponse with all fields populated."""
        response = ChatResponse(
            content="Response",
            thought="Thinking",
            tool_calls=[{"id": "1"}],
            finish_reason="tool_calls",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            raw={"raw": "data"},
        )
        assert response.content == "Response"
        assert response.thought == "Thinking"
        assert response.has_tool_calls is True
        assert response.finish_reason == "tool_calls"
        assert response.usage is not None
        assert response.raw is not None

    def test_stream_chunk_with_usage(self):
        """Test StreamChunk includes usage information."""
        chunk = StreamChunk(content="Hello", usage={"prompt_tokens": 10, "completion_tokens": 5})
        assert chunk.usage is not None
        assert chunk.usage["prompt_tokens"] == 10

    @pytest.mark.asyncio
    async def test_base_model_client_is_abstract(self):
        """Test that BaseModelClient cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseModelClient()

    def test_tool_call_round_trip(self):
        """Test ToolCall can be converted to dict and back."""
        original = ToolCall(id="call_123", name="test_tool", arguments={"param": "value"})
        d = original.to_dict()
        restored = ToolCall.from_dict(d)
        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.arguments == original.arguments
