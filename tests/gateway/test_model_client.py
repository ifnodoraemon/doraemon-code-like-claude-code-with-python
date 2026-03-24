"""Tests for the unified model client."""

from unittest.mock import patch

from src.core.model_client import (
    ModelClient,
)
from src.core.model_utils import (
    ChatResponse,
    ClientConfig,
    ClientMode,
    Message,
    ToolCall,
    ToolDefinition,
)


class TestClientConfig:
    """Tests for ClientConfig."""

    def test_from_env_direct_mode(self):
        """Test config from project config in direct mode."""
        with patch(
            "src.core.config.load_config",
            return_value={
                "model": "gemini-2.5-flash",
                "google_api_key": "test_google_key",
                "openai_api_key": "test_openai_key",
            },
        ):
            config = ClientConfig.from_env()

            assert config.mode == ClientMode.DIRECT
            assert config.google_api_key == "test_google_key"
            assert config.openai_api_key == "test_openai_key"
            assert config.model == "gemini-2.5-flash"

    def test_from_env_gateway_mode(self):
        """Test config from project config in gateway mode."""
        with patch(
            "src.core.config.load_config",
            return_value={
                "model": "gpt-4o",
                "gateway_url": "http://localhost:8000",
                "gateway_key": "test_gateway_key",
            },
        ):
            config = ClientConfig.from_env()

            assert config.mode == ClientMode.GATEWAY
            assert config.gateway_url == "http://localhost:8000"
            assert config.gateway_key == "test_gateway_key"


class TestMessage:
    """Tests for Message."""

    def test_to_dict_simple(self):
        """Test message to dict conversion."""
        msg = Message(role="user", content="Hello")
        d = msg.to_dict()

        assert d["role"] == "user"
        assert d["content"] == "Hello"

    def test_to_dict_with_tool_calls(self):
        """Test message with tool calls."""
        msg = Message(
            role="assistant",
            content=None,
            tool_calls=[{"name": "read_file", "arguments": {"path": "/test"}}],
        )
        d = msg.to_dict()

        assert d["role"] == "assistant"
        assert "content" not in d
        assert len(d["tool_calls"]) == 1


class TestToolDefinition:
    """Tests for ToolDefinition."""

    def test_to_openai_format(self):
        """Test conversion to OpenAI format."""
        tool = ToolDefinition(
            name="read_file",
            description="Read a file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        )

        openai_format = tool.to_openai_format()

        assert openai_format["type"] == "function"
        assert openai_format["function"]["name"] == "read_file"
        assert openai_format["function"]["description"] == "Read a file"


class TestChatResponse:
    """Tests for ChatResponse."""

    def test_has_tool_calls(self):
        """Test tool call detection."""
        response_with_tools = ChatResponse(
            content="",
            tool_calls=[ToolCall(id="1", name="test", arguments={})],
        )
        assert response_with_tools.has_tool_calls

        response_without_tools = ChatResponse(content="Hello")
        assert not response_without_tools.has_tool_calls


class TestModelClient:
    """Tests for ModelClient factory."""

    def test_get_mode_direct(self):
        """Test mode detection for direct mode."""
        with patch("src.core.config.load_config", return_value={"model": "gemini-2.5-flash"}):
            mode = ModelClient.get_mode()
            assert mode == ClientMode.DIRECT

    def test_get_mode_gateway(self):
        """Test mode detection for gateway mode."""
        with patch(
            "src.core.config.load_config",
            return_value={
                "model": "gpt-4o",
                "gateway_url": "http://localhost:8000",
            },
        ):
            mode = ModelClient.get_mode()
            assert mode == ClientMode.GATEWAY

    def test_get_mode_info(self):
        """Test mode info retrieval."""
        with patch(
            "src.core.config.load_config",
            return_value={
                "model": "gemini-2.5-flash",
                "google_api_key": "test_key",
            },
        ):
            info = ModelClient.get_mode_info()

            assert info["mode"] == "direct"
            assert "google" in info["providers"]


class TestToolCall:
    """Tests for ToolCall."""

    def test_from_dict(self):
        """Test creating ToolCall from dict."""
        data = {
            "id": "call_123",
            "name": "read_file",
            "arguments": {"path": "/test.txt"},
        }
        tool_call = ToolCall.from_dict(data)

        assert tool_call.id == "call_123"
        assert tool_call.name == "read_file"
        assert tool_call.arguments["path"] == "/test.txt"

    def test_to_dict(self):
        """Test converting ToolCall to dict."""
        tool_call = ToolCall(
            id="call_123",
            name="read_file",
            arguments={"path": "/test.txt"},
        )
        d = tool_call.to_dict()

        assert d["id"] == "call_123"
        assert d["name"] == "read_file"
