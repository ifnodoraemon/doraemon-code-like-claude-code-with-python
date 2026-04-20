"""
Additional unit tests for model_client.py to increase coverage.

Tests streaming, tool definitions, and edge cases.
"""

from unittest.mock import patch

import pytest

from src.core.llm.model_client import (
    GatewayModelClient,
)
from src.core.llm.model_utils import ClientConfig, ClientMode, Message, StreamChunk, ToolDefinition


class TestToolDefinition:
    """Tests for ToolDefinition class."""

    def test_to_openai_format(self):
        """Test conversion to OpenAI format."""
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
        )

        openai_format = tool.to_openai_format()

        assert openai_format["type"] == "function"
        assert openai_format["function"]["name"] == "test_tool"
        assert openai_format["function"]["description"] == "A test tool"

    def test_to_genai_format(self):
        """Test conversion to GenAI format."""
        tool = ToolDefinition(
            name="test_tool", description="A test tool", parameters={"type": "object"}
        )

        with patch("google.genai.types.FunctionDeclaration") as mock_func:
            tool.to_genai_format()
            mock_func.assert_called_once()


class TestStreamChunk:
    """Tests for StreamChunk class."""

    def test_stream_chunk_creation(self):
        """Test creating a stream chunk."""
        chunk = StreamChunk(content="Hello", thought="Thinking...", finish_reason="stop")

        assert chunk.content == "Hello"
        assert chunk.thought == "Thinking..."
        assert chunk.finish_reason == "stop"


class TestGatewayClientStreaming:
    """Tests for GatewayModelClient streaming."""

    @pytest.mark.asyncio
    async def test_chat_stream_basic(self):
        """Test basic chat_stream functionality."""
        config = ClientConfig(
            mode=ClientMode.GATEWAY,
            gateway_url="http://test.com",
            model="test-model",
        )
        client = GatewayModelClient(config)

        # For now, just test that the method exists and can be called
        # Full streaming test requires complex async mock setup
        assert hasattr(client, "chat_stream")
        assert callable(client.chat_stream)


class TestClientConfigValidation:
    """Tests for ClientConfig validation."""

    def test_gateway_mode_requires_url(self):
        """Test that gateway mode validates URL."""
        config = ClientConfig(
            mode=ClientMode.GATEWAY,
            gateway_url=None,
            model="test-model",
        )
        client = GatewayModelClient(config)

        with pytest.raises(ValueError, match="Gateway URL must be set"):
            import asyncio

            asyncio.run(client.connect())

    def test_from_env_detects_gateway_mode(self):
        """Test that from_env detects gateway mode."""
        with patch(
            "src.core.config.load_config",
            return_value={
                "model": "gpt-4o",
                "gateway_url": "http://test.com",
            },
        ):
            config = ClientConfig.from_env()
            assert config.mode == ClientMode.GATEWAY
            assert config.gateway_url == "http://test.com"

    def test_from_env_detects_direct_mode(self):
        """Test that from_env detects direct mode."""
        with patch("src.core.config.load_config", return_value={"model": "test-model"}):
            config = ClientConfig.from_env()
            assert config.mode == ClientMode.DIRECT

    def test_from_env_loads_protocol_and_capability_overrides(self):
        """Test that direct provider protocol/capability settings are loaded."""
        with patch(
            "src.core.config.load_config",
            return_value={
                "model": "gpt-5.4",
                "openai_protocol": "chat_completions",
                "openai_capabilities": {"tools": False, "streaming": False},
                "anthropic_protocol": "messages",
                "anthropic_capabilities": {"tools": False, "streaming": True},
            },
        ):
            config = ClientConfig.from_env()
            assert config.openai_protocol == "chat_completions"
            assert config.openai_capabilities.tools is False
            assert config.openai_capabilities.streaming is False
            assert config.anthropic_protocol == "messages"
            assert config.anthropic_capabilities.tools is False
            assert config.anthropic_capabilities.streaming is True


class TestMessageSerialization:
    """Tests for Message serialization."""

    def test_message_to_dict_with_all_fields(self):
        """Test message serialization with all fields."""
        msg = Message(
            role="assistant",
            content="Hello",
            thought="Thinking",
            tool_calls=[{"id": "1", "name": "test"}],
            tool_call_id="call_123",
            name="test_tool",
        )

        d = msg.to_dict()

        assert d["role"] == "assistant"
        assert d["content"] == "Hello"
        assert d["thought"] == "Thinking"
        assert len(d["tool_calls"]) == 1
        assert d["tool_call_id"] == "call_123"
        assert d["name"] == "test_tool"

    def test_message_to_dict_minimal(self):
        """Test message serialization with minimal fields."""
        msg = Message(role="user", content="Hi")
        d = msg.to_dict()

        assert "role" in d
        assert "content" in d
        assert "thought" not in d
        assert "tool_calls" not in d
