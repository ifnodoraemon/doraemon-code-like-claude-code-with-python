"""
Unit tests for model_client.py

Tests the unified model client interface, retry logic, and error handling.
"""

import base64
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.errors import ConfigurationError, TransientError
from src.core.llm.model_client import (
    DirectModelClient,
    GatewayModelClient,
    ModelClient,
)
from src.core.llm.model_client_direct import _OPENAI_PROTOCOL_CACHE
from src.core.llm.model_utils import (
    ChatResponse,
    ClientConfig,
    ClientMode,
    Message,
    Provider,
    StreamChunk,
    ToolDefinition,
)
from src.core.llm.provider_adapters import OpenAIAdapter


class TestGatewayModelClient:
    """Tests for GatewayModelClient."""

    @pytest.mark.asyncio
    async def test_context_manager_connects_and_closes(self):
        """Test that context manager properly connects and closes client."""
        config = ClientConfig(
            mode=ClientMode.GATEWAY,
            gateway_url="http://test.com",
            model="test-model",
        )
        client = GatewayModelClient(config)

        # Mock the HTTP client
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            async with client:
                # Should have connected
                assert client._client is not None

            # Should have closed
            mock_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self):
        """Test that client retries on rate limit with backoff."""
        config = ClientConfig(
            mode=ClientMode.GATEWAY,
            gateway_url="http://test.com",
            model="test-model",
        )
        client = GatewayModelClient(config)

        # Mock HTTP client
        mock_client = AsyncMock()
        client._client = mock_client

        # First call returns 429, second succeeds
        from httpx import HTTPStatusError, Request, Response

        rate_limit_response = Response(
            status_code=429,
            headers={"Retry-After": "1"},
            request=Request("POST", "http://test.com/v1/chat/completions"),
        )
        success_response = Response(
            status_code=200,
            json={"choices": [{"message": {"content": "Success"}, "finish_reason": "stop"}]},
            request=Request("POST", "http://test.com/v1/chat/completions"),
        )

        mock_client.post.side_effect = [
            HTTPStatusError(
                "Rate limited", request=rate_limit_response.request, response=rate_limit_response
            ),
            success_response,
        ]

        # Mock response.json()
        success_response.json = lambda: {
            "choices": [{"message": {"content": "Success"}, "finish_reason": "stop"}]
        }

        messages = [Message(role="user", content="Test")]

        # Should retry and succeed
        with patch("asyncio.sleep"):  # Speed up test by mocking sleep
            response = await client.chat(messages)

        assert response.content == "Success"
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_server_error(self):
        """Test that client retries on 5xx server errors."""
        config = ClientConfig(
            mode=ClientMode.GATEWAY,
            gateway_url="http://test.com",
            model="test-model",
        )
        client = GatewayModelClient(config)

        mock_client = AsyncMock()
        client._client = mock_client

        from httpx import HTTPStatusError, Request, Response

        server_error_response = Response(
            status_code=503,
            request=Request("POST", "http://test.com/v1/chat/completions"),
        )
        success_response = Response(
            status_code=200,
            json={"choices": [{"message": {"content": "Success"}, "finish_reason": "stop"}]},
            request=Request("POST", "http://test.com/v1/chat/completions"),
        )

        mock_client.post.side_effect = [
            HTTPStatusError(
                "Server error",
                request=server_error_response.request,
                response=server_error_response,
            ),
            success_response,
        ]

        success_response.json = lambda: {
            "choices": [{"message": {"content": "Success"}, "finish_reason": "stop"}]
        }

        messages = [Message(role="user", content="Test")]

        with patch("asyncio.sleep"):
            response = await client.chat(messages)

        assert response.content == "Success"
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_client_error(self):
        """Test that client does NOT retry on 4xx client errors."""
        config = ClientConfig(
            mode=ClientMode.GATEWAY,
            gateway_url="http://test.com",
            model="test-model",
        )
        client = GatewayModelClient(config)

        mock_client = AsyncMock()
        client._client = mock_client

        from httpx import HTTPStatusError, Request, Response

        from src.core.errors import AgentError

        client_error_response = Response(
            status_code=400,
            text="Bad request",
            request=Request("POST", "http://test.com/v1/chat/completions"),
        )

        mock_client.post.side_effect = HTTPStatusError(
            "Bad request", request=client_error_response.request, response=client_error_response
        )

        messages = [Message(role="user", content="Test")]

        # Should raise AgentError without retry
        with pytest.raises(AgentError):
            await client.chat(messages)

        # Should only call once (no retry)
        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_raises_config_error_if_client_fails_to_initialize(self):
        """Test that ConfigurationError is raised if client initialization fails."""
        config = ClientConfig(
            mode=ClientMode.GATEWAY,
            gateway_url="http://test.com",
            model="test-model",
        )
        client = GatewayModelClient(config)

        # Force client to remain None
        with patch.object(client, "connect", new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = None
            client._client = None

            messages = [Message(role="user", content="Test")]

            with pytest.raises(ConfigurationError, match="Failed to initialize HTTP client"):
                await client.chat(messages)


class TestDirectModelClient:
    """Tests for DirectModelClient."""

    @pytest.mark.asyncio
    async def test_context_manager_connects_and_closes(self):
        """Test that context manager properly connects and closes providers."""
        from unittest.mock import AsyncMock

        config = ClientConfig(
            mode=ClientMode.DIRECT,
            google_api_key="test-key",
            model="gemini-test",
        )
        client = DirectModelClient(config)

        with patch("google.genai.Client") as mock_genai:
            mock_client = MagicMock()
            mock_client.aclose = AsyncMock()
            mock_genai.return_value = mock_client

            async with client:
                # Should have connected
                assert len(client._providers) > 0

            # Should have cleared providers
            assert len(client._providers) == 0

    @pytest.mark.asyncio
    async def test_raises_error_if_no_providers_available(self):
        """Test that error is raised if no providers are configured."""
        config = ClientConfig(
            mode=ClientMode.DIRECT,
            model="test-model",
        )
        client = DirectModelClient(config)

        # Mock httpx so direct OpenAI-compatible initialization also fails
        with patch("httpx.AsyncClient", side_effect=Exception("No httpx")):
            with pytest.raises(RuntimeError, match="No providers available"):
                await client.connect()

    @pytest.mark.asyncio
    async def test_chat_openai_prefers_responses_api(self):
        """Test that OpenAI direct chat prefers the Responses API."""
        config = ClientConfig(
            mode=ClientMode.DIRECT,
            openai_api_key="test-key",
            model="gpt-5.4",
        )
        client = DirectModelClient(config)

        mock_provider = AsyncMock()
        mock_provider.responses.create.return_value = types.SimpleNamespace(
            output_text="pong",
            output=[],
            usage=types.SimpleNamespace(input_tokens=3, output_tokens=1),
            status="completed",
        )
        client._providers[Provider.OPENAI] = mock_provider

        response = await client._chat_openai([Message(role="user", content="ping")])

        mock_provider.responses.create.assert_called_once()
        mock_provider.chat.completions.create.assert_not_called()
        assert response.content == "pong"
        assert response.usage == {
            "prompt_tokens": 3,
            "completion_tokens": 1,
            "total_tokens": 4,
        }

    def test_build_openai_responses_params_uses_structured_input_items(self):
        """Responses API input should use official structured content blocks."""
        params = DirectModelClient._build_openai_responses_params(
            model="gpt-5.4",
            msg_list=[
                {"role": "user", "content": "Reply with exactly pong."},
                {"role": "system", "content": "Be concise."},
            ],
            tools=None,
            temperature=0.0,
            max_output_tokens=None,
            include=["reasoning.encrypted_content"],
            stream=True,
        )

        assert params["input"] == [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Reply with exactly pong."}],
            },
            {
                "role": "system",
                "content": [{"type": "input_text", "text": "Be concise."}],
            },
        ]
        assert "temperature" not in params
        assert params["include"] == ["reasoning.encrypted_content"]
        assert params["stream"] is True

    def test_build_openai_responses_params_encodes_tool_history_for_responses(self):
        """Assistant tool calls and tool outputs should be encoded in Responses format."""
        params = DirectModelClient._build_openai_responses_params(
            model="gpt-5.4",
            msg_list=[
                {
                    "role": "assistant",
                    "content": "I'll create the file now.",
                    "tool_calls": [
                        {
                            "id": "call_write_1",
                            "type": "function",
                            "function": {
                                "name": "write",
                                "arguments": '{"path":"add.py","content":"def add(a, b):\\n    return a + b"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_write_1",
                    "name": "write",
                    "content": "Successfully wrote 32 bytes to add.py",
                },
            ],
            tools=None,
            temperature=0.7,
            max_output_tokens=None,
        )

        assert params["input"] == [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "I'll create the file now."}],
            },
            {
                "type": "function_call",
                "call_id": "call_write_1",
                "name": "write",
                "arguments": '{"path":"add.py","content":"def add(a, b):\\n    return a + b"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_write_1",
                "output": "Successfully wrote 32 bytes to add.py",
            },
        ]

    def test_build_openai_responses_params_prefers_provider_items_for_replay(self):
        """Stored provider items should be replayed verbatim for Responses tool loops."""
        params = DirectModelClient._build_openai_responses_params(
            model="gpt-5.4",
            msg_list=[
                {
                    "role": "assistant",
                    "provider_items": [
                        {"type": "reasoning", "id": "rs_1", "encrypted_content": "enc"},
                        {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "echo",
                            "arguments": "{\"text\":\"hi\"}",
                        },
                    ],
                }
            ],
            tools=None,
            temperature=0.7,
            max_output_tokens=None,
        )

        assert params["input"] == [
            {"type": "reasoning", "id": "rs_1", "encrypted_content": "enc"},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "echo",
                "arguments": "{\"text\":\"hi\"}",
            },
        ]

    def test_build_openai_responses_tools_sets_non_strict_mode(self):
        """Responses tool definitions should explicitly disable strict mode for compatibility."""
        tools = DirectModelClient._build_openai_responses_tools(
            [
                ToolDefinition(
                    name="echo",
                    description="Echo text",
                    parameters={"type": "object", "properties": {"text": {"type": "string"}}},
                )
            ]
        )

        assert tools == [
            {
                "type": "function",
                "name": "echo",
                "description": "Echo text",
                "parameters": {"type": "object", "properties": {"text": {"type": "string"}}},
                "strict": False,
            }
        ]

    @pytest.mark.asyncio
    async def test_chat_openai_skips_responses_for_custom_base_url(self):
        """Third-party OpenAI-compatible upstreams should go straight to chat.completions."""
        config = ClientConfig(
            mode=ClientMode.DIRECT,
            openai_api_key="test-key",
            openai_api_base="https://api-inference.modelscope.cn/v1",
            model="ZhipuAI/GLM-5",
        )
        client = DirectModelClient(config)

        mock_provider = AsyncMock()
        mock_provider.chat.completions.create.return_value = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content="pong", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=types.SimpleNamespace(prompt_tokens=3, completion_tokens=1, total_tokens=4),
        )
        client._providers[Provider.OPENAI] = mock_provider

        response = await client._chat_openai([Message(role="user", content="ping")])

        mock_provider.responses.create.assert_not_called()
        mock_provider.chat.completions.create.assert_called_once()
        assert response.content == "pong"

    @pytest.mark.asyncio
    async def test_chat_openai_respects_forced_chat_completions_protocol(self):
        """Explicit chat_completions config should skip Responses even on official URLs."""
        config = ClientConfig(
            mode=ClientMode.DIRECT,
            openai_api_key="test-key",
            openai_api_base="https://api.openai.com/v1",
            openai_protocol="chat_completions",
            model="gpt-5.4",
        )
        client = DirectModelClient(config)

        mock_provider = AsyncMock()
        mock_provider.chat.completions.create.return_value = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content="pong", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=types.SimpleNamespace(prompt_tokens=3, completion_tokens=1, total_tokens=4),
        )
        client._providers[Provider.OPENAI] = mock_provider

        response = await client._chat_openai([Message(role="user", content="ping")])

        mock_provider.responses.create.assert_not_called()
        mock_provider.chat.completions.create.assert_called_once()
        assert response.content == "pong"

    @pytest.mark.asyncio
    async def test_chat_openai_forced_responses_does_not_fallback(self):
        """Explicit responses config should fail fast instead of silently falling back."""
        config = ClientConfig(
            mode=ClientMode.DIRECT,
            openai_api_key="test-key",
            openai_api_base="https://api.openai.com/v1",
            openai_protocol="responses",
            model="gpt-5.4",
        )
        client = DirectModelClient(config)

        mock_provider = AsyncMock()
        mock_provider.responses.create.side_effect = RuntimeError("400 bad_response_status_code")
        client._providers[Provider.OPENAI] = mock_provider

        with pytest.raises(RuntimeError, match="bad_response_status_code"):
            await client._chat_openai([Message(role="user", content="ping")])

        mock_provider.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_chat_openai_drops_tools_when_config_disables_tool_calling(self):
        """OpenAI-compatible requests should omit tools when disabled by config."""
        config = ClientConfig(
            mode=ClientMode.DIRECT,
            openai_api_key="test-key",
            openai_api_base="https://compat.example/v1",
            openai_protocol="chat_completions",
            openai_capabilities=types.SimpleNamespace(tools=False, streaming=True),
            model="gpt-5.4",
        )
        client = DirectModelClient(config)

        mock_provider = AsyncMock()
        mock_provider.chat.completions.create.return_value = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content="pong", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=types.SimpleNamespace(prompt_tokens=3, completion_tokens=1, total_tokens=4),
        )
        client._providers[Provider.OPENAI] = mock_provider

        await client._chat_openai(
            [Message(role="user", content="ping")],
            tools=[
                ToolDefinition(
                    name="echo",
                    description="Echo text",
                    parameters={"type": "object", "properties": {"text": {"type": "string"}}},
                )
            ],
        )

        params = mock_provider.chat.completions.create.call_args.kwargs
        assert "tools" not in params

    @pytest.mark.asyncio
    async def test_chat_stream_falls_back_when_openai_streaming_disabled(self):
        """Streaming-disabled upstreams should use one-shot chat and emit a single chunk."""
        config = ClientConfig(
            mode=ClientMode.DIRECT,
            openai_api_key="test-key",
            openai_api_base="https://compat.example/v1",
            openai_protocol="chat_completions",
            openai_capabilities=types.SimpleNamespace(tools=True, streaming=False),
            model="gpt-5.4",
        )
        client = DirectModelClient(config)

        mock_provider = AsyncMock()
        mock_provider.chat.completions.create.return_value = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content="pong", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=types.SimpleNamespace(prompt_tokens=3, completion_tokens=1, total_tokens=4),
        )
        client._providers[Provider.OPENAI] = mock_provider

        chunks = []
        async for chunk in client.chat_stream([Message(role="user", content="ping")]):
            chunks.append(chunk)

        assert chunks == [
            StreamChunk(
                content="pong",
                thought=None,
                tool_calls=None,
                finish_reason="stop",
                usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            )
        ]
        mock_provider.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_anthropic_drops_tools_when_config_disables_tool_calling(self):
        """Anthropic-compatible requests should omit tools when disabled by config."""
        config = ClientConfig(
            mode=ClientMode.DIRECT,
            anthropic_api_key="test-key",
            anthropic_protocol="messages",
            anthropic_capabilities=types.SimpleNamespace(tools=False, streaming=True),
            model="claude-sonnet-4-20250514",
        )
        client = DirectModelClient(config)

        mock_provider = AsyncMock()
        mock_provider.messages.create.return_value = types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text="pong")],
            usage=types.SimpleNamespace(input_tokens=3, output_tokens=1),
        )
        client._providers[Provider.ANTHROPIC] = mock_provider

        await client._chat_anthropic(
            [Message(role="user", content="ping")],
            tools=[
                ToolDefinition(
                    name="echo",
                    description="Echo text",
                    parameters={"type": "object", "properties": {"text": {"type": "string"}}},
                )
            ],
        )

        params = mock_provider.messages.create.call_args.kwargs
        assert "tools" not in params

    @pytest.mark.asyncio
    async def test_chat_openai_allows_responses_for_official_openai_url(self):
        """Official OpenAI URL should use Responses API."""
        config = ClientConfig(
            mode=ClientMode.DIRECT,
            openai_api_key="test-key",
            openai_api_base="https://api.openai.com/v1",
            model="gpt-5.4",
        )
        client = DirectModelClient(config)

        mock_provider = AsyncMock()
        mock_provider.responses.create.return_value = types.SimpleNamespace(
            output_text="pong",
            output=[],
            usage=types.SimpleNamespace(input_tokens=3, output_tokens=1),
            status="completed",
        )
        client._providers[Provider.OPENAI] = mock_provider

        response = await client._chat_openai([Message(role="user", content="ping")])

        mock_provider.responses.create.assert_called_once()
        mock_provider.chat.completions.create.assert_not_called()
        assert response.content == "pong"

    @pytest.mark.asyncio
    async def test_chat_openai_locks_responses_protocol_after_success(self):
        """Once Responses succeeds, later turns should stay on Responses."""
        config = ClientConfig(
            mode=ClientMode.DIRECT,
            openai_api_key="test-key",
            openai_api_base="https://api.openai.com/v1",
            model="gpt-5.4",
        )
        client = DirectModelClient(config)

        mock_provider = AsyncMock()
        mock_provider.responses.create.side_effect = [
            types.SimpleNamespace(
                output_text="first",
                output=[],
                usage=types.SimpleNamespace(input_tokens=3, output_tokens=1),
                status="completed",
            ),
            types.SimpleNamespace(
                output_text="second",
                output=[],
                usage=types.SimpleNamespace(input_tokens=4, output_tokens=1),
                status="completed",
            ),
        ]
        client._providers[Provider.OPENAI] = mock_provider

        first = await client._chat_openai([Message(role="user", content="ping")])
        second = await client._chat_openai(
            [
                Message(
                    role="assistant",
                    content="calling tool",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read", "arguments": "{}"},
                        }
                    ],
                ),
                Message(role="tool", content="result", tool_call_id="call_1", name="read"),
            ]
        )

        assert first.content == "first"
        assert second.content == "second"
        assert client._openai_protocol == "responses"
        assert mock_provider.responses.create.call_count == 2
        mock_provider.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_chat_openai_locks_chat_completions_after_responses_failure(self):
        """Once fallback to chat.completions succeeds, later turns should stay there."""
        config = ClientConfig(
            mode=ClientMode.DIRECT,
            openai_api_key="test-key",
            openai_api_base="https://api-inference.modelscope.cn/v1",
            model="ZhipuAI/GLM-5",
        )
        client = DirectModelClient(config)

        mock_provider = AsyncMock()
        mock_provider.chat.completions.create.side_effect = [
            types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content="first", tool_calls=None),
                        finish_reason="stop",
                    )
                ],
                usage=types.SimpleNamespace(prompt_tokens=3, completion_tokens=1, total_tokens=4),
            ),
            types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content="second", tool_calls=None),
                        finish_reason="stop",
                    )
                ],
                usage=types.SimpleNamespace(prompt_tokens=4, completion_tokens=1, total_tokens=5),
            ),
        ]
        client._providers[Provider.OPENAI] = mock_provider

        first = await client._chat_openai([Message(role="user", content="ping")])
        second = await client._chat_openai([Message(role="user", content="pong")])

        assert first.content == "first"
        assert second.content == "second"
        assert client._openai_protocol == "chat_completions"
        mock_provider.responses.create.assert_not_called()
        assert mock_provider.chat.completions.create.call_count == 2

    def test_openai_protocol_cache_seeds_new_client(self):
        """A successful probe should seed later client sessions for the same upstream."""
        key = ("https://api.openai.com/v1", "gpt-5.4")
        _OPENAI_PROTOCOL_CACHE[key] = "responses"

        try:
            client = DirectModelClient(
                ClientConfig(
                    mode=ClientMode.DIRECT,
                    openai_api_key="test-key",
                    openai_api_base="https://api.openai.com/v1",
                    model="gpt-5.4",
                )
            )
            assert client._openai_protocol == "responses"
        finally:
            _OPENAI_PROTOCOL_CACHE.pop(key, None)

    @pytest.mark.asyncio
    async def test_close_keeps_global_protocol_cache(self):
        """Closing a client should not erase the shared provider capability cache."""
        key = ("https://api.openai.com/v1", "gpt-5.4")
        _OPENAI_PROTOCOL_CACHE.pop(key, None)

        client = DirectModelClient(
            ClientConfig(
                mode=ClientMode.DIRECT,
                openai_api_key="test-key",
                openai_api_base="https://api.openai.com/v1",
                model="gpt-5.4",
            )
        )
        client._providers[Provider.OPENAI] = AsyncMock()
        client._openai_protocol = "responses"
        _OPENAI_PROTOCOL_CACHE[key] = "responses"

        await client.close()

        assert _OPENAI_PROTOCOL_CACHE[key] == "responses"
        _OPENAI_PROTOCOL_CACHE.pop(key, None)

    @pytest.mark.asyncio
    async def test_chat_openai_uses_responses_for_tool_history(self):
        """Chat-style tool history should be encoded for Responses API replay."""
        config = ClientConfig(
            mode=ClientMode.DIRECT,
            openai_api_key="test-key",
            model="gpt-5.4",
        )
        client = DirectModelClient(config)

        mock_provider = AsyncMock()
        mock_provider.responses.create.return_value = types.SimpleNamespace(
            output_text="done",
            output=[],
            usage=types.SimpleNamespace(input_tokens=3, output_tokens=1),
            status="completed",
        )
        client._providers[Provider.OPENAI] = mock_provider

        response = await client._chat_openai(
            [
                Message(
                    role="assistant",
                    content="calling tool",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read", "arguments": "{}"},
                        }
                    ],
                ),
                Message(role="tool", content="result", tool_call_id="call_1", name="read"),
            ]
        )

        mock_provider.responses.create.assert_called_once()
        mock_provider.chat.completions.create.assert_not_called()
        assert response.content == "done"

    @pytest.mark.asyncio
    async def test_chat_openai_rejects_string_responses_payload(self):
        """Test that string payloads from responses API fail clearly."""
        config = ClientConfig(
            mode=ClientMode.DIRECT,
            openai_api_key="test-key",
            model="gpt-5.4",
        )
        client = DirectModelClient(config)

        mock_provider = AsyncMock()
        mock_provider.responses.create.return_value = "<html>not json</html>"
        client._providers[Provider.OPENAI] = mock_provider

        with pytest.raises(RuntimeError, match="non-OpenAI responses payload"):
            await client._chat_openai([Message(role="user", content="ping")])

    @pytest.mark.asyncio
    async def test_chat_openai_falls_back_when_responses_object_is_failed(self):
        """Test fallback when Responses API returns a failed response object."""
        from src.core.llm.model_utils import ToolDefinition

        config = ClientConfig(
            mode=ClientMode.DIRECT,
            openai_api_key="test-key",
            model="gpt-5.4",
        )
        client = DirectModelClient(config)

        mock_provider = AsyncMock()
        mock_provider.responses.create.return_value = types.SimpleNamespace(
            status="failed",
            error=types.SimpleNamespace(message="Unsupported model: glm-5"),
        )
        mock_provider.chat.completions.create.return_value = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content=None,
                        tool_calls=[
                            types.SimpleNamespace(
                                id="call_1",
                                function=types.SimpleNamespace(
                                    name="write", arguments='{"path":"a.txt","content":"hello"}'
                                ),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        client._providers[Provider.OPENAI] = mock_provider

        response = await client._chat_openai(
            [Message(role="user", content="use write")],
            tools=[
                ToolDefinition(
                    name="write",
                    description="Write file",
                    parameters={"type": "object"},
                )
            ],
        )

        mock_provider.responses.create.assert_called_once()
        mock_provider.chat.completions.create.assert_called_once()
        assert response.tool_calls is not None
        assert response.tool_calls[0]["function"]["name"] == "write"

    @pytest.mark.asyncio
    async def test_chat_openai_falls_back_when_responses_object_has_top_level_error_fields(self):
        """Test fallback when Responses API returns a malformed object with code/message fields."""
        config = ClientConfig(
            mode=ClientMode.DIRECT,
            openai_api_key="test-key",
            openai_api_base="https://api.openai.com/v1",
            model="gpt-5.4",
        )
        client = DirectModelClient(config)

        mock_provider = AsyncMock()
        mock_provider.responses.create.return_value = types.SimpleNamespace(
            status=None,
            error=None,
            code="InvalidParameter",
            message="Invalid role value: 'tool'",
            output=None,
        )
        mock_provider.chat.completions.create.return_value = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content="done", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        client._providers[Provider.OPENAI] = mock_provider

        response = await client._chat_openai(
            [
                Message(role="user", content="ping"),
            ]
        )

        mock_provider.responses.create.assert_called_once()
        mock_provider.chat.completions.create.assert_called_once()
        assert response.content == "done"

    @pytest.mark.asyncio
    async def test_chat_openai_preserves_gemini_thought_signature_in_tool_calls(self):
        """Thought signatures from OpenAI-compatible tool calls should survive replay."""
        config = ClientConfig(
            mode=ClientMode.DIRECT,
            openai_api_key="test-key",
            openai_protocol="chat_completions",
            model="gemini-3-flash-preview",
        )
        client = DirectModelClient(config)

        thought_signature = b"sig"
        mock_provider = AsyncMock()
        mock_provider.chat.completions.create.return_value = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content=None,
                        tool_calls=[
                            types.SimpleNamespace(
                                id="call_1",
                                thought_signature=thought_signature,
                                function=types.SimpleNamespace(
                                    name="run",
                                    arguments='{"command":"pwd"}',
                                ),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        client._providers[Provider.OPENAI] = mock_provider

        response = await client._chat_openai([Message(role="user", content="use run")])

        assert response.tool_calls is not None
        assert response.tool_calls[0]["thought_signature"] == "base64:" + base64.b64encode(
            thought_signature
        ).decode("ascii")
        assert (
            response.tool_calls[0]["function"]["thought_signature"]
            == response.tool_calls[0]["thought_signature"]
        )

    @pytest.mark.asyncio
    async def test_stream_openai_preserves_gemini_thought_signature_in_tool_deltas(self):
        """Streaming OpenAI-compatible tool deltas should include thought signatures."""
        config = ClientConfig(
            mode=ClientMode.DIRECT,
            openai_api_key="test-key",
            openai_protocol="chat_completions",
            model="gemini-3-flash-preview",
        )
        client = DirectModelClient(config)

        thought_signature = b"sig"

        class MockStream:
            def __aiter__(self):
                async def gen():
                    yield types.SimpleNamespace(
                        choices=[
                            types.SimpleNamespace(
                                delta=types.SimpleNamespace(
                                    content=None,
                                    tool_calls=[
                                        types.SimpleNamespace(
                                            index=0,
                                            id="call_1",
                                            thought_signature=thought_signature,
                                            function=types.SimpleNamespace(
                                                name="run",
                                                arguments='{"command":"pwd"}',
                                                thought_signature=thought_signature,
                                            ),
                                        )
                                    ],
                                ),
                                finish_reason=None,
                            )
                        ],
                        usage=None,
                    )
                    yield types.SimpleNamespace(
                        choices=[
                            types.SimpleNamespace(
                                delta=types.SimpleNamespace(content=None, tool_calls=None),
                                finish_reason="tool_calls",
                            )
                        ],
                        usage=None,
                    )

                return gen()

        mock_provider = AsyncMock()
        mock_provider.chat.completions.create.return_value = MockStream()
        client._providers[Provider.OPENAI] = mock_provider

        chunks = []
        async for chunk in client._stream_openai([Message(role="user", content="use run")]):
            chunks.append(chunk)

        assert chunks[0].tool_calls is not None


class TestOpenAIAdapter:
    """Tests for OpenAI adapter helpers."""

    def test_build_params_injects_dummy_thought_signature_for_gemini_3_tool_history(self):
        params = OpenAIAdapter.build_params(
            model="gemini-3-flash-preview",
            msg_list=[
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "run", "arguments": '{"command":"pwd"}'},
                        }
                    ],
                }
            ],
            tools=None,
            temperature=0.1,
            max_tokens=None,
        )

        tool_call = params["messages"][0]["tool_calls"][0]
        assert tool_call["thought_signature"] == "skip_thought_signature_validator"
        assert tool_call["function"]["thought_signature"] == "skip_thought_signature_validator"

    def test_build_params_preserves_existing_thought_signature_for_gemini_3(self):
        params = OpenAIAdapter.build_params(
            model="gemini-3-flash-preview",
            msg_list=[
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "thought_signature": "base64:c2ln",
                            "function": {
                                "name": "run",
                                "arguments": '{"command":"pwd"}',
                                "thought_signature": "base64:c2ln",
                            },
                        }
                    ],
                }
            ],
            tools=None,
            temperature=0.1,
            max_tokens=None,
        )

        tool_call = params["messages"][0]["tool_calls"][0]
        assert tool_call["thought_signature"] == "base64:c2ln"
        assert tool_call["function"]["thought_signature"] == "base64:c2ln"


class TestModelClient:
    """Tests for ModelClient factory."""

    @pytest.mark.asyncio
    async def test_create_gateway_client(self):
        """Test creating a gateway mode client."""
        config = ClientConfig(
            mode=ClientMode.GATEWAY,
            gateway_url="http://test.com",
            model="test-model",
        )

        with patch("httpx.AsyncClient"):
            client = await ModelClient.create(config)
            assert isinstance(client, GatewayModelClient)

    @pytest.mark.asyncio
    async def test_create_direct_client(self):
        """Test creating a direct mode client."""
        config = ClientConfig(
            mode=ClientMode.DIRECT,
            google_api_key="test-key",
            model="gemini-test",
        )

        with patch("google.genai.Client"):
            client = await ModelClient.create(config)
            assert isinstance(client, DirectModelClient)

    def test_get_mode_detects_gateway(self):
        """Test that get_mode detects gateway mode from config."""
        with patch(
            "src.core.config.load_config",
            return_value={
                "model": "gpt-4o",
                "gateway_url": "http://test.com",
            },
        ):
            mode = ModelClient.get_mode()
            assert mode == ClientMode.GATEWAY

    def test_get_mode_detects_direct(self):
        """Test that get_mode detects direct mode when no gateway URL."""
        with patch("src.core.config.load_config", return_value={"model": "test-model"}):
            mode = ModelClient.get_mode()
            assert mode == ClientMode.DIRECT


class TestMessage:
    """Tests for Message class."""

    def test_to_dict_includes_all_fields(self):
        """Test that to_dict includes all non-None fields."""
        msg = Message(
            role="assistant",
            content="Hello",
            thought="Thinking...",
            tool_calls=[{"id": "1", "name": "test"}],
        )
        d = msg.to_dict()

        assert d["role"] == "assistant"
        assert d["content"] == "Hello"
        assert d["thought"] == "Thinking..."
        assert d["tool_calls"] == [{"id": "1", "name": "test"}]

    def test_to_dict_excludes_none_fields(self):
        """Test that to_dict excludes None fields."""
        msg = Message(role="user", content="Hello")
        d = msg.to_dict()

        assert "role" in d
        assert "content" in d
        assert "thought" not in d
        assert "tool_calls" not in d


class TestChatResponse:
    """Tests for ChatResponse class."""

    def test_has_tool_calls_true(self):
        """Test has_tool_calls returns True when tool calls present."""
        response = ChatResponse(
            content=None,
            tool_calls=[{"id": "1", "name": "test"}],
        )
        assert response.has_tool_calls is True

    def test_has_tool_calls_false(self):
        """Test has_tool_calls returns False when no tool calls."""
        response = ChatResponse(content="Hello")
        assert response.has_tool_calls is False


# Additional comprehensive tests for model_client.py


class TestGatewayModelClientErrorHandling:
    """Tests for error handling in GatewayModelClient."""

    @pytest.mark.asyncio
    async def test_network_error_raises_transient_error(self):
        """Test that network errors are raised as TransientError."""
        config = ClientConfig(
            mode=ClientMode.GATEWAY,
            gateway_url="http://test.com",
            model="test-model",
        )
        client = GatewayModelClient(config)
        mock_client = AsyncMock()
        client._client = mock_client

        from httpx import RequestError

        mock_client.post.side_effect = RequestError("Connection refused")

        messages = [Message(role="user", content="Test")]

        with patch("asyncio.sleep"):
            with pytest.raises(TransientError):
                await client.chat(messages)

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self):
        """Test that max retries are exhausted and error is raised."""
        config = ClientConfig(
            mode=ClientMode.GATEWAY,
            gateway_url="http://test.com",
            model="test-model",
        )
        client = GatewayModelClient(config)
        mock_client = AsyncMock()
        client._client = mock_client

        from httpx import HTTPStatusError, Request, Response

        server_error_response = Response(
            status_code=503,
            request=Request("POST", "http://test.com/v1/chat/completions"),
        )

        mock_client.post.side_effect = HTTPStatusError(
            "Server error", request=server_error_response.request, response=server_error_response
        )

        messages = [Message(role="user", content="Test")]

        with patch("asyncio.sleep"):
            with pytest.raises(TransientError):
                await client.chat(messages)

    @pytest.mark.asyncio
    async def test_rate_limit_with_retry_after_header(self):
        """Test rate limit respects Retry-After header."""
        config = ClientConfig(
            mode=ClientMode.GATEWAY,
            gateway_url="http://test.com",
            model="test-model",
        )
        client = GatewayModelClient(config)
        mock_client = AsyncMock()
        client._client = mock_client

        from httpx import HTTPStatusError, Request, Response

        rate_limit_response = Response(
            status_code=429,
            headers={"Retry-After": "5"},
            request=Request("POST", "http://test.com/v1/chat/completions"),
        )
        success_response = Response(
            status_code=200,
            request=Request("POST", "http://test.com/v1/chat/completions"),
        )
        success_response.json = lambda: {
            "choices": [{"message": {"content": "Success"}, "finish_reason": "stop"}]
        }

        mock_client.post.side_effect = [
            HTTPStatusError(
                "Rate limited", request=rate_limit_response.request, response=rate_limit_response
            ),
            success_response,
        ]

        messages = [Message(role="user", content="Test")]

        with patch("asyncio.sleep") as mock_sleep:
            response = await client.chat(messages)
            # Verify sleep was called with retry_after value
            mock_sleep.assert_called()

        assert response.content == "Success"

    @pytest.mark.asyncio
    async def test_empty_choices_in_response(self):
        """Test handling of empty choices in API response."""
        config = ClientConfig(
            mode=ClientMode.GATEWAY,
            gateway_url="http://test.com",
            model="test-model",
        )
        client = GatewayModelClient(config)
        mock_client = AsyncMock()
        client._client = mock_client

        from httpx import Request, Response

        response_obj = Response(
            status_code=200,
            request=Request("POST", "http://test.com/v1/chat/completions"),
        )
        response_obj.json = lambda: {"choices": [], "usage": {"total_tokens": 10}}

        mock_client.post.return_value = response_obj

        messages = [Message(role="user", content="Test")]
        response = await client.chat(messages)

        assert response.content is None
        assert response.tool_calls is None
        assert response.finish_reason == "error"

    @pytest.mark.asyncio
    async def test_chat_with_tools_normalization(self):
        """Test that tools are properly normalized in chat request."""
        config = ClientConfig(
            mode=ClientMode.GATEWAY,
            gateway_url="http://test.com",
            model="test-model",
        )
        client = GatewayModelClient(config)
        mock_client = AsyncMock()
        client._client = mock_client

        from httpx import Request, Response

        response_obj = Response(
            status_code=200,
            request=Request("POST", "http://test.com/v1/chat/completions"),
        )
        response_obj.json = lambda: {
            "choices": [{"message": {"content": "Success"}, "finish_reason": "stop"}]
        }
        mock_client.post.return_value = response_obj

        from src.core.llm.model_utils import ToolDefinition

        tools = [
            ToolDefinition(
                name="test_tool",
                description="A test tool",
                parameters={"type": "object", "properties": {}},
            )
        ]

        messages = [Message(role="user", content="Test")]
        await client.chat(messages, tools=tools)

        # Verify tools were included in the request
        call_args = mock_client.post.call_args
        assert call_args is not None
        payload = call_args.kwargs.get("json", {})
        assert "tools" in payload
        assert len(payload["tools"]) == 1

    @pytest.mark.asyncio
    async def test_chat_with_dict_messages(self):
        """Test chat with dict-formatted messages."""
        config = ClientConfig(
            mode=ClientMode.GATEWAY,
            gateway_url="http://test.com",
            model="test-model",
        )
        client = GatewayModelClient(config)
        mock_client = AsyncMock()
        client._client = mock_client

        from httpx import Request, Response

        response_obj = Response(
            status_code=200,
            request=Request("POST", "http://test.com/v1/chat/completions"),
        )
        response_obj.json = lambda: {
            "choices": [{"message": {"content": "Success"}, "finish_reason": "stop"}]
        }
        mock_client.post.return_value = response_obj

        messages = [{"role": "user", "content": "Test"}]
        response = await client.chat(messages)

        assert response.content == "Success"

    @pytest.mark.asyncio
    async def test_chat_with_custom_temperature(self):
        """Test chat with custom temperature parameter."""
        config = ClientConfig(
            mode=ClientMode.GATEWAY,
            gateway_url="http://test.com",
            model="test-model",
            temperature=0.5,
        )
        client = GatewayModelClient(config)
        mock_client = AsyncMock()
        client._client = mock_client

        from httpx import Request, Response

        response_obj = Response(
            status_code=200,
            request=Request("POST", "http://test.com/v1/chat/completions"),
        )
        response_obj.json = lambda: {
            "choices": [{"message": {"content": "Success"}, "finish_reason": "stop"}]
        }
        mock_client.post.return_value = response_obj

        messages = [Message(role="user", content="Test")]
        await client.chat(messages, temperature=0.9)

        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json", {})
        assert payload["temperature"] == 0.9

    @pytest.mark.asyncio
    async def test_chat_with_max_tokens(self):
        """Test chat with max_tokens configuration."""
        config = ClientConfig(
            mode=ClientMode.GATEWAY,
            gateway_url="http://test.com",
            model="test-model",
            max_tokens=100,
        )
        client = GatewayModelClient(config)
        mock_client = AsyncMock()
        client._client = mock_client

        from httpx import Request, Response

        response_obj = Response(
            status_code=200,
            request=Request("POST", "http://test.com/v1/chat/completions"),
        )
        response_obj.json = lambda: {
            "choices": [{"message": {"content": "Success"}, "finish_reason": "stop"}]
        }
        mock_client.post.return_value = response_obj

        messages = [Message(role="user", content="Test")]
        await client.chat(messages)

        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json", {})
        assert payload["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_list_models_from_gateway(self):
        """Test listing models from gateway."""
        config = ClientConfig(
            mode=ClientMode.GATEWAY,
            gateway_url="http://test.com",
            model="test-model",
        )
        client = GatewayModelClient(config)
        mock_client = AsyncMock()
        client._client = mock_client

        from httpx import Request, Response

        response_obj = Response(
            status_code=200,
            request=Request("GET", "http://test.com/v1/models"),
        )
        response_obj.json = lambda: {
            "data": [
                {"id": "model-1", "name": "Model 1"},
                {"id": "model-2", "name": "Model 2"},
            ]
        }
        mock_client.get.return_value = response_obj

        models = await client.list_models()

        assert len(models) == 2
        assert models[0]["id"] == "model-1"

    @pytest.mark.asyncio
    async def test_gateway_url_validation(self):
        """Test that gateway URL is required."""
        config = ClientConfig(
            mode=ClientMode.GATEWAY,
            model="test-model",
        )
        client = GatewayModelClient(config)

        with pytest.raises(ValueError, match="Gateway URL must be set"):
            await client.connect()

    @pytest.mark.asyncio
    async def test_gateway_with_authorization_header(self):
        """Test that authorization header is set when gateway_key is provided."""
        config = ClientConfig(
            mode=ClientMode.GATEWAY,
            gateway_url="http://test.com",
            gateway_key="test-key-123",
            model="test-model",
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            client = GatewayModelClient(config)
            await client.connect()

            # Verify AsyncClient was called with correct headers
            call_args = mock_client_class.call_args
            assert call_args is not None
            headers = call_args.kwargs.get("headers", {})
            assert headers.get("Authorization") == "Bearer test-key-123"

    @pytest.mark.asyncio
    async def test_chat_stream_with_valid_response(self):
        """Test streaming chat with valid SSE response."""
        config = ClientConfig(
            mode=ClientMode.GATEWAY,
            gateway_url="http://test.com",
            model="test-model",
        )
        client = GatewayModelClient(config)
        mock_client = AsyncMock()
        client._client = mock_client

        # Mock streaming response
        class MockResponse:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def raise_for_status(self):
                pass

            async def aiter_lines(self):
                yield 'data: {"choices": [{"delta": {"content": "Hello"}, "finish_reason": null}]}'
                yield 'data: {"choices": [{"delta": {"content": " world"}, "finish_reason": null}]}'
                yield "data: [DONE]"

        mock_client.stream = MagicMock(return_value=MockResponse())

        messages = [Message(role="user", content="Test")]
        chunks = []
        async for chunk in client.chat_stream(messages):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].content == "Hello"
        assert chunks[1].content == " world"

    @pytest.mark.asyncio
    async def test_chat_stream_with_malformed_json(self):
        """Test streaming chat handles malformed JSON gracefully."""
        config = ClientConfig(
            mode=ClientMode.GATEWAY,
            gateway_url="http://test.com",
            model="test-model",
        )
        client = GatewayModelClient(config)
        mock_client = AsyncMock()
        client._client = mock_client

        class MockResponse:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def raise_for_status(self):
                pass

            async def aiter_lines(self):
                yield "data: {invalid json}"
                yield 'data: {"choices": [{"delta": {"content": "Valid"}, "finish_reason": null}]}'
                yield "data: [DONE]"

        mock_client.stream = MagicMock(return_value=MockResponse())

        messages = [Message(role="user", content="Test")]
        chunks = []
        async for chunk in client.chat_stream(messages):
            chunks.append(chunk)

        # Should skip malformed JSON and continue
        assert len(chunks) == 1
        assert chunks[0].content == "Valid"

    @pytest.mark.asyncio
    async def test_chat_stream_with_tool_calls(self):
        """Test streaming chat with tool calls in response."""
        config = ClientConfig(
            mode=ClientMode.GATEWAY,
            gateway_url="http://test.com",
            model="test-model",
        )
        client = GatewayModelClient(config)
        mock_client = AsyncMock()
        client._client = mock_client

        class MockResponse:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def raise_for_status(self):
                pass

            async def aiter_lines(self):
                yield 'data: {"choices": [{"delta": {"tool_calls": [{"id": "1", "name": "test"}]}, "finish_reason": "tool_calls"}]}'
                yield "data: [DONE]"

        mock_client.stream = MagicMock(return_value=MockResponse())

        messages = [Message(role="user", content="Test")]
        chunks = []
        async for chunk in client.chat_stream(messages):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].tool_calls is not None


class TestDirectModelClientProviderDetection:
    """Tests for provider detection in DirectModelClient."""

    @pytest.mark.asyncio
    async def test_detect_google_provider(self):
        """Test detection of Google provider from model name."""
        config = ClientConfig(
            mode=ClientMode.DIRECT,
            google_api_key="test-key",
            model="gemini-2.5-flash",
        )
        client = DirectModelClient(config)

        with patch("google.genai.Client"):
            await client.connect()
            provider = client._detect_provider("gemini-2.5-flash")
            assert provider == Provider.GOOGLE

    @pytest.mark.asyncio
    async def test_google_openai_compatible_base_initializes_google_provider(self):
        """Gemini OpenAI-compatible config should still prefer the native Google provider."""
        config = ClientConfig(
            mode=ClientMode.DIRECT,
            openai_api_key="AIza-test-key",
            openai_api_base="https://generativelanguage.googleapis.com/v1beta/openai",
            model="gemini-3-flash-preview",
        )
        client = DirectModelClient(config)

        with patch("google.genai.Client") as google_client, patch("openai.AsyncOpenAI"):
            await client.connect()
            google_client.assert_called_once_with(api_key="AIza-test-key")
            provider = client._detect_provider("gemini-3-flash-preview")
            assert provider == Provider.GOOGLE

    @pytest.mark.asyncio
    async def test_detect_openai_provider(self):
        """Test detection of OpenAI provider from model name."""
        config = ClientConfig(
            mode=ClientMode.DIRECT,
            openai_api_key="test-key",
            model="gpt-4",
        )
        client = DirectModelClient(config)

        with patch("openai.AsyncOpenAI"):
            await client.connect()
            provider = client._detect_provider("gpt-4")
            assert provider == Provider.OPENAI

    @pytest.mark.asyncio
    async def test_detect_anthropic_provider(self):
        """Test detection of Anthropic provider from model name."""
        config = ClientConfig(
            mode=ClientMode.DIRECT,
            anthropic_api_key="test-key",
            model="claude-3-sonnet",
        )
        client = DirectModelClient(config)

        anthropic_module = types.SimpleNamespace(AsyncAnthropic=MagicMock())
        with patch.dict(sys.modules, {"anthropic": anthropic_module}):
            await client.connect()
            provider = client._detect_provider("claude-3-sonnet")
            assert provider == Provider.ANTHROPIC

    @pytest.mark.asyncio
    async def test_provider_fallback_to_first_available(self):
        """Test fallback to first available provider when model not recognized."""
        config = ClientConfig(
            mode=ClientMode.DIRECT,
            google_api_key="test-key",
            model="unknown-model",
        )
        client = DirectModelClient(config)

        with patch("google.genai.Client"):
            await client.connect()
            provider = client._detect_provider("unknown-model")
            # Should return first available provider
            assert provider in client._providers

    @pytest.mark.asyncio
    async def test_no_provider_raises_error(self):
        """Test that error is raised when no provider is available."""
        config = ClientConfig(
            mode=ClientMode.DIRECT,
            model="test-model",
        )
        client = DirectModelClient(config)

        with pytest.raises(RuntimeError, match="No providers available"):
            client._detect_provider("test-model")


class TestClientConfiguration:
    """Tests for ClientConfig and configuration validation."""

    def test_config_from_env_gateway_mode(self):
        """Test loading config from project config in gateway mode."""
        with patch(
            "src.core.config.load_config",
            return_value={
                "model": "test-model",
                "gateway_url": "http://gateway.test.com",
                "gateway_key": "test-key",
            },
        ):
            config = ClientConfig.from_env()
            assert config.mode == ClientMode.GATEWAY
            assert config.gateway_url == "http://gateway.test.com"
            assert config.gateway_key == "test-key"
            assert config.model == "test-model"

    def test_config_from_env_direct_mode(self):
        """Test loading config from project config in direct mode."""
        with patch(
            "src.core.config.load_config",
            return_value={
                "model": "gemini-2.5-flash",
                "google_api_key": "google-key",
                "openai_api_key": "openai-key",
            },
        ):
            config = ClientConfig.from_env()
            assert config.mode == ClientMode.DIRECT
            assert config.google_api_key == "google-key"
            assert config.openai_api_key == "openai-key"

    def test_config_defaults(self):
        """Test default configuration values."""
        config = ClientConfig()
        assert config.mode == ClientMode.DIRECT
        assert config.temperature == 0.7
        assert config.model is None
        assert config.max_tokens is None

    def test_config_custom_values(self):
        """Test configuration with custom values."""
        config = ClientConfig(
            mode=ClientMode.GATEWAY,
            gateway_url="http://test.com",
            temperature=0.5,
            max_tokens=500,
            system_prompt="You are helpful",
        )
        assert config.temperature == 0.5
        assert config.max_tokens == 500
        assert config.system_prompt == "You are helpful"


class TestMessageHandling:
    """Tests for Message class and message handling."""

    def test_message_with_tool_call_id(self):
        """Test message with tool_call_id field."""
        msg = Message(
            role="tool",
            content="Tool result",
            tool_call_id="call_123",
            name="my_tool",
        )
        d = msg.to_dict()
        assert d["tool_call_id"] == "call_123"
        assert d["name"] == "my_tool"

    def test_message_with_empty_content(self):
        """Test message with None content."""
        msg = Message(role="assistant", content=None)
        d = msg.to_dict()
        assert "content" not in d

    def test_message_with_all_fields(self):
        """Test message with all fields populated."""
        msg = Message(
            role="assistant",
            content="Response",
            thought="Thinking...",
            tool_calls=[{"id": "1", "name": "tool"}],
            tool_call_id="call_1",
            name="tool_name",
        )
        d = msg.to_dict()
        assert len(d) == 6
        assert d["role"] == "assistant"
        assert d["content"] == "Response"
        assert d["thought"] == "Thinking..."
        assert d["tool_calls"] == [{"id": "1", "name": "tool"}]
        assert d["tool_call_id"] == "call_1"
        assert d["name"] == "tool_name"


class TestToolDefinition:
    """Tests for ToolDefinition class."""

    def test_tool_to_openai_format(self):
        """Test converting tool to OpenAI format."""
        from src.core.llm.model_utils import ToolDefinition

        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {"arg": {"type": "string"}}},
        )
        openai_format = tool.to_openai_format()
        assert openai_format["type"] == "function"
        assert openai_format["function"]["name"] == "test_tool"
        assert openai_format["function"]["description"] == "A test tool"

    def test_tool_to_genai_format(self):
        """Test converting tool to Google GenAI format."""
        from src.core.llm.model_utils import ToolDefinition

        with patch("google.genai.types.FunctionDeclaration") as mock_func_decl:
            tool = ToolDefinition(
                name="test_tool", description="A test tool", parameters={"type": "object"}
            )
            tool.to_genai_format()
            mock_func_decl.assert_called_once()


class TestChatResponseHandling:
    """Tests for ChatResponse class."""

    def test_response_with_usage_info(self):
        """Test response with usage information."""
        response = ChatResponse(
            content="Hello",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        )
        assert response.usage["total_tokens"] == 15

    def test_response_with_raw_data(self):
        """Test response with raw provider data."""
        raw_data = {"custom": "data"}
        response = ChatResponse(
            content="Hello",
            raw=raw_data,
        )
        assert response.raw == raw_data

    def test_response_finish_reason_variations(self):
        """Test different finish reason values."""
        for reason in ["stop", "tool_calls", "length", "error"]:
            response = ChatResponse(content="Test", finish_reason=reason)
            assert response.finish_reason == reason


class TestStreamChunk:
    """Tests for StreamChunk class."""

    def test_stream_chunk_with_content(self):
        """Test stream chunk with content."""
        from src.core.llm.model_utils import StreamChunk

        chunk = StreamChunk(content="Hello")
        assert chunk.content == "Hello"

    def test_stream_chunk_with_tool_calls(self):
        """Test stream chunk with tool calls."""
        from src.core.llm.model_utils import StreamChunk

        chunk = StreamChunk(tool_calls=[{"id": "1", "name": "tool"}], finish_reason="tool_calls")
        assert chunk.tool_calls is not None
        assert len(chunk.tool_calls) == 1

    def test_stream_chunk_with_usage(self):
        """Test stream chunk with usage information."""
        from src.core.llm.model_utils import StreamChunk

        chunk = StreamChunk(content="Test", usage={"total_tokens": 10})
        assert chunk.usage["total_tokens"] == 10


class TestToolCall:
    """Tests for ToolCall class."""

    def test_tool_call_to_dict(self):
        """Test converting ToolCall to dict."""
        from src.core.llm.model_utils import ToolCall

        tc = ToolCall(id="call_123", name="my_tool", arguments={"arg1": "value1"})
        d = tc.to_dict()
        assert d["id"] == "call_123"
        assert d["name"] == "my_tool"
        assert d["arguments"] == {"arg1": "value1"}

    def test_tool_call_from_dict(self):
        """Test creating ToolCall from dict."""
        from src.core.llm.model_utils import ToolCall

        data = {"id": "call_456", "name": "another_tool", "arguments": {"key": "value"}}
        tc = ToolCall.from_dict(data)
        assert tc.id == "call_456"
        assert tc.name == "another_tool"
        assert tc.arguments == {"key": "value"}

    def test_tool_call_from_dict_with_missing_fields(self):
        """Test ToolCall.from_dict handles missing fields."""
        from src.core.llm.model_utils import ToolCall

        data = {"id": "call_789"}
        tc = ToolCall.from_dict(data)
        assert tc.id == "call_789"
        assert tc.name == ""
        assert tc.arguments == {}


class TestModelClientFactory:
    """Tests for ModelClient factory methods."""

    def test_get_mode_info_gateway(self):
        """Test get_mode_info returns gateway info."""
        with patch(
            "src.core.config.load_config",
            return_value={
                "model": "test-model",
                "gateway_url": "http://test.com",
            },
        ):
            info = ModelClient.get_mode_info()
            assert info["mode"] == "gateway"
            assert info["gateway_url"] == "http://test.com"

    def test_get_mode_info_direct(self):
        """Test get_mode_info returns direct mode info."""
        with patch("src.core.config.load_config", return_value={"model": "test-model"}):
            info = ModelClient.get_mode_info()
            assert info["mode"] == "direct"
            assert "providers" in info

    @pytest.mark.asyncio
    async def test_create_with_default_config(self):
        """Test creating client from config file settings."""
        with patch("google.genai.Client"):
            with patch(
                "src.core.config.load_config",
                return_value={
                    "model": "gemini-test",
                    "google_api_key": "google-key",
                },
            ):
                client = await ModelClient.create()
                assert isinstance(client, DirectModelClient)

    @pytest.mark.asyncio
    async def test_create_loads_config_from_env(self):
        """Test that create loads config from the project config file."""
        with patch("httpx.AsyncClient"):
            with patch(
                "src.core.config.load_config",
                return_value={
                    "model": "test-model",
                    "gateway_url": "http://test.com",
                },
            ):
                client = await ModelClient.create()
                assert isinstance(client, GatewayModelClient)
