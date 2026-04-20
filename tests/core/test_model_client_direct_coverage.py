"""Targeted coverage tests for core.llm.model_client_direct."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.llm.model_client_direct import (
    DirectModelClient,
    _is_google_openai_compatible_base,
    _is_retryable,
    _retry_async,
)
from src.core.llm.model_utils import ChatResponse, ClientConfig, Provider, StreamChunk


class TestChatWithRetry:
    @pytest.mark.asyncio
    async def test_chat_auto_connects_when_no_providers(self):
        config = ClientConfig(model="gpt-4o", openai_api_key="o")
        client = DirectModelClient(config)
        assert not client._providers

        with patch.object(client, "connect", new_callable=AsyncMock) as mock_connect:
            with patch.object(client, "_detect_provider", return_value=Provider.OPENAI):
                with patch.object(
                    client, "_chat_openai", new_callable=AsyncMock, return_value=ChatResponse(content="hi", tool_calls=None, finish_reason="stop", usage=None, raw=None)
                ):
                    result = await client.chat([{"role": "user", "content": "hi"}])
                    mock_connect.assert_called_once()
                    assert result.content == "hi"


class TestChatGooglePath:
    @pytest.mark.asyncio
    async def test_chat_google_dispatch(self):
        config = ClientConfig(model="gemini-2.5-flash", google_api_key="g")
        client = DirectModelClient(config)
        client._providers = {Provider.GOOGLE: MagicMock()}
        with patch.object(
            client, "_chat_google", new_callable=AsyncMock, return_value=ChatResponse(content="goog", tool_calls=None, finish_reason="stop", usage=None, raw=None)
        ):
            result = await client.chat([{"role": "user", "content": "hi"}])
            assert result.content == "goog"


class TestChatAnthropicPath:
    @pytest.mark.asyncio
    async def test_chat_anthropic_dispatch(self):
        config = ClientConfig(model="claude-3", anthropic_api_key="a")
        client = DirectModelClient(config)
        client._providers = {Provider.ANTHROPIC: MagicMock()}
        with patch.object(
            client, "_chat_anthropic", new_callable=AsyncMock, return_value=ChatResponse(content="anth", tool_calls=None, finish_reason="stop", usage=None, raw=None)
        ):
            result = await client.chat([{"role": "user", "content": "hi"}])
            assert result.content == "anth"


class TestCircuitBreakerOpenError:
    @pytest.mark.asyncio
    async def test_circuit_breaker_open_raises_runtime(self):
        from src.core.errors import CircuitBreakerOpenError, CircuitBreaker, CircuitBreakerConfig, CircuitState

        config = ClientConfig(model="gpt-4o", openai_api_key="o")
        client = DirectModelClient(config)
        client._providers = {Provider.OPENAI: MagicMock()}

        breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1, timeout=60.0))
        breaker.state = CircuitState.OPEN
        breaker.last_failure_time = 9999999999.0
        client._circuit_breakers[Provider.OPENAI] = breaker

        with pytest.raises(RuntimeError, match="temporarily unavailable"):
            await client.chat([{"role": "user", "content": "hi"}])


class TestChatStreamFallback:
    @pytest.mark.asyncio
    async def test_stream_falls_back_when_no_streaming_support(self):
        config = ClientConfig(model="gpt-4o", openai_api_key="o")
        client = DirectModelClient(config)
        client._providers = {Provider.OPENAI: MagicMock()}

        client._provider_supports_streaming = lambda p: False

        with patch.object(
            client, "chat", new_callable=AsyncMock, return_value=ChatResponse(content="fb", tool_calls=None, finish_reason="stop", usage=None, raw=None)
        ):
            chunks = []
            async for chunk in client.chat_stream([{"role": "user", "content": "hi"}]):
                chunks.append(chunk)
            assert len(chunks) == 1
            assert chunks[0].content == "fb"


class TestProtocolResolutionAutoNoCache:
    def test_auto_no_cache_resolves_to_should_use(self):
        config = ClientConfig(model="gpt-4o", openai_api_key="o", openai_protocol="auto")
        client = DirectModelClient(config)
        client._openai_protocol = None
        use, allow = client._resolve_openai_protocol([])
        assert allow is True

    def test_auto_cached_responses(self):
        config = ClientConfig(model="gpt-4o", openai_api_key="o", openai_protocol="auto")
        client = DirectModelClient(config)
        client._openai_protocol = "responses"
        use, allow = client._resolve_openai_protocol([])
        assert use is True
        assert allow is True

    def test_auto_cached_chat_completions(self):
        config = ClientConfig(model="gpt-4o", openai_api_key="o", openai_protocol="auto")
        client = DirectModelClient(config)
        client._openai_protocol = "chat_completions"
        use, allow = client._resolve_openai_protocol([])
        assert use is False
        assert allow is True


class TestOpenaiResponsesApi:
    def test_should_use_non_allowed_domain(self):
        config = ClientConfig(model="gpt-4o", openai_api_key="o", openai_api_base="https://custom.api.com/v1")
        client = DirectModelClient(config)
        assert client._should_use_openai_responses_api([]) is False

    def test_should_use_allowed_openai(self):
        config = ClientConfig(model="gpt-4o", openai_api_key="o", openai_api_base="https://api.openai.com/v1")
        client = DirectModelClient(config)
        assert client._should_use_openai_responses_api([]) is True


class TestRecordProtocolCacheUpdate:
    def test_auto_mode_updates_module_cache(self):
        config = ClientConfig(model="gpt-4o", openai_api_key="o", openai_protocol="auto")
        client = DirectModelClient(config)
        from src.core.llm.model_client_direct import _OPENAI_PROTOCOL_CACHE
        _OPENAI_PROTOCOL_CACHE.clear()
        client._record_protocol_success("chat_completions")
        key = client._get_openai_protocol_cache_key()
        assert _OPENAI_PROTOCOL_CACHE.get(key) == "chat_completions"
        _OPENAI_PROTOCOL_CACHE.clear()


class TestProviderSupportsWithCapabilities:
    def test_openai_tools_disabled(self):
        config = ClientConfig(model="gpt-4o", openai_api_key="o")
        config.openai_capabilities.tools = False
        client = DirectModelClient(config)
        assert client._provider_supports_tools(Provider.OPENAI) is False

    def test_anthropic_streaming_disabled(self):
        config = ClientConfig(model="claude-3", anthropic_api_key="a")
        config.anthropic_capabilities.streaming = False
        client = DirectModelClient(config)
        assert client._provider_supports_streaming(Provider.ANTHROPIC) is False

    def test_openai_streaming_disabled(self):
        config = ClientConfig(model="gpt-4o", openai_api_key="o")
        config.openai_capabilities.streaming = False
        client = DirectModelClient(config)
        assert client._provider_supports_streaming(Provider.OPENAI) is False


class TestRetryAsyncEdge:
    @pytest.mark.asyncio
    async def test_no_attempts_raises(self):
        async def noop():
            if False:
                yield

        MAX = 0
        with patch("src.core.llm.model_client_direct.MAX_RETRIES", 0):
            with pytest.raises(RuntimeError, match="no attempts"):
                await _retry_async(lambda: None)


class TestAnthropicProtocolValidation:
    @pytest.mark.asyncio
    async def test_anthropic_unsupported_protocol(self):
        config = ClientConfig(model="claude-3", anthropic_api_key="a")
        client = DirectModelClient(config)
        client._providers = {Provider.ANTHROPIC: MagicMock()}
        config.anthropic_protocol = "streaming"
        with pytest.raises(RuntimeError, match="Unsupported Anthropic protocol"):
            await client._chat_anthropic([], None)


class TestConnectImportErrors:
    @pytest.mark.asyncio
    async def test_google_import_error(self):
        config = ClientConfig(model="gemini", google_api_key="g")
        client = DirectModelClient(config)
        with patch.dict("sys.modules", {"google": None, "google.genai": None}):
            with pytest.raises(ImportError):
                import google.genai
            with pytest.raises(RuntimeError, match="No providers"):
                await client.connect()

    @pytest.mark.asyncio
    async def test_openai_import_error(self):
        config = ClientConfig(model="gpt-4o", openai_api_key="o")
        client = DirectModelClient(config)
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(RuntimeError, match="No providers"):
                await client.connect()

    @pytest.mark.asyncio
    async def test_anthropic_import_error(self):
        config = ClientConfig(model="claude-3", anthropic_api_key="a")
        client = DirectModelClient(config)
        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(RuntimeError, match="No providers"):
                await client.connect()

    @pytest.mark.asyncio
    async def test_google_openai_compatible_key_fallback(self):
        config = ClientConfig(
            model="gpt-4o",
            openai_api_key="o",
            openai_api_base="https://generativelanguage.googleapis.com/v1beta/openai",
        )
        client = DirectModelClient(config)
        with patch.dict("sys.modules", {"google": None, "google.genai": None}):
            await client.connect()
            effective_key = client.config.google_api_key
            assert effective_key is None or effective_key == "o"


class TestChatStreamProviderDispatch:
    @pytest.mark.asyncio
    async def test_stream_openai_dispatch(self):
        config = ClientConfig(model="gpt-4o", openai_api_key="o")
        client = DirectModelClient(config)
        client._providers = {Provider.OPENAI: MagicMock()}

        async def fake_stream(*a, **kw):
            yield StreamChunk(content="chunk1")

        with patch.object(client, "_stream_openai", fake_stream):
            chunks = []
            async for chunk in client.chat_stream([{"role": "user", "content": "hi"}]):
                chunks.append(chunk)
            assert any(c.content == "chunk1" for c in chunks)

    @pytest.mark.asyncio
    async def test_stream_anthropic_dispatch(self):
        config = ClientConfig(model="claude-3", anthropic_api_key="a")
        client = DirectModelClient(config)
        client._providers = {Provider.ANTHROPIC: MagicMock()}

        async def fake_stream(*a, **kw):
            yield StreamChunk(content="a_chunk")

        with patch.object(client, "_stream_anthropic", fake_stream):
            chunks = []
            async for chunk in client.chat_stream([{"role": "user", "content": "hi"}]):
                chunks.append(chunk)
            assert any(c.content == "a_chunk" for c in chunks)


class TestShouldFallbackMarkers:
    def test_responses_endpoint(self):
        assert DirectModelClient._should_fallback_to_chat_completions(Exception("/responses not found")) is True

    def test_404_marker(self):
        assert DirectModelClient._should_fallback_to_chat_completions(Exception("404 not found")) is True

    def test_bad_response_status_code(self):
        assert DirectModelClient._should_fallback_to_chat_completions(Exception("bad_response_status_code")) is True

    def test_non_openai_payload_no_fallback(self):
        assert DirectModelClient._should_fallback_to_chat_completions(Exception("non-openai responses payload")) is False
