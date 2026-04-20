"""Tests for core.llm.model_client_direct."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.llm.model_client_direct import (
    _is_google_openai_compatible_base,
    _is_retryable,
    _retry_async,
    DirectModelClient,
)
from src.core.llm.model_utils import ClientConfig, Provider


class TestIsGoogleOpenaiCompatibleBase:
    def test_none(self):
        assert _is_google_openai_compatible_base(None) is False

    def test_empty(self):
        assert _is_google_openai_compatible_base("") is False

    def test_google_openai_compatible(self):
        url = "https://generativelanguage.googleapis.com/v1beta/openai"
        assert _is_google_openai_compatible_base(url) is True

    def test_google_non_openai(self):
        url = "https://generativelanguage.googleapis.com/v1beta"
        assert _is_google_openai_compatible_base(url) is False

    def test_other_url(self):
        assert _is_google_openai_compatible_base("https://api.openai.com") is False

    def test_trailing_slash(self):
        url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        assert _is_google_openai_compatible_base(url) is True


class TestIsRetryable:
    def test_rate_limit(self):
        assert _is_retryable(Exception("rate limit exceeded")) is True

    def test_429(self):
        assert _is_retryable(Exception("429 too many requests")) is True

    def test_server_error_500(self):
        assert _is_retryable(Exception("500 internal server error")) is True

    def test_server_error_503(self):
        assert _is_retryable(Exception("503 service unavailable")) is True

    def test_timeout(self):
        assert _is_retryable(Exception("connection timed out")) is True

    def test_connection_reset(self):
        assert _is_retryable(Exception("connection reset by peer")) is True

    def test_resource_exhausted(self):
        assert _is_retryable(Exception("resource exhausted")) is True

    def test_non_retryable(self):
        assert _is_retryable(Exception("invalid API key")) is False

    def test_auth_error(self):
        assert _is_retryable(Exception("401 unauthorized")) is False


class TestRetryAsync:
    @pytest.mark.asyncio
    async def test_success_first_try(self):
        call_count = 0

        async def coro():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await _retry_async(coro)
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_non_retryable_raises(self):
        async def coro():
            raise ValueError("bad")

        with pytest.raises(ValueError, match="bad"):
            await _retry_async(coro)


class TestDirectModelClientInit:
    def test_init(self):
        config = ClientConfig(
            model="gpt-4o",
            google_api_key="g",
            openai_api_key="o",
            anthropic_api_key="a",
        )
        client = DirectModelClient(config)
        assert client.model == "gpt-4o"
        assert Provider.GOOGLE in client._circuit_breakers
        assert Provider.OPENAI in client._circuit_breakers
        assert Provider.ANTHROPIC in client._circuit_breakers

    def test_detect_provider_google(self):
        config = ClientConfig(model="gemini-2.5-flash", google_api_key="g", openai_api_key="o")
        client = DirectModelClient(config)
        client._providers = {Provider.GOOGLE: MagicMock(), Provider.OPENAI: MagicMock()}
        assert client._detect_provider("gemini-2.5-flash") == Provider.GOOGLE

    def test_detect_provider_openai(self):
        config = ClientConfig(model="gpt-4o", openai_api_key="o")
        client = DirectModelClient(config)
        client._providers = {Provider.OPENAI: MagicMock()}
        assert client._detect_provider("gpt-4o") == Provider.OPENAI

    def test_detect_provider_anthropic(self):
        config = ClientConfig(model="claude-3", anthropic_api_key="a")
        client = DirectModelClient(config)
        client._providers = {Provider.ANTHROPIC: MagicMock()}
        assert client._detect_provider("claude-3") == Provider.ANTHROPIC

    def test_detect_provider_fallback(self):
        config = ClientConfig(model="unknown-model", openai_api_key="o")
        client = DirectModelClient(config)
        client._providers = {Provider.OPENAI: MagicMock()}
        assert client._detect_provider("unknown-model") == Provider.OPENAI

    def test_detect_provider_no_providers(self):
        config = ClientConfig(model="gpt-4o")
        client = DirectModelClient(config)
        with pytest.raises(RuntimeError, match="No providers"):
            client._detect_provider("gpt-4o")

    def test_should_fallback_to_chat_completions(self):
        config = ClientConfig(model="gpt-4o")
        client = DirectModelClient(config)
        assert (
            DirectModelClient._should_fallback_to_chat_completions(Exception("404 not found"))
            is True
        )
        assert (
            DirectModelClient._should_fallback_to_chat_completions(
                Exception("bad_response_status_code")
            )
            is True
        )
        assert (
            DirectModelClient._should_fallback_to_chat_completions(Exception("invalid key"))
            is False
        )
        assert (
            DirectModelClient._should_fallback_to_chat_completions(
                Exception("non-openai responses payload")
            )
            is False
        )

    def test_resolve_openai_protocol_responses(self):
        config = ClientConfig(model="gpt-4o", openai_protocol="responses")
        client = DirectModelClient(config)
        use, allow = client._resolve_openai_protocol([])
        assert use is True
        assert allow is False

    def test_resolve_openai_protocol_chat_completions(self):
        config = ClientConfig(model="gpt-4o", openai_protocol="chat_completions")
        client = DirectModelClient(config)
        use, allow = client._resolve_openai_protocol([])
        assert use is False
        assert allow is False

    def test_provider_supports_tools(self):
        config = ClientConfig(model="gpt-4o", openai_api_key="o")
        client = DirectModelClient(config)
        assert client._provider_supports_tools(Provider.OPENAI) is True
        assert client._provider_supports_tools(Provider.GOOGLE) is True

    def test_provider_supports_streaming(self):
        config = ClientConfig(model="gpt-4o", openai_api_key="o")
        client = DirectModelClient(config)
        assert client._provider_supports_streaming(Provider.OPENAI) is True

    @pytest.mark.asyncio
    async def test_list_models(self):
        config = ClientConfig(
            model="gpt-4o", openai_api_key="o", google_api_key="g", anthropic_api_key="a"
        )
        client = DirectModelClient(config)
        client._providers = {
            Provider.OPENAI: MagicMock(),
            Provider.GOOGLE: MagicMock(),
            Provider.ANTHROPIC: MagicMock(),
        }
        models = await client.list_models()
        ids = [m["id"] for m in models]
        assert "gpt-4o" in ids
        assert "gemini-2.5-flash-preview" in ids


class TestDetectProviderAdvanced:
    def test_detect_provider_matched_but_not_configured(self):
        config = ClientConfig(model="gemini-flash", openai_api_key="o")
        client = DirectModelClient(config)
        client._providers = {Provider.OPENAI: MagicMock()}
        result = client._detect_provider("gemini-flash")
        assert result == Provider.OPENAI

    def test_detect_provider_o1(self):
        config = ClientConfig(model="o1", openai_api_key="o")
        client = DirectModelClient(config)
        client._providers = {Provider.OPENAI: MagicMock()}
        assert client._detect_provider("o1") == Provider.OPENAI

    def test_detect_provider_o3(self):
        config = ClientConfig(model="o3-mini", openai_api_key="o")
        client = DirectModelClient(config)
        client._providers = {Provider.OPENAI: MagicMock()}
        assert client._detect_provider("o3-mini") == Provider.OPENAI

    def test_detect_provider_palm(self):
        config = ClientConfig(model="palm-2", google_api_key="g")
        client = DirectModelClient(config)
        client._providers = {Provider.GOOGLE: MagicMock()}
        assert client._detect_provider("palm-2") == Provider.GOOGLE


class TestRetryAsyncAdvanced:
    @pytest.mark.asyncio
    async def test_retryable_succeeds_after_retry(self, monkeypatch):
        call_count = 0

        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("429 rate limit")
            return "ok"

        monkeypatch.setattr("src.core.llm.model_client_direct.INITIAL_DELAY", 0.01)
        monkeypatch.setattr("src.core.llm.model_client_direct.MAX_DELAY", 0.01)
        result = await _retry_async(flaky)
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retryable_exhausted(self, monkeypatch):
        async def always_fail():
            raise Exception("429 rate limit")

        monkeypatch.setattr("src.core.llm.model_client_direct.INITIAL_DELAY", 0.01)
        monkeypatch.setattr("src.core.llm.model_client_direct.MAX_DELAY", 0.01)
        with pytest.raises(Exception, match="429"):
            await _retry_async(always_fail)


class TestIsGoogleOpenaiCompatibleBaseAdvanced:
    def test_case_insensitive(self):
        url = "https://GENERATIVELANGUAGE.GOOGLEAPIS.COM/v1beta/OpenAI"
        assert _is_google_openai_compatible_base(url) is True

    def test_partial_match_not_openai(self):
        url = "https://generativelanguage.googleapis.com/v1beta/chat"
        assert _is_google_openai_compatible_base(url) is False


class TestProviderSupportsToolsAndStreaming:
    def test_google_always_supports_tools(self):
        config = ClientConfig(model="gemini")
        client = DirectModelClient(config)
        assert client._provider_supports_tools(Provider.GOOGLE) is True

    def test_google_always_supports_streaming(self):
        config = ClientConfig(model="gemini")
        client = DirectModelClient(config)
        assert client._provider_supports_streaming(Provider.GOOGLE) is True

    def test_anthropic_default_supports_tools(self):
        config = ClientConfig(model="claude", anthropic_api_key="a")
        client = DirectModelClient(config)
        assert client._provider_supports_tools(Provider.ANTHROPIC) is True

    def test_anthropic_default_supports_streaming(self):
        config = ClientConfig(model="claude", anthropic_api_key="a")
        client = DirectModelClient(config)
        assert client._provider_supports_streaming(Provider.ANTHROPIC) is True


class TestResolveOpenaiProtocolAuto:
    def test_auto_mode_with_cache(self):
        config = ClientConfig(model="gpt-4o", openai_api_key="o")
        client = DirectModelClient(config)
        client._openai_protocol = "responses"
        use, allow = client._resolve_openai_protocol([])
        assert use is True
        assert allow is True

    def test_auto_mode_no_cache(self, monkeypatch):
        config = ClientConfig(model="gpt-4o", openai_api_key="o", openai_api_base="https://api.openai.com/v1")
        client = DirectModelClient(config)
        client._openai_protocol = None
        use, allow = client._resolve_openai_protocol([])
        assert allow is True


class TestShouldUseOpenaiResponsesApi:
    def test_non_allowed_domain(self):
        config = ClientConfig(model="gpt-4o", openai_api_key="o", openai_api_base="https://custom.api.com/v1")
        client = DirectModelClient(config)
        assert client._should_use_openai_responses_api([]) is False

    def test_allowed_domain_openai(self):
        config = ClientConfig(model="gpt-4o", openai_api_key="o", openai_api_base="https://api.openai.com/v1")
        client = DirectModelClient(config)
        assert client._should_use_openai_responses_api([]) is True

    def test_allowed_domain_azure(self):
        config = ClientConfig(model="gpt-4o", openai_api_key="o", openai_api_base="https://my.openai.azure.com/v1")
        client = DirectModelClient(config)
        assert client._should_use_openai_responses_api([]) is True

    def test_no_base_url(self):
        config = ClientConfig(model="gpt-4o", openai_api_key="o")
        client = DirectModelClient(config)
        assert client._should_use_openai_responses_api([]) is True


class TestRecordProtocolSuccess:
    def test_records_to_instance(self):
        config = ClientConfig(model="gpt-4o", openai_api_key="o")
        client = DirectModelClient(config)
        client._record_protocol_success("responses")
        assert client._openai_protocol == "responses"

    def test_auto_mode_caches(self, monkeypatch):
        config = ClientConfig(model="gpt-4o", openai_api_key="o", openai_protocol="auto")
        client = DirectModelClient(config)
        client._record_protocol_success("chat_completions")
        assert client._openai_protocol == "chat_completions"


class TestPrepareToolsForProvider:
    def test_tools_disabled(self):
        config = ClientConfig(model="gpt-4o", openai_api_key="o")
        client = DirectModelClient(config)
        client._provider_supports_tools = MagicMock(return_value=False)
        result = client._prepare_tools_for_provider(Provider.OPENAI, [{"name": "fn"}])
        assert result is None

    def test_tools_enabled(self):
        config = ClientConfig(model="gpt-4o", openai_api_key="o")
        client = DirectModelClient(config)
        client._provider_supports_tools = MagicMock(return_value=True)
        result = client._prepare_tools_for_provider(Provider.OPENAI, [{"name": "fn"}])
        assert result is not None


class TestContextManager:
    @pytest.mark.asyncio
    async def test_aenter_calls_connect(self):
        config = ClientConfig(model="gpt-4o", openai_api_key="o")
        client = DirectModelClient(config)
        with patch.object(client, "connect", new_callable=AsyncMock) as mock_connect:
            with patch.object(client, "close", new_callable=AsyncMock):
                result = await client.__aenter__()
                mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_aexit_calls_close(self):
        config = ClientConfig(model="gpt-4o", openai_api_key="o")
        client = DirectModelClient(config)
        with patch.object(client, "close", new_callable=AsyncMock) as mock_close:
            result = await client.__aexit__(None, None, None)
            mock_close.assert_called_once()
            assert result is False


class TestClose:
    @pytest.mark.asyncio
    async def test_close_clears_providers_and_resets_breakers(self):
        config = ClientConfig(model="gpt-4o", openai_api_key="o")
        client = DirectModelClient(config)
        mock_provider = MagicMock()
        mock_provider.aclose = AsyncMock()
        client._providers = {Provider.OPENAI: mock_provider}

        await client.close()
        assert len(client._providers) == 0
        for breaker in client._circuit_breakers.values():
            assert breaker.state.value == "closed"


class TestIsRetryableAdvanced:
    def test_502(self):
        assert _is_retryable(Exception("502 bad gateway")) is True

    def test_504(self):
        assert _is_retryable(Exception("504 gateway timeout")) is True

    def test_server_error_string(self):
        assert _is_retryable(Exception("server error")) is True

    def test_internal_error_string(self):
        assert _is_retryable(Exception("internal error")) is True

    def test_connection_refused(self):
        assert _is_retryable(Exception("connection refused")) is True

    def test_connection_error(self):
        assert _is_retryable(Exception("connection error")) is True

    def test_timed_out(self):
        assert _is_retryable(Exception("request timed out")) is True

    def test_non_retryable_400(self):
        assert _is_retryable(Exception("400 bad request")) is False

    def test_non_retryable_403(self):
        assert _is_retryable(Exception("403 forbidden")) is False
