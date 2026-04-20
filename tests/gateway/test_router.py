"""Tests for gateway.router — ModelRouter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.gateway.adapters.base import AdapterConfig, BaseAdapter
from src.gateway.schema import ChatMessage, ChatRequest, ChatResponse, Choice, ModelInfo, Role
from src.gateway.router import ModelRouter


def _make_mock_adapter(provider: str, models: list[ModelInfo] | None = None) -> BaseAdapter:
    adapter = AsyncMock(spec=BaseAdapter)
    adapter.provider_name = provider
    if models is None:
        models = [
            ModelInfo(
                id=f"{provider}-model",
                name=f"{provider.title()} Model",
                provider=provider,
                aliases=[f"{provider}-alias"],
            )
        ]
    adapter.get_models.return_value = models
    adapter.health_check = AsyncMock(return_value=True)
    adapter.close = AsyncMock()
    return adapter


class TestModelRouter:
    def _make_router(self, config=None):
        if config is None:
            config = {}
        return ModelRouter(config)

    def test_init(self):
        router = self._make_router()
        assert router._adapters == {}
        assert router._model_cache == {}

    def test_get_providers_empty(self):
        router = self._make_router()
        assert router.get_providers() == []

    @pytest.mark.asyncio
    async def test_initialize_skips_disabled(self):
        config = {"openai": {"enabled": False}, "anthropic": {"enabled": True, "api_key": "k"}}
        router = ModelRouter(config)
        with patch.object(router, "_adapters", {}):
            with patch("src.gateway.router.OpenAIAdapter") as mock_cls:
                await router.initialize()
                mock_cls.assert_not_called()

    def test_get_adapter_by_cache(self):
        router = self._make_router()
        adapter = _make_mock_adapter("openai")
        router._adapters = {"openai": adapter}
        router._model_cache = {"gpt-4o": "openai"}
        assert router._get_adapter("gpt-4o") == adapter

    def test_get_adapter_by_pattern(self):
        router = self._make_router()
        adapter = _make_mock_adapter("openai")
        router._adapters = {"openai": adapter}
        assert router._get_adapter("gpt-4o-turbo") == adapter

    def test_get_adapter_preferred_provider(self):
        router = self._make_router()
        adapter = _make_mock_adapter("openai")
        router._adapters = {"openai": adapter}
        assert router._get_adapter("anything", preferred_provider="openai") == adapter

    def test_get_adapter_not_found(self):
        router = self._make_router()
        assert router._get_adapter("unknown-model") is None

    @pytest.mark.asyncio
    async def test_chat_model_not_found(self):
        router = self._make_router()
        req = ChatRequest(model="nonexistent", messages=[ChatMessage(role=Role.USER, content="hi")])
        result = await router.chat(req)
        assert hasattr(result, "error")
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_chat_success(self):
        router = self._make_router()
        adapter = _make_mock_adapter("openai")
        mock_resp = ChatResponse(id="r1", model="gpt-4o", choices=[])
        adapter.chat = AsyncMock(return_value=mock_resp)
        router._adapters = {"openai": adapter}
        router._model_cache = {"gpt-4o": "openai"}

        req = ChatRequest(model="gpt-4o", messages=[ChatMessage(role=Role.USER, content="hi")])
        result = await router.chat(req)
        assert result.id == "r1"

    @pytest.mark.asyncio
    async def test_chat_exception(self):
        router = self._make_router()
        adapter = _make_mock_adapter("openai")
        adapter.chat = AsyncMock(side_effect=RuntimeError("API down"))
        router._adapters = {"openai": adapter}
        router._model_cache = {"gpt-4o": "openai"}

        req = ChatRequest(model="gpt-4o", messages=[ChatMessage(role=Role.USER, content="hi")])
        result = await router.chat(req)
        assert hasattr(result, "error")
        assert "API down" in result.error

    @pytest.mark.asyncio
    async def test_chat_stream_model_not_found(self):
        router = self._make_router()
        req = ChatRequest(model="nope", messages=[ChatMessage(role=Role.USER, content="hi")])
        chunks = []
        async for chunk in router.chat_stream(req):
            chunks.append(chunk)
        assert len(chunks) == 1
        assert hasattr(chunks[0], "error")

    @pytest.mark.asyncio
    async def test_chat_stream_success(self):
        from src.gateway.schema import StreamChunk

        router = self._make_router()
        adapter = _make_mock_adapter("openai")

        async def mock_stream(request):
            yield StreamChunk(id="c1", model="gpt-4o", delta_content="hi")
            yield StreamChunk(id="c1", model="gpt-4o", finish_reason="stop")

        adapter.chat_stream = mock_stream
        router._adapters = {"openai": adapter}
        router._model_cache = {"gpt-4o": "openai"}

        req = ChatRequest(model="gpt-4o", messages=[ChatMessage(role=Role.USER, content="hi")])
        chunks = []
        async for chunk in router.chat_stream(req):
            chunks.append(chunk)
        assert len(chunks) == 2
        assert chunks[0].delta_content == "hi"

    @pytest.mark.asyncio
    async def test_chat_stream_exception(self):
        router = self._make_router()
        adapter = _make_mock_adapter("openai")

        async def failing_stream(request):
            yield StreamChunk(id="c1", model="gpt-4o", delta_content="hi")
            raise RuntimeError("stream broke")

        adapter.chat_stream = failing_stream
        router._adapters = {"openai": adapter}
        router._model_cache = {"gpt-4o": "openai"}

        req = ChatRequest(model="gpt-4o", messages=[ChatMessage(role=Role.USER, content="hi")])
        chunks = []
        async for chunk in router.chat_stream(req):
            chunks.append(chunk)
        assert len(chunks) >= 1
        assert hasattr(chunks[-1], "error")

    def test_list_models(self):
        router = self._make_router()
        adapter = _make_mock_adapter(
            "openai", [ModelInfo(id="gpt-4o", name="GPT-4o", provider="openai")]
        )
        adapter2 = _make_mock_adapter(
            "anthropic", [ModelInfo(id="claude", name="Claude", provider="anthropic")]
        )
        router._adapters = {"openai": adapter, "anthropic": adapter2}

        models = router.list_models()
        assert len(models) == 2

    def test_list_models_filter_provider(self):
        router = self._make_router()
        adapter = _make_mock_adapter(
            "openai", [ModelInfo(id="gpt-4o", name="GPT-4o", provider="openai")]
        )
        adapter2 = _make_mock_adapter(
            "anthropic", [ModelInfo(id="claude", name="Claude", provider="anthropic")]
        )
        router._adapters = {"openai": adapter, "anthropic": adapter2}

        models = router.list_models(provider="openai")
        assert len(models) == 1
        assert models[0].id == "gpt-4o"

    def test_get_model_info(self):
        router = self._make_router()
        model = ModelInfo(id="gpt-4o", name="GPT-4o", provider="openai", aliases=["4o"])
        adapter = _make_mock_adapter("openai", [model])
        router._adapters = {"openai": adapter}
        router._model_cache = {"gpt-4o": "openai", "4o": "openai"}

        info = router.get_model_info("gpt-4o")
        assert info is not None
        assert info.id == "gpt-4o"

        info_alias = router.get_model_info("4o")
        assert info_alias is not None

    def test_get_model_info_not_found(self):
        router = self._make_router()
        assert router.get_model_info("nope") is None

    @pytest.mark.asyncio
    async def test_health_check(self):
        router = self._make_router()
        adapter = _make_mock_adapter("openai")
        adapter2 = _make_mock_adapter("anthropic")
        adapter.health_check = AsyncMock(return_value=True)
        adapter2.health_check = AsyncMock(return_value=False)
        router._adapters = {"openai": adapter, "anthropic": adapter2}

        result = await router.health_check()
        assert result["openai"] is True
        assert result["anthropic"] is False

    @pytest.mark.asyncio
    async def test_close(self):
        router = self._make_router()
        adapter = _make_mock_adapter("openai")
        adapter.close = AsyncMock()
        router._adapters = {"openai": adapter}
        router._model_cache = {"gpt-4o": "openai"}

        await router.close()
        assert router._adapters == {}
        assert router._model_cache == {}

    @pytest.mark.asyncio
    async def test_close_handles_exception(self):
        router = self._make_router()
        adapter = _make_mock_adapter("openai")
        adapter.close = AsyncMock(side_effect=RuntimeError("fail"))
        router._adapters = {"openai": adapter}

        await router.close()
        assert router._adapters == {}
