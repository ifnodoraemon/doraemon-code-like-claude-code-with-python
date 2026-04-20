"""Model Router - Routes requests to appropriate provider adapter."""

import logging
from collections.abc import AsyncIterator
from typing import Any

from .adapters.anthropic_adapter import AnthropicAdapter
from .adapters.base import AdapterConfig, BaseAdapter
from .adapters.google_adapter import GoogleAdapter
from .adapters.openai_adapter import OpenAIAdapter
from .schema import ChatRequest, ChatResponse, ErrorResponse, ModelInfo, StreamChunk

logger = logging.getLogger(__name__)


class ModelRouter:
    """Routes chat requests to the appropriate provider adapter."""

    PROVIDER_PATTERNS = {
        "google": ["gemini-", "palm-"],
        "openai": ["gpt-", "o1", "o3"],
        "anthropic": ["claude-"],
    }

    def __init__(self, config: dict[str, Any]):
        self._config = config
        self._adapters: dict[str, BaseAdapter] = {}
        self._model_cache: dict[str, str] = {}

    async def initialize(self) -> None:
        """Initialize all enabled adapters."""
        adapter_classes = {
            "google": GoogleAdapter,
            "openai": OpenAIAdapter,
            "anthropic": AnthropicAdapter,
        }

        for provider, adapter_class in adapter_classes.items():
            provider_config = self._config.get(provider, {})
            if not provider_config.get("enabled", False):
                continue

            try:
                adapter_config = AdapterConfig(
                    api_key=provider_config.get("api_key"),
                    api_base=provider_config.get("api_base"),
                    timeout=provider_config.get("timeout", 60.0),
                    max_retries=provider_config.get("max_retries", 3),
                )
                adapter = adapter_class(adapter_config)
                await adapter.initialize()
                self._adapters[provider] = adapter

                for model in adapter.get_models():
                    self._model_cache[model.id] = provider
                    for alias in model.aliases:
                        self._model_cache[alias] = provider
            except Exception as e:
                logger.error("Failed to initialize %s: %s", provider, type(e).__name__)

    def _get_adapter(
        self, model_id: str, preferred_provider: str | None = None
    ) -> BaseAdapter | None:
        if preferred_provider and preferred_provider in self._adapters:
            return self._adapters[preferred_provider]

        provider = self._model_cache.get(model_id)
        if not provider:
            for p, patterns in self.PROVIDER_PATTERNS.items():
                if p in self._adapters:
                    for pat in patterns:
                        if model_id.startswith(pat):
                            provider = p
                            break
        return self._adapters.get(provider) if provider else None

    async def chat(
        self,
        request: ChatRequest,
        preferred_provider: str | None = None,
    ) -> ChatResponse | ErrorResponse:
        adapter = self._get_adapter(request.model, preferred_provider=preferred_provider)
        if not adapter:
            return ErrorResponse(error=f"Model not found: {request.model}", code="model_not_found")
        try:
            return await adapter.chat(request)
        except Exception as e:
            return ErrorResponse(error=str(e), code="provider_error")

    async def chat_stream(
        self,
        request: ChatRequest,
        preferred_provider: str | None = None,
    ) -> AsyncIterator[StreamChunk | ErrorResponse]:
        adapter = self._get_adapter(request.model, preferred_provider=preferred_provider)
        if not adapter:
            yield ErrorResponse(error=f"Model not found: {request.model}", code="model_not_found")
            return
        try:
            async for chunk in adapter.chat_stream(request):
                yield chunk
        except Exception as e:
            logger.error("Stream error for %s: %s", request.model, type(e).__name__)
            yield ErrorResponse(error=str(e), code="provider_error")

    def list_models(self, provider: str | None = None) -> list[ModelInfo]:
        models = []
        for p, adapter in self._adapters.items():
            if provider and p != provider:
                continue
            models.extend(adapter.get_models())
        return models

    def get_model_info(self, model_id: str) -> ModelInfo | None:
        adapter = self._get_adapter(model_id)
        if adapter:
            for m in adapter.get_models():
                if m.id == model_id or model_id in m.aliases:
                    return m
        return None

    async def health_check(self) -> dict[str, bool]:
        import asyncio

        results = await asyncio.gather(
            *[a.health_check() for a in self._adapters.values()],
            return_exceptions=True,
        )
        return {
            p: (r if isinstance(r, bool) else False)
            for p, r in zip(self._adapters, results, strict=False)
        }

    def get_providers(self) -> list[str]:
        return list(self._adapters.keys())

    async def close(self) -> None:
        """Close all adapters and release resources."""
        for provider, adapter in self._adapters.items():
            try:
                await adapter.close()
            except Exception as e:
                logger.error("Error closing %s adapter: %s", provider, type(e).__name__)
        self._adapters.clear()
        self._model_cache.clear()
