"""
Base Adapter Interface

All provider adapters must implement this interface.
"""

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from ..schema import (
    ChatRequest,
    ChatResponse,
    ModelInfo,
    StreamChunk,
)

logger = logging.getLogger(__name__)


@dataclass
class AdapterConfig:
    """Configuration for an adapter."""

    api_key: str | None = None
    api_base: str | None = None
    timeout: float = 60.0
    max_retries: int = 3
    extra: dict[str, Any] = field(default_factory=dict)


class BaseAdapter(ABC):
    """
    Base class for all provider adapters.

    Each adapter is responsible for:
    1. Translating unified ChatRequest to provider-specific format
    2. Making the API call
    3. Translating provider response to unified ChatResponse
    4. Handling streaming responses
    """

    provider_name: str = "base"

    def __init__(self, config: AdapterConfig):
        self.config = config
        self._client: Any = None
        self._health_cache: bool | None = None
        self._health_cache_time: float = 0.0
        self._health_cache_ttl: float = 30.0

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider client."""
        pass

    @abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Send a chat request and get a response.

        Args:
            request: Unified chat request

        Returns:
            Unified chat response
        """
        pass

    @abstractmethod
    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """
        Send a chat request and stream the response.

        Args:
            request: Unified chat request

        Yields:
            Stream chunks
        """
        if False:
            yield StreamChunk(id="", model="")
        pass

    @abstractmethod
    def get_models(self) -> list[ModelInfo]:
        """
        Get list of available models for this provider.

        Returns:
            List of model info
        """
        pass

    def supports_model(self, model_id: str) -> bool:
        """Check if this adapter supports a model."""
        models = self.get_models()
        for model in models:
            if model.id == model_id or model_id in model.aliases:
                return True
        return False

    def resolve_model(self, model_id: str) -> str | None:
        """Resolve model ID or alias to actual model ID."""
        models = self.get_models()
        for model in models:
            if model.id == model_id:
                return model.id
            if model_id in model.aliases:
                return model.id
        return None

    async def health_check(self) -> bool:
        """Check if the adapter is healthy (cached for 30 seconds)."""
        now = time.monotonic()
        if (
            self._health_cache is not None
            and (now - self._health_cache_time) < self._health_cache_ttl
        ):
            return self._health_cache
        try:
            models = self.get_models()
            self._health_cache = len(models) > 0
        except Exception as e:
            logger.error("Health check failed for %s: %s", self.provider_name, type(e).__name__)
            self._health_cache = False
        self._health_cache_time = now
        return self._health_cache

    async def close(self) -> None:
        """Clean up adapter resources. Override in subclasses if needed."""
        self._client = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(provider={self.provider_name})"
