"""
Base Model Client

Provides the abstract base class for all model clients.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence

from src.core.llm.model_utils import (
    ChatResponse,
    Message,
    StreamChunk,
    ToolDefinition,
)


class BaseModelClient(ABC):
    """Base class for model clients."""

    @abstractmethod
    async def chat(
        self,
        messages: Sequence[Message | dict],
        tools: Sequence[ToolDefinition | dict] | None = None,
        **kwargs,
    ) -> ChatResponse:
        """Send a chat request."""
        pass

    @abstractmethod
    def chat_stream(
        self,
        messages: Sequence[Message | dict],
        tools: Sequence[ToolDefinition | dict] | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Send a streaming chat request."""
        pass

    @abstractmethod
    async def list_models(self) -> list[dict]:
        """List available models."""
        pass

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the model provider/gateway."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the client."""
        pass
