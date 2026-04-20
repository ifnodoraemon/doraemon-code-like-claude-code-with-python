"""
Provider Adapter Base Interface

Defines the typed interface that all LLM provider adapters must implement.
Each adapter converts between the unified message format and a
provider-specific API format.
"""

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from src.core.llm.model_utils import ToolDefinition


@runtime_checkable
class ProviderAdapter(Protocol):
    """
    Typed interface for LLM provider adapters.

    Every adapter must be able to convert unified messages into its
    provider-native format and build the corresponding request parameters.
    Streaming and response parsing are provider-specific and exposed as
    optional extensions beyond this core interface.
    """

    @staticmethod
    def convert_messages(
        messages: Sequence[Any],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Convert unified messages into provider-native format.

        Returns:
            Provider-specific message representation.  The concrete return
            type varies by provider (e.g. tuple for Google/Anthropic,
            list for OpenAI).
        """
        ...

    @staticmethod
    def build_params(
        model: str,
        msg_list: Any,
        tools: Sequence[ToolDefinition | dict] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build provider-specific request parameters.

        Returns:
            Dict of keyword arguments suitable for the provider SDK call.
        """
        ...
