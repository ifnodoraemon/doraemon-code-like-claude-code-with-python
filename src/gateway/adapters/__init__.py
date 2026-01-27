"""
Provider Adapters

Each adapter translates the unified API format to provider-specific format.
"""

from .anthropic_adapter import AnthropicAdapter
from .base import AdapterConfig, BaseAdapter
from .google_adapter import GoogleAdapter
from .ollama_adapter import OllamaAdapter
from .openai_adapter import OpenAIAdapter

__all__ = [
    "BaseAdapter",
    "AdapterConfig",
    "GoogleAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "OllamaAdapter",
]
