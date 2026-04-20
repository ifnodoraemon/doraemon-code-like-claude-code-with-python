"""
Provider Adapters Package

Provides distinct adapter modules for each LLM provider behind a
typed ProviderAdapter interface.  All public names are re-exported
here so that existing ``from src.core.llm.provider_adapters import …``
paths continue to work via the compatibility shim.
"""

from src.core.llm.providers.anthropic import AnthropicAdapter, build_anthropic_content_parts
from src.core.llm.providers.base import ProviderAdapter
from src.core.llm.providers.google import (
    GoogleAdapter,
    _deserialize_gemini_thought_signature,
    _serialize_gemini_thought_signature,
    build_google_content_parts,
)
from src.core.llm.providers.openai import OpenAIAdapter, build_openai_content_parts

__all__ = [
    "ProviderAdapter",
    "GoogleAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "build_google_content_parts",
    "build_openai_content_parts",
    "build_anthropic_content_parts",
    "_serialize_gemini_thought_signature",
    "_deserialize_gemini_thought_signature",
]
