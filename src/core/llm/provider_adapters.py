"""
Provider Adapters (Compatibility Shim)

All provider adapter classes have been moved to the
``src.core.llm.providers`` package.  This module re-exports every
public name so that existing imports continue to work unchanged.
"""

from src.core.llm.providers import *  # noqa: F403
from src.core.llm.providers import (  # noqa: F401
    AnthropicAdapter,
    GoogleAdapter,
    OpenAIAdapter,
    ProviderAdapter,
    _deserialize_gemini_thought_signature,
    _serialize_gemini_thought_signature,
    build_anthropic_content_parts,
    build_google_content_parts,
    build_openai_content_parts,
)

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
