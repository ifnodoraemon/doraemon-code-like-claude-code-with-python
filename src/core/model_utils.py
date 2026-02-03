"""
Model Client Utilities

Provides utility functions and data classes for the unified model client.
Includes message conversion, tool definitions, and response handling.
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ClientMode(Enum):
    """Client connection mode."""
    GATEWAY = "gateway"
    DIRECT = "direct"


class Provider(Enum):
    """Supported providers for direct mode."""
    GOOGLE = "google"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


@dataclass
class Message:
    """Unified message format."""
    role: str
    content: str | None = None
    thought: str | None = None  # Reasoning/thought process
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict:
        result = {"role": self.role}
        if self.content is not None:
            result["content"] = self.content
        if self.thought is not None:
            result["thought"] = self.thought
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.name:
            result["name"] = self.name
        return result


@dataclass
class ToolDefinition:
    """Tool/function definition."""
    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai_format(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_genai_format(self):
        """Convert to Google GenAI format."""
        from google.genai import types
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


@dataclass
class ChatResponse:
    """Unified chat response."""
    content: str | None = None
    thought: str | None = None  # Reasoning/thought process
    tool_calls: list[dict] | None = None
    finish_reason: str | None = None
    usage: dict | None = None
    raw: Any = None  # Original response for provider-specific access

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class StreamChunk:
    """Streaming response chunk."""
    content: str | None = None
    thought: str | None = None  # Streaming thought content
    tool_calls: list[dict] | None = None
    finish_reason: str | None = None
    usage: dict | None = None


@dataclass
class ToolCall:
    """A tool/function call from the model."""
    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ToolCall":
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            arguments=data.get("arguments", {}),
        )


@dataclass
class ClientConfig:
    """Model client configuration."""
    mode: ClientMode = ClientMode.DIRECT
    model: str = "gemini-2.5-flash-preview"
    temperature: float = 0.7
    max_tokens: int | None = None
    system_prompt: str | None = None

    # Gateway mode settings
    gateway_url: str | None = None
    gateway_key: str | None = None

    # Direct mode settings - provider API keys
    google_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"

    @classmethod
    def from_env(cls) -> "ClientConfig":
        """Load configuration from environment variables."""
        import os

        # Check if gateway mode is configured
        gateway_url = os.getenv("DORAEMON_GATEWAY_URL")

        if gateway_url:
            mode = ClientMode.GATEWAY
        else:
            mode = ClientMode.DIRECT

        return cls(
            mode=mode,
            model=os.getenv("DORAEMON_MODEL", "gemini-3-pro-preview"),
            # Gateway settings
            gateway_url=gateway_url,
            gateway_key=os.getenv("DORAEMON_API_KEY"),
            # Direct mode API keys
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            ollama_base_url=os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
        )
