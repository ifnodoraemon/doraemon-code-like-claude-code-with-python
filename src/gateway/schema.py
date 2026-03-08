"""
Unified Schema for Model Gateway

Defines the common request/response format that works across all providers.
Inspired by OpenAI's API format for familiarity.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Role(str, Enum):
    """Message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(str, Enum):
    """Reason for completion."""

    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    LENGTH = "length"
    ERROR = "error"


@dataclass
class ToolCall:
    """A tool/function call from the model."""

    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ToolCall":
        func = data.get("function", {})
        return cls(
            id=data.get("id", ""),
            name=func.get("name", ""),
            arguments=func.get("arguments", {}),
        )


@dataclass
class ToolResult:
    """Result from a tool execution."""

    tool_call_id: str
    content: str

    def to_dict(self) -> dict:
        return {
            "tool_call_id": self.tool_call_id,
            "content": self.content,
        }


@dataclass
class ChatMessage:
    """A chat message in the unified format."""

    role: Role | str
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # For tool response messages
    name: str | None = None  # Optional name for the message author

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "role": self.role.value if isinstance(self.role, Role) else self.role
        }
        if self.content is not None:
            result["content"] = self.content
        if self.tool_calls:
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.name:
            result["name"] = self.name
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "ChatMessage":
        tool_calls = None
        if "tool_calls" in data:
            tool_calls = [ToolCall.from_dict(tc) for tc in data["tool_calls"]]

        return cls(
            role=data.get("role", "user"),
            content=data.get("content"),
            tool_calls=tool_calls,
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
        )


@dataclass
class ToolDefinition:
    """Definition of a tool/function that the model can call."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema format

    def to_dict(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ChatRequest:
    """Unified chat request format."""

    model: str
    messages: list[ChatMessage]
    tools: list[ToolDefinition] | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    stream: bool = False
    stop: list[str] | None = None
    # Extended options
    top_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None

    def to_dict(self) -> dict:
        result = {
            "model": self.model,
            "messages": [m.to_dict() for m in self.messages],
            "temperature": self.temperature,
            "stream": self.stream,
        }
        if self.tools:
            result["tools"] = [t.to_dict() for t in self.tools]
        if self.max_tokens:
            result["max_tokens"] = self.max_tokens
        if self.stop:
            result["stop"] = self.stop
        if self.top_p is not None:
            result["top_p"] = self.top_p
        if self.presence_penalty is not None:
            result["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty is not None:
            result["frequency_penalty"] = self.frequency_penalty
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "ChatRequest":
        messages = [ChatMessage.from_dict(m) for m in data.get("messages", [])]
        tools = None
        if "tools" in data:
            tools = [
                ToolDefinition(
                    name=t["function"]["name"],
                    description=t["function"].get("description", ""),
                    parameters=t["function"].get("parameters", {}),
                )
                for t in data["tools"]
            ]

        return cls(
            model=data.get("model", ""),
            messages=messages,
            tools=tools,
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens"),
            stream=data.get("stream", False),
            stop=data.get("stop"),
            top_p=data.get("top_p"),
            presence_penalty=data.get("presence_penalty"),
            frequency_penalty=data.get("frequency_penalty"),
        )


@dataclass
class Usage:
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class Choice:
    """A single completion choice."""

    index: int
    message: ChatMessage
    finish_reason: FinishReason | str | None = None

    def to_dict(self) -> dict:
        reason = self.finish_reason
        if isinstance(reason, FinishReason):
            reason = reason.value
        return {
            "index": self.index,
            "message": self.message.to_dict(),
            "finish_reason": reason,
        }


@dataclass
class ChatResponse:
    """Unified chat response format."""

    id: str
    model: str
    choices: list[Choice]
    usage: Usage | None = None
    created: int = field(default_factory=lambda: int(datetime.now().timestamp()))

    def to_dict(self) -> dict:
        result = {
            "id": self.id,
            "object": "chat.completion",
            "model": self.model,
            "created": self.created,
            "choices": [c.to_dict() for c in self.choices],
        }
        if self.usage:
            result["usage"] = self.usage.to_dict()
        return result

    @property
    def message(self) -> ChatMessage | None:
        """Get the first choice's message (convenience)."""
        if self.choices:
            return self.choices[0].message
        return None

    @property
    def content(self) -> str | None:
        """Get the first choice's content (convenience)."""
        if self.message:
            return self.message.content
        return None

    @property
    def tool_calls(self) -> list[ToolCall] | None:
        """Get tool calls from first choice (convenience)."""
        if self.message:
            return self.message.tool_calls
        return None


@dataclass
class StreamChunk:
    """A streaming response chunk."""

    id: str
    model: str
    delta_content: str | None = None
    delta_tool_calls: list[ToolCall] | None = None
    finish_reason: str | None = None
    usage: Usage | None = None

    def to_dict(self) -> dict[str, Any]:
        delta: dict[str, Any] = {}
        if self.delta_content:
            delta["content"] = self.delta_content
        if self.delta_tool_calls:
            delta["tool_calls"] = [tc.to_dict() for tc in self.delta_tool_calls]

        return {
            "id": self.id,
            "object": "chat.completion.chunk",
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": self.finish_reason,
                }
            ],
            "usage": self.usage.to_dict() if self.usage else None,
        }


@dataclass
class ModelInfo:
    """Information about an available model."""

    id: str
    name: str
    provider: str
    description: str = ""
    context_window: int = 0
    max_output: int = 0
    input_price: float = 0.0  # per 1M tokens
    output_price: float = 0.0  # per 1M tokens
    capabilities: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "description": self.description,
            "context_window": self.context_window,
            "max_output": self.max_output,
            "input_price": self.input_price,
            "output_price": self.output_price,
            "capabilities": self.capabilities,
            "aliases": self.aliases,
        }


@dataclass
class ErrorResponse:
    """Error response format."""

    error: str
    code: str
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "error": {
                "message": self.error,
                "code": self.code,
            }
        }
        if self.details:
            result["error"]["details"] = self.details
        return result
