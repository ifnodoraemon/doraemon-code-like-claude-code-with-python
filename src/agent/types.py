"""
Agent Types - Core Data Structures

Defines the core types used across the agent system.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentStatus(Enum):
    """Agent execution status."""

    IDLE = "idle"
    RUNNING = "running"
    THINKING = "thinking"
    ACTING = "acting"
    WAITING = "waiting"
    FINISHED = "finished"
    ERROR = "error"


class ActionType(Enum):
    """Types of actions an agent can take."""

    TOOL_CALL = "tool_call"
    RESPOND = "respond"
    ASK_USER = "ask_user"
    ERROR = "error"
    FINISH = "finish"


@dataclass
class Message:
    """A message in the conversation."""

    role: str  # "user", "assistant", "tool", "system"
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    thought: str | None = None

    def to_api_format(self) -> dict[str, Any]:
        """Convert to API format."""
        result = {"role": self.role}
        if self.content:
            result["content"] = self.content
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.name:
            result["name"] = self.name
        return result


@dataclass
class ToolCall:
    """A tool call made by the agent."""

    id: str
    name: str
    arguments: dict[str, Any]
    result: str | None = None
    error: str | None = None
    duration: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "result": self.result,
            "error": self.error,
            "duration": self.duration,
        }


@dataclass
class Observation:
    """What the agent observes from the environment."""

    user_input: str | None = None
    tool_results: list[ToolCall] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class Thought:
    """Agent's reasoning about what to do next."""

    reasoning: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    response: str | None = None
    is_finished: bool = False


@dataclass
class Action:
    """An action the agent decides to take."""

    type: ActionType
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_call_id: str | None = None
    response: str | None = None
    error: str | None = None

    @classmethod
    def tool_call(cls, name: str, args: dict, call_id: str) -> "Action":
        return cls(
            type=ActionType.TOOL_CALL,
            tool_name=name,
            tool_args=args,
            tool_call_id=call_id,
        )

    @classmethod
    def respond(cls, response: str) -> "Action":
        return cls(type=ActionType.RESPOND, response=response)

    @classmethod
    def ask_user(cls, question: str) -> "Action":
        return cls(type=ActionType.ASK_USER, response=question)

    @classmethod
    def finish(cls, response: str | None = None) -> "Action":
        return cls(type=ActionType.FINISH, response=response)

    @classmethod
    def error(cls, error: str) -> "Action":
        return cls(type=ActionType.ERROR, error=error)

    def to_dict(self) -> dict[str, Any]:
        """Convert action to dictionary."""
        result = {"type": self.type.value}
        if self.tool_name:
            result["tool_name"] = self.tool_name
        if self.tool_args:
            result["tool_args"] = self.tool_args
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.response:
            result["response"] = self.response
        if self.error:
            result["error"] = self.error
        return result


@dataclass
class AgentResult:
    """Result of agent execution."""

    success: bool
    response: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    tokens_used: int = 0
    duration: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "response": self.response,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "tokens_used": self.tokens_used,
            "duration": self.duration,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class ToolDefinition:
    """Definition of a tool available to the agent."""

    name: str
    description: str
    parameters: dict[str, Any]
    sensitive: bool = False

    def to_api_format(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
