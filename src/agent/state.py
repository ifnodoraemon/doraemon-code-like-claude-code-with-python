"""
Agent State - Unified State Management

Replaces multiple managers with a single cohesive state object.
"""

from dataclasses import dataclass, field
from typing import Any

from .types import AgentStatus, Message, ToolCall


@dataclass
class AgentState:
    """
    Unified agent state.

    This replaces multiple managers:
    - ContextManager -> messages, summaries
    - CheckpointManager -> in save/restore methods
    - CostTracker -> usage tracking
    - SessionManager -> at higher level

    Key principles:
    - Pull-based state access (agent observes when needed)
    - Single source of truth
    - Clear lifecycle management
    - Memory bounded (max 50 messages)
    """

    messages: list[Message] = field(default_factory=list)
    tool_history: list[ToolCall] = field(default_factory=list)

    mode: str = "build"
    max_turns: int = 100
    turn_count: int = 0

    max_context_tokens: int = 128000
    estimated_tokens: int = 0

    max_messages: int = 50
    _compressed_summary: str = field(default="")

    goal: str | None = None
    is_finished: bool = False
    status: str = "idle"  # Use AgentStatus values

    user_input: str | None = None
    last_response: str | None = None
    last_error: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, message: Message) -> None:
        """Add a message to history."""
        self.messages.append(message)
        self._enforce_message_limit()
        self._update_token_estimate()

    def _enforce_message_limit(self) -> None:
        """Enforce message limit to prevent memory bloat."""
        if len(self.messages) > self.max_messages:
            old_messages = self.messages[: -self.max_messages]
            self._compress_messages(old_messages)
            self.messages = self.messages[-self.max_messages :]

    def _compress_messages(self, old_messages: list[Message]) -> None:
        """Create a structured summary of compressed messages.

        Preserves key information: user intents, tool names, and outcomes.
        Full semantic summarization is handled by the Agent's LLM via _compress_context.
        """
        if not old_messages:
            return

        user_intents = [m.content for m in old_messages if m.role == "user" and m.content]
        tool_names = []
        for m in old_messages:
            if m.tool_calls:
                for tc in m.tool_calls:
                    name = tc.get("name") or tc.get("function", {}).get("name", "")
                    if name:
                        tool_names.append(name)

        summary_parts = [f"[Context: {len(old_messages)} messages archived]"]
        if user_intents:
            recent_intents = user_intents[-3:]
            summary_parts.append(f"User intents: {'; '.join(recent_intents)}")
        if tool_names:
            summary_parts.append(f"Tools used: {', '.join(dict.fromkeys(tool_names))}")

        self._compressed_summary += "\n" + "\n".join(summary_parts)

    def get_compressed_summary(self) -> str:
        """Get summary of compressed messages."""
        return self._compressed_summary

    def add_user_message(self, content: str) -> None:
        """Add a user message."""
        self.add_message(Message(role="user", content=content))
        self.user_input = content

    def add_assistant_message(
        self,
        content: str | None = None,
        provider_items: list[dict[str, Any]] | None = None,
        tool_calls: list[dict] | None = None,
        thought: str | None = None,
    ) -> None:
        """Add an assistant message."""
        self.add_message(
            Message(
                role="assistant",
                content=content,
                provider_items=provider_items,
                tool_calls=tool_calls,
                thought=thought,
            )
        )
        self.last_response = content

    def add_tool_result(
        self,
        tool_call_id: str,
        name: str,
        result: str,
    ) -> None:
        """Add a tool result message."""
        self.add_message(
            Message(
                role="tool",
                content=result,
                tool_call_id=tool_call_id,
                name=name,
            )
        )

    def add_tool_call(self, tool_call: ToolCall) -> None:
        """Record a tool call in history."""
        self.tool_history.append(tool_call)

    def _update_token_estimate(self) -> None:
        """Estimate token count from messages."""
        total = 0
        for msg in self.messages:
            if msg.content:
                total += len(msg.content) // 4
            if msg.tool_calls:
                total += len(str(msg.tool_calls)) // 4
        self.estimated_tokens = total

    def needs_compression(self) -> bool:
        """Check if context needs compression."""
        threshold = self.max_context_tokens * 0.9
        return self.estimated_tokens > threshold

    def get_recent_messages(self, n: int = 12) -> list[Message]:
        """Get the N most recent messages."""
        return self.messages[-n:] if len(self.messages) > n else self.messages

    def get_history_for_api(self) -> list[dict]:
        """Get messages in API format."""
        return [msg.to_api_format() for msg in self.messages]

    def clear_history(self) -> None:
        """Clear message history."""
        self.messages.clear()
        self.tool_history.clear()
        self.estimated_tokens = 0
        self.turn_count = 0
        self.goal = None
        self.is_finished = False
        self.status = AgentStatus.IDLE.value
        self.user_input = None
        self.last_response = None
        self.last_error = None

    def set_goal(self, goal: str) -> None:
        """Set the agent's goal."""
        self.goal = goal
        self.is_finished = False
        self.status = AgentStatus.RUNNING.value

    def mark_finished(self) -> None:
        """Mark the agent as finished."""
        self.is_finished = True
        self.status = AgentStatus.FINISHED.value

    def mark_error(self, error: str) -> None:
        """Mark an error state."""
        self.last_error = error
        self.status = AgentStatus.ERROR.value

    def increment_turn(self) -> bool:
        """Increment turn counter, return True if within limits."""
        self.turn_count += 1
        return self.turn_count < self.max_turns

    def get_tool_call_count(self) -> int:
        """Get total number of tool calls."""
        return len(self.tool_history)

    def get_successful_tool_calls(self) -> list[ToolCall]:
        """Get tool calls that succeeded."""
        return [tc for tc in self.tool_history if tc.error is None]

    def get_failed_tool_calls(self) -> list[ToolCall]:
        """Get tool calls that failed."""
        return [tc for tc in self.tool_history if tc.error is not None]

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "mode": self.mode,
            "max_turns": self.max_turns,
            "turn_count": self.turn_count,
            "is_finished": self.is_finished,
            "status": self.status,
            "goal": self.goal,
            "max_context_tokens": self.max_context_tokens,
            "estimated_tokens": self.estimated_tokens,
            "user_input": self.user_input,
            "last_response": self.last_response,
            "last_error": self.last_error,
            "metadata": self.metadata,
            "tool_call_count": self.get_tool_call_count(),
            "message_count": len(self.messages),
            "messages": [msg.to_api_format() for msg in self.messages],
            "tool_history": [tc.to_dict() for tc in self.tool_history],
            "compressed_summary": self._compressed_summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentState":
        """Deserialize state from dictionary."""
        messages = [
            Message(
                role=m.get("role", "user"),
                content=m.get("content"),
                provider_items=m.get("provider_items"),
                tool_calls=m.get("tool_calls"),
                tool_call_id=m.get("tool_call_id"),
                name=m.get("name"),
                thought=m.get("thought"),
            )
            for m in data.get("messages", [])
        ]
        tool_history = [
            ToolCall(
                id=tc.get("id", ""),
                name=tc.get("name", ""),
                arguments=tc.get("arguments", {}),
                result=tc.get("result"),
                error=tc.get("error"),
                duration=tc.get("duration", 0.0),
            )
            for tc in data.get("tool_history", [])
        ]
        state = cls(
            mode=data.get("mode", "build"),
            max_turns=data.get("max_turns", 100),
            turn_count=data.get("turn_count", 0),
            max_context_tokens=data.get("max_context_tokens", 128000),
            estimated_tokens=data.get("estimated_tokens", 0),
            goal=data.get("goal"),
            is_finished=data.get("is_finished", False),
            status=data.get("status", "idle"),
            user_input=data.get("user_input"),
            last_response=data.get("last_response"),
            last_error=data.get("last_error"),
            metadata=data.get("metadata", {}),
            messages=messages,
            tool_history=tool_history,
        )
        state._compressed_summary = data.get("compressed_summary", "")
        return state

    def create_checkpoint(self) -> dict[str, Any]:
        """Create a checkpoint for recovery."""
        return {
            "messages": [msg.to_api_format() for msg in self.messages],
            "tool_history": [tc.to_dict() for tc in self.tool_history],
            "state": self.to_dict(),
        }

    def restore_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Restore from a checkpoint."""
        if "state" in checkpoint:
            state_data = checkpoint["state"]
            self.mode = state_data.get("mode", self.mode)
            self.max_turns = state_data.get("max_turns", self.max_turns)
            self.turn_count = state_data.get("turn_count", 0)
            self.is_finished = state_data.get("is_finished", False)
            self.status = state_data.get("status", "idle")
            self.goal = state_data.get("goal")
            self.max_context_tokens = state_data.get("max_context_tokens", self.max_context_tokens)
            self.estimated_tokens = state_data.get("estimated_tokens", 0)
            self.user_input = state_data.get("user_input")
            self.last_response = state_data.get("last_response")
            self.last_error = state_data.get("last_error")
            self.metadata = state_data.get("metadata", {})

        if "messages" in checkpoint:
            self.messages = [
                Message(
                    role=m.get("role", "user"),
                    content=m.get("content"),
                    provider_items=m.get("provider_items"),
                    tool_calls=m.get("tool_calls"),
                    tool_call_id=m.get("tool_call_id"),
                    name=m.get("name"),
                    thought=m.get("thought"),
                )
                for m in checkpoint["messages"]
            ]

        if "tool_history" in checkpoint:
            self.tool_history = [
                ToolCall(
                    id=tc.get("id", ""),
                    name=tc.get("name", ""),
                    arguments=tc.get("arguments", {}),
                    result=tc.get("result"),
                    error=tc.get("error"),
                    duration=tc.get("duration", 0.0),
                )
                for tc in checkpoint["tool_history"]
            ]
