"""
Agent State - Unified State Management

Replaces multiple managers with a single cohesive state object.
"""

from dataclasses import dataclass, field
from typing import Any

from .types import Message, ToolCall


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

    max_messages: int = 50  # 内存限制
    _compressed_summary: str = ""

    goal: str | None = None
    is_finished: bool = False
    status: str = "idle"

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
        """Compress old messages into summary."""
        if not old_messages:
            return

        summary_parts = []
        for msg in old_messages:
            if msg.content:
                role = msg.role
                content_preview = msg.content[:200]
                summary_parts.append(f"[{role}]: {content_preview}")

        if summary_parts:
            self._compressed_summary += f"\n[Earlier conversation]\n" + "\n".join(
                summary_parts[-5:]
            )

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
        tool_calls: list[dict] | None = None,
        thought: str | None = None,
    ) -> None:
        """Add an assistant message."""
        self.add_message(
            Message(
                role="assistant",
                content=content,
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

    def set_goal(self, goal: str) -> None:
        """Set the agent's goal."""
        self.goal = goal
        self.is_finished = False
        self.status = "running"

    def mark_finished(self) -> None:
        """Mark the agent as finished."""
        self.is_finished = True
        self.status = "finished"

    def mark_error(self, error: str) -> None:
        """Mark an error state."""
        self.last_error = error
        self.status = "error"

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
            "turn_count": self.turn_count,
            "is_finished": self.is_finished,
            "status": self.status,
            "goal": self.goal,
            "estimated_tokens": self.estimated_tokens,
            "tool_call_count": self.get_tool_call_count(),
            "message_count": len(self.messages),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentState":
        """Deserialize state from dictionary."""
        return cls(
            mode=data.get("mode", "build"),
            max_turns=data.get("max_turns", 100),
            turn_count=data.get("turn_count", 0),
            max_context_tokens=data.get("max_context_tokens", 128000),
            goal=data.get("goal"),
            is_finished=data.get("is_finished", False),
            status=data.get("status", "idle"),
            metadata=data.get("metadata", {}),
        )

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
            self.turn_count = state_data.get("turn_count", 0)
            self.is_finished = state_data.get("is_finished", False)
            self.status = state_data.get("status", "idle")
            self.goal = state_data.get("goal")
