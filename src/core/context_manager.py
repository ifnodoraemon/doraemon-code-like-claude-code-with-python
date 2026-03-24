"""
Conversation Context Management

Provides automatic conversation summarization and context window management
to enable effectively unlimited conversation length.

Key improvements over naive approaches:
1. Uses actual token counts from API responses (not character estimates)
2. Uses LLM for intelligent summarization
3. Manages history in sync with Gemini chat
4. Injects summary as user context (not system instruction)
"""

import json
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ========================================
# Data Structures
# ========================================


@dataclass
class Message:
    """A conversation message. content can be str or list[dict] for multimodal."""

    role: str  # "user", "assistant", "model"
    content: str | list[dict]
    timestamp: float = field(default_factory=time.time)
    token_count: int | None = None  # Actual token count from API
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Extract plain text from content (works for both str and multimodal)."""
        from src.core.model_utils import get_content_text

        return get_content_text(self.content)

    def to_dict(self) -> dict[str, Any]:
        # For persistence: strip image base64 data, keep only path placeholders
        content_for_save = self.content
        if isinstance(self.content, list):
            content_for_save = []
            for part in self.content:
                if part.get("type") == "image":
                    # Save only a reference, not the base64 blob
                    path = part.get("path", "unknown")
                    content_for_save.append({"type": "text", "text": f"[Image: {path}]"})
                else:
                    content_for_save.append(part)
        return {
            "role": self.role,
            "content": content_for_save,
            "timestamp": self.timestamp,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            token_count=data.get("token_count"),
            metadata=data.get("metadata", {}),
        )

    def to_api_format(self) -> dict[str, Any]:
        """Convert to API message format."""
        # Normalize role: Gemini uses "model" instead of "assistant"
        role = "model" if self.role == "assistant" else self.role
        if isinstance(self.content, list):
            parts = []
            for part in self.content:
                if part.get("type") == "text":
                    parts.append({"text": part["text"]})
                # Image parts are handled by provider-specific adapters
            return {"role": role, "parts": parts or [{"text": ""}]}
        return {"role": role, "parts": [{"text": self.content}]}


@dataclass
class ConversationSummary:
    """Summary of a conversation segment."""

    content: str
    message_count: int
    token_count: int  # Tokens summarized
    created_at: float = field(default_factory=time.time)
    key_points: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "message_count": self.message_count,
            "token_count": self.token_count,
            "created_at": self.created_at,
            "key_points": self.key_points,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationSummary":
        return cls(
            content=data["content"],
            message_count=data["message_count"],
            token_count=data.get("token_count", 0),
            created_at=data.get("created_at", time.time()),
            key_points=data.get("key_points", []),
        )


# ========================================
# Configuration
# ========================================


@dataclass
class ContextConfig:
    """Configuration for context management."""

    # Token limits (dynamically set based on model)
    max_context_tokens: int = 100_000
    summarize_threshold: float = 0.7  # Summarize at 70% of max
    target_after_summary: float = 0.4  # Target 40% after summarization

    # Message handling
    keep_recent_messages: int = 6  # Always keep last N messages (3 turns)
    min_messages_to_summarize: int = 4

    # Fallback token estimation (when API count unavailable)
    chars_per_token_estimate: float = 3.0  # Conservative for mixed content

    # Persistence
    auto_save: bool = True
    save_directory: str = ".agent/conversations"


# Model context window sizes (latest generation only, from official docs 2026-03)
# Source: developers.openai.com, platform.claude.com, ai.google.dev, api-docs.deepseek.com
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # Google Gemini 3.x (latest gen, 2026)
    "gemini-3.1-pro": 1_000_000,  # gemini-3.1-pro-preview
    "gemini-3-flash": 1_000_000,  # gemini-3-flash-preview
    "gemini-3.1-flash-lite": 1_000_000,  # gemini-3.1-flash-lite-preview
    # OpenAI (latest gen, 2026)
    "gpt-5.3": 400_000,  # 128K max output
    "gpt-5.2": 400_000,  # 128K max output
    # Anthropic Claude (latest gen, 2026)
    "claude-opus-4-6": 200_000,  # 128K max output, 1M beta
    "claude-sonnet-4-6": 200_000,  # 64K max output, 1M beta
    "claude-haiku-4-5": 200_000,  # 64K max output (Haiku 最新就是 4.5)
    # DeepSeek V3.2 (latest gen, 2025-2026)
    "deepseek-chat": 128_000,  # 8K max output
    "deepseek-reasoner": 128_000,  # 64K max output
}

DEFAULT_CONTEXT_WINDOW = 200_000


def get_context_window_for_model(model_name: str) -> int:
    """
    Get the context window size for a model.

    Uses prefix matching to handle versioned model names like
    'gemini-2.5-flash-preview-05-20' or 'claude-sonnet-4-20250514'.
    """
    if not model_name:
        return DEFAULT_CONTEXT_WINDOW

    model_lower = model_name.lower()

    # Try exact match first
    if model_lower in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[model_lower]

    # Try prefix matching (longest match wins)
    best_match = ""
    best_size = DEFAULT_CONTEXT_WINDOW
    for prefix, size in MODEL_CONTEXT_WINDOWS.items():
        if model_lower.startswith(prefix) and len(prefix) > len(best_match):
            best_match = prefix
            best_size = size

    return best_size


DEFAULT_CONFIG = ContextConfig()


# ========================================
# Context Manager
# ========================================


class ContextManager:
    """
    Manages conversation context with automatic summarization.

    Key design decisions:
    1. Single source of truth for conversation history
    2. Uses actual token counts from API when available
    3. Supports LLM-based summarization
    4. Handles Gemini chat history format
    """

    def __init__(
        self,
        project: str = "default",
        config: ContextConfig | None = None,
        summarize_fn: Callable[[list[Message]], str] | None = None,
    ):
        """
        Initialize context manager.

        Args:
            project: Project name for persistence
            config: Context configuration
            summarize_fn: Optional function to generate summaries using LLM
        """
        self.project = project
        self.config = config or DEFAULT_CONFIG
        self.summarize_fn = summarize_fn

        # State
        self.messages: list[Message] = []
        self.summaries: list[ConversationSummary] = []
        self.total_messages_ever: int = 0
        self.session_id: str = uuid.uuid4().hex[:12]

        # Token tracking
        self._last_prompt_tokens: int = 0
        self._total_tokens_used: int = 0

        # Load existing state
        if self.config.auto_save:
            self._load_state()

    # ----------------------------------------
    # Public API
    # ----------------------------------------

    def add_user_message(self, content: str | list[dict]) -> Message:
        """Add a user message (text or multimodal content)."""
        msg = Message(role="user", content=content)
        self.messages.append(msg)
        self.total_messages_ever += 1
        self._auto_save()
        return msg

    def add_system_message(self, content: str) -> Message:
        """Add a system message."""
        msg = Message(role="system", content=content)
        self.messages.append(msg)
        self.total_messages_ever += 1
        self._auto_save()
        return msg

    def add_assistant_message(
        self,
        content: str,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> Message:
        """
        Add an assistant message with optional token counts from API.

        Args:
            content: The assistant's response
            prompt_tokens: Token count for the prompt (from usage_metadata)
            completion_tokens: Token count for the completion
        """
        msg = Message(
            role="assistant",
            content=content,
            token_count=completion_tokens,
        )
        self.messages.append(msg)
        self.total_messages_ever += 1

        # Update token tracking
        if prompt_tokens:
            self._last_prompt_tokens = prompt_tokens
            self._total_tokens_used += prompt_tokens + (completion_tokens or 0)

        # Check if summarization needed
        self._maybe_summarize()
        self._auto_save()

        return msg

    def get_history_for_api(self) -> list[dict]:
        """
        Get conversation history formatted for Gemini API.

        Returns list of messages in Gemini format, with summary prepended if exists.
        """
        history = []

        # Prepend summary context if we have summaries
        if self.summaries:
            summary_text = self._format_summaries_for_context()
            # Add as a context-setting exchange with clear marker
            history.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": f"[Previous Conversation Summary]\n\n{summary_text}\n\n[End of Summary - Continue from here]"
                        }
                    ],
                }
            )
            history.append(
                {
                    "role": "model",
                    "parts": [
                        {
                            "text": "Understood. I have the context from our previous conversation and will continue from here."
                        }
                    ],
                }
            )

        # Add actual messages
        for msg in self.messages:
            history.append(msg.to_api_format())

        return history

    def get_context_stats(self) -> dict[str, Any]:
        """Get statistics about current context state."""
        estimated_tokens = self._estimate_current_tokens()
        threshold = int(self.config.max_context_tokens * self.config.summarize_threshold)

        return {
            "messages": len(self.messages),
            "summaries": len(self.summaries),
            "total_messages_ever": self.total_messages_ever,
            "estimated_tokens": estimated_tokens,
            "last_prompt_tokens": self._last_prompt_tokens,
            "threshold_tokens": threshold,
            "usage_percent": round(
                estimated_tokens / max(self.config.max_context_tokens, 1) * 100, 1
            ),
            "session_id": self.session_id,
            "needs_summary": estimated_tokens > threshold,
        }

    def clear(self, keep_summaries: bool = True):
        """Clear current messages, optionally keeping summaries."""
        if keep_summaries and len(self.messages) >= self.config.min_messages_to_summarize:
            self._force_summarize()

        self.messages = []
        if not keep_summaries:
            self.summaries = []

        self._auto_save()

    def reset(self):
        """Complete reset - new session."""
        self.messages = []
        self.summaries = []
        self.total_messages_ever = 0
        self.session_id = uuid.uuid4().hex[:12]
        self._last_prompt_tokens = 0
        self._total_tokens_used = 0
        self._auto_save()

    # ----------------------------------------
    # Summarization
    # ----------------------------------------

    def _maybe_summarize(self):
        """Check if summarization is needed and perform it."""
        estimated_tokens = self._estimate_current_tokens()
        threshold = self.config.max_context_tokens * self.config.summarize_threshold

        if estimated_tokens > threshold:
            logger.info(
                f"Context size ({estimated_tokens} tokens) exceeds threshold "
                f"({threshold:.0f}), triggering summarization"
            )
            self._force_summarize()

    def _force_summarize(self):
        """Force summarization of older messages.

        Ensures tool_call/result message pairs are not split across the boundary.
        """
        if len(self.messages) <= self.config.keep_recent_messages:
            logger.debug("Not enough messages to summarize")
            return

        # Calculate initial split point
        split_idx = len(self.messages) - self.config.keep_recent_messages

        # Adjust split_idx to avoid breaking tool_call/result pairs.
        # If the message at split_idx looks like a tool result (role="tool" or
        # metadata hints), move split_idx backward to include the preceding
        # assistant message that triggered the tool call.
        max_backward_steps = 20  # Safety limit to prevent infinite loops
        steps = 0
        while split_idx > 0 and steps < max_backward_steps:
            msg = self.messages[split_idx]
            # Check if this is a tool result message
            is_tool_result = (
                msg.role == "tool"
                or msg.metadata.get("is_tool_result")
                or (msg.role == "model" and msg.metadata.get("tool_call_id"))
            )
            if is_tool_result:
                split_idx -= 1
                steps += 1
            else:
                break

        if split_idx < self.config.min_messages_to_summarize:
            logger.debug("Not enough messages to meet minimum for summarization")
            return

        messages_to_summarize = self.messages[:split_idx]
        messages_to_keep = self.messages[split_idx:]

        # Generate summary
        if self.summarize_fn:
            # Use LLM-based summarization
            try:
                summary_text = self.summarize_fn(messages_to_summarize)
            except Exception as e:
                logger.error(f"LLM summarization failed: {e}, using fallback")
                summary_text = self._simple_summarize(messages_to_summarize)
        else:
            # Fallback to simple extraction
            summary_text = self._simple_summarize(messages_to_summarize)

        # Calculate tokens summarized
        tokens_summarized = sum(
            m.token_count
            if isinstance(m.token_count, int) and m.token_count > 0
            else self._estimate_tokens(m.content)
            for m in messages_to_summarize
        )

        # Create summary object
        summary = ConversationSummary(
            content=summary_text,
            message_count=len(messages_to_summarize),
            token_count=tokens_summarized,
        )

        self.summaries.append(summary)
        self.messages = messages_to_keep

        logger.info(
            f"Summarized {summary.message_count} messages ({tokens_summarized} tokens). "
            f"Kept {len(messages_to_keep)} recent messages."
        )

    def _simple_summarize(self, messages: list[Message]) -> str:
        """Simple extractive summary as fallback."""
        parts = [f"=== Conversation Summary ({len(messages)} messages) ===\n"]

        # Extract user requests
        user_msgs = [m for m in messages if m.role == "user"]
        if user_msgs:
            parts.append("User requests/questions:")
            for i, m in enumerate(user_msgs[:5], 1):
                text = m.text  # handles both str and multimodal
                preview = text[:150].replace("\n", " ")
                if len(text) > 150:
                    preview += "..."
                parts.append(f"  {i}. {preview}")

        # Extract key actions from assistant
        assistant_msgs = [m for m in messages if m.role == "assistant"]
        action_keywords = [
            "created",
            "updated",
            "fixed",
            "implemented",
            "added",
            "removed",
            "changed",
        ]

        actions = []
        for m in assistant_msgs:
            content_lower = m.text.lower()
            for keyword in action_keywords:
                if keyword in content_lower:
                    # Extract sentence containing keyword
                    sentences = m.text.replace("\n", " ").split(". ")
                    for sent in sentences:
                        if keyword in sent.lower() and len(sent) < 200:
                            actions.append(sent.strip())
                            break
                    break

        if actions:
            parts.append("\nKey actions taken:")
            for i, action in enumerate(actions[:5], 1):
                parts.append(f"  {i}. {action}")

        return "\n".join(parts)

    def _format_summaries_for_context(self) -> str:
        """Format all summaries for injection into context."""
        if not self.summaries:
            return ""

        parts = []
        for i, summary in enumerate(self.summaries, 1):
            timestamp = datetime.fromtimestamp(summary.created_at).strftime("%H:%M")
            parts.append(f"[Segment {i} @ {timestamp}]")
            parts.append(summary.content)
            parts.append("")

        return "\n".join(parts)

    # ----------------------------------------
    # Token Estimation
    # ----------------------------------------

    def _estimate_tokens(self, text: str | list[dict]) -> int:
        """Estimate tokens for text or multimodal content."""
        if isinstance(text, list):
            from src.core.model_utils import TOKENS_PER_IMAGE

            total = 0
            for part in text:
                if part.get("type") == "text":
                    total += int(len(part.get("text", "")) / self.config.chars_per_token_estimate)
                elif part.get("type") == "image":
                    total += TOKENS_PER_IMAGE
            return total
        return int(len(text) / self.config.chars_per_token_estimate)

    def _estimate_current_tokens(self) -> int:
        """Estimate total tokens in current context."""
        # Use actual counts when available, estimate otherwise
        total = 0

        # Messages
        for msg in self.messages:
            if isinstance(msg.token_count, int) and msg.token_count > 0:
                total += msg.token_count
            else:
                total += self._estimate_tokens(msg.content)

        # Summaries (these become part of context)
        for summary in self.summaries:
            total += self._estimate_tokens(summary.content)

        return total

    # ----------------------------------------
    # Persistence
    # ----------------------------------------

    def _get_save_path(self) -> Path:
        """Get path for saving state."""
        save_dir = Path(self.config.save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir / f"{self.project}.json"

    def _auto_save(self):
        """Auto-save if enabled."""
        if self.config.auto_save:
            self._save_state()

    def _save_state(self):
        """Save conversation state to file."""
        try:
            path = self._get_save_path()
            data = {
                "version": 2,
                "session_id": self.session_id,
                "project": self.project,
                "total_messages_ever": self.total_messages_ever,
                "messages": [m.to_dict() for m in self.messages],
                "summaries": [s.to_dict() for s in self.summaries],
                "last_prompt_tokens": self._last_prompt_tokens,
                "saved_at": time.time(),
            }

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.debug(f"Saved context state to {path}")

        except Exception as e:
            logger.warning(f"Failed to save context state: {e}")

    def _load_state(self):
        """Load conversation state from file."""
        try:
            path = self._get_save_path()
            if not path.exists():
                return

            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            # Version check
            version = data.get("version", 1)
            if version < 2:
                logger.info("Old context format, starting fresh")
                return

            self.session_id = data.get("session_id", self.session_id)
            self.total_messages_ever = data.get("total_messages_ever", 0)
            self._last_prompt_tokens = data.get("last_prompt_tokens", 0)

            self.messages = [Message.from_dict(m) for m in data.get("messages", [])]
            self.summaries = [ConversationSummary.from_dict(s) for s in data.get("summaries", [])]

            logger.info(
                f"Loaded context: {len(self.messages)} messages, {len(self.summaries)} summaries"
            )

        except Exception as e:
            logger.warning(f"Failed to load context state: {e}")


# ========================================
# LLM Summarization Helper
# ========================================

SUMMARIZE_PROMPT = """Please summarize this conversation segment concisely.
Focus on:
1. What the user asked for or wanted to accomplish
2. What actions were taken (files created/modified, code written, etc.)
3. Any important decisions or conclusions reached
4. Any unresolved issues or next steps mentioned

Keep the summary under 500 words. Be factual and specific.

Conversation to summarize:
---
{conversation}
---

Summary:"""


def create_llm_summarizer(chat_fn: Callable[[str], str]) -> Callable[[list[Message]], str]:
    """
    Create an LLM-based summarizer function.

    Args:
        chat_fn: Function that sends a message and returns response
                 (e.g., lambda msg: chat.send_message(msg).text)

    Returns:
        Summarizer function compatible with ContextManager
    """

    def summarize(messages: list[Message]) -> str:
        # Format conversation
        conversation_parts = []
        for msg in messages:
            role = "User" if msg.role == "user" else "Assistant"
            text = msg.text  # handles both str and multimodal
            content = text[:1000]
            if len(text) > 1000:
                content += "... [truncated]"
            conversation_parts.append(f"{role}: {content}")

        conversation_text = "\n\n".join(conversation_parts)
        prompt = SUMMARIZE_PROMPT.format(conversation=conversation_text)

        # Get summary from LLM
        response = chat_fn(prompt)
        return response.strip()

    return summarize
