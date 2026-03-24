"""
Extended Thinking Mode

Deep reasoning mode for complex problems.

Features:
- Extended thinking with chain-of-thought
- Thinking budget management
- Reasoning trace display
- Problem decomposition
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ThinkingMode(Enum):
    """Thinking modes."""

    NORMAL = "normal"  # Standard response
    EXTENDED = "extended"  # Extended thinking with reasoning
    DEEP = "deep"  # Maximum reasoning depth


@dataclass
class ThinkingConfig:
    """Thinking configuration."""

    mode: ThinkingMode = ThinkingMode.NORMAL
    max_thinking_tokens: int = 8000  # Max tokens for thinking
    show_thinking: bool = True  # Show thinking process
    thinking_budget: int = 0  # 0 = unlimited


@dataclass
class ThinkingResult:
    """Result of thinking process."""

    thinking: str  # Thinking/reasoning trace
    response: str  # Final response
    thinking_tokens: int
    response_tokens: int
    duration: float
    mode: ThinkingMode

    def to_dict(self) -> dict[str, Any]:
        return {
            "thinking": self.thinking,
            "response": self.response,
            "thinking_tokens": self.thinking_tokens,
            "response_tokens": self.response_tokens,
            "duration": self.duration,
            "mode": self.mode.value,
        }


class ThinkingManager:
    """
    Manages extended thinking mode.

    Usage:
        thinking = ThinkingManager()

        # Enable extended thinking
        thinking.set_mode(ThinkingMode.EXTENDED)

        # Build thinking prompt
        prompt = thinking.build_prompt(user_message)

        # Parse response with thinking
        result = thinking.parse_response(response_text)
    """

    def __init__(self, config: ThinkingConfig | None = None):
        """
        Initialize thinking manager.

        Args:
            config: Thinking configuration
        """
        self.config = config or ThinkingConfig()
        self._mode = self.config.mode
        self._total_thinking_tokens = 0

    def get_mode(self) -> ThinkingMode:
        """Get current thinking mode."""
        return self._mode

    def set_mode(self, mode: ThinkingMode):
        """Set thinking mode."""
        self._mode = mode
        logger.info(f"Thinking mode set to: {mode.value}")

    def toggle_mode(self) -> ThinkingMode:
        """Toggle between normal and extended thinking."""
        if self._mode == ThinkingMode.NORMAL:
            self._mode = ThinkingMode.EXTENDED
        else:
            self._mode = ThinkingMode.NORMAL
        return self._mode

    def is_extended(self) -> bool:
        """Check if extended thinking is enabled."""
        return self._mode != ThinkingMode.NORMAL

    def build_prompt(self, user_message: str) -> str:
        """
        Build prompt with thinking instructions.

        Args:
            user_message: Original user message

        Returns:
            Enhanced prompt with thinking instructions
        """
        if self._mode == ThinkingMode.NORMAL:
            return user_message

        thinking_instruction = self._get_thinking_instruction()
        return f"{thinking_instruction}\n\nUser request: {user_message}"

    def _get_thinking_instruction(self) -> str:
        """Get thinking instruction based on mode."""
        if self._mode == ThinkingMode.EXTENDED:
            return """Before responding, think through this problem step by step.

<thinking>
1. Understand what is being asked
2. Break down the problem into parts
3. Consider different approaches
4. Evaluate trade-offs
5. Form a plan
</thinking>

After your thinking, provide your response."""

        elif self._mode == ThinkingMode.DEEP:
            return """This requires deep analysis. Think very carefully.

<thinking>
## Problem Analysis
- What exactly is being asked?
- What are the constraints?
- What information do we have?

## Approach Exploration
- What are all possible approaches?
- What are the pros and cons of each?

## Solution Design
- Which approach is best and why?
- What are the steps to implement?

## Verification
- How can we verify the solution is correct?
- What edge cases should we consider?
</thinking>

After your thorough analysis, provide your response."""

        return ""

    def parse_response(self, response: str, duration: float = 0) -> ThinkingResult:
        """
        Parse response to extract thinking and final response.

        Args:
            response: Raw response text
            duration: Response duration

        Returns:
            ThinkingResult with separated thinking and response
        """
        thinking = ""
        final_response = response

        # Extract thinking block
        if "<thinking>" in response and "</thinking>" in response:
            start = response.find("<thinking>")
            end = response.find("</thinking>") + len("</thinking>")
            thinking = response[start:end]

            # Remove thinking from response
            final_response = response[:start] + response[end:]
            final_response = final_response.strip()

        # Estimate tokens (rough approximation)
        thinking_tokens = len(thinking) // 4 if thinking else 0
        response_tokens = len(final_response) // 4

        # Track total thinking tokens
        self._total_thinking_tokens += thinking_tokens

        return ThinkingResult(
            thinking=thinking,
            response=final_response,
            thinking_tokens=thinking_tokens,
            response_tokens=response_tokens,
            duration=duration,
            mode=self._mode,
        )

    def get_thinking_usage(self) -> dict[str, Any]:
        """Get thinking token usage."""
        return {
            "mode": self._mode.value,
            "total_thinking_tokens": self._total_thinking_tokens,
            "budget": self.config.thinking_budget,
            "remaining": (
                self.config.thinking_budget - self._total_thinking_tokens
                if self.config.thinking_budget > 0
                else "unlimited"
            ),
        }

    def should_show_thinking(self) -> bool:
        """Check if thinking should be displayed."""
        return self.config.show_thinking and self._mode != ThinkingMode.NORMAL

    def reset_usage(self):
        """Reset thinking token usage."""
        self._total_thinking_tokens = 0

    def format_thinking(self, thinking: str) -> str:
        """
        Format thinking block for display.

        Args:
            thinking: Raw thinking text

        Returns:
            Formatted thinking text
        """
        if not thinking:
            return ""

        # Remove tags
        formatted = thinking.replace("<thinking>", "").replace("</thinking>", "")
        formatted = formatted.strip()

        # Add indentation
        lines = formatted.split("\n")
        formatted_lines = ["  │ " + line for line in lines]

        return "\n".join(formatted_lines)

    def get_mode_indicator(self) -> str:
        """Get mode indicator for prompt."""
        if self._mode == ThinkingMode.EXTENDED:
            return "🧠"
        elif self._mode == ThinkingMode.DEEP:
            return "🧠🧠"
        return ""
