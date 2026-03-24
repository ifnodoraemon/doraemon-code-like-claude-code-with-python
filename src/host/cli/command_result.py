"""Shared command result type for CLI command handlers."""

from dataclasses import dataclass


@dataclass
class CommandResult:
    """Command result object used across CLI handlers."""

    handled: bool = True
    mode: str = "build"
    tool_names: list | None = None
    tool_definitions: list | None = None
    system_prompt: str | None = None
    active_skills_content: str = ""
    conversation_history: list | None = None
    next_prompt: str | None = None

    @classmethod
    def default(
        cls,
        *,
        mode: str,
        tool_names: list,
        tool_definitions: list,
        active_skills_content: str,
        conversation_history: list,
    ) -> "CommandResult":
        """Build the default command result for the current loop state."""
        return cls(
            mode=mode,
            tool_names=tool_names,
            tool_definitions=tool_definitions,
            active_skills_content=active_skills_content,
            conversation_history=conversation_history,
        )
