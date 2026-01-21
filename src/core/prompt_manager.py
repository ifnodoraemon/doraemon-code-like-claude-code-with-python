"""
Enhanced prompt management with dual-layer system.

Supports both:
1. System Instruction (immutable, high priority)
2. User Prefix (dynamic, per-session)
"""


class PromptManager:
    """Manage system and user prompts."""

    def __init__(self):
        self.system_instruction = ""
        self.user_prefix = ""
        self.temporary_instructions = []

    def set_system_instruction(self, instruction: str):
        """Set the base system instruction (immutable after init)."""
        self.system_instruction = instruction

    def add_user_prefix(self, prefix: str):
        """Add a persistent user-level prefix."""
        self.user_prefix = prefix

    def add_temporary_instruction(self, instruction: str):
        """Add a temporary instruction (cleared after next message)."""
        self.temporary_instructions.append(instruction)

    def clear_temporary_instructions(self):
        """Clear all temporary instructions."""
        self.temporary_instructions = []

    def format_user_message(self, user_input: str) -> str:
        """
        Format user message with prefixes.

        Returns:
            Formatted message with user prefix and temporary instructions
        """
        parts = []

        # Add persistent user prefix if exists
        if self.user_prefix:
            parts.append(f"[User Context]\n{self.user_prefix}\n")

        # Add temporary instructions
        if self.temporary_instructions:
            temp = "\n".join(self.temporary_instructions)
            parts.append(f"[Temporary Instructions]\n{temp}\n")

        # Add actual user message
        parts.append(f"[User Request]\n{user_input}")

        return "\n".join(parts)

    def get_full_context(self) -> dict:
        """Get complete prompt context for debugging."""
        return {
            "system_instruction": self.system_instruction,
            "user_prefix": self.user_prefix,
            "temporary_instructions": self.temporary_instructions,
        }


# Example usage in CLI
"""
# In init_chat_model()
prompt_manager = PromptManager()
prompt_manager.set_system_instruction(sys_instruction)

# User can add prefix via /instruct command
> /instruct Focus on performance and avoid dependencies

# Next message will include this prefix
> How do I parse JSON in Python?
# Actual sent:
# [User Context]
# Focus on performance and avoid dependencies
#
# [User Request]
# How do I parse JSON in Python?
"""
