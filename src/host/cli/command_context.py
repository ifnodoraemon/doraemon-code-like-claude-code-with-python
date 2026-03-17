"""
Shared Command Context

Dataclass that encapsulates all shared dependencies for CLI command handlers,
eliminating the 12+ parameter constructors repeated across every handler class.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CommandContext:
    """All command handlers share these dependencies."""

    ctx: Any  # ContextManager
    tool_selector: Any
    registry: Any  # ToolRegistry
    skill_mgr: Any
    checkpoint_mgr: Any
    task_mgr: Any
    cost_tracker: Any
    cmd_history: Any
    session_mgr: Any
    hook_mgr: Any
    model_name: str
    project: str
    permission_mgr: Any = field(default=None)
    build_system_prompt: Any = field(default=None)
    convert_tools_to_definitions: Any = field(default=None)
