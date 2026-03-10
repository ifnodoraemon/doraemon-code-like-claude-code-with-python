"""
CLI Slash Commands Handler

Main entry point that delegates to specialized command handlers:
- commands_core.py: Core commands (help, clear, mode, reset, context, tools, debug, init, skills)
- commands_session.py: Session management (sessions, resume, rename, export, fork, checkpoints, rewind, tasks, task)
- commands_config.py: Configuration (model, theme, vim, thinking, doctor, workspace, add-dir, cost, agents, history, plugins, plugin)
"""

from typing import Any

from rich.console import Console

from src.host.cli.command_context import CommandContext
from src.host.cli.commands_config import ConfigCommandHandler
from src.host.cli.commands_core import MODE_COLORS, CoreCommandHandler
from src.host.cli.commands_session import SessionCommandHandler

console = Console()

# Re-export MODE_COLORS for backward compatibility
__all__ = ["CommandHandler", "MODE_COLORS"]


class CommandHandler:
    """Main handler that delegates to specialized command handlers."""

    def __init__(
        self,
        ctx,
        tool_selector,
        registry,
        skill_mgr,
        checkpoint_mgr,
        task_mgr,
        cost_tracker,
        cmd_history,
        session_mgr,
        hook_mgr,
        model_name: str,
        project: str,
        permission_mgr=None,
    ):
        # Build shared context once
        self.cc = CommandContext(
            ctx=ctx,
            tool_selector=tool_selector,
            registry=registry,
            skill_mgr=skill_mgr,
            checkpoint_mgr=checkpoint_mgr,
            task_mgr=task_mgr,
            cost_tracker=cost_tracker,
            cmd_history=cmd_history,
            session_mgr=session_mgr,
            hook_mgr=hook_mgr,
            model_name=model_name,
            project=project,
            permission_mgr=permission_mgr,
        )

        # Expose commonly accessed attributes for backward compatibility
        self.ctx = ctx
        self.tool_selector = tool_selector
        self.registry = registry
        self.skill_mgr = skill_mgr
        self.checkpoint_mgr = checkpoint_mgr
        self.task_mgr = task_mgr
        self.cost_tracker = cost_tracker
        self.cmd_history = cmd_history
        self.session_mgr = session_mgr
        self.hook_mgr = hook_mgr
        self.model_name = model_name
        self.project = project

        # Initialize specialized handlers with shared context
        self.core_handler = CoreCommandHandler(self.cc)
        self.session_handler = SessionCommandHandler(self.cc)
        self.config_handler = ConfigCommandHandler(self.cc)

    async def handle(
        self,
        cmd: str,
        cmd_args: list[str],
        mode: str,
        tool_names: list[str],
        tool_definitions: list,
        conversation_history: list,
        active_skills_content: str,
        build_system_prompt,
        convert_tools_to_definitions,
        sensitive_tools: set,
    ) -> dict[str, Any]:
        """
        Handle a slash command by delegating to appropriate handler.

        Returns:
            dict with keys:
            - handled: bool - whether command was handled
            - mode: str - updated mode
            - tool_names: list - updated tool names
            - tool_definitions: list - updated tool definitions
            - system_prompt: str | None - updated system prompt if changed
            - active_skills_content: str - updated skills content
            - conversation_history: list - updated history
        """
        result = {
            "handled": True,
            "mode": mode,
            "tool_names": tool_names,
            "tool_definitions": tool_definitions,
            "system_prompt": None,
            "active_skills_content": active_skills_content,
            "conversation_history": conversation_history,
        }

        # Try core commands first (they may modify mode/tools)
        core_result = await self.core_handler.handle_core_command(
            cmd,
            cmd_args,
            mode,
            tool_names,
            tool_definitions,
            conversation_history,
            active_skills_content,
            build_system_prompt,
            convert_tools_to_definitions,
            sensitive_tools,
        )
        if core_result:
            return core_result

        # Try session commands
        session_result = await self.session_handler.handle_session_command(cmd, cmd_args)
        if session_result:
            return session_result

        # Try config commands
        config_result = await self.config_handler.handle_config_command(cmd, cmd_args)
        if config_result:
            return config_result

        # Unknown command
        console.print(f"[yellow]Unknown command: {cmd}[/yellow]")
        return result
