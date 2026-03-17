"""CLI slash command router."""

from rich.console import Console

from src.host.cli.command_context import CommandContext
from src.host.cli.command_result import CommandResult
from src.host.cli.commands_config import ConfigCommandHandler
from src.host.cli.commands_core import CoreCommandHandler
from src.host.cli.commands_session import SessionCommandHandler

console = Console()


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
    ) -> CommandResult:
        """Handle a slash command by delegating to the appropriate handler."""
        fallback = CommandResult.default(
            mode=mode,
            tool_names=tool_names,
            tool_definitions=tool_definitions,
            active_skills_content=active_skills_content,
            conversation_history=conversation_history,
        )

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
        )
        if core_result:
            return core_result

        session_result = await self.session_handler.handle_session_command(cmd, cmd_args)
        if session_result:
            return session_result

        config_result = await self.config_handler.handle_config_command(cmd, cmd_args)
        if config_result:
            return config_result

        console.print(f"[yellow]Unknown command: {cmd}[/yellow]")
        return fallback
