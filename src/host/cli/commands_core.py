"""
Core CLI Commands Handler

Handles core commands: help, clear, mode, reset, context, tools, debug, init, skills
"""

from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

# Mode colors
MODE_COLORS = {
    "plan": "blue",
    "build": "green",
}


class CoreCommandHandler:
    """Handle core slash commands in the CLI."""

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
        self.permission_mgr = permission_mgr

    async def handle_core_command(
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
    ) -> dict | None:
        """
        Handle core commands.

        Returns:
            dict with updated state or None if command not handled
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

        if cmd == "help":
            self._show_help()

        elif cmd == "init":
            self._handle_init()

        elif cmd == "mode":
            if cmd_args:
                new_mode = cmd_args[0].lower()
                if new_mode in MODE_COLORS:
                    result["mode"] = new_mode
                    new_tool_names = self.tool_selector.get_tools_for_mode(new_mode)
                    genai_tools = self.registry.get_genai_tools(new_tool_names)
                    new_tool_definitions = convert_tools_to_definitions(genai_tools)
                    result["tool_names"] = new_tool_names
                    result["tool_definitions"] = new_tool_definitions
                    self.hook_mgr.permission_mode = new_mode
                    if self.permission_mgr:
                        self.permission_mgr.set_mode(new_mode)
                    result["system_prompt"] = build_system_prompt(new_mode, active_skills_content)
                    console.print(
                        f"[green]Switched to {new_mode} mode ({len(new_tool_definitions)} tools)[/green]"
                    )
                else:
                    console.print(f"[red]Unknown mode: {new_mode}[/red]")
            else:
                console.print(f"Current mode: {mode}")

        elif cmd == "context":
            self._show_context(mode, tool_names)

        elif cmd == "skills":
            self._show_skills()

        elif cmd == "clear":
            self.ctx.clear(keep_summaries=True)
            result["conversation_history"] = []
            console.print("[green]Conversation cleared (summaries preserved)[/green]")

        elif cmd == "compact":
            stats_before = self.ctx.get_context_stats()
            if stats_before["messages"] <= self.ctx.config.keep_recent_messages:
                console.print("[yellow]Not enough messages to compact.[/yellow]")
            else:
                self.ctx._force_summarize()
                stats_after = self.ctx.get_context_stats()
                result["conversation_history"] = []
                console.print(
                    f"[green]Context compacted: {stats_before['messages']} → {stats_after['messages']} messages, "
                    f"{stats_before['estimated_tokens']:,} → {stats_after['estimated_tokens']:,} tokens[/green]"
                )

        elif cmd == "reset":
            self.ctx.reset()
            result["active_skills_content"] = ""
            result["mode"] = "build"
            new_tool_names = self.tool_selector.get_tools_for_mode("build")
            genai_tools = self.registry.get_genai_tools(new_tool_names)
            result["tool_names"] = new_tool_names
            result["tool_definitions"] = convert_tools_to_definitions(genai_tools)
            result["system_prompt"] = build_system_prompt("build", "")
            result["conversation_history"] = []
            console.print("[green]Full reset complete[/green]")

        elif cmd == "tools":
            self._show_tools(mode, tool_names, sensitive_tools)

        elif cmd == "debug":
            self._show_debug(mode, tool_names)

        else:
            return None

        return result

    def _handle_init(self):
        """Initialize project with DORAEMON.md."""
        fname = "DORAEMON.md"
        path = Path.cwd() / fname
        if path.exists():
            console.print(f"[yellow]{fname} already exists.[/yellow]")
            return

        content = """# Doraemon Code Project Rules

Project specific rules for Doraemon Code.

## Tech Stack
- Language: Python 3.10+
- Framework: FastAPI

## Code Style
- 4 space indentation
- Type hints required

## Directory Structure
- src/: Source code
- tests/: Tests
"""
        try:
            path.write_text(content, encoding="utf-8")
            console.print(f"[green]Initialized project. Created {fname}[/green]")
            console.print("[dim]Edit this file to define project-specific rules.[/dim]")
        except Exception as e:
            console.print(f"[red]Failed to create {fname}: {e}[/red]")

    def _show_help(self):
        """Show help text."""
        console.print("""
[bold]Commands:[/bold]
  /init           - Initialize project (create DORAEMON.md)
  /mode <name>    - Switch mode (plan/build)
  /model [name]   - Switch/list AI models
  /context        - Show context/memory statistics
  /skills         - Show loaded skills
  /clear          - Clear conversation (keeps summaries)
  /compact        - Compress context (summarize older messages)
  /reset          - Full reset (clears everything)
  /tools          - List available tools
  /debug          - Show debug info

[bold]Session Commands:[/bold]
  /sessions       - List recent sessions
  /resume <id>    - Resume a session
  /rename <name>  - Rename current session
  /export [path]  - Export conversation
  /fork           - Fork current session

[bold]Checkpoint Commands:[/bold]
  /checkpoints    - List checkpoints
  /rewind [id]    - Rewind to checkpoint (or last)

[bold]Task Commands:[/bold]
  /tasks          - List background tasks
  /task <id>      - Show task output

[bold]Plugin Commands:[/bold]
  /plugins        - List installed plugins
  /plugin install <source> - Install a plugin
  /plugin enable/disable <name> - Enable/disable plugin

[bold]Configuration:[/bold]
  /theme [name]   - Switch/list themes
  /vim            - Toggle vim mode
  /thinking       - Toggle extended thinking mode
  /doctor         - Run health checks
  /workspace      - Show workspace directories
  /add-dir <path> - Add working directory

[bold]Other Commands:[/bold]
  /cost           - Show cost/usage statistics
  /agents         - List available subagents
  /history        - Show command history
  /exit           - Exit

[bold]Shortcuts:[/bold]
  !<cmd>          - Execute shell command directly (Bash mode)

[bold]Modes:[/bold]
  plan   - Analyze requirements, investigate code, create plans (read-only)
  build  - Implement solutions, write code, execute tasks
""")

    def _show_context(self, mode: str, tool_names: list):
        """Show context statistics."""
        stats = self.ctx.get_context_stats()
        table = Table(title="Context Statistics", show_header=False)
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Session ID", stats["session_id"])
        table.add_row("Messages", str(stats["messages"]))
        table.add_row("Summaries", str(stats["summaries"]))
        table.add_row("Total Ever", str(stats["total_messages_ever"]))
        table.add_row("Est. Tokens", f"{stats['estimated_tokens']:,}")
        table.add_row("Last Prompt", f"{stats['last_prompt_tokens']:,}")
        table.add_row("Threshold", f"{stats['threshold_tokens']:,}")
        table.add_row("Usage", f"{stats['usage_percent']}%")
        active = self.skill_mgr.get_active_skills()
        table.add_row("Active Skills", ", ".join(active) if active else "(none)")
        table.add_row("Mode", mode)
        table.add_row("Loaded Tools", f"{len(tool_names)}")
        if stats["needs_summary"]:
            table.add_row("Status", "[yellow]Summary needed[/yellow]")
        console.print(table)

    def _show_skills(self):
        """Show skills information."""
        console.print("[bold]Skills System[/bold]")
        active = self.skill_mgr.get_active_skills()
        if active:
            console.print(f"  [green]Active:[/green] {', '.join(active)}")
        else:
            console.print("  [dim]No skills currently active[/dim]")
        console.print("\n[dim]Skills are loaded automatically based on conversation context.[/dim]")
        console.print(
            "[dim]Put SKILL.md files in .doraemon/skills/<name>/ to add custom skills.[/dim]"
        )

    def _show_tools(self, mode: str, tool_names: list, sensitive_tools: set):
        """Show available tools."""
        categories = self.tool_selector.get_tool_categories()
        console.print(f"[bold]Tools (mode: {mode})[/bold]")

        for cat_name, cat_tools in categories.items():
            if not cat_tools:
                continue
            console.print(f"\n[cyan]{cat_name}:[/cyan]")
            for name in cat_tools:
                in_current = name in tool_names
                marker = "🔒" if name in sensitive_tools else "  "
                status = "[green]✓[/green]" if in_current else "[dim]○[/dim]"
                console.print(f"  {status}{marker} {name}")

        console.print(f"\n[dim]Current mode has {len(tool_names)} tools[/dim]")

    def _show_debug(self, mode: str, tool_names: list):
        """Show debug information."""
        console.print(f"Mode: {mode}")
        console.print(f"Tools: {len(tool_names)} loaded")
        console.print(f"MCP Tools: {self.tool_selector.mcp_tools or '(none)'}")
        console.print(f"Project: {self.project}")
        stats = self.ctx.get_context_stats()
        console.print(f"Context: {stats['messages']} msgs, {stats['summaries']} summaries")
        console.print(f"Checkpoints: {len(self.checkpoint_mgr.checkpoints)}")
        console.print(f"Background Tasks: {self.task_mgr.get_running_count()} running")
