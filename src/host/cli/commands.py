"""
CLI Slash Commands Handler

Handles all /command processing in the Doraemon CLI.
Extracted from cli.py for better maintainability.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from src.core.plugins import PluginManager
from src.core.themes import ThemeManager
from src.core.input_mode import InputManager, InputMode
from src.core.thinking import ThinkingManager
from src.core.doctor import Doctor
from src.core.workspace import WorkspaceManager
from src.core.model_manager import ModelManager

console = Console()

# Mode colors
MODE_COLORS = {
    "plan": "blue",
    "build": "green",
}


class CommandHandler:
    """Handle slash commands in the CLI."""
    
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
        Handle a slash command.
        
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
        
        if cmd == "help":
            self._show_help()
            
        elif cmd == "init":
            self._handle_init()
            
        elif cmd == "mode":
            if cmd_args:
                new_mode = cmd_args[0].lower()
                if new_mode in MODE_COLORS:
                    result["mode"] = new_mode
                    # Update tools for new mode
                    new_tool_names = self.tool_selector.get_tools_for_mode(new_mode)
                    genai_tools = self.registry.get_genai_tools(new_tool_names)
                    new_tool_definitions = convert_tools_to_definitions(genai_tools)
                    result["tool_names"] = new_tool_names
                    result["tool_definitions"] = new_tool_definitions
                    self.hook_mgr.permission_mode = new_mode
                    result["system_prompt"] = build_system_prompt(new_mode, active_skills_content)
                    console.print(f"[green]Switched to {new_mode} mode ({len(new_tool_definitions)} tools)[/green]")
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
            
        elif cmd == "sessions":
            self._show_sessions()
            
        elif cmd == "resume":
            self._resume_session(cmd_args)
            
        elif cmd == "rename":
            self._rename_session(cmd_args)
            
        elif cmd == "export":
            self._export_session(cmd_args)
            
        elif cmd == "fork":
            self._fork_session()
            
        elif cmd == "checkpoints":
            self._show_checkpoints()
            
        elif cmd == "rewind":
            self._rewind_checkpoint(cmd_args)
            
        elif cmd == "tasks":
            self._show_tasks()
            
        elif cmd == "task":
            self._show_task_output(cmd_args)
            
        elif cmd == "cost":
            self._show_cost()
            
        elif cmd == "agents":
            self._show_agents()
            
        elif cmd == "history":
            self._show_history()
            
        elif cmd == "model":
            self._handle_model(cmd_args)
            
        elif cmd == "plugins":
            self._show_plugins()
            
        elif cmd == "plugin":
            self._handle_plugin(cmd_args)
            
        elif cmd == "theme":
            self._handle_theme(cmd_args)
            
        elif cmd == "vim":
            self._toggle_vim()
            
        elif cmd == "thinking":
            self._toggle_thinking()
            
        elif cmd == "doctor":
            self._run_doctor()
            
        elif cmd == "workspace":
            self._show_workspace()
            
        elif cmd == "add-dir":
            self._add_directory(cmd_args)
            
        else:
            console.print(f"[yellow]Unknown command: {cmd}[/yellow]")
            
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
        console.print("[dim]Put SKILL.md files in .doraemon/skills/<name>/ to add custom skills.[/dim]")
        
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
        
    def _show_sessions(self):
        """List recent sessions."""
        sessions = self.session_mgr.list_sessions(project=self.project, limit=10)
        if sessions:
            table = Table(title="Recent Sessions")
            table.add_column("ID", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Messages", style="yellow")
            table.add_column("Updated")
            for s in sessions:
                updated = datetime.fromtimestamp(s.updated_at).strftime("%Y-%m-%d %H:%M")
                table.add_row(s.id[:8], s.name or "-", str(s.message_count), updated)
            console.print(table)
        else:
            console.print("[dim]No sessions found[/dim]")
            
    def _resume_session(self, cmd_args: list):
        """Resume a session."""
        if cmd_args:
            session_data = self.session_mgr.resume_session(cmd_args[0])
            if session_data:
                console.print(f"[green]Resumed: {session_data.metadata.get_display_name()}[/green]")
            else:
                console.print(f"[red]Session not found: {cmd_args[0]}[/red]")
        else:
            console.print("[yellow]Usage: /resume <session_id or name>[/yellow]")
            
    def _rename_session(self, cmd_args: list):
        """Rename current session."""
        if cmd_args:
            new_name = " ".join(cmd_args)
            self.session_mgr.rename_session(self.ctx.session_id, new_name)
            console.print(f"[green]Session renamed to: {new_name}[/green]")
        else:
            console.print("[yellow]Usage: /rename <new_name>[/yellow]")
            
    def _export_session(self, cmd_args: list):
        """Export session."""
        try:
            path = cmd_args[0] if cmd_args else f"session_{self.ctx.session_id[:8]}.md"
            self.session_mgr.export_session(self.ctx.session_id, format="markdown", path=path)
            console.print(f"[green]Exported to: {path}[/green]")
        except Exception as e:
            console.print(f"[red]Export failed: {e}[/red]")
            
    def _fork_session(self):
        """Fork current session."""
        try:
            forked = self.session_mgr.fork_session(self.ctx.session_id)
            if forked:
                console.print(f"[green]Forked session: {forked.metadata.id}[/green]")
            else:
                console.print("[red]Fork failed[/red]")
        except Exception as e:
            console.print(f"[red]Fork failed: {e}[/red]")
            
    def _show_checkpoints(self):
        """List checkpoints."""
        cps = self.checkpoint_mgr.list_checkpoints(limit=10)
        if cps:
            table = Table(title="Checkpoints")
            table.add_column("ID", style="cyan")
            table.add_column("Time", style="green")
            table.add_column("Files", style="yellow")
            table.add_column("Prompt")
            for cp in cps:
                table.add_row(
                    cp["id"][:8],
                    cp["created_at"].split("T")[1][:8],
                    str(cp["files_count"]),
                    cp["prompt"][:40] + "..." if len(cp["prompt"]) > 40 else cp["prompt"],
                )
            console.print(table)
        else:
            console.print("[dim]No checkpoints yet[/dim]")
            
    def _rewind_checkpoint(self, cmd_args: list):
        """Rewind to checkpoint."""
        try:
            if cmd_args:
                result = self.checkpoint_mgr.rewind(cmd_args[0], mode="code")
            else:
                result = self.checkpoint_mgr.rewind_last(mode="code")
            
            if result:
                console.print(f"[green]Rewound: {len(result['restored_files'])} files restored[/green]")
                if result["failed_files"]:
                    console.print(f"[yellow]Failed: {len(result['failed_files'])} files[/yellow]")
            else:
                console.print("[yellow]No checkpoint to rewind to[/yellow]")
        except Exception as e:
            console.print(f"[red]Rewind failed: {e}[/red]")
            
    def _show_tasks(self):
        """List background tasks."""
        tasks = self.task_mgr.list_tasks(limit=10)
        if tasks:
            table = Table(title="Background Tasks")
            table.add_column("ID", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Status", style="yellow")
            table.add_column("Progress")
            table.add_column("Duration")
            for t in tasks:
                status_color = {
                    "running": "cyan",
                    "completed": "green",
                    "failed": "red",
                    "cancelled": "yellow",
                }.get(t["status"], "white")
                table.add_row(
                    t["id"],
                    t["name"][:20],
                    f"[{status_color}]{t['status']}[/{status_color}]",
                    f"{t['progress']}%",
                    f"{t['duration'] or 0:.1f}s",
                )
            console.print(table)
        else:
            console.print("[dim]No tasks[/dim]")
            
    def _show_task_output(self, cmd_args: list):
        """Show task output."""
        from rich.panel import Panel
        if cmd_args:
            output = self.task_mgr.get_task_output(cmd_args[0])
            if output:
                console.print(Panel(output, title=f"Task {cmd_args[0]}", border_style="cyan"))
            else:
                console.print(f"[red]Task not found: {cmd_args[0]}[/red]")
        else:
            console.print("[yellow]Usage: /task <task_id>[/yellow]")
            
    def _show_cost(self):
        """Show cost statistics."""
        summary = self.cost_tracker.get_cost_summary()
        table = Table(title="Usage & Cost", show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Session", style="green")
        table.add_column("Today", style="yellow")
        
        session = summary["session"]
        today = summary["today"]
        
        table.add_row("Tokens", f"{session['total_tokens']:,}", f"{today['total_tokens']:,}")
        table.add_row("  Input", f"{session['total_input_tokens']:,}", f"{today['total_input_tokens']:,}")
        table.add_row("  Output", f"{session['total_output_tokens']:,}", f"{today['total_output_tokens']:,}")
        table.add_row("Cost (USD)", f"${session['total_cost_usd']:.4f}", f"${today['total_cost_usd']:.4f}")
        table.add_row("Requests", str(session['request_count']), str(today['request_count']))
        
        console.print(table)
        
        if summary["budget"]["warning"]:
            console.print(f"\n[yellow]⚠️ {summary['budget']['warning']}[/yellow]")
            
    def _show_agents(self):
        """Show available subagents."""
        try:
            from src.core.subagents import BUILTIN_AGENTS
            console.print("[bold]Available Subagents:[/bold]")
            for name, config in BUILTIN_AGENTS.items():
                console.print(f"  [cyan]{name}[/cyan]: {config.description}")
        except Exception as e:
            console.print(f"[red]Error loading agents: {e}[/red]")
            
    def _show_history(self):
        """Show command history."""
        recent = self.cmd_history.get_recent(20)
        if recent:
            console.print("[bold]Recent Commands:[/bold]")
            for i, cmd in enumerate(recent, 1):
                console.print(f"  {i}. {cmd[:60]}{'...' if len(cmd) > 60 else ''}")
        else:
            console.print("[dim]No history[/dim]")
            
    def _handle_model(self, cmd_args: list):
        """Handle model switching."""
        model_mgr = ModelManager(default_model=self.model_name)
        if cmd_args:
            new_model = cmd_args[0]
            if model_mgr.switch_model(new_model):
                self.model_name = model_mgr.get_current_model()
                console.print(f"[green]Switched to model: {self.model_name}[/green]")
            else:
                console.print(f"[red]Unknown model: {new_model}[/red]")
                console.print(model_mgr.format_model_list())
        else:
            console.print(model_mgr.format_model_list())
            
    def _show_plugins(self):
        """Show installed plugins."""
        plugin_mgr = PluginManager(project_dir=Path.cwd())
        summary = plugin_mgr.get_summary()
        console.print(f"[bold]Plugins ({summary['enabled']} enabled, {summary['disabled']} disabled):[/bold]")
        for p in summary['plugins']:
            status = "[green]✓[/green]" if p['enabled'] else "[dim]○[/dim]"
            console.print(f"  {status} {p['name']} v{p['version']} ({p['scope']})")
        if not summary['plugins']:
            console.print("[dim]No plugins installed[/dim]")
            
    def _handle_plugin(self, cmd_args: list):
        """Handle plugin management."""
        plugin_mgr = PluginManager(project_dir=Path.cwd())
        if len(cmd_args) >= 2:
            action = cmd_args[0]
            target = cmd_args[1]
            if action == "install":
                result = plugin_mgr.install(target)
                if result:
                    console.print(f"[green]Installed: {result.manifest.name}[/green]")
                else:
                    console.print("[red]Installation failed[/red]")
            elif action == "enable":
                if plugin_mgr.enable(target):
                    console.print(f"[green]Enabled: {target}[/green]")
                else:
                    console.print(f"[red]Plugin not found: {target}[/red]")
            elif action == "disable":
                if plugin_mgr.disable(target):
                    console.print(f"[yellow]Disabled: {target}[/yellow]")
                else:
                    console.print(f"[red]Plugin not found: {target}[/red]")
            elif action == "uninstall":
                if plugin_mgr.uninstall(target):
                    console.print(f"[green]Uninstalled: {target}[/green]")
                else:
                    console.print(f"[red]Uninstall failed: {target}[/red]")
        else:
            console.print("[yellow]Usage: /plugin <install|enable|disable|uninstall> <name>[/yellow]")
            
    def _handle_theme(self, cmd_args: list):
        """Handle theme switching."""
        theme_mgr = ThemeManager()
        if cmd_args:
            theme_name = cmd_args[0]
            if theme_mgr.set_theme(theme_name):
                console.print(f"[green]Theme set to: {theme_name}[/green]")
            else:
                console.print(f"[red]Theme not found: {theme_name}[/red]")
        else:
            console.print(theme_mgr.format_theme_list())
            
    def _toggle_vim(self):
        """Toggle vim mode."""
        input_mgr = InputManager()
        new_mode = input_mgr.toggle_mode()
        console.print(f"[green]Input mode: {new_mode.value}[/green]")
        if new_mode == InputMode.VI:
            console.print("[dim]Press Esc for normal mode, i for insert mode[/dim]")
            
    def _toggle_thinking(self):
        """Toggle extended thinking mode."""
        thinking_mgr = ThinkingManager()
        new_mode = thinking_mgr.toggle_mode()
        indicator = thinking_mgr.get_mode_indicator()
        console.print(f"[green]Thinking mode: {new_mode.value} {indicator}[/green]")
        
    def _run_doctor(self):
        """Run health checks."""
        doctor = Doctor(project_dir=Path.cwd())
        console.print("[bold]Running health checks...[/bold]")
        results = doctor.run_all_checks()
        console.print(doctor.format_results(results))
        
    def _show_workspace(self):
        """Show workspace directories."""
        workspace_mgr = WorkspaceManager(primary_dir=Path.cwd())
        console.print("[bold]Workspace Directories:[/bold]")
        console.print(workspace_mgr.get_summary())
        
    def _add_directory(self, cmd_args: list):
        """Add a directory to workspace."""
        workspace_mgr = WorkspaceManager(primary_dir=Path.cwd())
        if cmd_args:
            dir_path = cmd_args[0]
            alias = cmd_args[1] if len(cmd_args) > 1 else None
            if workspace_mgr.add_directory(dir_path, alias=alias):
                console.print(f"[green]Added: {dir_path}[/green]")
            else:
                console.print(f"[red]Failed to add: {dir_path}[/red]")
        else:
            console.print("[yellow]Usage: /add-dir <path> [alias][/yellow]")
