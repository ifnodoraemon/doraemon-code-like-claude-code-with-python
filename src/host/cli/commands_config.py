"""
Configuration CLI Commands Handler

Handles configuration commands: model, theme, vim, thinking, doctor, workspace, add-dir, cost, agents, history, plugins, plugin
"""

from pathlib import Path
from rich.console import Console
from rich.table import Table

from src.core.doctor import Doctor
from src.core.input_mode import InputManager, InputMode
from src.core.model_manager import ModelManager
from src.core.plugins import PluginManager
from src.core.themes import ThemeManager
from src.core.thinking import ThinkingManager
from src.core.workspace import WorkspaceManager

console = Console()


class ConfigCommandHandler:
    """Handle configuration slash commands in the CLI."""

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

    async def handle_config_command(
        self,
        cmd: str,
        cmd_args: list[str],
    ) -> dict | None:
        """
        Handle configuration commands.

        Returns:
            dict with handled status or None if command not handled
        """
        result = {"handled": True}

        if cmd == "model":
            self._handle_model(cmd_args)

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

        elif cmd == "cost":
            self._show_cost()

        elif cmd == "agents":
            self._show_agents()

        elif cmd == "history":
            self._show_history()

        elif cmd == "plugins":
            self._show_plugins()

        elif cmd == "plugin":
            self._handle_plugin(cmd_args)

        else:
            return None

        return result

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
