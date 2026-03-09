"""
Session Management CLI Commands Handler

Handles session commands: sessions, resume, rename, export, fork, checkpoints, rewind, tasks, task
"""

from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.host.cli.command_context import CommandContext

console = Console()


class SessionCommandHandler:
    """Handle session management slash commands in the CLI."""

    def __init__(self, cc: CommandContext):
        self.cc = cc
        self.ctx = cc.ctx
        self.tool_selector = cc.tool_selector
        self.registry = cc.registry
        self.skill_mgr = cc.skill_mgr
        self.checkpoint_mgr = cc.checkpoint_mgr
        self.task_mgr = cc.task_mgr
        self.cost_tracker = cc.cost_tracker
        self.cmd_history = cc.cmd_history
        self.session_mgr = cc.session_mgr
        self.hook_mgr = cc.hook_mgr
        self.model_name = cc.model_name
        self.project = cc.project

    async def handle_session_command(
        self,
        cmd: str,
        cmd_args: list[str],
    ) -> dict | None:
        """
        Handle session management commands.

        Returns:
            dict with handled status or None if command not handled
        """
        result = {"handled": True}

        if cmd == "sessions":
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

        else:
            return None

        return result

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
        if cmd_args:
            output = self.task_mgr.get_task_output(cmd_args[0])
            if output:
                console.print(Panel(output, title=f"Task {cmd_args[0]}", border_style="cyan"))
            else:
                console.print(f"[red]Task not found: {cmd_args[0]}[/red]")
        else:
            console.print("[yellow]Usage: /task <task_id>[/yellow]")
