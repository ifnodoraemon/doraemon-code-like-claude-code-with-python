"""
Core CLI Commands Handler

Handles core commands: help, clear, mode, reset, init, commit
"""

import os
import subprocess
from pathlib import Path

from rich.console import Console

from src.core.paths import memory_path
from src.core.rules import create_default_rules
from src.host.cli.command_context import CommandContext
from src.host.cli.command_result import CommandResult
from src.servers.memory import (
    delete_note,
    export_notes,
    get_note,
    get_note_file_path,
    get_user_persona,
    list_notes,
    save_note,
    search_notes,
    update_user_persona,
)

console = Console()

# Mode colors
MODE_COLORS = {
    "plan": "blue",
    "build": "green",
}


class CoreCommandHandler:
    """Handle core slash commands in the CLI."""

    def __init__(self, cc: CommandContext):
        self.cc = cc

    def __getattr__(self, name):
        """Proxy attribute access to the underlying CommandContext."""
        return getattr(self.cc, name)

    async def handle(
        self,
        cmd: str,
        cmd_args: list[str],
        mode: str,
        tool_names: list[str],
        tool_definitions: list,
        conversation_history: list,
        active_skills_content: str,
    ) -> CommandResult | None:
        """Handle core commands."""
        result = CommandResult.default(
            mode=mode,
            tool_names=tool_names,
            tool_definitions=tool_definitions,
            active_skills_content=active_skills_content,
            conversation_history=conversation_history,
        )

        async_handlers = {
            "commit": lambda: self._handle_commit(cmd_args),
            "review": lambda: self._handle_review(cmd_args, conversation_history),
            "memory": lambda: self._handle_memory(cmd_args),
        }
        sync_handlers = {
            "help": self._show_help,
            "init": self._handle_init,
            "mode": lambda: self._handle_mode_command(
                cmd_args,
                result,
                active_skills_content,
            ),
            "clear": lambda: self._handle_clear_command(result),
            "compact": lambda: self._handle_compact_command(result),
            "reset": lambda: self._handle_reset_command(result),
        }

        if cmd in async_handlers:
            async_result = await async_handlers[cmd]()
            return async_result or result
        if cmd in sync_handlers:
            sync_handlers[cmd]()
            return result
        return None

    def _handle_mode_command(
        self,
        cmd_args: list[str],
        result: CommandResult,
        active_skills_content: str,
    ) -> None:
        """Switch mode and refresh tooling."""
        if not cmd_args:
            console.print(f"Current mode: {result.mode}")
            return

        new_mode = cmd_args[0].lower()
        if new_mode not in MODE_COLORS:
            console.print(f"[red]Unknown mode: {new_mode}[/red]")
            return

        result.mode = new_mode
        new_tool_names = self.tool_selector.get_tools_for_mode(new_mode)
        genai_tools = self.registry.get_genai_tools(new_tool_names)
        new_tool_definitions = self.convert_tools_to_definitions(genai_tools)
        result.tool_names = new_tool_names
        result.tool_definitions = new_tool_definitions
        result.system_prompt = self.build_system_prompt(new_mode, active_skills_content)
        self.hook_mgr.permission_mode = new_mode
        if self.permission_mgr:
            self.permission_mgr.set_mode(new_mode)
        console.print(
            f"[green]Switched to {new_mode} mode ({len(new_tool_definitions)} tools)[/green]"
        )

    def _handle_clear_command(self, result: CommandResult) -> None:
        """Clear conversation while preserving summaries."""
        self.ctx.clear(keep_summaries=True)
        result.conversation_history = []
        console.print("[green]Conversation cleared (summaries preserved)[/green]")

    def _handle_compact_command(self, result: CommandResult) -> None:
        """Compact context and clear conversation history."""
        stats_before = self.ctx.get_context_stats()
        if stats_before["messages"] <= self.ctx.config.keep_recent_messages:
            console.print("[yellow]Not enough messages to compact.[/yellow]")
            return

        self.ctx._force_summarize()
        stats_after = self.ctx.get_context_stats()
        result.conversation_history = []
        console.print(
            f"[green]Context compacted: {stats_before['messages']} → {stats_after['messages']} messages, "
            f"{stats_before['estimated_tokens']:,} → {stats_after['estimated_tokens']:,} tokens[/green]"
        )

    def _handle_reset_command(
        self,
        result: CommandResult,
    ) -> None:
        """Reset context and restore build-mode defaults."""
        self.ctx.reset()
        result.active_skills_content = ""
        result.mode = "build"
        new_tool_names = self.tool_selector.get_tools_for_mode("build")
        genai_tools = self.registry.get_genai_tools(new_tool_names)
        result.tool_names = new_tool_names
        result.tool_definitions = self.convert_tools_to_definitions(genai_tools)
        result.system_prompt = self.build_system_prompt("build", "")
        result.conversation_history = []
        console.print("[green]Full reset complete[/green]")

    def _handle_init(self):
        """Initialize project with AGENTS.md."""
        fname = "AGENTS.md"
        path = Path.cwd() / fname
        if path.exists():
            console.print(f"[yellow]{fname} already exists.[/yellow]")
            return

        try:
            create_default_rules(Path.cwd())
            console.print(f"[green]Initialized project. Created {fname}[/green]")
            console.print("[dim]Edit this file to define project-specific rules.[/dim]")
        except Exception as e:
            console.print(f"[red]Failed to create {fname}: {e}[/red]")

    def _show_help(self):
        """Show help text."""
        console.print("""
[bold]Commands:[/bold]
  /help
  /init
  /mode <plan|build>
  /clear
  /compact
  /reset
  /memory
  /commit <message>
  /commit --amend
  /review [n|goto <n>|search <q>|all]
  /exit

[bold]Shell:[/bold]
  !<cmd>
""")

    async def _handle_commit(self, cmd_args: list[str]):
        if not self._command_available("git", ["rev-parse", "--git-dir"]):
            console.print("[red]Error: Not in a git repository[/red]")
            return

        amend = "--amend" in cmd_args
        message_args = [a for a in cmd_args if a != "--amend"]
        status_output = self._run_command("git", ["status", "--porcelain"])
        if not status_output.strip():
            console.print("[yellow]No changes to commit.[/yellow]")
            return
        commit_msg = " ".join(message_args).strip()
        if not amend and not commit_msg:
            console.print("[red]Usage: /commit <message> or /commit --amend[/red]")
            return

        from rich.prompt import Confirm

        if not Confirm.ask("Proceed with commit?", default=True):
            console.print("[yellow]Commit cancelled.[/yellow]")
            return

        stage_result = self._run_command("git", ["add", "-A"])
        if stage_result is None:
            console.print("[red]Error staging files[/red]")
            return

        commit_args = ["commit"]
        if amend:
            commit_args.append("--amend")
            if commit_msg:
                commit_args.extend(["-m", commit_msg])
        else:
            commit_args.extend(["-m", commit_msg])

        result = self._run_command("git", commit_args)
        if result is not None:
            console.print("\n[green]Committed successfully![/green]")
            log_output = self._run_command("git", ["log", "-1", "--oneline"])
            if log_output:
                console.print(f"[dim]{log_output}[/dim]")
        else:
            console.print("[red]Commit failed. Check git status.[/red]")

    def _command_available(self, program: str, args: list[str]) -> bool:
        """Check whether a command succeeds."""
        return self._run_command(program, args, timeout=5) is not None

    def _run_command(
        self,
        program: str,
        args: list[str],
        *,
        timeout: int = 30,
    ) -> str | None:
        """Run a command and return trimmed stdout on success."""
        try:
            result = subprocess.run(
                [program] + [str(arg) for arg in args],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except Exception:
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    async def _handle_review(
        self,
        cmd_args: list[str],
        conversation_history: list,
    ) -> CommandResult | None:
        """Handle /review command - view and navigate conversation history."""

        messages = self.ctx.messages

        if not messages:
            console.print("[yellow]No conversation history yet.[/yellow]")
            return None

        if not cmd_args:
            self._show_conversation_history(messages, limit=10)
            return None

        subcommand = cmd_args[0].lower()
        if subcommand.isdigit():
            self._show_conversation_history(messages, limit=int(subcommand))
            return None

        if subcommand == "goto":
            if len(cmd_args) < 2 or not cmd_args[1].isdigit():
                console.print("[red]Usage: /review goto <turn_number>[/red]")
                return None
            return self._goto_turn(messages, int(cmd_args[1]), conversation_history)
        if subcommand == "search":
            if len(cmd_args) < 2:
                console.print("[red]Usage: /review search <keyword>[/red]")
                return None
            self._search_conversation(messages, " ".join(cmd_args[1:]))
            return None
        if subcommand == "all":
            self._show_conversation_history(messages, limit=None)
            return None

        if subcommand not in {"goto", "search", "all"}:
            console.print(f"[red]Unknown subcommand: {subcommand}[/red]")
            console.print("[dim]Usage: /review [n|goto <n>|search <q>|all][/dim]")
            return None
        return None

    def _show_conversation_history(self, messages: list, limit: int | None = 10):
        """Display conversation history with turn numbers."""

        if limit:
            display_messages = messages[-limit * 2 :]  # 2 messages per turn (user + assistant)
            start_idx = max(0, len(messages) - limit * 2)
        else:
            display_messages = messages
            start_idx = 0

        console.print(
            f"\n[bold cyan]Conversation History[/bold cyan] ({len(messages)} messages total)\n"
        )

        turn_num = start_idx // 2 + 1
        for _i, msg in enumerate(display_messages):
            from datetime import datetime as dt

            timestamp = dt.fromtimestamp(msg.timestamp).strftime("%H:%M:%S")

            if msg.role == "user":
                console.print(f"[bold yellow]Turn {turn_num}[/bold yellow] [dim]{timestamp}[/dim]")
                console.print(
                    f"[cyan]You:[/cyan] {msg.content[:200]}{'...' if len(msg.content) > 200 else ''}"
                )
            elif msg.role in ("assistant", "model"):
                content_preview = msg.content[:300] if msg.content else "(no content)"
                console.print(
                    f"[green]AI:[/green] {content_preview}{'...' if len(msg.content) > 300 else ''}"
                )
                console.print()
                turn_num += 1

        console.print("[dim]Use /review goto <turn> to go back to a specific turn[/dim]")

    def _goto_turn(
        self,
        messages: list,
        target_turn: int,
        conversation_history: list,
    ) -> CommandResult:
        """Go back to a specific turn, discarding later messages."""
        target_idx = target_turn * 2

        if target_idx > len(messages):
            console.print(
                f"[red]Turn {target_turn} doesn't exist. Max turn: {len(messages) // 2}[/red]"
            )
            return CommandResult()

        if target_idx <= 0:
            console.print("[red]Turn number must be positive.[/red]")
            return CommandResult()

        from rich.prompt import Confirm

        msgs_to_remove = len(messages) - target_idx
        if msgs_to_remove > 0:
            if not Confirm.ask(
                f"This will remove {msgs_to_remove} messages (turns {target_turn + 1} onwards). Continue?",
                default=False,
            ):
                console.print("[yellow]Cancelled.[/yellow]")
                return CommandResult()

            self.ctx.messages = messages[:target_idx]
            self.ctx._auto_save()
            conversation_history.clear()
            console.print(f"[green]Returned to turn {target_turn}. Later messages removed.[/green]")
        else:
            console.print(f"[yellow]Already at or before turn {target_turn}.[/yellow]")

        return CommandResult(conversation_history=[])

    def _search_conversation(self, messages: list, query: str):
        """Search conversation history for a keyword."""
        query_lower = query.lower()
        results = []

        for i, msg in enumerate(messages):
            if query_lower in msg.content.lower():
                turn_num = i // 2 + 1
                results.append((turn_num, msg))

        if not results:
            console.print(f"[yellow]No matches found for '{query}'[/yellow]")
            return

        console.print(
            f"\n[bold cyan]Search Results for '{query}'[/bold cyan] ({len(results)} matches)\n"
        )

        for turn_num, msg in results[:20]:  # Limit to 20 results
            role_color = "cyan" if msg.role == "user" else "green"
            role_name = "You" if msg.role == "user" else "AI"

            # Highlight the match in context
            content = msg.content
            idx = content.lower().find(query_lower)
            if idx != -1:
                start = max(0, idx - 50)
                end = min(len(content), idx + len(query) + 50)
                snippet = content[start:end]
                if start > 0:
                    snippet = "..." + snippet
                if end < len(content):
                    snippet = snippet + "..."
            else:
                snippet = content[:100]

            console.print(
                f"[bold]Turn {turn_num}[/bold] [{role_color}]{role_name}[/{role_color}]: {snippet}"
            )

        console.print("\n[dim]Use /review goto <turn> to jump to a specific turn[/dim]")

    async def _handle_memory(self, cmd_args: list[str]):
        """Handle /memory command - edit MEMORY.md files and inspect note memory."""
        project_memory = memory_path()

        if not cmd_args or cmd_args[0] == "project":
            self._open_memory_file(project_memory)
            return None

        handlers = {
            "show": lambda: self._show_project_memory(project_memory),
            "notes": self._list_project_notes,
            "search": lambda: self._search_project_notes(cmd_args),
            "save": lambda: self._save_project_note(cmd_args),
            "open": lambda: self._open_project_note(cmd_args),
            "delete": lambda: self._delete_project_note(cmd_args),
            "export": lambda: self._export_project_notes(cmd_args),
            "persona": lambda: self._handle_memory_persona(cmd_args),
        }
        handler = handlers.get(cmd_args[0])
        if handler is None:
            console.print(
                "[red]Usage: /memory [project|show|notes|search <query>|save <title> <content>|open <title>|delete <title>|export [path]|persona ...][/red]"
            )
            return None
        handler()
        return None

    def _open_with_editor(self, target: Path, saved_message: str) -> None:
        """Open a file in `$EDITOR` with consistent fallback handling."""
        editor = os.getenv("EDITOR", "nano")
        console.print(f"[dim]Opening {target} in {editor}...[/dim]")
        try:
            subprocess.run([editor, str(target)], timeout=600)
            console.print(f"[green]{saved_message}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to open editor: {e}[/red]")
            console.print(f"[dim]Edit manually: {target}[/dim]")

    def _open_memory_file(self, target: Path) -> None:
        """Open project MEMORY.md, creating it if needed."""
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text("# Memory\n\nAdd notes here that should persist across sessions.\n")
        self._open_with_editor(target, "Memory file saved.")

    def _show_project_memory(self, project_memory: Path) -> None:
        """Display project memory contents."""
        console.print("[bold cyan]Project Memory (.agent/MEMORY.md):[/bold cyan]")
        if project_memory.exists():
            console.print(project_memory.read_text())
        else:
            console.print("[dim](not found)[/dim]")

    def _list_project_notes(self) -> None:
        """List project note memory."""
        console.print(list_notes(collection_name=self.project))

    def _search_project_notes(self, cmd_args: list[str]) -> None:
        """Search project note memory."""
        if len(cmd_args) < 2:
            console.print("[red]Usage: /memory search <query>[/red]")
            return
        console.print(search_notes(query=" ".join(cmd_args[1:]), collection_name=self.project))

    def _save_project_note(self, cmd_args: list[str]) -> None:
        """Save a project note."""
        if len(cmd_args) < 3:
            console.print("[red]Usage: /memory save <title> <content>[/red]")
            return
        console.print(
            save_note(
                title=cmd_args[1],
                content=" ".join(cmd_args[2:]),
                collection_name=self.project,
            )
        )

    def _open_project_note(self, cmd_args: list[str]) -> None:
        """Open a project note in `$EDITOR`."""
        if len(cmd_args) < 2:
            console.print("[red]Usage: /memory open <title>[/red]")
            return
        title = " ".join(cmd_args[1:])
        note_path = Path(get_note_file_path(title=title, collection_name=self.project))
        if not note_path.exists():
            console.print(get_note(title=title, collection_name=self.project))
            return
        self._open_with_editor(note_path, "Note file saved.")

    def _delete_project_note(self, cmd_args: list[str]) -> None:
        """Delete a project note."""
        if len(cmd_args) < 2:
            console.print("[red]Usage: /memory delete <title>[/red]")
            return
        console.print(delete_note(title=" ".join(cmd_args[1:]), collection_name=self.project))

    def _export_project_notes(self, cmd_args: list[str]) -> None:
        """Export project notes as JSON."""
        export_data = export_notes(collection_name=self.project, export_format="json")
        if len(cmd_args) == 1:
            console.print(export_data)
            return
        output_path = Path(cmd_args[1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(export_data, encoding="utf-8")
        console.print(f"[green]Exported notes to {output_path}[/green]")

    def _handle_memory_persona(self, cmd_args: list[str]) -> None:
        """Handle `/memory persona`."""
        if len(cmd_args) == 1 or cmd_args[1] == "show":
            console.print(get_user_persona())
            return
        if len(cmd_args) >= 4 and cmd_args[1] == "set":
            console.print(update_user_persona(key=cmd_args[2], value=" ".join(cmd_args[3:])))
            return
        console.print("[red]Usage: /memory persona [show|set <key> <value>][/red]")
