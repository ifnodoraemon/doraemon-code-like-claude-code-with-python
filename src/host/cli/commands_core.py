"""
Core CLI Commands Handler

Handles core commands: help, clear, mode, reset, context, tools, debug, init, skills, commit
"""

import subprocess
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.core.paths import config_path, memory_path, skills_dir
from src.core.rules import create_default_rules
from src.host.cli.command_context import CommandContext

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
        # Shortcut aliases for backward compatibility
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
        self.permission_mgr = cc.permission_mgr


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

        elif cmd == "commit":
            await self._handle_commit(cmd_args)

        elif cmd == "review-pr":
            await self._handle_review_pr(cmd_args)

        elif cmd == "review":
            result = await self._handle_review(cmd_args, conversation_history)

        elif cmd == "status":
            self._show_status(mode, tool_names)

        elif cmd == "config":
            await self._handle_config(cmd_args)

        elif cmd == "memory":
            await self._handle_memory(cmd_args)

        elif cmd == "doctor":
            self._run_doctor()

        else:
            return None

        return result

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
  /help           - Show this help
  /init           - Initialize project (create AGENTS.md)
  /mode <name>    - Switch mode (plan/build)
  /model [name]   - Switch/list AI models
  /status         - Show system status
  /config         - Configure settings
  /context        - Show context/memory statistics
  /skills         - Show loaded skills
  /clear          - Clear conversation (keeps summaries)
  /compact        - Compress context (summarize older messages)
  /reset          - Full reset (clears everything)
  /tools          - List available tools
  /debug          - Show debug info
  /doctor         - Run diagnostic checks
  /memory         - Edit MEMORY.md files

[bold]Git Commands:[/bold]
  /commit [msg]   - Smart commit (auto-generates message if not provided)
  /commit --amend - Amend the last commit
  /review-pr [n]  - View PR details (current branch or by number)
  /review-pr --diff - Show PR with diff

[bold]Conversation History:[/bold]
  /review         - Show recent conversation turns
  /review <n>     - Show last n turns
  /review goto <n> - Go back to turn n (discard later messages)
  /review search <q> - Search conversation for keyword
  /review all     - Show all messages

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
            f"[dim]Put SKILL.md files in {skills_dir()}/<name>/ to add custom skills.[/dim]"
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

    async def _handle_commit(self, cmd_args: list[str]):
        """
        Handle /commit command - smart git commit workflow.

        Usage:
            /commit              - Auto-generate commit message from changes
            /commit <message>    - Use provided commit message
            /commit --amend      - Amend the last commit
        """
        # Check if in a git repo
        if not self._is_git_repo():
            console.print("[red]Error: Not in a git repository[/red]")
            return

        # Parse arguments
        amend = "--amend" in cmd_args
        message_args = [a for a in cmd_args if a != "--amend"]
        custom_message = " ".join(message_args) if message_args else None

        # Get git status
        status_output = self._run_git(["status", "--porcelain"])
        if not status_output.strip():
            console.print("[yellow]No changes to commit.[/yellow]")
            return

        # Show current changes
        console.print("\n[bold cyan]Changes to commit:[/bold cyan]")
        diff_stat = self._run_git(["diff", "--stat", "HEAD"])
        if diff_stat:
            console.print(diff_stat)
        else:
            # Show untracked files
            console.print(status_output)

        # Get detailed diff for message generation
        diff_output = self._run_git(["diff", "HEAD"])
        if not diff_output:
            diff_output = self._run_git(["diff", "--cached"])

        # Generate or use commit message
        if custom_message:
            commit_msg = custom_message
        else:
            commit_msg = self._generate_commit_message(status_output, diff_output)

        # Show the commit message
        console.print(Panel(commit_msg, title="Commit Message", border_style="green"))

        # Ask for confirmation
        from rich.prompt import Confirm
        if not Confirm.ask("Proceed with commit?", default=True):
            console.print("[yellow]Commit cancelled.[/yellow]")
            return

        # Stage all changes
        stage_result = self._run_git(["add", "-A"])
        if stage_result is None:
            console.print("[red]Error staging files[/red]")
            return

        # Commit
        commit_args = ["commit", "-m", commit_msg]
        if amend:
            commit_args.append("--amend")

        result = self._run_git(commit_args)
        if result is not None:
            console.print("\n[green]Committed successfully![/green]")
            # Show the commit
            log_output = self._run_git(["log", "-1", "--oneline"])
            if log_output:
                console.print(f"[dim]{log_output}[/dim]")
        else:
            console.print("[red]Commit failed. Check git status.[/red]")

    def _is_git_repo(self) -> bool:
        """Check if current directory is a git repository."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _run_git(self, args: list[str]) -> str | None:
        """Run a git command and return output."""
        try:
            result = subprocess.run(
                ["git"] + [str(a) for a in args],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return None
        except Exception:
            return None

    def _generate_commit_message(self, status: str, diff: str) -> str:
        """Generate a commit message based on changes."""
        lines = status.strip().split("\n")

        # Categorize changes
        added = []
        modified = []
        deleted = []
        renamed = []

        for line in lines:
            if not line.strip():
                continue
            status_code = line[:2]
            file_path = line[3:].strip()

            if status_code.startswith("A") or status_code == "??":
                added.append(file_path)
            elif status_code.startswith("M") or (len(status_code) > 1 and status_code[1] == "M"):
                modified.append(file_path)
            elif status_code.startswith("D"):
                deleted.append(file_path)
            elif status_code.startswith("R"):
                renamed.append(file_path)

        # Determine commit type
        if added and not modified and not deleted:
            prefix = "feat"
            action = "add"
        elif deleted and not added and not modified:
            prefix = "chore"
            action = "remove"
        elif modified and not added and not deleted:
            # Check if it's a fix or feature
            if diff and ("fix" in diff.lower() or "bug" in diff.lower()):
                prefix = "fix"
                action = "fix"
            else:
                prefix = "refactor"
                action = "update"
        else:
            prefix = "chore"
            action = "update"

        # Generate message
        all_files = added + modified + deleted + renamed
        if len(all_files) == 1:
            file_name = Path(all_files[0]).name
            return f"{prefix}: {action} {file_name}"
        elif len(all_files) <= 3:
            file_names = ", ".join(Path(f).name for f in all_files)
            return f"{prefix}: {action} {file_names}"
        else:
            # Group by directory
            dirs = {str(Path(f).parent) for f in all_files if Path(f).parent != Path(".")}
            if dirs:
                dir_name = list(dirs)[0] if len(dirs) == 1 else "multiple files"
                return f"{prefix}: {action} {dir_name} ({len(all_files)} files)"
            return f"{prefix}: {action} {len(all_files)} files"

    async def _handle_review_pr(self, cmd_args: list[str]):
        """
        Handle /review-pr command - view and review Pull Requests.

        Usage:
            /review-pr           - View PR for current branch
            /review-pr <number>  - View specific PR by number
            /review-pr --diff    - Show PR diff
            /review-pr --files   - Show changed files only
        """
        # Check if gh CLI is available
        if not self._is_gh_available():
            console.print("[red]Error: GitHub CLI (gh) is not installed.[/red]")
            console.print("[dim]Install from: https://cli.github.com/[/dim]")
            return

        # Parse arguments
        show_diff = "--diff" in cmd_args
        show_files = "--files" in cmd_args
        pr_args = [a for a in cmd_args if not a.startswith("--")]
        pr_number = pr_args[0] if pr_args else None

        # Get PR info
        view_args = ["pr", "view"]
        if pr_number:
            view_args.append(pr_number)

        pr_info = self._run_gh(view_args)
        if not pr_info:
            if pr_number:
                console.print(f"[red]Error: PR #{pr_number} not found[/red]")
            else:
                console.print("[yellow]No PR found for current branch.[/yellow]")
                console.print("[dim]Use /review-pr <number> to view a specific PR.[/dim]")
            return

        # Display PR info
        console.print(Panel(pr_info, title="Pull Request", border_style="cyan"))

        # Show diff if requested
        if show_diff:
            diff_args = ["pr", "diff"]
            if pr_number:
                diff_args.append(pr_number)
            diff_output = self._run_gh(diff_args)
            if diff_output:
                console.print("\n[bold cyan]Diff:[/bold cyan]")
                # Truncate if too long
                if len(diff_output) > 5000:
                    console.print(diff_output[:5000])
                    console.print(f"\n[dim]... (truncated, {len(diff_output)} chars total)[/dim]")
                else:
                    console.print(diff_output)

        # Show files if requested
        if show_files:
            files_args = ["pr", "diff", "--stat"]
            if pr_number:
                files_args.append(pr_number)
            files_output = self._run_gh(files_args)
            if files_output:
                console.print("\n[bold cyan]Changed Files:[/bold cyan]")
                console.print(files_output)

    def _is_gh_available(self) -> bool:
        """Check if GitHub CLI is available."""
        try:
            result = subprocess.run(
                ["gh", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _run_gh(self, args: list[str]) -> str | None:
        """Run a gh command and return output."""
        try:
            result = subprocess.run(
                ["gh"] + args,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return None
        except Exception:
            return None

    async def _handle_review(self, cmd_args: list[str], conversation_history: list) -> dict | None:
        """
        Handle /review command - view and navigate conversation history.

        Usage:
            /review              - Show recent conversation turns
            /review <n>          - Show last n turns
            /review goto <n>     - Go back to turn n (discard later messages)
            /review search <q>   - Search conversation for keyword
        """

        messages = self.ctx.messages

        if not messages:
            console.print("[yellow]No conversation history yet.[/yellow]")
            return None

        # Parse subcommand
        if not cmd_args:
            # Default: show last 10 turns
            self._show_conversation_history(messages, limit=10)
            return None

        subcommand = cmd_args[0].lower()

        if subcommand.isdigit():
            # /review <n> - show last n turns
            limit = int(subcommand)
            self._show_conversation_history(messages, limit=limit)
            return None

        elif subcommand == "goto":
            # /review goto <n> - go back to turn n
            if len(cmd_args) < 2 or not cmd_args[1].isdigit():
                console.print("[red]Usage: /review goto <turn_number>[/red]")
                return None

            target_turn = int(cmd_args[1])
            return self._goto_turn(messages, target_turn, conversation_history)

        elif subcommand == "search":
            # /review search <query>
            if len(cmd_args) < 2:
                console.print("[red]Usage: /review search <keyword>[/red]")
                return None

            query = " ".join(cmd_args[1:])
            self._search_conversation(messages, query)
            return None

        elif subcommand == "all":
            # /review all - show all messages
            self._show_conversation_history(messages, limit=None)
            return None

        else:
            console.print(f"[red]Unknown subcommand: {subcommand}[/red]")
            console.print("[dim]Usage: /review [n|goto <n>|search <q>|all][/dim]")
            return None

    def _show_conversation_history(self, messages: list, limit: int | None = 10):
        """Display conversation history with turn numbers."""

        if limit:
            display_messages = messages[-limit * 2:]  # 2 messages per turn (user + assistant)
            start_idx = max(0, len(messages) - limit * 2)
        else:
            display_messages = messages
            start_idx = 0

        console.print(f"\n[bold cyan]Conversation History[/bold cyan] ({len(messages)} messages total)\n")

        turn_num = start_idx // 2 + 1
        for _i, msg in enumerate(display_messages):
            from datetime import datetime as dt
            timestamp = dt.fromtimestamp(msg.timestamp).strftime("%H:%M:%S")

            if msg.role == "user":
                console.print(f"[bold yellow]Turn {turn_num}[/bold yellow] [dim]{timestamp}[/dim]")
                console.print(f"[cyan]You:[/cyan] {msg.content[:200]}{'...' if len(msg.content) > 200 else ''}")
            elif msg.role in ("assistant", "model"):
                content_preview = msg.content[:300] if msg.content else "(no content)"
                console.print(f"[green]AI:[/green] {content_preview}{'...' if len(msg.content) > 300 else ''}")
                console.print()
                turn_num += 1

        console.print("[dim]Use /review goto <turn> to go back to a specific turn[/dim]")

    def _goto_turn(self, messages: list, target_turn: int, conversation_history: list) -> dict:
        """Go back to a specific turn, discarding later messages."""
        # Calculate message index (2 messages per turn: user + assistant)
        target_idx = target_turn * 2

        if target_idx > len(messages):
            console.print(f"[red]Turn {target_turn} doesn't exist. Max turn: {len(messages) // 2}[/red]")
            return self._default_result()

        if target_idx <= 0:
            console.print("[red]Turn number must be positive.[/red]")
            return self._default_result()

        # Confirm with user
        from rich.prompt import Confirm
        msgs_to_remove = len(messages) - target_idx
        if msgs_to_remove > 0:
            if not Confirm.ask(
                f"This will remove {msgs_to_remove} messages (turns {target_turn + 1} onwards). Continue?",
                default=False
            ):
                console.print("[yellow]Cancelled.[/yellow]")
                return {"handled": True}

            # Truncate messages
            self.ctx.messages = messages[:target_idx]
            self.ctx._auto_save()

            # Clear and rebuild conversation_history
            conversation_history.clear()

            console.print(f"[green]Returned to turn {target_turn}. Later messages removed.[/green]")
        else:
            console.print(f"[yellow]Already at or before turn {target_turn}.[/yellow]")

        return {
            "handled": True,
            "conversation_history": [],  # Signal to rebuild from ctx
        }

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

        console.print(f"\n[bold cyan]Search Results for '{query}'[/bold cyan] ({len(results)} matches)\n")

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

            console.print(f"[bold]Turn {turn_num}[/bold] [{role_color}]{role_name}[/{role_color}]: {snippet}")

        console.print("\n[dim]Use /review goto <turn> to jump to a specific turn[/dim]")

    def _show_status(self, mode: str, tool_names: list):
        """Show system status information."""
        import os

        table = Table(title="System Status", show_header=False)
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")

        # Session info
        table.add_row("Session ID", self.ctx.session_id)
        table.add_row("Project", self.project)
        table.add_row("Mode", mode)

        # Model info
        model = os.getenv("DORAEMON_MODEL", "gemini-3-pro-preview")
        table.add_row("Model", model)

        gateway = os.getenv("DORAEMON_GATEWAY_URL")
        if gateway:
            table.add_row("Gateway", gateway)
        else:
            table.add_row("Mode", "Direct API")

        # Context stats
        stats = self.ctx.get_context_stats()
        table.add_row("Messages", str(stats["messages"]))
        table.add_row("Summaries", str(stats["summaries"]))
        table.add_row("Est. Tokens", f"{stats['estimated_tokens']:,}")
        table.add_row("Context Usage", f"{stats['usage_percent']}%")

        # Tools
        table.add_row("Tools Loaded", str(len(tool_names)))

        # Cost
        cost_stats = self.cost_tracker.get_stats()
        table.add_row("Session Cost", f"${cost_stats.get('session_cost', 0):.4f}")

        console.print(table)

    async def _handle_config(self, cmd_args: list[str]):
        """
        Handle /config command - interactive configuration.

        Usage:
            /config              - Show current config
            /config set <k> <v>  - Set a config value
            /config model        - Change model interactively
        """
        import json
        import os

        config_file = config_path()
        config_file.parent.mkdir(parents=True, exist_ok=True)

        # Load config
        if config_file.exists():
            config_data = json.loads(config_file.read_text())
        else:
            config_data = {}

        if not cmd_args:
            # Show current config
            table = Table(title="Configuration", show_header=True)
            table.add_column("Key", style="cyan")
            table.add_column("Value", style="green")
            table.add_column("Source", style="dim")

            # Show env vars and config
            table.add_row("model", os.getenv("DORAEMON_MODEL", "gemini-3-pro-preview"), "env")
            table.add_row("gateway_url", os.getenv("DORAEMON_GATEWAY_URL", "(not set)"), "env")
            table.add_row("daily_budget", os.getenv("DORAEMON_DAILY_BUDGET", "0"), "env")
            table.add_row("session_budget", os.getenv("DORAEMON_SESSION_BUDGET", "0"), "env")

            for k, v in config_data.items():
                table.add_row(k, str(v), "config.json")

            console.print(table)
            console.print("\n[dim]Use /config set <key> <value> to change settings[/dim]")
            return

        subcommand = cmd_args[0].lower()

        if subcommand == "set" and len(cmd_args) >= 3:
            key = cmd_args[1]
            value = " ".join(cmd_args[2:])
            config_data[key] = value
            config_file.write_text(json.dumps(config_data, indent=2))
            console.print(f"[green]Set {key} = {value}[/green]")

        elif subcommand == "model":
            # Interactive model selection
            from rich.prompt import Prompt
            models = [
                "gemini-3-pro-preview",
                "gemini-2.5-flash",
                "gpt-4o",
                "gpt-4o-mini",
                "claude-sonnet-4-5-20250929",
                "claude-opus-4-6",
            ]
            console.print("[bold]Available Models:[/bold]")
            for i, m in enumerate(models, 1):
                console.print(f"  {i}. {m}")
            choice = Prompt.ask("Select model (number or name)", default="1")
            if choice.isdigit() and 1 <= int(choice) <= len(models):
                selected = models[int(choice) - 1]
            else:
                selected = choice
            console.print(f"[green]Selected: {selected}[/green]")
            console.print(f"[dim]Set DORAEMON_MODEL={selected} in your environment[/dim]")

        else:
            console.print("[red]Usage: /config [set <key> <value> | model][/red]")

    async def _handle_memory(self, cmd_args: list[str]):
        """
        Handle /memory command - edit MEMORY.md files.

        Usage:
            /memory              - Edit project MEMORY.md
            /memory show         - Show current memory content
        """
        import os

        project_memory = memory_path()

        if not cmd_args or cmd_args[0] == "project":
            target = project_memory
            target.parent.mkdir(parents=True, exist_ok=True)
        elif cmd_args[0] == "show":
            console.print("[bold cyan]Project Memory (.agent/MEMORY.md):[/bold cyan]")
            if project_memory.exists():
                console.print(project_memory.read_text())
            else:
                console.print("[dim](not found)[/dim]")
            return
        else:
            console.print("[red]Usage: /memory [project|show][/red]")
            return

        # Create file if not exists
        if not target.exists():
            target.write_text("# Memory\n\nAdd notes here that should persist across sessions.\n")

        # Open in editor
        editor = os.getenv("EDITOR", "nano")
        console.print(f"[dim]Opening {target} in {editor}...[/dim]")

        import subprocess
        try:
            subprocess.run([editor, str(target)], timeout=600)
            console.print("[green]Memory file saved.[/green]")
        except Exception as e:
            console.print(f"[red]Failed to open editor: {e}[/red]")
            console.print(f"[dim]Edit manually: {target}[/dim]")

    def _run_doctor(self):
        """Run diagnostic checks (same as CLI doctor command)."""
        import os
        import sys

        console.print("[bold]🔍 Doraemon Code Diagnostics[/bold]\n")

        checks = []

        # Python version
        py_version = sys.version_info
        py_ok = py_version >= (3, 10)
        checks.append(("Python", f"{py_version.major}.{py_version.minor}.{py_version.micro}", py_ok))

        # API keys
        google_key = bool(os.getenv("GOOGLE_API_KEY"))
        openai_key = bool(os.getenv("OPENAI_API_KEY"))
        anthropic_key = bool(os.getenv("ANTHROPIC_API_KEY"))
        gateway_url = os.getenv("DORAEMON_GATEWAY_URL")

        if gateway_url:
            checks.append(("Gateway", gateway_url, True))
        else:
            checks.append(("GOOGLE_API_KEY", "✓" if google_key else "✗", google_key))
            checks.append(("OPENAI_API_KEY", "✓" if openai_key else "✗", openai_key))
            checks.append(("ANTHROPIC_API_KEY", "✓" if anthropic_key else "✗", anthropic_key))

        # Directories
        checks.append((".agent/", "✓" if Path(".agent").exists() else "Will create", True))
        checks.append(("AGENTS.md", "✓" if Path("AGENTS.md").exists() else "Use /init", Path("AGENTS.md").exists()))

        # Git
        try:
            result = subprocess.run(["git", "--version"], capture_output=True, text=True, timeout=5)
            checks.append(("Git", "✓" if result.returncode == 0 else "✗", result.returncode == 0))
        except Exception:
            checks.append(("Git", "✗", False))

        # Display
        table = Table(show_header=True)
        table.add_column("Check", style="cyan")
        table.add_column("Status")
        table.add_column("", width=3)

        for name, status, ok in checks:
            icon = "✅" if ok else "❌"
            style = "green" if ok else "red"
            table.add_row(name, f"[{style}]{status}[/{style}]", icon)

        console.print(table)

    # ── Spec Mode ─────────────────────────────────────────────

    def _make_tool_state(self, mode: str, tool_names: list[str],
                         active_skills_content: str,
                         build_system_prompt, convert_tools_to_definitions,
                         extra_prompt: str = "") -> dict:
        """Build the state dict for a mode switch (tools + system prompt)."""
        genai_tools = self.registry.get_genai_tools(tool_names)
        prompt = build_system_prompt(mode, active_skills_content)
        if extra_prompt:
            prompt += f"\n\n{extra_prompt}"
        return {
            "mode": mode,
            "tool_names": tool_names,
            "tool_definitions": convert_tools_to_definitions(genai_tools),
            "system_prompt": prompt,
        }
