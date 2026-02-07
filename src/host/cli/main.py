"""
Doraemon Code CLI - Main Entry Point

Provides the primary interactive interface for Doraemon Code AI agent.
Features:
- Multi-mode support (plan/build)
- HITL (Human-in-the-loop) approval for sensitive operations
- Rich terminal UI with markdown rendering
- Vector memory (ChromaDB) for long-term recall
- Direct tool calls (no subprocess overhead)
- Automatic context summarization for unlimited conversation length
- Checkpointing and rollback
- Background tasks
- Subagents
- Hooks system
- Session management
- Cost tracking
- Command history and Bash mode
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from src.host.cli.chat_loop import chat_loop
from src.core.session import SessionManager

logger = logging.getLogger(__name__)

# Fix encoding
for stream in [sys.stdin, sys.stdout, sys.stderr]:
    if stream and hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

app = typer.Typer()
console = Console()


def interactive_session_picker(project: str = "default") -> str | None:
    """
    Show interactive session picker using Rich.

    Returns:
        Selected session ID or None if cancelled
    """
    from rich.prompt import Prompt

    mgr = SessionManager()
    sessions_list = mgr.list_sessions(project=project, limit=20)

    if not sessions_list:
        console.print("[yellow]No sessions found.[/yellow]")
        return None

    # Display sessions
    console.print("\n[bold cyan]Recent Sessions[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", style="dim", width=3)
    table.add_column("ID", style="cyan", width=10)
    table.add_column("Name/Description", style="green")
    table.add_column("Messages", style="yellow", width=8)
    table.add_column("Updated", style="dim")
    table.add_column("Project", style="dim")

    for i, s in enumerate(sessions_list, 1):
        updated = datetime.fromtimestamp(s.updated_at).strftime("%m-%d %H:%M")
        name = s.name or s.description[:30] or f"Session {s.id[:8]}"
        table.add_row(
            str(i),
            s.id[:8],
            name[:40],
            str(s.message_count),
            updated,
            s.project,
        )

    console.print(table)
    console.print()

    # Get user selection
    choice = Prompt.ask(
        "Select session (number, ID, or 'q' to cancel)",
        default="1"
    )

    if choice.lower() in ('q', 'quit', 'cancel', ''):
        return None

    # Try as number
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(sessions_list):
            return sessions_list[idx].id

    # Try as ID (partial match)
    for s in sessions_list:
        if s.id.startswith(choice) or (s.name and s.name == choice):
            return s.id

    console.print(f"[red]Session not found: {choice}[/red]")
    return None


def get_most_recent_session(project: str = "default") -> str | None:
    """Get the most recent session ID."""
    mgr = SessionManager()
    sessions_list = mgr.list_sessions(project=project, limit=1)
    if sessions_list:
        return sessions_list[0].id
    return None


@app.command()
def start(
    project: str = typer.Option("default", "--project", "-p", help="Project name"),
    resume: str = typer.Option(None, "--resume", "-r", help="Resume session (ID, name, or empty for picker)"),
    continue_session: bool = typer.Option(False, "--continue", "-c", help="Continue most recent session"),
    name: str = typer.Option(None, "--name", "-n", help="Name for new session"),
    prompt: str = typer.Option(None, "--prompt", "-P", help="Initial prompt"),
    print_mode: bool = typer.Option(False, "--print", help="Print mode (non-interactive, exit after response)"),
    max_turns: int = typer.Option(0, "--max-turns", help="Maximum conversation turns (0 = unlimited)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    allowed_tools: str = typer.Option(None, "--allowedTools", help="Comma-separated list of allowed tools"),
    disallowed_tools: str = typer.Option(None, "--disallowedTools", help="Comma-separated list of disallowed tools"),
):
    """Start Doraemon Code CLI."""
    # Setup logging
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s - %(levelname)s - %(message)s")

    # Handle --continue: use most recent session
    if continue_session:
        resume = get_most_recent_session(project)
        if not resume:
            console.print("[yellow]No previous session found. Starting new session.[/yellow]")

    # Handle --resume without ID: show interactive picker
    # Note: typer passes empty string when flag is used without value
    if resume == "":
        resume = interactive_session_picker(project)
        if not resume:
            console.print("[dim]Starting new session.[/dim]")

    # Parse tool permissions
    tool_config = {}
    if allowed_tools:
        tool_config["allowed"] = [t.strip() for t in allowed_tools.split(",")]
    if disallowed_tools:
        tool_config["disallowed"] = [t.strip() for t in disallowed_tools.split(",")]

    asyncio.run(chat_loop(
        project=project,
        resume_session=resume,
        session_name=name,
        prompt=prompt,
        print_mode=print_mode,
        max_turns=max_turns if max_turns > 0 else None,
        tool_config=tool_config if tool_config else None,
    ))


@app.command()
def sessions(
    project: str = typer.Option("default", "--project", "-p", help="Project name"),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of sessions to show"),
    all_projects: bool = typer.Option(False, "--all", "-a", help="Show sessions from all projects"),
):
    """List recent sessions."""
    mgr = SessionManager()

    if all_projects:
        sessions_list = mgr.list_sessions(limit=limit)
    else:
        sessions_list = mgr.list_sessions(project=project, limit=limit)

    if sessions_list:
        table = Table(title="Sessions")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Messages", style="yellow")
        table.add_column("Project", style="dim")
        table.add_column("Updated")

        for s in sessions_list:
            updated = datetime.fromtimestamp(s.updated_at).strftime("%Y-%m-%d %H:%M")
            table.add_row(s.id[:8], s.name or "-", str(s.message_count), s.project, updated)

        console.print(table)
    else:
        console.print("[dim]No sessions found[/dim]")


@app.command()
def config(
    action: str = typer.Argument(None, help="Action: set, get, add, remove, list"),
    key: str = typer.Argument(None, help="Configuration key"),
    value: str = typer.Argument(None, help="Configuration value"),
):
    """Manage configuration settings."""
    config_path = Path.home() / ".doraemon" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing config
    if config_path.exists():
        import json
        config_data = json.loads(config_path.read_text())
    else:
        config_data = {}

    if action is None or action == "list":
        # List all config
        if config_data:
            table = Table(title="Configuration")
            table.add_column("Key", style="cyan")
            table.add_column("Value", style="green")
            for k, v in config_data.items():
                table.add_row(k, str(v))
            console.print(table)
        else:
            console.print("[dim]No configuration set[/dim]")
        return

    if action == "get":
        if key and key in config_data:
            console.print(f"{key} = {config_data[key]}")
        else:
            console.print(f"[yellow]Key not found: {key}[/yellow]")
        return

    if action == "set":
        if key and value:
            config_data[key] = value
            import json
            config_path.write_text(json.dumps(config_data, indent=2))
            console.print(f"[green]Set {key} = {value}[/green]")
        else:
            console.print("[red]Usage: doraemon config set <key> <value>[/red]")
        return

    if action == "add":
        if key and value:
            if key not in config_data:
                config_data[key] = []
            if isinstance(config_data[key], list):
                config_data[key].append(value)
                import json
                config_path.write_text(json.dumps(config_data, indent=2))
                console.print(f"[green]Added {value} to {key}[/green]")
            else:
                console.print(f"[red]{key} is not a list[/red]")
        return

    if action == "remove":
        if key and value:
            if key in config_data and isinstance(config_data[key], list):
                if value in config_data[key]:
                    config_data[key].remove(value)
                    import json
                    config_path.write_text(json.dumps(config_data, indent=2))
                    console.print(f"[green]Removed {value} from {key}[/green]")
                else:
                    console.print(f"[yellow]{value} not in {key}[/yellow]")
        return

    console.print(f"[red]Unknown action: {action}[/red]")


@app.command()
def version():
    """Show version information."""
    console.print("[bold]🤖 Doraemon Code v0.9.0[/bold]")
    console.print("[dim]Features: Web UI, Gateway, checkpointing, subagents, hooks, cost tracking[/dim]")
    gateway_url = os.getenv("DORAEMON_GATEWAY_URL")
    if gateway_url:
        console.print(f"[cyan]Mode: Gateway ({gateway_url})[/cyan]")
    else:
        console.print("[green]Mode: Direct (using API keys)[/green]")


@app.command()
def doctor():
    """Run diagnostic checks."""
    console.print("[bold]🔍 Doraemon Code Diagnostics[/bold]\n")

    checks = []

    # Check Python version
    py_version = sys.version_info
    py_ok = py_version >= (3, 10)
    checks.append(("Python version", f"{py_version.major}.{py_version.minor}.{py_version.micro}", py_ok))

    # Check API keys
    google_key = bool(os.getenv("GOOGLE_API_KEY"))
    openai_key = bool(os.getenv("OPENAI_API_KEY"))
    anthropic_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    gateway_url = os.getenv("DORAEMON_GATEWAY_URL")

    if gateway_url:
        checks.append(("Gateway URL", gateway_url, True))
    else:
        checks.append(("GOOGLE_API_KEY", "✓ Set" if google_key else "✗ Not set", google_key))
        checks.append(("OPENAI_API_KEY", "✓ Set" if openai_key else "✗ Not set", openai_key))
        checks.append(("ANTHROPIC_API_KEY", "✓ Set" if anthropic_key else "✗ Not set", anthropic_key))

        if not any([google_key, openai_key, anthropic_key]):
            checks.append(("API Keys", "⚠ No API keys configured", False))

    # Check directories
    doraemon_dir = Path(".doraemon")
    checks.append((".doraemon directory", "✓ Exists" if doraemon_dir.exists() else "Will be created", True))

    home_doraemon = Path.home() / ".doraemon"
    checks.append(("~/.doraemon directory", "✓ Exists" if home_doraemon.exists() else "Will be created", True))

    # Check DORAEMON.md
    doraemon_md = Path("DORAEMON.md")
    checks.append(("DORAEMON.md", "✓ Found" if doraemon_md.exists() else "Not found (use /init)", doraemon_md.exists()))

    # Check git
    import subprocess
    try:
        result = subprocess.run(["git", "--version"], capture_output=True, text=True, timeout=5)
        git_ok = result.returncode == 0
        git_version = result.stdout.strip() if git_ok else "Not found"
    except Exception:
        git_ok = False
        git_version = "Not found"
    checks.append(("Git", git_version, git_ok))

    # Check gh CLI
    try:
        result = subprocess.run(["gh", "--version"], capture_output=True, text=True, timeout=5)
        gh_ok = result.returncode == 0
        gh_version = result.stdout.split("\n")[0] if gh_ok else "Not found"
    except Exception:
        gh_ok = False
        gh_version = "Not found (optional)"
    checks.append(("GitHub CLI", gh_version, True))  # Optional, so always "ok"

    # Display results
    table = Table(show_header=True)
    table.add_column("Check", style="cyan")
    table.add_column("Status")
    table.add_column("", width=3)

    all_ok = True
    for name, status, ok in checks:
        icon = "✅" if ok else "❌"
        style = "green" if ok else "red"
        table.add_row(name, f"[{style}]{status}[/{style}]", icon)
        if not ok:
            all_ok = False

    console.print(table)

    if all_ok:
        console.print("\n[green]All checks passed! ✨[/green]")
    else:
        console.print("\n[yellow]Some checks failed. Please review the issues above.[/yellow]")


def entry_point():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    app()
