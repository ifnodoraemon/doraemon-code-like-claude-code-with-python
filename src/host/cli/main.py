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


@app.command()
def start(
    project: str = typer.Option("default", "--project", "-p", help="Project name"),
    resume: str = typer.Option(None, "--resume", "-r", help="Resume session by ID or name"),
    name: str = typer.Option(None, "--name", "-n", help="Name for new session"),
    prompt: str = typer.Option(None, "--prompt", "-P", help="Initial prompt (enables headless mode)"),
):
    """Start Doraemon Code CLI."""
    asyncio.run(chat_loop(project=project, resume_session=resume, session_name=name, prompt=prompt))


@app.command()
def sessions(
    project: str = typer.Option("default", "--project", "-p", help="Project name"),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of sessions to show"),
):
    """List recent sessions."""
    mgr = SessionManager()
    sessions_list = mgr.list_sessions(project=project, limit=limit)

    if sessions_list:
        table = Table(title="Sessions")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Messages", style="yellow")
        table.add_column("Updated")

        for s in sessions_list:
            updated = datetime.fromtimestamp(s.updated_at).strftime("%Y-%m-%d %H:%M")
            table.add_row(s.id[:8], s.name or "-", str(s.message_count), updated)

        console.print(table)
    else:
        console.print("[dim]No sessions found[/dim]")


@app.command()
def version():
    """Show version information."""
    console.print("[bold]🤖 Doraemon Code v0.8.0[/bold]")
    console.print("[dim]Features: Web UI, Gateway, checkpointing, subagents, hooks, cost tracking[/dim]")
    gateway_url = os.getenv("DORAEMON_GATEWAY_URL")
    if gateway_url:
        console.print(f"[cyan]Mode: Gateway ({gateway_url})[/cyan]")
    else:
        console.print("[green]Mode: Direct (using API keys)[/green]")


def entry_point():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    app()
