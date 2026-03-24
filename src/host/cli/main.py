"""
Doraemon Code CLI - Simplified Entry Point

Uses the new Agent architecture for cleaner code.
"""

import asyncio
import logging
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from src.agent import (
    AgentSession,
    AgentState,
    AgentTurnResult,
    create_doraemon_agent_with_mcp,
)
from src.core.config import load_config
from src.core.logger import configure_root_logger

logger = logging.getLogger(__name__)
console = Console()

for stream in [sys.stdin, sys.stdout, sys.stderr]:
    if stream and hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

app = typer.Typer()


async def run_chat_loop(
    project: str,
    mode: str,
    max_turns: int,
    headless: bool,
    initial_prompt: str | None,
):
    """Main chat loop using Agent architecture."""
    config_path = Path.cwd() / ".agent" / "config.json"

    from src.core.model_client import ModelClient
    from src.core.checkpoint import CheckpointManager
    from src.core.skills import SkillManager
    from src.core.hooks import HookManager

    model_client = await ModelClient.create()
    checkpoint_mgr = CheckpointManager(project=project)
    skill_mgr = SkillManager(project_dir=Path.cwd())
    hook_mgr = HookManager(project_dir=Path.cwd())

    session = AgentSession(
        model_client=model_client,
        registry=None,
        mode=mode,
        hook_mgr=hook_mgr,
        checkpoint_mgr=checkpoint_mgr,
        skill_mgr=skill_mgr,
        max_turns=max_turns,
    )
    await session.initialize()

    if initial_prompt:
        result = await session.turn(initial_prompt)
        if result.response:
            console.print(Markdown(result.response))
        if headless:
            return

    while True:
        try:
            user_input = await asyncio.to_thread(
                Prompt.ask,
                f"[bold {('green' if mode == 'build' else 'blue')}]doraemon[/bold {('green' if mode == 'build' else 'blue')}]",
            )
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Goodbye![/yellow]")
            break

        if not user_input.strip():
            continue

        if user_input.lower() in {"exit", "quit", "/exit", "/quit"}:
            console.print("[yellow]Goodbye![/yellow]")
            break

        if user_input.startswith("/"):
            result = await handle_command(user_input, session, mode)
            if result == "exit":
                break
            continue

        console.print()
        result = await session.turn(user_input)

        if result.response:
            console.print(
                Panel(
                    Markdown(result.response),
                    border_style="green" if result.success else "red",
                    expand=False,
                )
            )

        if result.error:
            console.print(f"[red]Error: {result.error}[/red]")

        console.print(
            f"[dim]Turn {session._state.turn_count if session._state else 0}, Tools: {len(result.tool_calls)}[/dim]"
        )
        console.print()


async def handle_command(cmd: str, session: AgentSession, mode: str) -> str | None:
    """Handle slash commands."""
    parts = cmd[1:].split()
    command = parts[0].lower() if parts else ""
    args = parts[1:] if len(parts) > 1 else []

    if command in {"help", "h", "?"}:
        console.print("""
[bold]Commands:[/bold]
  /help, /h, /?     Show this help
  /mode <mode>      Switch mode (plan/build)
  /clear            Clear conversation
  /reset            Reset agent state
  /exit, /quit      Exit the CLI

[bold]Tips:[/bold]
  Start with ! to run bash commands: !ls -la
""")

    elif command == "mode":
        if args and args[0] in {"plan", "build"}:
            new_mode = args[0]
            session.set_mode(new_mode)
            mode = new_mode
            console.print(f"[green]Switched to {new_mode} mode[/green]")
        else:
            console.print(f"[yellow]Current mode: {mode}[/yellow]")
            console.print("Usage: /mode plan | /mode build")

    elif command == "clear":
        session._state.messages.clear() if session._state else None
        console.print("[green]Conversation cleared[/green]")

    elif command == "reset":
        session.reset()
        console.print("[green]Agent reset[/green]")

    elif command in {"exit", "quit"}:
        return "exit"

    else:
        console.print(f"[red]Unknown command: /{command}[/red]")

    return None


@app.command()
def start(
    project: str = typer.Option("default", "--project", "-p", help="Project name"),
    mode: str = typer.Option("build", "--mode", "-m", help="Agent mode (plan/build)"),
    prompt: str = typer.Option(None, "--prompt", "-P", help="Initial prompt"),
    print_mode: bool = typer.Option(False, "--print", help="Print mode (non-interactive)"),
    max_turns: int = typer.Option(100, "--max-turns", help="Max turns per conversation"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """Start Doraemon Code CLI."""
    configure_root_logger(level="DEBUG" if verbose else "INFO")

    load_config()

    headless = print_mode or bool(prompt)

    if mode not in {"plan", "build"}:
        console.print(f"[red]Invalid mode: {mode}. Use 'plan' or 'build'.[/red]")
        raise typer.Exit(1)

    console.print(f"[bold cyan]Doraemon Code[/bold cyan] - Mode: [bold]{mode}[/bold]")
    console.print()

    try:
        asyncio.run(
            run_chat_loop(
                project=project,
                mode=mode,
                max_turns=max_turns,
                headless=headless,
                initial_prompt=prompt,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")


@app.command()
def version():
    """Show version."""
    console.print("[bold cyan]Doraemon Code[/bold cyan] v0.1.0")


def entry_point():
    """Entry point for package script."""
    app()


if __name__ == "__main__":
    app()
