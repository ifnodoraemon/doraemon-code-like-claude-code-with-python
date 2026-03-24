"""
Doraemon Code CLI - Simplified Entry Point

Uses the new Agent architecture for cleaner code.

Storage:
    ~/.doraemon/          # User level (settings, global skills)
    .agent/               # Project level (sessions, traces, checkpoints)
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
from src.core.config.config import load_config
from src.core.home import (
    get_agent_dir,
    get_project_config_path,
    get_user_settings_path,
    load_user_settings,
    save_user_settings,
    set_project_dir,
)
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
    project_dir = Path.cwd()
    set_project_dir(project_dir)

    config_path = get_project_config_path()

    from src.core.llm.model_client import ModelClient
    from src.core.checkpoint import CheckpointManager
    from src.core.skills import SkillManager
    from src.core.hooks import HookManager

    model_client = await ModelClient.create()
    checkpoints = CheckpointManager(project=project)
    skills = SkillManager(project_dir=project_dir)
    hooks = HookManager(project_dir=project_dir)

    session = AgentSession(
        model_client=model_client,
        registry=None,
        mode=mode,
        hooks=hooks,
        checkpoints=checkpoints,
        skills=skills,
        max_turns=max_turns,
        project_dir=project_dir,
        enable_trace=True,
    )
    await session.initialize()

    console.print(f"[dim]Session: {session.session_id}[/dim]")
    console.print()

    if initial_prompt:
        result = await session.turn(initial_prompt)
        if result.response:
            console.print(Markdown(result.response))
        if headless:
            trace_path = session.close()
            if trace_path:
                console.print(f"[dim]Trace saved: {trace_path}[/dim]")
            return

    try:
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

    finally:
        trace_path = session.close()
        if trace_path:
            console.print(f"[dim]Trace saved: {trace_path}[/dim]")


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
  /trace            Show trace info
  /session          Show session ID
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

    elif command == "trace":
        trace = session.get_trace()
        if trace:
            console.print(f"[cyan]Session ID:[/cyan] {trace.session_id}")
            console.print(f"[cyan]Events:[/cyan] {len(trace.events)}")
            tool_calls = len([e for e in trace.events if e.type == "tool_call"])
            llm_calls = len([e for e in trace.events if e.type == "llm_call"])
            errors = len([e for e in trace.events if e.type == "error"])
            console.print(f"[cyan]Tool calls:[/cyan] {tool_calls}")
            console.print(f"[cyan]LLM calls:[/cyan] {llm_calls}")
            console.print(f"[cyan]Errors:[/cyan] {errors}")
        else:
            console.print("[yellow]No trace available[/yellow]")

    elif command == "session":
        console.print(f"[cyan]Session ID:[/cyan] {session.session_id}")
        console.print(f"[cyan]Project:[/cyan] {session.project_dir}")

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
    no_trace: bool = typer.Option(False, "--no-trace", help="Disable trace recording"),
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
