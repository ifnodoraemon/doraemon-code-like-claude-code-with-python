"""
Doraemon Code CLI - Simplified Entry Point

Uses the new Agent architecture for cleaner code.

Storage:
    ~/.doraemon/          # User level (settings, global skills)
    .agent/               # Project level (sessions, traces, checkpoints)
"""

import asyncio
import logging
import shlex
import sys
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from src.agent import (
    AgentSession,
)
from src.core.commands import CommandDefinition, CommandManager, CommandResult
from src.core.config.config import load_config
from src.core.home import (
    set_project_dir,
)
from src.core.logger import configure_root_logger
from src.core.session import SessionManager

logger = logging.getLogger(__name__)
console = Console()

for stream in [sys.stdin, sys.stdout, sys.stderr]:
    if stream and hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

app = typer.Typer()

_TOP_LEVEL_COMMANDS = {"start", "version"}
_TOP_LEVEL_OPTIONS = {"--help", "--install-completion", "--show-completion"}


def _argv_with_default_start(argv: list[str]) -> list[str]:
    """Route `doraemon --prompt ...` and bare `doraemon` to the start command."""
    if len(argv) <= 1:
        return [argv[0], "start"]

    first_arg = argv[1]
    if first_arg in _TOP_LEVEL_COMMANDS or first_arg in _TOP_LEVEL_OPTIONS:
        return argv

    if first_arg.startswith("-"):
        return [argv[0], "start", *argv[1:]]

    return argv


def _parse_orchestrate_args(args: list[str]) -> tuple[int, str] | None:
    """Parse `/orchestrate` arguments into max workers and goal."""
    if not args:
        return None

    max_workers = 2
    goal_parts = args
    if len(args) >= 2 and args[0] in {"--workers", "-w"}:
        try:
            max_workers = max(1, int(args[1]))
        except ValueError:
            return None
        goal_parts = args[2:]

    goal = " ".join(goal_parts).strip()
    if not goal:
        return None
    return max_workers, goal


def _format_task_tree(nodes: list[dict[str, Any]], level: int = 0) -> list[str]:
    """Render the task tree into compact CLI lines."""
    lines: list[str] = []
    indent = "  " * level
    for node in nodes:
        status = node.get("status", "pending")
        ready = " ready" if node.get("ready") else ""
        assigned = f" @{node['assigned_agent']}" if node.get("assigned_agent") else ""
        lines.append(
            f"{indent}- [{status}] {node.get('title', node.get('id', 'task'))} "
            f"({node.get('id', '?')}){ready}{assigned}"
        )
        for child in node.get("children", []):
            lines.extend(_format_task_tree([child], level + 1))
    return lines


def _find_orchestration_run(session: Any, run_id: str) -> dict[str, Any] | None:
    get_runs = getattr(session, "get_orchestration_runs", None)
    if not callable(get_runs):
        return None
    for run in reversed(get_runs() or []):
        if run.get("run_id") == run_id:
            return run
    return None


def _resolve_task_root(session: Any, run_id: str | None = None) -> tuple[str | None, str | None]:
    if run_id:
        run = _find_orchestration_run(session, run_id)
        if run is None:
            return None, f"Unknown run: {run_id}"
        return run.get("root_task_id"), None

    orchestration_state = getattr(session, "get_orchestration_state", lambda: {})()
    if isinstance(orchestration_state, dict):
        return orchestration_state.get("root_task_id"), None
    return None, None


def _parse_resume_args(args: list[str]) -> tuple[str, int] | None:
    if not args:
        return None

    run_id = args[0].strip()
    if not run_id:
        return None

    max_workers = 2
    remaining = args[1:]
    if len(remaining) >= 2 and remaining[0] in {"--workers", "-w"}:
        try:
            max_workers = max(1, int(remaining[1]))
        except ValueError:
            return None
    elif remaining:
        return None

    return run_id, max_workers


def _session_project_dir(session: Any) -> Path:
    """Return the project directory associated with a session-like object."""
    project_dir = getattr(session, "project_dir", None)
    if project_dir is None:
        return Path.cwd()
    return Path(project_dir)


def _parse_custom_command_args(
    command: CommandDefinition,
    args: list[str],
) -> dict[str, str] | None:
    """Map slash command args to command variables.

    Supports positional args in declaration order plus KEY=value, --KEY value,
    and --KEY=value forms.
    """
    parsed: dict[str, str] = {}
    positionals: list[str] = []
    i = 0
    while i < len(args):
        token = args[i]
        if token.startswith("--") and "=" in token:
            key, value = token[2:].split("=", 1)
            if not key:
                return None
            parsed[key] = value
        elif token.startswith("--"):
            key = token[2:]
            if not key or i + 1 >= len(args):
                return None
            parsed[key] = args[i + 1]
            i += 1
        elif "=" in token:
            key, value = token.split("=", 1)
            if not key:
                return None
            parsed[key] = value
        else:
            positionals.append(token)
        i += 1

    positional_index = 0
    for argument in command.arguments:
        if argument.name in parsed:
            continue
        if argument.name == "ARGUMENTS":
            parsed[argument.name] = " ".join(positionals[positional_index:])
            positional_index = len(positionals)
            continue
        if positional_index < len(positionals):
            parsed[argument.name] = positionals[positional_index]
            positional_index += 1

    if not command.arguments and positionals:
        parsed["ARGUMENTS"] = " ".join(positionals)
        positional_index = len(positionals)

    if positional_index < len(positionals):
        return None
    return parsed


def _print_command_result(result: CommandResult) -> None:
    color = "green" if result.success else "red"
    status = "completed" if result.success else "failed"
    console.print(f"[{color}]Command {status}:[/{color}] /{result.command_name}")
    for output in result.outputs:
        if output:
            console.print(output)
    for error in result.errors:
        if error:
            console.print(f"[red]{error}[/red]")


def _print_cli_help() -> None:
    console.print("""
[bold]Commands:[/bold]
  /help, /h, /?     Show this help
  /mode <mode>      Switch mode (plan/build)
  /commands         List project custom commands
  /commands help X  Show help for a custom command
  /orchestrate ...  Run lead/worker orchestration
  /runs             List orchestration runs in this session
  /resume <run_id>  Resume a blocked orchestration run
  /tasks ...        Show runtime task graph
  /clear            Clear conversation
  /reset            Reset agent state
  /trace            Show trace info
  /session          Show session ID
  /exit, /quit      Exit the CLI

[bold]Tips:[/bold]
  Start with ! to run bash commands: !ls -la
  Create custom commands in .agent/commands/*.md
  Example: /orchestrate --workers 3 implement auth flow
""")


def _print_custom_commands(project_dir: Path, args: list[str]) -> None:
    manager = CommandManager(project_dir)
    if args[:1] == ["help"] and len(args) == 2:
        help_text = manager.get_command_help(args[1])
        if help_text:
            console.print(help_text)
        else:
            console.print(f"[yellow]Command not found: /{args[1]}[/yellow]")
        return
    if args:
        console.print("Usage: /commands | /commands help <name>")
        return

    commands = manager.list_commands()
    if not commands:
        console.print("[yellow]No custom commands found in .agent/commands[/yellow]")
        return
    console.print("[cyan]Custom commands:[/cyan]")
    for item in commands:
        args_text = " ".join(f"${name}" for name in item.get("arguments", []))
        suffix = f" {args_text}" if args_text else ""
        description = item.get("description") or ""
        console.print(f"  /{item['name']}{suffix}  {description}".rstrip())


def _session_manager_for_project(project_dir: Path) -> SessionManager:
    return SessionManager(base_dir=project_dir / ".agent" / "sessions")


def _print_session_list(project_dir: Path, project: str, limit: int = 20) -> None:
    manager = _session_manager_for_project(project_dir)
    sessions = manager.list_sessions(project=project, limit=limit)
    if not sessions:
        console.print(f"[yellow]No sessions found for project: {project}[/yellow]")
        return

    console.print(f"[cyan]Recent sessions for {project}:[/cyan]")
    for session in sessions:
        name = session.name or "Untitled Session"
        console.print(
            f"  {session.id}  {name}  "
            f"({session.message_count} messages, mode={session.mode})"
        )


def _resolve_resume_session_id(
    project_dir: Path,
    project: str,
    *,
    resume: str | None,
    continue_last: bool,
) -> str | None:
    manager = _session_manager_for_project(project_dir)

    if continue_last:
        sessions = manager.list_sessions(project=project, limit=1)
        if not sessions:
            console.print(f"[yellow]No sessions found for project: {project}[/yellow]")
            return None
        latest = sessions[0]
        console.print(f"[cyan]Resuming latest session:[/cyan] {latest.id}")
        return latest.id

    if not resume:
        return None

    session = manager.load_session(resume)
    if session is None:
        matching = [meta for meta in manager.search_sessions(resume, project=project) if meta.name == resume]
        matching.sort(key=lambda meta: meta.updated_at, reverse=True)
        if matching:
            session = manager.load_session(matching[0].id)
    if session is None:
        console.print(f"[red]Session not found:[/red] {resume}")
        return None
    console.print(f"[cyan]Resuming session:[/cyan] {session.metadata.id}")
    return session.metadata.id


async def _handle_project_command(
    name: str,
    args: list[str],
    session: Any,
) -> bool:
    """Execute a project-local custom slash command if it exists."""
    manager = CommandManager(_session_project_dir(session))
    command = manager.loader.get_command(name)
    if command is None:
        return False

    parsed_args = _parse_custom_command_args(command, args)
    if parsed_args is None:
        console.print(f"Usage: {manager.get_command_help(name) or f'/{name}'}")
        return True

    if not command.steps and command.prompt:
        prompt, error = manager.render_command_prompt(command, parsed_args)
        if error:
            console.print(f"Usage: {manager.get_command_help(name) or f'/{name}'}")
            return True
        if not prompt:
            console.print(f"[yellow]Command has no prompt content: /{name}[/yellow]")
            return True
        result = await session.turn(prompt)
        if getattr(result, "response", None):
            console.print(
                Panel(
                    Markdown(result.response),
                    border_style="green" if getattr(result, "success", True) else "red",
                    expand=False,
                )
            )
        if getattr(result, "error", None):
            console.print(f"[red]Error: {result.error}[/red]")
        return True

    result = await manager.run_command(name, parsed_args)
    _print_command_result(result)
    return True


async def _handle_bang_command(user_input: str, project_dir: Path) -> None:
    """Run a shell command from the interactive CLI `!cmd` shortcut."""
    command = user_input[1:].strip()
    if not command:
        console.print("Usage: !<shell command>")
        return

    from src.servers.run import run

    output = await asyncio.to_thread(
        run,
        command,
        mode="shell",
        working_dir=str(project_dir),
    )
    console.print(output)


async def _handle_pre_session_initial_prompt(
    initial_prompt: str,
    project_dir: Path,
) -> bool:
    """Handle one-shot local commands before model/MCP session initialization."""
    if initial_prompt.startswith("!"):
        await _handle_bang_command(initial_prompt, project_dir)
        return True

    if not initial_prompt.startswith("/"):
        return False

    try:
        parts = shlex.split(initial_prompt[1:])
    except ValueError as exc:
        console.print(f"[red]Invalid command syntax:[/red] {exc}")
        return True

    command = parts[0].lower() if parts else ""
    args = parts[1:] if len(parts) > 1 else []

    if command in {"help", "h", "?"}:
        _print_cli_help()
        return True

    if command in {"commands", "cmds"}:
        _print_custom_commands(project_dir, args)
        return True

    if command in {"exit", "quit"}:
        return True

    return False


async def _handle_initial_prompt(
    initial_prompt: str,
    session: AgentSession,
    project_dir: Path,
) -> str | None:
    """Route print-mode input through the same command paths as interactive input."""
    if initial_prompt.startswith("/"):
        return await handle_command(initial_prompt, session)

    if initial_prompt.startswith("!"):
        await _handle_bang_command(initial_prompt, project_dir)
        return None

    result = await session.turn(initial_prompt)
    if result.response:
        console.print(Markdown(result.response))
    if result.error:
        console.print(f"[red]Error: {result.error}[/red]")
    return None


async def run_chat_loop(
    project: str,
    mode: str,
    max_turns: int,
    headless: bool,
    initial_prompt: str | None,
    enable_trace: bool,
    resume: str | None = None,
    continue_last: bool = False,
    list_sessions: bool = False,
):
    """Main chat loop using Agent architecture."""
    project_dir = Path.cwd()
    set_project_dir(project_dir)

    if list_sessions:
        _print_session_list(project_dir, project)
        return

    resume_session_id = _resolve_resume_session_id(
        project_dir,
        project,
        resume=resume,
        continue_last=continue_last,
    )
    if (resume or continue_last) and resume_session_id is None:
        return

    if headless and initial_prompt:
        if await _handle_pre_session_initial_prompt(initial_prompt, project_dir):
            return

    session = AgentSession(
        model_client=None,
        registry=None,
        project=project,
        mode=mode,
        max_turns=max_turns,
        project_dir=project_dir,
        enable_trace=enable_trace,
        session_id=resume_session_id,
    )
    await session.initialize()

    console.print(f"[dim]Session: {session.session_id}[/dim]")
    console.print()

    if initial_prompt:
        await _handle_initial_prompt(initial_prompt, session, project_dir)
    if headless:
        trace_path = await session.aclose()
        if trace_path:
            console.print(f"[dim]Trace saved: {trace_path}[/dim]")
        return

    try:
        while True:
            try:
                user_input = await asyncio.to_thread(
                    Prompt.ask,
                    (
                        f"[bold {('green' if session.mode == 'build' else 'blue')}]"
                        f"doraemon[/bold {('green' if session.mode == 'build' else 'blue')}]"
                    ),
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
                result = await handle_command(user_input, session)
                if result == "exit":
                    break
                continue

            if user_input.startswith("!"):
                await _handle_bang_command(user_input, project_dir)
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
        trace_path = await session.aclose()
        if trace_path:
            console.print(f"[dim]Trace saved: {trace_path}[/dim]")


async def handle_command(cmd: str, session: AgentSession) -> str | None:
    """Handle slash commands."""
    try:
        parts = shlex.split(cmd[1:])
    except ValueError as exc:
        console.print(f"[red]Invalid command syntax:[/red] {exc}")
        return None

    command = parts[0].lower() if parts else ""
    args = parts[1:] if len(parts) > 1 else []

    if command in {"help", "h", "?"}:
        _print_cli_help()

    elif command in {"commands", "cmds"}:
        _print_custom_commands(_session_project_dir(session), args)

    elif command == "mode":
        if args and args[0] in {"plan", "build"}:
            new_mode = args[0]
            await session.set_mode(new_mode)
            console.print(f"[green]Switched to {new_mode} mode[/green]")
        else:
            console.print(f"[yellow]Current mode: {session.mode}[/yellow]")
            console.print("Usage: /mode plan | /mode build")

    elif command == "clear":
        if session._state:
            session._state.clear_history()
        console.print("[green]Conversation cleared[/green]")

    elif command == "orchestrate":
        parsed = _parse_orchestrate_args(args)
        if parsed is None:
            console.print("Usage: /orchestrate [--workers N] <goal>")
            return None

        max_workers, goal = parsed
        console.print(
            f"[cyan]Running orchestration[/cyan] with {max_workers} worker(s): {goal}"
        )
        try:
            result = await session.orchestrate(goal, max_workers=max_workers)
        except Exception as exc:
            console.print(f"[red]Orchestration failed:[/red] {exc}")
            return None
        color = "green" if result.success else "red"
        console.print(f"[{color}]Summary:[/{color}] {result.summary}")
        console.print(f"[cyan]Root task:[/cyan] {result.root_task_id}")
        console.print(f"[cyan]Completed:[/cyan] {len(result.completed_task_ids)}")
        if result.failed_task_ids:
            console.print(f"[red]Failed:[/red] {len(result.failed_task_ids)}")
        if result.worker_assignments:
            console.print("[cyan]Worker assignments:[/cyan]")
            for task_id, assignment in result.worker_assignments.items():
                console.print(
                    f"  - {task_id}: {assignment['role']} via {assignment['worker_session_id']}"
                )

    elif command == "runs":
        get_runs = getattr(session, "get_orchestration_runs", None)
        if not callable(get_runs):
            console.print("[yellow]No orchestration history available[/yellow]")
            return None
        runs = get_runs() or []
        if not runs:
            console.print("[yellow]No orchestration runs[/yellow]")
            return None
        active_run_id = getattr(session, "get_active_orchestration_run_id", lambda: None)()
        console.print("[cyan]Runs:[/cyan]")
        for run in reversed(runs):
            marker = "*" if run.get("run_id") == active_run_id else "-"
            status = "completed" if run.get("success") else "blocked"
            console.print(
                f"{marker} [{status}] {run.get('run_id') or '?'} "
                f"{run.get('goal') or run.get('summary') or ''}".rstrip()
            )

    elif command == "resume":
        parsed = _parse_resume_args(args)
        if parsed is None:
            console.print("Usage: /resume <run_id> [--workers N]")
            return None
        run_id, max_workers = parsed
        console.print(f"[cyan]Resuming run[/cyan] {run_id} with {max_workers} worker(s)")
        try:
            result = await session.orchestrate(
                "",
                max_workers=max_workers,
                resume_run_id=run_id,
            )
        except Exception as exc:
            console.print(f"[red]Resume failed:[/red] {exc}")
            return None
        color = "green" if result.success else "red"
        console.print(f"[{color}]Summary:[/{color}] {result.summary}")
        console.print(f"[cyan]Root task:[/cyan] {result.root_task_id}")

    elif command == "tasks":
        task_manager = session.get_task_manager()
        if task_manager is None:
            console.print("[yellow]No task manager available[/yellow]")
            return None
        ready_only = bool(args and args[0] == "ready")
        run_arg = args[1] if ready_only and len(args) > 1 else args[0] if args and not ready_only else None
        active_root_task_id, error = _resolve_task_root(session, run_arg)
        if error:
            console.print(f"[red]{error}[/red]")
            return None

        if ready_only:
            ready_tasks = [
                task
                for task in task_manager.list_ready_tasks()
                if active_root_task_id is None or task.parent_id == active_root_task_id
            ]
            if not ready_tasks:
                console.print("[yellow]No ready tasks[/yellow]")
                return None
            for task in ready_tasks:
                console.print(f"- [{task.status.value}] {task.title} ({task.id})")
            return None

        lines = _format_task_tree(task_manager.get_task_tree(active_root_task_id))
        if not lines:
            console.print("[yellow]No tasks[/yellow]")
            return None
        console.print("[cyan]Task graph:[/cyan]")
        for line in lines:
            console.print(line)

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
        if await _handle_project_command(command, args, session):
            return None
        console.print(f"[red]Unknown command: /{command}[/red]")

    return None


@app.command()
def start(
    project: str = typer.Option("default", "--project", "-p", help="Project name"),
    mode: str = typer.Option("build", "--mode", "-m", help="Agent mode (plan/build)"),
    prompt: str = typer.Option(None, "--prompt", "-P", help="Initial prompt"),
    print_mode: bool = typer.Option(False, "--print", help="Print mode (non-interactive)"),
    max_turns: int = typer.Option(100, "--max-turns", help="Max turns per conversation"),
    resume: str | None = typer.Option(None, "--resume", "-r", help="Resume session by ID or name"),
    continue_last: bool = typer.Option(False, "--continue", help="Resume the latest session"),
    list_sessions: bool = typer.Option(False, "--list-sessions", help="List recent sessions and exit"),
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
                enable_trace=not no_trace,
                resume=resume,
                continue_last=continue_last,
                list_sessions=list_sessions,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")


@app.command()
def version():
    """Show version."""
    console.print("[bold cyan]Doraemon Code[/bold cyan] v0.8.0")


def entry_point():
    """Entry point for package script."""
    sys.argv = _argv_with_default_start(sys.argv)
    app()


if __name__ == "__main__":
    entry_point()
