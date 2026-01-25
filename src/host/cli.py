"""
Polymath CLI - Main Command Line Interface

Provides the primary interactive interface for Polymath AI agent.
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
import json
import os
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

# Direct imports (no DI container needed)
from src.core.config import load_config
from src.core.context_manager import ContextConfig, ContextManager
from src.core.diff import print_diff
from src.core.prompts import get_system_prompt
from src.core.rules import format_instructions_for_prompt, load_all_instructions
from src.core.skills import SkillManager
from src.core.tool_selector import ToolSelector
from src.host.tools import get_default_registry

# New feature imports
from src.core.checkpoint import CheckpointManager, CheckpointConfig
from src.core.background_tasks import BackgroundTaskManager, TaskStatus, get_task_manager
from src.core.hooks import HookManager, HookEvent
from src.core.cost_tracker import CostTracker, BudgetConfig
from src.core.command_history import CommandHistory, BashModeExecutor
from src.core.session import SessionManager
from src.core.plugins import PluginManager
from src.core.workspace import WorkspaceManager
from src.core.model_manager import ModelManager
from src.core.input_mode import InputManager, InputMode
from src.core.thinking import ThinkingManager, ThinkingMode
from src.core.doctor import Doctor
from src.core.themes import ThemeManager

# Unified Model Client (supports Gateway and Direct modes)
from src.core.model_client import (
    ModelClient,
    ClientConfig,
    ClientMode,
    Message,
    ToolDefinition,
    ChatResponse,
)

# Fix encoding
for stream in [sys.stdin, sys.stdout, sys.stderr]:
    if stream and hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

app = typer.Typer()
console = Console()

# Mode colors (only two modes: plan and build)
MODE_COLORS = {
    "plan": "blue",
    "build": "green",
}


def build_system_prompt(mode: str, skills_content: str = "") -> str:
    """Build the system prompt with mode, rules, and skills."""
    config = load_config()
    persona = config.get("persona", {})

    # Build system prompt
    system_prompt = get_system_prompt(mode, persona)

    # Add project rules (AGENTS.md)
    instructions = load_all_instructions(config)
    if instructions:
        system_prompt += format_instructions_for_prompt(instructions)

    # Add active skills (loaded on-demand based on context)
    if skills_content:
        system_prompt += f"\n\n{skills_content}"

    return system_prompt


def convert_tools_to_definitions(registry_tools: list) -> list[ToolDefinition]:
    """Convert registry tools to ToolDefinition format."""
    definitions = []
    for tool in registry_tools:
        # Handle both FunctionDeclaration and dict formats
        if hasattr(tool, "name"):
            definitions.append(ToolDefinition(
                name=tool.name,
                description=getattr(tool, "description", ""),
                parameters=getattr(tool, "parameters", {}) or {},
            ))
        elif isinstance(tool, dict):
            definitions.append(ToolDefinition(
                name=tool.get("name", ""),
                description=tool.get("description", ""),
                parameters=tool.get("parameters", {}),
            ))
    return definitions


async def chat_loop(
    project: str = "default",
    resume_session: str | None = None,
    session_name: str | None = None,
):
    """Main chat loop with automatic context management."""

    # 1. Detect mode and check configuration
    client_mode = ModelClient.get_mode()
    mode_info = ModelClient.get_mode_info()

    if client_mode == ClientMode.GATEWAY:
        if not mode_info.get("gateway_url"):
            console.print("[red]Error: POLYMATH_GATEWAY_URL not set[/red]")
            return
    else:
        # Direct mode - check for at least one provider
        providers = mode_info.get("providers", {})
        if not any(providers.values()):
            console.print("[red]Error: No API keys configured[/red]")
            console.print("[dim]Set at least one of: GOOGLE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY[/dim]")
            console.print("[dim]Or configure Gateway mode: POLYMATH_GATEWAY_URL[/dim]")
            return

    # 2. Initialize unified model client
    try:
        model_client = await ModelClient.create()
    except Exception as e:
        console.print(f"[red]Failed to initialize client: {e}[/red]")
        return

    # 3. Initialize tools (direct function calls, no MCP)
    registry = get_default_registry()
    sensitive_tools = registry.get_sensitive_tools()

    # 4. Initialize tool selector (按模式分配工具)
    tool_selector = ToolSelector()

    # 5. Initialize context manager
    ctx_config = ContextConfig(
        max_context_tokens=100_000,
        summarize_threshold=0.7,
        keep_recent_messages=6,
        auto_save=True,
    )
    ctx = ContextManager(project=project, config=ctx_config)

    # 6. Initialize skill manager
    skill_mgr = SkillManager(project_dir=Path.cwd(), max_skill_tokens=5000)
    active_skills_content = ""  # Will be populated based on context

    # 7. Initialize new systems
    # Checkpoint manager
    checkpoint_mgr = CheckpointManager(
        project=project,
        config=CheckpointConfig(enabled=True, retention_days=30),
    )

    # Background task manager
    task_mgr = get_task_manager()

    # Hook manager
    hook_mgr = HookManager(
        project_dir=Path.cwd(),
        session_id=ctx.session_id,
        permission_mode="default",
    )
    # Load hooks from config if exists
    hooks_file = Path(".polymath/hooks.json")
    if hooks_file.exists():
        hook_mgr.load_from_file(hooks_file)

    # Cost tracker
    budget_config = BudgetConfig(
        daily_limit_usd=float(os.getenv("POLYMATH_DAILY_BUDGET", "0")) or None,
        session_limit_usd=float(os.getenv("POLYMATH_SESSION_BUDGET", "0")) or None,
    )
    cost_tracker = CostTracker(
        project=project,
        session_id=ctx.session_id,
        budget=budget_config,
    )

    # Command history
    cmd_history = CommandHistory(project=project)
    cmd_history.setup_readline()

    # Bash mode executor
    bash_executor = BashModeExecutor(cwd=Path.cwd())

    # Session manager
    session_mgr = SessionManager()

    # Handle session resume
    if resume_session:
        session_data = session_mgr.resume_session(resume_session)
        if session_data:
            console.print(f"[green]Resumed session: {session_data.metadata.get_display_name()}[/green]")
            # Restore messages to context
            for msg in session_data.messages:
                if msg.get("role") == "user":
                    ctx.messages.append(ctx.messages[0].__class__(
                        role="user",
                        content=msg.get("content", ""),
                    ))
        else:
            console.print(f"[yellow]Session not found: {resume_session}[/yellow]")

    # Show startup info
    if client_mode == ClientMode.GATEWAY:
        console.print(f"[bold cyan]Mode: Gateway[/bold cyan] ({mode_info.get('gateway_url')})")
    else:
        active_providers = [p for p, v in mode_info.get("providers", {}).items() if v]
        console.print(f"[bold green]Mode: Direct[/bold green] ({', '.join(active_providers)})")
    console.print(f"[bold yellow]Project: {project}[/bold yellow]")

    stats = ctx.get_context_stats()
    if stats["messages"] > 0 or stats["summaries"] > 0:
        console.print(
            f"[dim]Restored context: {stats['messages']} messages, "
            f"{stats['summaries']} summaries[/dim]"
        )

    # 8. State
    mode = "build"  # Default to build mode
    turn_count = 0
    model_name = os.getenv("POLYMATH_MODEL", "gemini-2.5-flash-preview")

    # Get tools for current mode
    tool_names = tool_selector.get_tools_for_mode(mode)
    genai_tools = registry.get_genai_tools(tool_names)
    tool_definitions = convert_tools_to_definitions(genai_tools)
    console.print(f"[dim]Tools: {len(tool_definitions)} ({mode} mode)[/dim]")

    # Build system prompt
    system_prompt = build_system_prompt(mode, active_skills_content)

    # Conversation history for unified client
    conversation_history: list[Message] = []

    # Restore history from context
    history = ctx.get_history_for_api()
    for msg in history:
        if hasattr(msg, "role") and hasattr(msg, "parts"):
            role = msg.role
            content = ""
            for part in msg.parts:
                if hasattr(part, "text"):
                    content += part.text
            conversation_history.append(Message(role=role, content=content))

    # Trigger SessionStart hook
    await hook_mgr.trigger(
        HookEvent.SESSION_START,
        message_count=len(ctx.messages),
    )

    console.print(
        Panel.fit(
            f"[bold blue]Polymath[/bold blue]\n[dim]Type /help for commands. Mode: {mode}[/dim]",
            border_style="blue",
        )
    )

    # 9. Main loop
    while True:
        mode_color = MODE_COLORS.get(mode, "yellow")

        # Show running tasks
        running_tasks = task_mgr.get_running_tasks()
        if running_tasks:
            console.print(f"[dim cyan]⏳ {len(running_tasks)} background task(s) running[/dim cyan]")

        # Check budget warning
        budget_status = cost_tracker.check_budget()
        if budget_status.get("warning"):
            console.print(f"[yellow]⚠️ {budget_status['warning']}[/yellow]")

        user_input = Prompt.ask(f"\n[bold {mode_color}]You ({mode})[/bold {mode_color}]")

        # Exit
        if user_input.lower() in ["exit", "quit", "/exit"]:
            # Trigger SessionEnd hook
            await hook_mgr.trigger(HookEvent.SESSION_END, message_count=len(ctx.messages))
            # Close model client
            await model_client.close()
            break

        # Bash mode (! prefix)
        if user_input.startswith("!"):
            bash_cmd = user_input[1:].strip()
            if bash_cmd:
                console.print(f"[dim]$ {bash_cmd}[/dim]")
                result = bash_executor.execute(bash_cmd)
                if result["output"]:
                    console.print(result["output"])
                if result["error"]:
                    console.print(f"[red]{result['error']}[/red]")
                # Add to context
                ctx.add_user_message(bash_executor.execute_for_context(bash_cmd))
                cmd_history.add(user_input)
            continue

        # Slash commands
        if user_input.startswith("/"):
            cmd_parts = user_input[1:].split()
            cmd = cmd_parts[0].lower() if cmd_parts else ""
            cmd_args = cmd_parts[1:] if len(cmd_parts) > 1 else []

            if cmd == "help":
                console.print("""
[bold]Commands:[/bold]
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
                continue

            elif cmd == "mode":
                if cmd_args:
                    new_mode = cmd_args[0].lower()
                    if new_mode in MODE_COLORS:
                        mode = new_mode
                        # Update tools for new mode
                        tool_names = tool_selector.get_tools_for_mode(mode)
                        genai_tools = registry.get_genai_tools(tool_names)
                        tool_definitions = convert_tools_to_definitions(genai_tools)
                        hook_mgr.permission_mode = mode
                        # Rebuild system prompt with new mode
                        system_prompt = build_system_prompt(mode, active_skills_content)
                        console.print(f"[green]Switched to {mode} mode ({len(tool_definitions)} tools)[/green]")
                    else:
                        console.print(f"[red]Unknown mode: {new_mode}[/red]")
                else:
                    console.print(f"Current mode: {mode}")
                continue

            elif cmd == "context":
                stats = ctx.get_context_stats()
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
                active = skill_mgr.get_active_skills()
                table.add_row("Active Skills", ", ".join(active) if active else "(none)")
                table.add_row("Mode", mode)
                table.add_row("Loaded Tools", f"{len(tool_names)}")
                if stats["needs_summary"]:
                    table.add_row("Status", "[yellow]Summary needed[/yellow]")
                console.print(table)
                continue

            elif cmd == "skills":
                console.print("[bold]Skills System[/bold]")
                active = skill_mgr.get_active_skills()
                if active:
                    console.print(f"  [green]Active:[/green] {', '.join(active)}")
                else:
                    console.print("  [dim]No skills currently active[/dim]")
                console.print(
                    "\n[dim]Skills are loaded automatically based on conversation context.[/dim]"
                )
                console.print(
                    "[dim]Put SKILL.md files in .polymath/skills/<name>/ to add custom skills.[/dim]"
                )
                continue

            elif cmd == "clear":
                ctx.clear(keep_summaries=True)
                # Clear conversation history but keep system prompt
                conversation_history.clear()
                console.print("[green]Conversation cleared (summaries preserved)[/green]")
                continue

            elif cmd == "reset":
                ctx.reset()
                active_skills_content = ""
                mode = "build"
                tool_names = tool_selector.get_tools_for_mode(mode)
                genai_tools = registry.get_genai_tools(tool_names)
                tool_definitions = convert_tools_to_definitions(genai_tools)
                system_prompt = build_system_prompt(mode, "")
                conversation_history.clear()
                turn_count = 0
                console.print("[green]Full reset complete[/green]")
                continue

            elif cmd == "tools":
                categories = tool_selector.get_tool_categories()
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
                continue

            elif cmd == "debug":
                console.print(f"Mode: {mode}")
                console.print(f"Turn: {turn_count}")
                console.print(f"Tools: {len(tool_names)} loaded")
                console.print(f"MCP Tools: {tool_selector.mcp_tools or '(none)'}")
                console.print(f"Project: {project}")
                stats = ctx.get_context_stats()
                console.print(f"Context: {stats['messages']} msgs, {stats['summaries']} summaries")
                console.print(f"Checkpoints: {len(checkpoint_mgr.checkpoints)}")
                console.print(f"Background Tasks: {task_mgr.get_running_count()} running")
                continue

            # Session commands
            elif cmd == "sessions":
                sessions = session_mgr.list_sessions(project=project, limit=10)
                if sessions:
                    table = Table(title="Recent Sessions")
                    table.add_column("ID", style="cyan")
                    table.add_column("Name", style="green")
                    table.add_column("Messages", style="yellow")
                    table.add_column("Updated")
                    for s in sessions:
                        from datetime import datetime
                        updated = datetime.fromtimestamp(s.updated_at).strftime("%Y-%m-%d %H:%M")
                        table.add_row(s.id[:8], s.name or "-", str(s.message_count), updated)
                    console.print(table)
                else:
                    console.print("[dim]No sessions found[/dim]")
                continue

            elif cmd == "resume":
                if cmd_args:
                    session_data = session_mgr.resume_session(cmd_args[0])
                    if session_data:
                        console.print(f"[green]Resumed: {session_data.metadata.get_display_name()}[/green]")
                    else:
                        console.print(f"[red]Session not found: {cmd_args[0]}[/red]")
                else:
                    console.print("[yellow]Usage: /resume <session_id or name>[/yellow]")
                continue

            elif cmd == "rename":
                if cmd_args:
                    new_name = " ".join(cmd_args)
                    session_mgr.rename_session(ctx.session_id, new_name)
                    console.print(f"[green]Session renamed to: {new_name}[/green]")
                else:
                    console.print("[yellow]Usage: /rename <new_name>[/yellow]")
                continue

            elif cmd == "export":
                try:
                    path = cmd_args[0] if cmd_args else f"session_{ctx.session_id[:8]}.md"
                    content = session_mgr.export_session(ctx.session_id, format="markdown", path=path)
                    console.print(f"[green]Exported to: {path}[/green]")
                except Exception as e:
                    console.print(f"[red]Export failed: {e}[/red]")
                continue

            elif cmd == "fork":
                try:
                    forked = session_mgr.fork_session(ctx.session_id)
                    if forked:
                        console.print(f"[green]Forked session: {forked.metadata.id}[/green]")
                    else:
                        console.print("[red]Fork failed[/red]")
                except Exception as e:
                    console.print(f"[red]Fork failed: {e}[/red]")
                continue

            # Checkpoint commands
            elif cmd == "checkpoints":
                cps = checkpoint_mgr.list_checkpoints(limit=10)
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
                continue

            elif cmd == "rewind":
                try:
                    if cmd_args:
                        result = checkpoint_mgr.rewind(cmd_args[0], mode="code")
                    else:
                        result = checkpoint_mgr.rewind_last(mode="code")

                    if result:
                        console.print(f"[green]Rewound: {len(result['restored_files'])} files restored[/green]")
                        if result["failed_files"]:
                            console.print(f"[yellow]Failed: {len(result['failed_files'])} files[/yellow]")
                    else:
                        console.print("[yellow]No checkpoint to rewind to[/yellow]")
                except Exception as e:
                    console.print(f"[red]Rewind failed: {e}[/red]")
                continue

            # Task commands
            elif cmd == "tasks":
                tasks = task_mgr.list_tasks(limit=10)
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
                continue

            elif cmd == "task":
                if cmd_args:
                    output = task_mgr.get_task_output(cmd_args[0])
                    if output:
                        console.print(Panel(output, title=f"Task {cmd_args[0]}", border_style="cyan"))
                    else:
                        console.print(f"[red]Task not found: {cmd_args[0]}[/red]")
                else:
                    console.print("[yellow]Usage: /task <task_id>[/yellow]")
                continue

            # Cost commands
            elif cmd == "cost":
                summary = cost_tracker.get_cost_summary()
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
                continue

            # Agent commands
            elif cmd == "agents":
                try:
                    from src.core.subagents import BUILTIN_AGENTS
                    console.print("[bold]Available Subagents:[/bold]")
                    for name, config in BUILTIN_AGENTS.items():
                        console.print(f"  [cyan]{name}[/cyan]: {config.description}")
                except Exception as e:
                    console.print(f"[red]Error loading agents: {e}[/red]")
                continue

            # History commands
            elif cmd == "history":
                recent = cmd_history.get_recent(20)
                if recent:
                    console.print("[bold]Recent Commands:[/bold]")
                    for i, cmd in enumerate(recent, 1):
                        console.print(f"  {i}. {cmd[:60]}{'...' if len(cmd) > 60 else ''}")
                else:
                    console.print("[dim]No history[/dim]")
                continue

            # Model commands
            elif cmd == "model":
                model_mgr = ModelManager(default_model=model_name)
                if cmd_args:
                    new_model = cmd_args[0]
                    if model_mgr.switch_model(new_model):
                        model_name = model_mgr.get_current_model()
                        console.print(f"[green]Switched to model: {model_name}[/green]")
                    else:
                        console.print(f"[red]Unknown model: {new_model}[/red]")
                        console.print(model_mgr.format_model_list())
                else:
                    console.print(model_mgr.format_model_list())
                continue

            # Plugin commands
            elif cmd == "plugins":
                plugin_mgr = PluginManager(project_dir=Path.cwd())
                summary = plugin_mgr.get_summary()
                console.print(f"[bold]Plugins ({summary['enabled']} enabled, {summary['disabled']} disabled):[/bold]")
                for p in summary['plugins']:
                    status = "[green]✓[/green]" if p['enabled'] else "[dim]○[/dim]"
                    console.print(f"  {status} {p['name']} v{p['version']} ({p['scope']})")
                if not summary['plugins']:
                    console.print("[dim]No plugins installed[/dim]")
                continue

            elif cmd == "plugin":
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
                continue

            # Theme commands
            elif cmd == "theme":
                theme_mgr = ThemeManager()
                if cmd_args:
                    theme_name = cmd_args[0]
                    if theme_mgr.set_theme(theme_name):
                        console.print(f"[green]Theme set to: {theme_name}[/green]")
                    else:
                        console.print(f"[red]Theme not found: {theme_name}[/red]")
                else:
                    console.print(theme_mgr.format_theme_list())
                continue

            # Vim mode
            elif cmd == "vim":
                input_mgr = InputManager()
                new_mode = input_mgr.toggle_mode()
                console.print(f"[green]Input mode: {new_mode.value}[/green]")
                if new_mode == InputMode.VI:
                    console.print("[dim]Press Esc for normal mode, i for insert mode[/dim]")
                continue

            # Extended thinking
            elif cmd == "thinking":
                thinking_mgr = ThinkingManager()
                new_mode = thinking_mgr.toggle_mode()
                indicator = thinking_mgr.get_mode_indicator()
                console.print(f"[green]Thinking mode: {new_mode.value} {indicator}[/green]")
                continue

            # Doctor/health check
            elif cmd == "doctor":
                doctor = Doctor(project_dir=Path.cwd())
                console.print("[bold]Running health checks...[/bold]")
                results = doctor.run_all_checks()
                console.print(doctor.format_results(results))
                continue

            # Workspace commands
            elif cmd == "workspace":
                workspace_mgr = WorkspaceManager(primary_dir=Path.cwd())
                console.print("[bold]Workspace Directories:[/bold]")
                console.print(workspace_mgr.get_summary())
                continue

            elif cmd == "add-dir":
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
                continue

            else:
                console.print(f"[yellow]Unknown command: {cmd}[/yellow]")
                continue

        # Add to command history
        cmd_history.add(user_input)

        # Trigger UserPromptSubmit hook
        hook_result = await hook_mgr.trigger(
            HookEvent.USER_PROMPT_SUBMIT,
            user_prompt=user_input,
            message_count=len(ctx.messages),
        )

        if not hook_result.continue_processing:
            if hook_result.reason:
                console.print(f"[yellow]{hook_result.reason}[/yellow]")
            continue

        # Start checkpoint before processing
        checkpoint_mgr.begin_checkpoint(user_input, message_count=len(ctx.messages))

        # Track user message in context
        ctx.add_user_message(user_input)

        # Check if we need to load/update skills based on user input
        new_skills_content = skill_mgr.get_skills_for_context(user_input)
        if new_skills_content != active_skills_content:
            active_skills_content = new_skills_content
            new_active = skill_mgr.get_active_skills()
            if new_active:
                console.print(f"[dim cyan]Skills loaded: {', '.join(new_active)}[/dim cyan]")
            # Rebuild system prompt with new skills
            system_prompt = build_system_prompt(mode, active_skills_content)

        # Add user message to conversation history
        conversation_history.append(Message(role="user", content=user_input))

        # Build messages with system prompt
        messages_for_api = [Message(role="system", content=system_prompt)] + conversation_history

        # Send message using unified client
        try:
            with console.status(f"[bold {mode_color}]Thinking...[/bold {mode_color}]"):
                response = await model_client.chat(
                    messages_for_api,
                    tools=tool_definitions,
                    model=model_name,
                )
        except Exception as e:
            console.print(f"[red]API Error: {e}[/red]")
            # Remove the user message we just added
            conversation_history.pop()
            checkpoint_mgr.discard_checkpoint()
            continue

        # Process response (tool loop)
        accumulated_text = ""
        files_modified = []

        while True:
            # Check for empty response
            if not response.content and not response.tool_calls:
                console.print("[red]Empty response[/red]")
                break

            has_tool_call = response.has_tool_calls
            tool_results = []

            # Text response
            if response.content:
                accumulated_text += response.content
                console.print(
                    Panel(
                        Markdown(response.content),
                        title="[bold purple]Response[/bold purple]",
                        border_style="purple",
                        expand=False,
                    )
                )

            # Tool calls
            if response.tool_calls:
                for tc in response.tool_calls:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    args_str = func.get("arguments", "{}")
                    tool_call_id = tc.get("id", "")

                    # Parse arguments
                    try:
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except json.JSONDecodeError:
                        args = {}

                    # Trigger PreToolUse hook
                    pre_hook = await hook_mgr.trigger(
                        HookEvent.PRE_TOOL_USE,
                        tool_name=tool_name,
                        tool_input=args,
                    )

                    if pre_hook.decision.value == "deny":
                        tool_results.append({
                            "tool_call_id": tool_call_id,
                            "name": tool_name,
                            "result": f"Blocked: {pre_hook.reason}",
                        })
                        continue

                    if pre_hook.modified_input:
                        args = pre_hook.modified_input

                    # Inject project context for memory tools
                    if tool_name in ["save_note", "search_notes"]:
                        args["collection_name"] = project

                    # Snapshot file before modification
                    if tool_name in ["write_file", "edit_file"] and "path" in args:
                        checkpoint_mgr.snapshot_file(args["path"])
                        files_modified.append(args["path"])

                    # Show diff for write operations
                    if tool_name == "write_file" and "content" in args and "path" in args:
                        console.print(
                            f"\n[bold yellow]📝 Proposing changes:[/bold yellow] {args['path']}"
                        )
                        print_diff(args["path"], args["content"])

                    # HITL approval for sensitive tools
                    tool_result = None
                    if tool_name in sensitive_tools:
                        console.print(f"\n[bold red]⚠️ Sensitive:[/bold red] {tool_name}")
                        if tool_name != "write_file":
                            console.print(
                                f"[dim]{json.dumps(args, indent=2, ensure_ascii=False)}[/dim]"
                            )

                        if Prompt.ask("Execute?", choices=["y", "n"], default="n") != "y":
                            tool_result = "User denied the operation."
                            console.print("[red]Cancelled[/red]")
                        else:
                            console.print(f"[cyan]Running {tool_name}...[/cyan]")
                            tool_result = await registry.call_tool(tool_name, args)
                    else:
                        console.print(f"[cyan]Running {tool_name}...[/cyan]")
                        tool_result = await registry.call_tool(tool_name, args)

                    # Trigger PostToolUse hook
                    await hook_mgr.trigger(
                        HookEvent.POST_TOOL_USE,
                        tool_name=tool_name,
                        tool_input=args,
                        tool_output=tool_result,
                    )

                    tool_results.append({
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "result": str(tool_result) if tool_result else "",
                    })

            # No more tool calls - done with this turn
            if not has_tool_call:
                turn_count += 1
                usage = response.usage

                # Track usage and costs
                if usage:
                    cost_tracker.track(
                        model=model_name,
                        input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0),
                        session_id=ctx.session_id,
                    )

                # Track assistant response in context
                prompt_tokens = usage.get("prompt_tokens") if usage else None
                completion_tokens = usage.get("completion_tokens") if usage else None

                # Add assistant message to conversation history
                conversation_history.append(Message(role="assistant", content=accumulated_text))

                # Record messages before summary check
                prev_summary_count = len(ctx.summaries)
                ctx.add_assistant_message(
                    accumulated_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

                # Check if summarization occurred
                if len(ctx.summaries) > prev_summary_count:
                    await hook_mgr.trigger(HookEvent.PRE_COMPACT)
                    console.print(
                        "[dim yellow]Context summarized to save memory.[/dim yellow]"
                    )
                    # Rebuild conversation history from context
                    conversation_history.clear()
                    history = ctx.get_history_for_api()
                    for msg in history:
                        if hasattr(msg, "role") and hasattr(msg, "parts"):
                            content = ""
                            for part in msg.parts:
                                if hasattr(part, "text"):
                                    content += part.text
                            conversation_history.append(Message(role=msg.role, content=content))

                # Finalize checkpoint if files were modified
                if files_modified:
                    checkpoint_mgr.finalize_checkpoint(
                        description=f"Modified: {', '.join(files_modified)}"
                    )
                else:
                    checkpoint_mgr.discard_checkpoint()

                # Trigger Stop hook
                await hook_mgr.trigger(HookEvent.STOP, message_count=len(ctx.messages))

                # Show usage stats
                if usage:
                    stats = ctx.get_context_stats()
                    cost = cost_tracker.calculate_cost(
                        model_name,
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0),
                    )
                    console.print(
                        f"\n[dim]Turn {turn_count} | "
                        f"In: {usage.get('prompt_tokens', 0):,} | "
                        f"Out: {usage.get('completion_tokens', 0):,} | "
                        f"Cost: ${cost:.4f} | "
                        f"Ctx: {stats['usage_percent']}%[/dim]"
                    )
                break

            # Send tool results back - add assistant message with tool calls, then tool results
            conversation_history.append(Message(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls,
            ))

            for tr in tool_results:
                conversation_history.append(Message(
                    role="tool",
                    content=tr["result"],
                    tool_call_id=tr["tool_call_id"],
                    name=tr["name"],
                ))

            # Continue the conversation
            messages_for_api = [Message(role="system", content=system_prompt)] + conversation_history

            with console.status("[cyan]Processing...[/cyan]"):
                response = await model_client.chat(
                    messages_for_api,
                    tools=tool_definitions,
                    model=model_name,
                )


@app.command()
def start(
    project: str = typer.Option("default", "--project", "-p", help="Project name"),
    resume: str = typer.Option(None, "--resume", "-r", help="Resume session by ID or name"),
    name: str = typer.Option(None, "--name", "-n", help="Name for new session"),
):
    """Start Polymath CLI."""
    asyncio.run(chat_loop(project=project, resume_session=resume, session_name=name))


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
            from datetime import datetime
            updated = datetime.fromtimestamp(s.updated_at).strftime("%Y-%m-%d %H:%M")
            table.add_row(s.id[:8], s.name or "-", str(s.message_count), updated)

        console.print(table)
    else:
        console.print("[dim]No sessions found[/dim]")


@app.command()
def version():
    """Show version information."""
    console.print("[bold]Polymath v0.7.0[/bold]")
    console.print("[dim]Features: Gateway support, checkpointing, subagents, hooks, cost tracking[/dim]")
    # Show mode
    gateway_url = os.getenv("POLYMATH_GATEWAY_URL")
    if gateway_url:
        console.print(f"[cyan]Mode: Gateway ({gateway_url})[/cyan]")
    else:
        console.print("[green]Mode: Direct (using API keys)[/green]")


def entry_point():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    app()
