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
import json
import logging
import os
import sys
from pathlib import Path

import typer

logger = logging.getLogger(__name__)
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from src.core.background_tasks import get_task_manager

# New feature imports
from src.core.checkpoint import CheckpointConfig, CheckpointManager
from src.core.command_history import BashModeExecutor, CommandHistory

# Direct imports (no DI container needed)
from src.core.config import load_config
from src.core.context_manager import ContextConfig, ContextManager
from src.core.cost_tracker import BudgetConfig, CostTracker
from src.core.diff import print_diff
from src.core.hooks import HookEvent, HookManager
from src.core.input_mode import InputManager, InputMode

# Unified Model Client (supports Gateway and Direct modes)
from src.core.model_client import (
    ClientMode,
    Message,
    ModelClient,
    ToolDefinition,
    ChatResponse,
)
from src.core.prompts import get_system_prompt
from src.core.rules import format_instructions_for_prompt, load_all_instructions
from src.core.session import SessionManager
from src.core.skills import SkillManager
from src.core.tool_selector import ToolSelector
from src.host.cli.commands import CommandHandler, MODE_COLORS
from src.host.tools import get_default_registry

# Fix encoding
for stream in [sys.stdin, sys.stdout, sys.stderr]:
    if stream and hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

app = typer.Typer()
console = Console()


def build_system_prompt(mode: str, skills_content: str = "") -> str:
    """Build the system prompt with mode, rules, and skills."""
    config = load_config()
    persona = config.get("persona", {})

    # Build system prompt
    system_prompt = get_system_prompt(mode, persona)

    # Add project rules (DORAEMON.md)
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
    prompt: str | None = None,
):
    """Main chat loop with automatic context management."""

    # Check for piped input (Headless detection)
    piped_input = None
    if not sys.stdin.isatty():
        try:
            piped_input = sys.stdin.read().strip()
        except Exception:
            pass

    initial_prompt = prompt
    if piped_input:
        if initial_prompt:
            initial_prompt = f"{initial_prompt}\n{piped_input}"
        else:
            initial_prompt = piped_input

    headless = bool(initial_prompt)
    if headless:
        console.print("[dim cyan]Running in headless mode[/dim cyan]")

    client_mode = ModelClient.get_mode()
    mode_info = ModelClient.get_mode_info()

    if client_mode == ClientMode.GATEWAY:
        if not mode_info.get("gateway_url"):
            console.print("[red]Error: DORAEMON_GATEWAY_URL not set[/red]")
            return
    else:
        # Direct mode - check for at least one provider
        providers = mode_info.get("providers", {})
        if not any(providers.values()):
            console.print("[red]Error: No API keys configured[/red]")
            console.print("[dim]Set at least one of: GOOGLE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY[/dim]")
            console.print("[dim]Or configure Gateway mode: DORAEMON_GATEWAY_URL[/dim]")
            return

    # Initialize unified model client
    try:
        model_client = await ModelClient.create()
    except Exception as e:
        console.print(f"[red]Failed to initialize client: {e}[/red]")
        return

    # Initialize tools (direct function calls, no MCP)
    registry = get_default_registry()
    sensitive_tools = registry.get_sensitive_tools()

    # Initialize tool selector (按模式分配工具)
    tool_selector = ToolSelector()

    # Initialize context manager
    ctx_config = ContextConfig(
        max_context_tokens=100_000,
        summarize_threshold=0.7,
        keep_recent_messages=6,
        auto_save=True,
    )
    ctx = ContextManager(project=project, config=ctx_config)

    # Initialize skill manager
    skill_mgr = SkillManager(project_dir=Path.cwd(), max_skill_tokens=5000)
    active_skills_content = ""

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
    hooks_file = Path(".doraemon/hooks.json")
    if hooks_file.exists():
        hook_mgr.load_from_file(hooks_file)

    # Cost tracker
    budget_config = BudgetConfig(
        daily_limit_usd=float(os.getenv("DORAEMON_DAILY_BUDGET", "0")) or None,
        session_limit_usd=float(os.getenv("DORAEMON_SESSION_BUDGET", "0")) or None,
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
            for msg in session_data.messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role and content:
                    ctx.add_user_message(content) if role == "user" else ctx.add_assistant_message(content)
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

    # State
    mode = "build"
    turn_count = 0
    model_name = os.getenv("DORAEMON_MODEL", "gemini-3-pro-preview")

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

    # Initialize command handler
    cmd_handler = CommandHandler(
        ctx=ctx,
        tool_selector=tool_selector,
        registry=registry,
        skill_mgr=skill_mgr,
        checkpoint_mgr=checkpoint_mgr,
        task_mgr=task_mgr,
        cost_tracker=cost_tracker,
        cmd_history=cmd_history,
        session_mgr=session_mgr,
        hook_mgr=hook_mgr,
        model_name=model_name,
        project=project,
    )

    console.print(
        Panel.fit(
            f"[bold blue]🤖 Doraemon Code[/bold blue]\n[dim]Type /help for commands. Mode: {mode}[/dim]",
            border_style="blue",
        )
    )

    # Main loop
    try:
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

            # Determine user input
            if initial_prompt:
                user_input = initial_prompt
                initial_prompt = None
                console.print(f"\n[bold {mode_color}]> {user_input}[/bold {mode_color}]")
            elif headless:
                break
            else:
                user_input = Prompt.ask(f"\n[bold {mode_color}]You ({mode})[/bold {mode_color}]")

            # Exit
            if user_input.lower() in ["exit", "quit", "/exit"]:
                await hook_mgr.trigger(HookEvent.SESSION_END, message_count=len(ctx.messages))
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
                    ctx.add_user_message(bash_executor.execute_for_context(bash_cmd))
                    cmd_history.add(user_input)
                continue

            # Slash commands
            if user_input.startswith("/"):
                cmd_parts = user_input[1:].split()
                cmd = cmd_parts[0].lower() if cmd_parts else ""
                cmd_args = cmd_parts[1:] if len(cmd_parts) > 1 else []

                result = await cmd_handler.handle(
                    cmd=cmd,
                    cmd_args=cmd_args,
                    mode=mode,
                    tool_names=tool_names,
                    tool_definitions=tool_definitions,
                    conversation_history=conversation_history,
                    active_skills_content=active_skills_content,
                    build_system_prompt=build_system_prompt,
                    convert_tools_to_definitions=convert_tools_to_definitions,
                    sensitive_tools=sensitive_tools,
                )

                # Update state from command result
                mode = result["mode"]
                tool_names = result["tool_names"]
                tool_definitions = result["tool_definitions"]
                active_skills_content = result["active_skills_content"]
                conversation_history = result["conversation_history"]
                if result["system_prompt"]:
                    system_prompt = result["system_prompt"]
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

            # Check if we need to load/update skills
            new_skills_content = skill_mgr.get_skills_for_context(user_input)
            if new_skills_content != active_skills_content:
                active_skills_content = new_skills_content
                new_active = skill_mgr.get_active_skills()
                if new_active:
                    console.print(f"[dim cyan]Skills loaded: {', '.join(new_active)}[/dim cyan]")
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
                conversation_history.pop()
                checkpoint_mgr.discard_checkpoint()
                continue
            
            # Process response (tool loop)
            accumulated_text = ""
            files_modified = []
            tool_steps = 0
            MAX_TOOL_STEPS = 15  # Prevent infinite tool loops
            previous_tool_calls = []  # Specific for loop detection
            
            while True:
                if tool_steps >= MAX_TOOL_STEPS:
                    console.print(f"[red]⚠️ Max tool steps ({MAX_TOOL_STEPS}) reached. Stopping to prevent infinite loop.[/red]")
                    # Add a system message to context explaining why it stopped
                    ctx.add_system_message(f"System: Execution stopped because maximum tool steps ({MAX_TOOL_STEPS}) were exceeded.")
                    break

                tool_steps += 1
                if not response.content and not response.tool_calls:
                    console.print("[red]Empty response[/red]")
                    break

                has_tool_call = response.has_tool_calls
                tool_results = []

                # Thought display
                if response.thought:
                    console.print(
                        Panel(
                            Markdown(response.thought),
                            title="[bold dim]Thinking[/bold dim]",
                            border_style="dim white",
                            expand=False,
                        )
                    )

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
                        tool_call_id = tc.get("id", "")

                        try:
                            # Extract arguments
                            args_raw = func.get("arguments", {})
                            if isinstance(args_raw, str):
                                args = json.loads(args_raw)
                            else:
                                args = args_raw

                            # Normalize for comparison
                            args_str_normalized = json.dumps(args, sort_keys=True)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse tool arguments for {tool_name}: {args_raw}, error: {e}")
                            args = {}
                            args_str_normalized = "{}"

                        # Loop Detection
                        current_call_signature = f"{tool_name}:{args_str_normalized}"
                        previous_tool_calls.append(current_call_signature)
                        
                        # Check last 3 calls
                        if len(previous_tool_calls) >= 3:
                            last_three = previous_tool_calls[-3:]
                            if all(s == current_call_signature for s in last_three):
                                console.print(f"[red]⚠️ Loop detected: {tool_name} called repeatedly with same args.[/red]")
                                tool_results.append({
                                    "tool_call_id": tool_call_id,
                                    "name": tool_name,
                                    "result": "Error: Loop detected. You have called this tool with the same arguments 3 times in a row. Stop and rethink your approach.",
                                })
                                continue

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
                            console.print(f"\n[bold yellow]📝 Proposing changes:[/bold yellow] {args['path']}")
                            print_diff(args["path"], args["content"])

                        # HITL approval for sensitive tools
                        tool_result = None
                        if tool_name in sensitive_tools:
                            console.print(f"\n[bold red]⚠️ Sensitive:[/bold red] {tool_name}")
                            if tool_name != "write_file":
                                # Pretty print content if present
                                if "content" in args and isinstance(args["content"], str):
                                    display_args = args.copy()
                                    content = display_args.pop("content")
                                    console.print(f"[dim]{json.dumps(display_args, indent=2, ensure_ascii=False)}[/dim]")
                                    console.print(Panel(Markdown(content), title="[bold]Content Preview[/bold]", border_style="dim"))
                                else:
                                    console.print(f"[dim]{json.dumps(args, indent=2, ensure_ascii=False)}[/dim]")

                            if headless:
                                console.print("[red]Headless mode: Sensitive tool denied[/red]")
                                tool_result = "Headless mode: Sensitive tool execution denied."
                            elif Prompt.ask("Execute?", choices=["y", "n"], default="n") != "y":
                                tool_result = "User denied the operation."
                                console.print("[red]Cancelled[/red]")
                            else:
                                console.print(f"[cyan]Running {tool_name}...[/cyan]")
                                tool_result = await registry.call_tool(tool_name, args)
                        else:
                            console.print(f"[cyan]Running {tool_name}...[/cyan]")
                            try:
                                tool_result = await registry.call_tool(tool_name, args)
                            except Exception as e:
                                logger.error(f"Tool {tool_name} failed: {e}", exc_info=True)
                                tool_result = f"Error executing {tool_name}: {e}"

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

                        # Special handling for switch_mode
                        if tool_name == "switch_mode" and "mode" in args:
                            new_mode = args["mode"]
                            if new_mode in ["plan", "build"]:
                                mode = new_mode
                                console.print(f"[bold cyan]🔄 Automatic mode switch to: {mode}[/bold cyan]")
                                # Update available tools immediately
                                tool_names = tool_selector.get_tools_for_mode(mode)
                                genai_tools = registry.get_genai_tools(tool_names)
                                tool_definitions = convert_tools_to_definitions(genai_tools)
                                system_prompt = build_system_prompt(mode, active_skills_content)
                                console.print(f"[dim]Tools updated for {mode} mode ({len(tool_definitions)} tools)[/dim]")

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

                    prev_summary_count = len(ctx.summaries)
                    ctx.add_assistant_message(
                        accumulated_text,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )

                    # Check if summarization occurred
                    if len(ctx.summaries) > prev_summary_count:
                        await hook_mgr.trigger(HookEvent.PRE_COMPACT)
                        console.print("[dim yellow]Context summarized to save memory.[/dim yellow]")
                        conversation_history.clear()
                        history = ctx.get_history_for_api()
                        for msg in history:
                            if hasattr(msg, "role") and hasattr(msg, "parts"):
                                content = ""
                                for part in msg.parts:
                                    if hasattr(part, "text"):
                                        content += part.text
                                conversation_history.append(Message(role=msg.role, content=content))

                    # Finalize checkpoint
                    if files_modified:
                        checkpoint_mgr.finalize_checkpoint(description=f"Modified: {', '.join(files_modified)}")
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

                # Send tool results back
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

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
    except Exception as e:
        console.print(f"[red]Error in chat loop: {e}[/red]")
        import traceback
        traceback.print_exc()
    finally:
        try:
            await hook_mgr.trigger(HookEvent.SESSION_END, message_count=len(ctx.messages))
        except Exception:
            pass
        try:
            if model_client:
                await model_client.close()
        except Exception:
            pass
        console.print("[dim]Session ended[/dim]")


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
            from datetime import datetime
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
