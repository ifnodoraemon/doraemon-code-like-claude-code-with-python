"""
Main Chat Loop

Core conversation loop with tool execution, context management, and response handling.
Extracted from main.py for better maintainability.
"""

import asyncio
import json
import logging
import sys

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from src.core.config import load_config
from src.core.hooks import HookEvent
from src.core.model_client import ClientMode, Message, ToolDefinition
from src.core.model_utils import ChatResponse
from src.core.parallel_executor import DependencyAnalyzer
from src.core.parallel_executor import ToolCall as PToolCall
from src.core.prompts import get_system_prompt
from src.core.rules import (
    format_instructions_for_prompt,
    format_memory_for_prompt,
    load_all_instructions,
    load_global_memory,
    load_project_memory,
)
from src.host.cli.commands import MODE_COLORS, CommandHandler
from src.host.cli.initialization import initialize_all_managers
from src.host.cli.tool_execution import (
    detect_tool_loop,
    execute_tool,
    parse_tool_arguments,
)

logger = logging.getLogger(__name__)
console = Console()

MAX_TOOL_STEPS = 15  # Prevent infinite tool loops


def _is_context_overflow(error: Exception) -> bool:
    """Check if an error is caused by context window overflow."""
    msg = str(error).lower()
    indicators = [
        "context length",
        "context window",
        "token limit",
        "max.*token",
        "too many tokens",
        "request too large",
        "content too large",
        "maximum context",
        "exceeds the model",
        "prompt is too long",
        "input is too long",
    ]
    return any(indicator in msg for indicator in indicators)


def expand_file_references(text: str) -> str:
    """
    Expand @file references in user input.

    Supports:
    - @./path/to/file - Include file content
    - @./directory/ - Include directory listing
    - @file.txt - Relative to current directory

    Returns:
        Text with file references expanded
    """
    import re
    from pathlib import Path

    # Pattern: @ followed by path (not starting with space)
    pattern = r'@(\.?/?[\w\-./]+)'

    def replace_reference(match):
        ref_path = match.group(1)
        path = Path(ref_path)

        try:
            # Security: resolve and check path is within cwd
            resolved = path.resolve()
            cwd = Path.cwd().resolve()
            if not str(resolved).startswith(str(cwd)):
                logger.warning(f"Blocked @reference outside workspace: {ref_path}")
                return match.group(0)

            if path.is_file():
                # Read file content
                content = path.read_text(encoding="utf-8", errors="replace")
                # Truncate if too large
                if len(content) > 50000:
                    content = content[:50000] + "\n... [truncated]"
                return f"\n```{path.suffix[1:] if path.suffix else 'text'}\n# File: {path}\n{content}\n```\n"

            elif path.is_dir():
                # List directory
                files = sorted(path.iterdir())
                listing = []
                for f in files[:100]:  # Limit to 100 entries
                    prefix = "📁 " if f.is_dir() else "📄 "
                    listing.append(f"{prefix}{f.name}")
                if len(files) > 100:
                    listing.append(f"... and {len(files) - 100} more")
                return f"\n```\n# Directory: {path}/\n" + "\n".join(listing) + "\n```\n"

            else:
                # Path doesn't exist, keep original
                return match.group(0)

        except Exception as e:
            logger.warning(f"Failed to expand @{ref_path}: {e}")
            return match.group(0)

    return re.sub(pattern, replace_reference, text)


def build_system_prompt(mode: str, skills_content: str = "") -> str:
    """Build the system prompt with mode, rules, memory, and skills."""
    config = load_config()
    persona = config.get("persona", {})

    # Build system prompt
    system_prompt = get_system_prompt(mode, persona)

    # Add project rules (DORAEMON.md)
    instructions = load_all_instructions(config)
    if instructions:
        system_prompt += format_instructions_for_prompt(instructions)

    # Add project memory (.doraemon/MEMORY.md and ~/.doraemon/MEMORY.md)
    project_memory = load_project_memory()
    global_memory = load_global_memory()
    combined_memory = ""
    if project_memory:
        combined_memory += project_memory
    if global_memory:
        if combined_memory:
            combined_memory += "\n\n"
        combined_memory += global_memory
    if combined_memory:
        system_prompt += format_memory_for_prompt(combined_memory)

    # Add active skills (loaded on-demand based on context)
    if skills_content:
        system_prompt += f"\n\n{skills_content}"

    return system_prompt


def convert_tools_to_definitions(registry_tools: list) -> list:
    """Convert registry tools to ToolDefinition format."""

    definitions = []
    for tool in registry_tools:
        # Handle both FunctionDeclaration and dict formats
        if hasattr(tool, "name"):
            definitions.append(
                ToolDefinition(
                    name=tool.name,
                    description=getattr(tool, "description", ""),
                    parameters=getattr(tool, "parameters", {}) or {},
                )
            )
        elif isinstance(tool, dict):
            definitions.append(
                ToolDefinition(
                    name=tool.get("name", ""),
                    description=tool.get("description", ""),
                    parameters=tool.get("parameters", {}),
                )
            )
    return definitions


def check_piped_input() -> tuple[str | None, bool]:
    """
    Check for piped input (headless mode detection).

    Returns:
        tuple of (piped_input, is_headless)
    """
    piped_input = None
    if not sys.stdin.isatty():
        try:
            piped_input = sys.stdin.read().strip()
        except Exception:
            pass

    return piped_input, bool(piped_input)


def validate_client_mode(model_client) -> bool:
    """
    Validate that the model client is properly configured.

    Returns:
        True if valid, False otherwise
    """
    client_mode = model_client.get_mode()
    mode_info = model_client.get_mode_info()

    if client_mode == ClientMode.GATEWAY:
        if not mode_info.get("gateway_url"):
            console.print("[red]Error: DORAEMON_GATEWAY_URL not set[/red]")
            return False
    else:
        # Direct mode - check for at least one provider
        providers = mode_info.get("providers", {})
        if not any(providers.values()):
            console.print("[red]Error: No API keys configured[/red]")
            console.print(
                "[dim]Set at least one of: GOOGLE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY[/dim]"
            )
            console.print("[dim]Or configure Gateway mode: DORAEMON_GATEWAY_URL[/dim]")
            return False

    return True


def restore_session_history(session_mgr, ctx, resume_session: str | None):
    """Restore session history if resuming a session."""
    if resume_session:
        session_data = session_mgr.resume_session(resume_session)
        if session_data:
            console.print(
                f"[green]Resumed session: {session_data.metadata.get_display_name()}[/green]"
            )
            for msg in session_data.messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role and content:
                    ctx.add_user_message(content) if role == "user" else ctx.add_assistant_message(
                        content
                    )
        else:
            console.print(f"[yellow]Session not found: {resume_session}[/yellow]")


def show_startup_info(model_client, project: str, ctx):
    """Display startup information."""
    client_mode = model_client.get_mode()
    mode_info = model_client.get_mode_info()

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


def restore_conversation_history(ctx) -> list[Message]:
    """Restore conversation history from context."""
    conversation_history: list[Message] = []
    history = ctx.get_history_for_api()
    for msg in history:
        if hasattr(msg, "role") and hasattr(msg, "parts"):
            role = msg.role
            content = ""
            for part in msg.parts:
                if hasattr(part, "text"):
                    content += part.text
            conversation_history.append(Message(role=role, content=content))
    return conversation_history


async def handle_bash_mode(user_input: str, bash_executor, ctx, cmd_history):
    """Handle bash mode (! prefix) input."""
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


def _merge_tool_call_delta(accumulated: list[dict], delta: dict):
    """Merge a streaming tool_call delta into accumulated list."""
    index = delta.get("index", len(accumulated))
    while len(accumulated) <= index:
        accumulated.append(
            {
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            }
        )
    tc = accumulated[index]
    if delta.get("id"):
        tc["id"] = delta["id"]
    func_delta = delta.get("function", {})
    if func_delta.get("name"):
        tc["function"]["name"] += func_delta["name"]
    if func_delta.get("arguments"):
        tc["function"]["arguments"] += func_delta["arguments"]


async def stream_model_response(
    model_client,
    messages: list[Message],
    tools,
    model_name: str,
) -> ChatResponse:
    """
    Stream model response, printing text in real-time.

    Accumulates the full response (text + tool_calls) for agentic loop.
    Falls back to non-streaming on error.

    Returns:
        ChatResponse with accumulated content, tool_calls, usage.
    """
    accumulated_content = ""
    accumulated_thought = ""
    accumulated_tool_calls: list[dict] = []
    last_usage = None
    finish_reason = None

    try:
        stream = model_client.chat_stream(messages, tools=tools, model=model_name)

        with Live("", console=console, refresh_per_second=12, transient=True) as live:
            async for chunk in stream:
                if chunk.thought:
                    accumulated_thought += chunk.thought
                if chunk.content:
                    accumulated_content += chunk.content
                    live.update(Markdown(accumulated_content))
                if chunk.tool_calls:
                    for tc_delta in chunk.tool_calls:
                        _merge_tool_call_delta(accumulated_tool_calls, tc_delta)
                if chunk.finish_reason:
                    finish_reason = chunk.finish_reason
                if chunk.usage:
                    last_usage = chunk.usage

        # Parse accumulated tool_call arguments from JSON strings
        for tc in accumulated_tool_calls:
            func = tc.get("function", {})
            args = func.get("arguments", "")
            if isinstance(args, str) and args:
                try:
                    func["arguments"] = json.loads(args)
                except json.JSONDecodeError:
                    pass

    except Exception as e:
        logger.debug(f"Streaming failed, falling back to non-streaming: {e}")
        response = await model_client.chat(messages, tools=tools, model=model_name)
        return response

    return ChatResponse(
        content=accumulated_content or None,
        thought=accumulated_thought or None,
        tool_calls=accumulated_tool_calls or None,
        finish_reason=finish_reason,
        usage=last_usage,
    )


async def process_tool_calls(
    response,
    registry,
    sensitive_tools: set,
    checkpoint_mgr,
    hook_mgr,
    ctx,
    headless: bool,
    model_name: str,
    cost_tracker,
    model_client=None,
    conversation_history: list[Message] | None = None,
    tool_definitions=None,
    system_prompt: str = "",
    permission_mgr=None,
) -> tuple[str, list[str], list[Message]]:
    """
    Process tool calls from model response with agentic loop.

    After executing tools, sends results back to the model for follow-up
    reasoning. Continues until the model responds with text only (no tool calls).

    Returns:
        tuple of (accumulated_text, files_modified, tool_results_messages)
    """
    accumulated_text = ""
    files_modified = []
    tool_steps = 0
    previous_tool_calls = []
    tool_results_messages = []
    last_usage = response.usage

    while True:
        if tool_steps >= MAX_TOOL_STEPS:
            console.print(
                f"[red]⚠️ Max tool steps ({MAX_TOOL_STEPS}) reached. Stopping to prevent infinite loop.[/red]"
            )
            ctx.add_system_message(
                f"System: Execution stopped because maximum tool steps ({MAX_TOOL_STEPS}) were exceeded."
            )
            break

        tool_steps += 1

        # Debug: Log response state
        logger.debug(
            f"Tool step {tool_steps}: content={bool(response.content)}, tool_calls={bool(response.tool_calls)}, thought={bool(response.thought)}"
        )

        if not response.content and not response.tool_calls:
            if tool_steps == 1:
                console.print("[red]Empty response[/red]")
                logger.warning(
                    f"Empty response received. Raw: {response.raw if hasattr(response, 'raw') else 'N/A'}"
                )
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
            # Pre-process: parse args and check loops
            pending_calls = []
            for tc in response.tool_calls:
                func = tc.get("function", {})
                tool_name = func.get("name", "")
                tool_call_id = tc.get("id", "")

                args_raw = func.get("arguments", {})
                args, args_str_normalized = parse_tool_arguments(args_raw)

                # Loop Detection
                is_loop, loop_msg = detect_tool_loop(tool_name, args, previous_tool_calls)
                if is_loop:
                    console.print(f"[red]⚠️ {loop_msg}[/red]")
                    tool_results.append(
                        {
                            "tool_call_id": tool_call_id,
                            "name": tool_name,
                            "result": f"Error: {loop_msg}",
                        }
                    )
                    continue

                pending_calls.append((tool_name, args, tool_call_id))

            # Execute tools: parallel for multiple independent calls, sequential for single/sensitive
            if len(pending_calls) > 1:
                # Use DependencyAnalyzer to find parallel stages
                analyzer = DependencyAnalyzer()
                p_calls = [
                    PToolCall(id=tc_id, name=tn, arguments=a) for tn, a, tc_id in pending_calls
                ]
                stages = analyzer.analyze(p_calls)

                for stage in stages:
                    if len(stage) == 1:
                        pc = stage[0]
                        a = next(a for tn, a, tc_id in pending_calls if tc_id == pc.id)
                        r = await execute_tool(
                            tool_name=pc.name,
                            args=a,
                            tool_call_id=pc.id,
                            registry=registry,
                            sensitive_tools=sensitive_tools,
                            checkpoint_mgr=checkpoint_mgr,
                            hook_mgr=hook_mgr,
                            headless=headless,
                            permission_mgr=permission_mgr,
                        )
                        tool_results.append(r)
                    else:
                        # Parallel execution of independent tools in this stage
                        async def _run_tool(pc, a):
                            return await execute_tool(
                                tool_name=pc.name,
                                args=a,
                                tool_call_id=pc.id,
                                registry=registry,
                                sensitive_tools=sensitive_tools,
                                checkpoint_mgr=checkpoint_mgr,
                                hook_mgr=hook_mgr,
                                headless=headless,
                                permission_mgr=permission_mgr,
                            )

                        tasks = []
                        for pc in stage:
                            a = next(a for tn, a, tc_id in pending_calls if tc_id == pc.id)
                            tasks.append(_run_tool(pc, a))
                        stage_results = await asyncio.gather(*tasks)
                        tool_results.extend(stage_results)
            else:
                # Single tool call - execute directly
                for tool_name, args, tool_call_id in pending_calls:
                    tool_result_dict = await execute_tool(
                        tool_name=tool_name,
                        args=args,
                        tool_call_id=tool_call_id,
                        registry=registry,
                        sensitive_tools=sensitive_tools,
                        checkpoint_mgr=checkpoint_mgr,
                        hook_mgr=hook_mgr,
                        headless=headless,
                        permission_mgr=permission_mgr,
                    )
                    tool_results.append(tool_result_dict)

            # Post-process: track modifications
            for tool_name, args, _tool_call_id in pending_calls:
                if tool_name in ["write_file", "edit_file"] and "path" in args:
                    files_modified.append(args["path"])

        # No more tool calls - done with this turn
        if not has_tool_call:
            logger.info(
                f"Turn complete: accumulated_text={len(accumulated_text)} chars, content={bool(response.content)}"
            )
            if not accumulated_text and not response.content:
                console.print("[yellow]⚠️ No text response generated.[/yellow]")

            # Track usage and costs
            if last_usage:
                cost_tracker.track(
                    model=model_name,
                    input_tokens=last_usage.get("prompt_tokens", 0),
                    output_tokens=last_usage.get("completion_tokens", 0),
                    session_id=ctx.session_id,
                )

            break

        # === Agentic loop: send tool results back to model ===

        # Add assistant message (with tool_calls) to conversation history
        if conversation_history is not None:
            conversation_history.append(
                Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                    thought=response.thought,
                )
            )

        # Add tool result messages
        for tr in tool_results:
            tool_msg = Message(
                role="tool",
                content=tr["result"],
                tool_call_id=tr["tool_call_id"],
                name=tr["name"],
            )
            tool_results_messages.append(tool_msg)
            if conversation_history is not None:
                conversation_history.append(tool_msg)

        # Re-call model with updated conversation history (streaming)
        if model_client and conversation_history is not None and tool_definitions is not None:
            messages_for_api = [
                Message(role="system", content=system_prompt)
            ] + conversation_history
            try:
                response = await stream_model_response(
                    model_client,
                    messages_for_api,
                    tool_definitions,
                    model_name,
                )
                last_usage = response.usage
            except Exception as e:
                if _is_context_overflow(e):
                    console.print("[yellow]Context overflow in tool loop, compacting...[/yellow]")
                    ctx._force_summarize()
                    # Rebuild conversation_history from compacted context
                    conversation_history.clear()
                    for msg in restore_conversation_history(ctx):
                        conversation_history.append(msg)
                    # Re-add the latest tool results
                    for tr in tool_results:
                        tool_msg = Message(
                            role="tool",
                            content=tr["result"],
                            tool_call_id=tr["tool_call_id"],
                            name=tr["name"],
                        )
                        conversation_history.append(tool_msg)
                    messages_for_api = [
                        Message(role="system", content=system_prompt)
                    ] + conversation_history
                    try:
                        response = await stream_model_response(
                            model_client,
                            messages_for_api,
                            tool_definitions,
                            model_name,
                        )
                        last_usage = response.usage
                    except Exception as retry_e:
                        console.print(f"[red]API Error after compaction: {retry_e}[/red]")
                        break
                else:
                    console.print(f"[red]API Error during tool loop: {e}[/red]")
                    break
        else:
            # Fallback: no model_client provided, break after first round
            logger.warning(
                "No model_client provided to process_tool_calls, cannot send tool results back to model"
            )
            break

    return accumulated_text, files_modified, tool_results_messages


async def chat_loop(
    project: str = "default",
    resume_session: str | None = None,
    session_name: str | None = None,
    prompt: str | None = None,
    print_mode: bool = False,
    max_turns: int | None = None,
    tool_config: dict | None = None,
):
    """Main chat loop with automatic context management."""

    # Check for piped input (Headless detection)
    piped_input, headless_from_pipe = check_piped_input()

    initial_prompt = prompt
    if piped_input:
        if initial_prompt:
            initial_prompt = f"{initial_prompt}\n{piped_input}"
        else:
            initial_prompt = piped_input

    # Print mode implies headless
    headless = bool(initial_prompt) or print_mode
    if headless and not print_mode:
        console.print("[dim cyan]Running in headless mode[/dim cyan]")

    # Initialize model client
    try:
        model_client = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: asyncio.run(
                (
                    lambda: __import__(
                        "src.host.cli.initialization", fromlist=["initialize_model_client"]
                    ).initialize_model_client()
                )()
            ),
        )
    except Exception:
        # Fallback: initialize directly
        from src.host.cli.initialization import initialize_model_client

        model_client = await initialize_model_client()

    if not validate_client_mode(model_client):
        return

    # Initialize all managers
    try:
        managers = await initialize_all_managers(project)
    except Exception as e:
        console.print(f"[red]Failed to initialize managers: {e}[/red]")
        return

    # Extract managers
    registry = managers["registry"]
    tool_selector = managers["tool_selector"]
    ctx = managers["ctx"]
    skill_mgr = managers["skill_mgr"]
    checkpoint_mgr = managers["checkpoint_mgr"]
    task_mgr = managers["task_mgr"]
    hook_mgr = managers["hook_mgr"]
    cost_tracker = managers["cost_tracker"]
    cmd_history = managers["cmd_history"]
    bash_executor = managers["bash_executor"]
    session_mgr = managers["session_mgr"]
    permission_mgr = managers.get("permission_mgr")

    sensitive_tools = registry.get_sensitive_tools()

    # Handle session resume
    restore_session_history(session_mgr, ctx, resume_session)

    # Show startup info
    show_startup_info(model_client, project, ctx)

    # State
    mode = "build"
    turn_count = 0
    import os

    model_name = os.getenv("DORAEMON_MODEL", "gemini-3-pro-preview")

    # Get tools for current mode
    tool_names = tool_selector.get_tools_for_mode(mode)
    genai_tools = registry.get_genai_tools(tool_names)
    tool_definitions = convert_tools_to_definitions(genai_tools)
    console.print(f"[dim]Tools: {len(tool_definitions)} ({mode} mode)[/dim]")

    # Build system prompt
    active_skills_content = ""
    system_prompt = build_system_prompt(mode, active_skills_content)

    # Restore conversation history
    conversation_history = restore_conversation_history(ctx)

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
        permission_mgr=permission_mgr,
    )

    # Setup tab completion for slash commands
    slash_commands = [
        "help", "init", "mode", "model", "status", "config", "context", "skills",
        "clear", "compact", "reset", "tools", "debug", "doctor", "memory",
        "commit", "review-pr", "review", "sessions", "resume", "rename", "export",
        "fork", "checkpoints", "rewind", "tasks", "task", "plugins", "plugin",
        "theme", "vim", "thinking", "workspace", "add-dir", "cost", "agents",
        "history", "exit",
    ]
    cmd_history.setup_completer(slash_commands)

    console.print(
        Panel.fit(
            f"[bold blue]🤖 Doraemon Code[/bold blue]\n[dim]Type /help for commands. Mode: {mode}[/dim]",
            border_style="blue",
        )
    )

    # Main loop
    _ctrl_c_count = 0
    try:
        while True:
            mode_color = MODE_COLORS.get(mode, "yellow")

            # Show running tasks
            running_tasks = task_mgr.get_running_tasks()
            if running_tasks:
                console.print(
                    f"[dim cyan]⏳ {len(running_tasks)} background task(s) running[/dim cyan]"
                )

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
                try:
                    _ctrl_c_count = 0
                    user_input = Prompt.ask(
                        f"\n[bold {mode_color}]You ({mode})[/bold {mode_color}]"
                    )

                    # Multi-line input: detect """ or ''' opener
                    for delim in ('"""', "'''"):
                        if user_input.strip().startswith(delim) and not user_input.strip().endswith(
                            delim
                        ):
                            lines = [user_input]
                            console.print(
                                "[dim]Multi-line mode (close with matching delimiter)...[/dim]"
                            )
                            while True:
                                line = Prompt.ask("[dim]...[/dim]")
                                lines.append(line)
                                if delim in line:
                                    break
                            user_input = "\n".join(lines)
                            break

                except KeyboardInterrupt:
                    _ctrl_c_count += 1
                    if _ctrl_c_count >= 2:
                        console.print("\n[yellow]Exiting...[/yellow]")
                        break
                    console.print("\n[dim]Press Ctrl+C again to exit, or type to continue.[/dim]")
                    continue

            # Exit
            if user_input.lower() in ["exit", "quit", "/exit"]:
                await hook_mgr.trigger(HookEvent.SESSION_END, message_count=len(ctx.messages))
                await model_client.close()
                break

            # Bash mode (! prefix)
            if user_input.startswith("!"):
                await handle_bash_mode(user_input, bash_executor, ctx, cmd_history)
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
                    sensitive_tools=set(sensitive_tools),
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

            # Expand @file references
            user_input = expand_file_references(user_input)

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
            messages_for_api = [
                Message(role="system", content=system_prompt)
            ] + conversation_history

            # Send message using unified client (streaming)
            try:
                response = await stream_model_response(
                    model_client,
                    messages_for_api,
                    tool_definitions,
                    model_name,
                )
            except KeyboardInterrupt:
                console.print("\n[yellow]Generation interrupted.[/yellow]")
                conversation_history.pop()
                checkpoint_mgr.discard_checkpoint()
                continue
            except Exception as e:
                if _is_context_overflow(e):
                    console.print("[yellow]Context too large, compacting and retrying...[/yellow]")
                    ctx._force_summarize()
                    conversation_history.clear()
                    conversation_history = restore_conversation_history(ctx)
                    conversation_history.append(Message(role="user", content=user_input))
                    messages_for_api = [
                        Message(role="system", content=system_prompt)
                    ] + conversation_history
                    try:
                        response = await stream_model_response(
                            model_client,
                            messages_for_api,
                            tool_definitions,
                            model_name,
                        )
                    except Exception as retry_e:
                        console.print(f"[red]API Error after compaction: {retry_e}[/red]")
                        conversation_history.pop()
                        checkpoint_mgr.discard_checkpoint()
                        continue
                else:
                    console.print(f"[red]API Error: {e}[/red]")
                    conversation_history.pop()
                    checkpoint_mgr.discard_checkpoint()
                    continue

            # Process tool calls (agentic loop: sends tool results back to model)
            accumulated_text, files_modified, tool_results_messages = await process_tool_calls(
                response=response,
                registry=registry,
                sensitive_tools=sensitive_tools,
                checkpoint_mgr=checkpoint_mgr,
                hook_mgr=hook_mgr,
                ctx=ctx,
                headless=headless,
                model_name=model_name,
                cost_tracker=cost_tracker,
                model_client=model_client,
                conversation_history=conversation_history,
                tool_definitions=tool_definitions,
                system_prompt=system_prompt,
                permission_mgr=permission_mgr,
            )

            # Add final assistant text to conversation history
            # (intermediate assistant+tool messages are already appended by process_tool_calls)
            if accumulated_text:
                conversation_history.append(Message(role="assistant", content=accumulated_text))

            # Track assistant response in context
            usage = response.usage
            prompt_tokens = usage.get("prompt_tokens") if usage else None
            completion_tokens = usage.get("completion_tokens") if usage else None

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
                conversation_history = restore_conversation_history(ctx)

            # Finalize checkpoint
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
                turn_count += 1
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

            # Print mode: exit after first response
            if print_mode:
                break

            # Max turns limit
            if max_turns and turn_count >= max_turns:
                console.print(f"[yellow]Reached max turns limit ({max_turns})[/yellow]")
                break

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
