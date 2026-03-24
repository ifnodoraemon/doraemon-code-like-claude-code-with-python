"""
Main Chat Loop

Core conversation loop with tool execution, context management, and response handling.
Extracted from main.py for better maintainability.
"""

import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from src.core.config import load_config
from src.core.context_manager import ConversationSummary
from src.core.hooks import HookEvent
from src.core.logger import configure_root_logger
from src.core.model_utils import ChatResponse, ClientMode, Message, ToolDefinition
from src.core.prompts import get_system_prompt
from src.core.rules import (
    format_instructions_for_prompt,
    format_memory_for_prompt,
    load_all_instructions,
    load_project_memory,
)
from src.host.cli.commands import CommandHandler
from src.host.cli.commands_core import MODE_COLORS
from src.host.cli.initialization import initialize_all_managers
from src.host.cli.tool_processor import (
    ToolCallContext,
    process_tool_calls as process_tool_calls_refactored,
)

logger = logging.getLogger(__name__)
console = Console()

MAX_INLINE_FILE_CHARS = 50_000
MAX_INLINE_DIR_ENTRIES = 100
_REFERENCE_PATTERN = r"@(\.?/?[\w\-./]+)"


@dataclass
class ChatLoopState:
    """Mutable chat-loop state shared across helpers."""

    mode: str
    tool_names: list[str]
    tool_definitions: list[ToolDefinition]
    active_skills_content: str
    system_prompt: str
    conversation_history: list[Message]
    initial_prompt: str | None
    turn_count: int
    session_data: dict | None


@dataclass
class InputResult:
    """Resolved input for one chat-loop iteration."""

    user_input: str | None
    ctrl_c_count: int
    should_exit: bool = False


@dataclass
class TurnMetrics:
    """Usage metrics returned after a completed turn."""

    prompt_tokens: int | None
    completion_tokens: int | None
    usage_available: bool


@dataclass
class ChatRuntime:
    """Long-lived runtime dependencies for the chat loop."""

    model_client: object
    registry: object
    tool_selector: object
    ctx: object
    skill_mgr: object
    checkpoint_mgr: object
    hook_mgr: object
    cost_tracker: object
    cmd_history: object
    bash_executor: object
    session_mgr: object
    permission_mgr: object | None
    sensitive_tools: set[str]
    model_name: str
    headless: bool


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


def _resolve_workspace_path(ref_path: str) -> Path | None:
    """Resolve a referenced path inside the current workspace."""
    path = Path(ref_path)
    try:
        resolved = path.resolve()
        resolved.relative_to(Path.cwd().resolve())
    except ValueError:
        logger.warning(f"Blocked @reference outside workspace: {ref_path}")
        return None
    except Exception as error:
        logger.warning(f"Failed to resolve @{ref_path}: {error}")
        return None
    return resolved if resolved.exists() else None


def _read_file_preview(path: Path) -> str:
    """Read a bounded preview of a text file."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        preview = handle.read(MAX_INLINE_FILE_CHARS + 1)
    if len(preview) > MAX_INLINE_FILE_CHARS:
        return preview[:MAX_INLINE_FILE_CHARS] + "\n... [truncated]"
    return preview


def _format_directory_preview(path: Path) -> str:
    """Render a bounded directory listing."""
    entries = sorted(path.iterdir())
    listing = []
    for entry in entries[:MAX_INLINE_DIR_ENTRIES]:
        prefix = "📁 " if entry.is_dir() else "📄 "
        listing.append(f"{prefix}{entry.name}")
    if len(entries) > MAX_INLINE_DIR_ENTRIES:
        listing.append(f"... and {len(entries) - MAX_INLINE_DIR_ENTRIES} more")
    return f"\n```\n# Directory: {path}/\n" + "\n".join(listing) + "\n```\n"


def resolve_input_references(text: str) -> tuple[str, list[str]]:
    """Resolve `@path` references into inline previews plus image attachments."""
    import re

    from src.core.model_utils import IMAGE_EXTENSIONS

    image_paths = []

    def replace_reference(match):
        ref_path = match.group(1)
        resolved = _resolve_workspace_path(ref_path)
        if resolved is None:
            return match.group(0)

        if resolved.is_file() and resolved.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(str(resolved))
            return ""
        if resolved.is_file():
            content = _read_file_preview(resolved)
            lang = resolved.suffix[1:] if resolved.suffix else "text"
            return f"\n```{lang}\n# File: {resolved}\n{content}\n```\n"
        if resolved.is_dir():
            return _format_directory_preview(resolved)
        return match.group(0)

    cleaned = re.sub(_REFERENCE_PATTERN, replace_reference, text).strip()
    return cleaned, image_paths


def build_system_prompt(mode: str, skills_content: str = "") -> str:
    """Build the system prompt with mode, rules, memory, and skills."""
    config = load_config()
    persona = config.get("persona", {})

    # Build system prompt
    system_prompt = get_system_prompt(mode, persona)

    # Add project instructions (AGENTS.md)
    instructions = load_all_instructions(config)
    if instructions:
        system_prompt += format_instructions_for_prompt(instructions)

    # Add project memory (.agent/MEMORY.md)
    project_memory = load_project_memory()
    if project_memory:
        system_prompt += format_memory_for_prompt(project_memory)

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


def resolve_tooling(tool_selector, registry, mode: str):
    """Resolve tool names and definitions for the current mode."""
    tool_names = tool_selector.get_tools_for_mode(mode)
    genai_tools = registry.get_genai_tools(tool_names)
    return tool_names, convert_tools_to_definitions(genai_tools)


def prepare_user_turn(
    user_input: str,
    image_paths: list[str],
    *,
    ctx,
    checkpoint_mgr,
    tool_selector,
    registry,
    mode: str,
    skill_mgr,
    active_skills_content: str,
    system_prompt: str,
) -> tuple[str | list[dict] | None, str, str, list, list]:
    """Prepare user input, checkpointing, context tracking, and skill refresh."""
    checkpoint_mgr.begin_checkpoint(user_input, message_count=len(ctx.messages))

    if image_paths:
        from src.core.model_utils import make_image_part, make_text_part

        content_parts = [make_text_part(user_input)]
        for img_path in image_paths:
            try:
                content_parts.append(make_image_part(img_path))
            except Exception as e:
                console.print(f"[red]Failed to load image {img_path}: {e}[/red]")
        user_content = content_parts
    else:
        user_content = user_input

    ctx.add_user_message(user_content)
    tool_names, tool_definitions = resolve_tooling(tool_selector, registry, mode)

    new_skills_content = skill_mgr.get_skills_for_context(user_input)
    updated_system_prompt = system_prompt
    if new_skills_content != active_skills_content:
        active_skills_content = new_skills_content
        new_active = skill_mgr.get_active_skills()
        if new_active:
            console.print(f"[dim cyan]Skills loaded: {', '.join(new_active)}[/dim cyan]")
        updated_system_prompt = build_system_prompt(mode, active_skills_content)

    return (
        user_content,
        active_skills_content,
        updated_system_prompt,
        tool_names,
        tool_definitions,
    )


def show_runtime_status(*, cost_tracker) -> None:
    """Render only actionable runtime warnings before prompting."""
    budget_status = cost_tracker.check_budget()
    if budget_status.get("warning"):
        console.print(f"[yellow]⚠️ {budget_status['warning']}[/yellow]")


def _read_multiline_input(first_line: str, delimiter: str) -> str:
    """Continue reading until the closing delimiter is seen."""
    lines = [first_line]
    while True:
        if len(lines) >= 1000:
            console.print(
                "[yellow]Multi-line input limit (1000 lines) reached, closing automatically.[/yellow]"
            )
            break
        line = Prompt.ask("[dim]...[/dim]")
        lines.append(line)
        if delimiter in line:
            break
    return "\n".join(lines)


def read_user_input(
    *,
    state: ChatLoopState,
    headless: bool,
    mode_color: str,
    ctrl_c_count: int,
) -> InputResult:
    """Read the next user input, honoring initial prompts and headless mode."""
    if state.initial_prompt:
        user_input = state.initial_prompt
        state.initial_prompt = None
        return InputResult(user_input=user_input, ctrl_c_count=ctrl_c_count)

    if headless:
        return InputResult(user_input=None, ctrl_c_count=ctrl_c_count, should_exit=True)

    try:
        user_input = Prompt.ask(f"\n[bold {mode_color}]>[/bold {mode_color}]")
        ctrl_c_count = 0

        for delim in ('"""', "'''"):
            stripped = user_input.strip()
            if not stripped.startswith(delim) or stripped.endswith(delim):
                continue
            user_input = _read_multiline_input(user_input, delim)
            break

        return InputResult(user_input=user_input, ctrl_c_count=ctrl_c_count)
    except KeyboardInterrupt:
        ctrl_c_count += 1
        if ctrl_c_count >= 2:
            return InputResult(user_input=None, ctrl_c_count=ctrl_c_count, should_exit=True)
        console.print("\n[dim]Ctrl+C again to exit[/dim]")
        return InputResult(user_input=None, ctrl_c_count=ctrl_c_count)


def apply_command_result(
    *,
    result,
    state: ChatLoopState,
) -> None:
    """Apply a command result to the mutable chat-loop state."""
    state.mode = result.mode
    state.tool_names = result.tool_names
    state.tool_definitions = result.tool_definitions
    state.active_skills_content = result.active_skills_content
    state.conversation_history = result.conversation_history
    if result.system_prompt:
        state.system_prompt = result.system_prompt
    if result.next_prompt:
        state.initial_prompt = result.next_prompt


async def send_model_request_with_retry(
    *,
    model_client,
    conversation_history: list[Message],
    user_content,
    system_prompt: str,
    tool_definitions,
    model_name: str,
    ctx,
    checkpoint_mgr,
) -> ChatResponse | None:
    """Send a model request and retry once after context compaction."""
    conversation_history.append(Message(role="user", content=user_content))
    messages_for_api = [Message(role="system", content=system_prompt)] + conversation_history

    try:
        return await stream_model_response(
            model_client,
            messages_for_api,
            tool_definitions,
            model_name,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Generation interrupted.[/yellow]")
        conversation_history.pop()
        checkpoint_mgr.discard_checkpoint()
        return None
    except Exception as e:
        if not _is_context_overflow(e):
            console.print(f"[red]API Error: {e}[/red]")
            conversation_history.pop()
            checkpoint_mgr.discard_checkpoint()
            return None

        console.print("[yellow]Context too large, compacting and retrying...[/yellow]")
        ctx._force_summarize()
        conversation_history.clear()
        conversation_history.extend(restore_conversation_history(ctx))
        conversation_history.append(Message(role="user", content=user_content))
        messages_for_api = [Message(role="system", content=system_prompt)] + conversation_history
        try:
            return await stream_model_response(
                model_client,
                messages_for_api,
                tool_definitions,
                model_name,
            )
        except Exception as retry_e:
            console.print(f"[red]API Error after compaction: {retry_e}[/red]")
            conversation_history.pop()
            checkpoint_mgr.discard_checkpoint()
            return None


async def finalize_turn(
    *,
    accumulated_text: str,
    files_modified: list[str],
    response,
    conversation_history: list[Message],
    ctx,
    hook_mgr,
    checkpoint_mgr,
) -> TurnMetrics:
    """Finalize one completed agent turn."""
    if accumulated_text:
        conversation_history.append(Message(role="assistant", content=accumulated_text))

    usage = response.usage
    prompt_tokens = usage.get("prompt_tokens") if usage else None
    completion_tokens = usage.get("completion_tokens") if usage else None

    prev_summary_count = len(ctx.summaries)
    ctx.add_assistant_message(
        accumulated_text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )

    if len(ctx.summaries) > prev_summary_count:
        await hook_mgr.trigger(HookEvent.PRE_COMPACT)
        console.print("[dim yellow]Context summarized to save memory.[/dim yellow]")
        conversation_history.clear()
        conversation_history.extend(restore_conversation_history(ctx))

    if files_modified:
        checkpoint_mgr.finalize_checkpoint(description=f"Modified: {', '.join(files_modified)}")
    else:
        checkpoint_mgr.discard_checkpoint()

    await hook_mgr.trigger(HookEvent.STOP, message_count=len(ctx.messages))

    return TurnMetrics(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        usage_available=bool(usage),
    )


async def execute_agent_turn(
    *,
    user_input: str,
    state: ChatLoopState,
    ctx,
    checkpoint_mgr,
    hook_mgr,
    tool_selector,
    registry,
    skill_mgr,
    model_client,
    model_name: str,
    project: str,
    sensitive_tools: set[str],
    headless: bool,
    cost_tracker,
    permission_mgr,
    session_mgr,
    session_name: str | None,
) -> TurnMetrics | None:
    """Execute one non-command agent turn and persist resulting session state."""
    user_input, image_paths = resolve_input_references(user_input)
    if image_paths:
        console.print(
            f"[dim cyan]Images: {', '.join(Path(p).name for p in image_paths)}[/dim cyan]"
        )
    hook_result = await hook_mgr.trigger(
        HookEvent.USER_PROMPT_SUBMIT,
        user_prompt=user_input,
        message_count=len(ctx.messages),
    )
    if not hook_result.continue_processing:
        if hook_result.reason:
            console.print(f"[yellow]{hook_result.reason}[/yellow]")
        return None

    (
        user_content,
        state.active_skills_content,
        state.system_prompt,
        state.tool_names,
        state.tool_definitions,
    ) = prepare_user_turn(
        user_input,
        image_paths,
        ctx=ctx,
        checkpoint_mgr=checkpoint_mgr,
        tool_selector=tool_selector,
        registry=registry,
        mode=state.mode,
        skill_mgr=skill_mgr,
        active_skills_content=state.active_skills_content,
        system_prompt=state.system_prompt,
    )

    response = await send_model_request_with_retry(
        model_client=model_client,
        conversation_history=state.conversation_history,
        user_content=user_content,
        system_prompt=state.system_prompt,
        tool_definitions=state.tool_definitions,
        model_name=model_name,
        ctx=ctx,
        checkpoint_mgr=checkpoint_mgr,
    )
    if response is None:
        return None

    tool_context = ToolCallContext(
        project=project,
        registry=registry,
        sensitive_tools=sensitive_tools,
        checkpoint_mgr=checkpoint_mgr,
        hook_mgr=hook_mgr,
        ctx=ctx,
        headless=headless,
        model_name=model_name,
        cost_tracker=cost_tracker,
        model_client=model_client,
        conversation_history=state.conversation_history,
        tool_definitions=state.tool_definitions,
        system_prompt=state.system_prompt,
        permission_mgr=permission_mgr,
    )
    process_result = await process_tool_calls_refactored(
        response=response,
        context=tool_context,
    )
    accumulated_text = process_result.accumulated_text
    files_modified = process_result.files_modified

    finalization = await finalize_turn(
        accumulated_text=accumulated_text,
        files_modified=files_modified,
        response=response,
        conversation_history=state.conversation_history,
        ctx=ctx,
        hook_mgr=hook_mgr,
        checkpoint_mgr=checkpoint_mgr,
    )
    state.session_data = persist_session_state(
        session_mgr,
        ctx,
        project,
        state.mode,
        session_name=session_name,
        session_data=state.session_data,
    )
    return finalization


def display_turn_metrics(*, metrics: TurnMetrics, state: ChatLoopState) -> None:
    """Print minimal turn usage stats."""
    if not metrics.usage_available:
        return

    state.turn_count += 1
    console.print(
        f"\n[dim]Turn {state.turn_count} | "
        f"In: {int(metrics.prompt_tokens or 0):,} | "
        f"Out: {int(metrics.completion_tokens or 0):,}[/dim]"
    )


def combine_initial_prompt(prompt: str | None, piped_input: str | None) -> str | None:
    """Merge CLI prompt and piped stdin content into one initial prompt."""
    if not piped_input:
        return prompt
    if prompt:
        return f"{prompt}\n{piped_input}"
    return piped_input


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
            console.print("[red]Error: gateway_url not set in .agent/config.json[/red]")
            return False
    else:
        # Direct mode - check for at least one provider
        providers = mode_info.get("providers", {})
        if not any(providers.values()):
            console.print("[red]Error: No API keys configured[/red]")
            console.print(
                "[dim]Set at least one of: google_api_key, openai_api_key, anthropic_api_key in .agent/config.json[/dim]"
            )
            console.print(
                "[dim]Or configure Gateway mode with gateway_url in .agent/config.json[/dim]"
            )
            return False

    return True


def restore_session_history(session_mgr, ctx, resume_session: str | None):
    """Restore session history if resuming a session."""
    if not resume_session:
        return None

    session_data = session_mgr.resume_session(resume_session)
    if session_data is None:
        console.print(f"[yellow]Session not found: {resume_session}[/yellow]")
        return None

    ctx.clear(keep_summaries=False)
    ctx.session_id = session_data.metadata.id
    ctx.summaries = [
        ConversationSummary.from_dict(summary) if isinstance(summary, dict) else summary
        for summary in session_data.summaries
    ]
    console.print(f"[green]Resumed session: {session_data.metadata.get_display_name()}[/green]")

    for msg in session_data.messages:
        role = msg.get("role")
        content = msg.get("content")
        if not role or content is None:
            continue
        if role == "user":
            ctx.add_user_message(content)
        else:
            ctx.add_assistant_message(content)
    return session_data


def _serialize_session_entries(entries: list) -> list[dict]:
    """Convert context messages/summaries into session-persistable dicts."""
    serialized = []
    for entry in entries:
        if isinstance(entry, dict):
            serialized.append(entry)
            continue
        to_dict = getattr(entry, "to_dict", None)
        if callable(to_dict):
            try:
                data = to_dict()
                if isinstance(data, dict):
                    serialized.append(data)
                    continue
            except Exception:
                pass
        serialized.append({"value": str(entry)})
    return serialized


def persist_session_state(
    session_mgr,
    ctx,
    project: str,
    mode: str,
    session_name: str | None = None,
    session_data=None,
):
    """Create/update the session record so session commands map to real persisted data."""
    try:
        session = session_data or session_mgr.load_session(ctx.session_id)
        if session is None:
            session = session_mgr.create_session(project=project, name=session_name, mode=mode)

        metadata = session.metadata
        ctx.session_id = metadata.id

        metadata.project = project
        metadata.mode = mode
        if session_name and not metadata.name:
            metadata.name = session_name

        stats = ctx.get_context_stats()
        metadata.message_count = len(ctx.messages)
        metadata.total_tokens = stats.get("estimated_tokens", 0)

        session.messages = _serialize_session_entries(ctx.messages)
        session.summaries = _serialize_session_entries(ctx.summaries)

        session_mgr.save_session(session)
        return session
    except Exception as e:
        logger.warning(f"Failed to persist session state: {e}")
        return session_data


async def initialize_chat_runtime(
    *,
    project: str,
    resume_session: str | None,
    session_name: str | None,
    prompt: str | None,
    print_mode: bool,
) -> tuple[ChatRuntime, ChatLoopState]:
    """Initialize model client, managers, session state, and prompt state."""
    piped_input, _ = check_piped_input()
    initial_prompt = combine_initial_prompt(prompt, piped_input)
    headless = bool(initial_prompt) or print_mode

    from src.host.cli.initialization import initialize_model_client

    model_client = await initialize_model_client()
    if not validate_client_mode(model_client):
        await model_client.close()
        raise RuntimeError("Invalid model client configuration")

    managers = await initialize_all_managers(project)
    registry = managers["registry"]
    tool_selector = managers["tool_selector"]
    ctx = managers["ctx"]
    skill_mgr = managers["skill_mgr"]
    checkpoint_mgr = managers["checkpoint_mgr"]
    hook_mgr = managers["hook_mgr"]
    cost_tracker = managers["cost_tracker"]
    cmd_history = managers["cmd_history"]
    bash_executor = managers["bash_executor"]
    session_mgr = managers["session_mgr"]
    permission_mgr = managers.get("permission_mgr")
    sensitive_tools = registry.get_sensitive_tools()

    session_data = restore_session_history(session_mgr, ctx, resume_session)
    mode = getattr(getattr(session_data, "metadata", None), "mode", "build")
    model_name = load_config().get("model")
    if not model_name:
        raise ValueError("Project config is missing required 'model'")

    session_data = persist_session_state(
        session_mgr,
        ctx,
        project,
        mode,
        session_name=session_name,
        session_data=session_data,
    )
    tool_names, tool_definitions = resolve_tooling(tool_selector, registry, mode)

    runtime = ChatRuntime(
        model_client=model_client,
        registry=registry,
        tool_selector=tool_selector,
        ctx=ctx,
        skill_mgr=skill_mgr,
        checkpoint_mgr=checkpoint_mgr,
        hook_mgr=hook_mgr,
        cost_tracker=cost_tracker,
        cmd_history=cmd_history,
        bash_executor=bash_executor,
        session_mgr=session_mgr,
        permission_mgr=permission_mgr,
        sensitive_tools=sensitive_tools,
        model_name=model_name,
        headless=headless,
    )
    state = ChatLoopState(
        mode=mode,
        tool_names=tool_names,
        tool_definitions=tool_definitions,
        active_skills_content="",
        system_prompt=build_system_prompt(mode, ""),
        conversation_history=restore_conversation_history(ctx),
        initial_prompt=initial_prompt,
        turn_count=0,
        session_data=session_data,
    )
    return runtime, state


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
        context_lines = [f"$ {bash_cmd}"]
        if result["output"]:
            context_lines.append(result["output"].rstrip())
        if result["error"]:
            context_lines.append(result["error"].rstrip())
        ctx.add_user_message("\n".join(context_lines))
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


def _append_tool_messages(
    conversation_history: list[Message],
    *,
    response,
    tool_results: list[dict[str, str]],
) -> None:
    """Append assistant tool-call output and tool results to history."""
    conversation_history.append(
        Message(
            role="assistant",
            content=response.content,
            tool_calls=response.tool_calls,
            thought=response.thought,
        )
    )
    for tool_result in tool_results:
        conversation_history.append(
            Message(
                role="tool",
                content=tool_result["result"],
                tool_call_id=tool_result["tool_call_id"],
                name=tool_result["name"],
            )
        )


async def request_tool_follow_up(
    *,
    model_client,
    conversation_history: list[Message],
    tool_results: list[dict[str, str]],
    response,
    ctx,
    tool_definitions,
    model_name: str,
    system_prompt: str,
) -> ChatResponse:
    """Re-query the model after tool execution, compacting context once if needed."""
    _append_tool_messages(
        conversation_history,
        response=response,
        tool_results=tool_results,
    )
    messages_for_api = [Message(role="system", content=system_prompt)] + conversation_history

    try:
        return await stream_model_response(
            model_client,
            messages_for_api,
            tool_definitions,
            model_name,
        )
    except Exception as error:
        if not _is_context_overflow(error):
            raise

    console.print("[yellow]Context overflow in tool loop, compacting...[/yellow]")
    ctx._force_summarize()
    conversation_history.clear()
    conversation_history.extend(restore_conversation_history(ctx))
    for tool_result in tool_results:
        conversation_history.append(
            Message(
                role="tool",
                content=tool_result["result"],
                tool_call_id=tool_result["tool_call_id"],
                name=tool_result["name"],
            )
        )

    messages_for_api = [Message(role="system", content=system_prompt)] + conversation_history
    return await stream_model_response(
        model_client,
        messages_for_api,
        tool_definitions,
        model_name,
    )


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
    content_parts: list[str] = []
    thought_parts: list[str] = []
    accumulated_tool_calls: list[dict] = []
    last_usage = None
    finish_reason = None

    try:
        stream = model_client.chat_stream(messages, tools=tools, model=model_name)

        with Live("", console=console, refresh_per_second=12, transient=True) as live:
            async for chunk in stream:
                if chunk.thought:
                    thought_parts.append(chunk.thought)
                if chunk.content:
                    content_parts.append(chunk.content)
                    live.update(Markdown("".join(content_parts)))
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

    accumulated_content = "".join(content_parts) if content_parts else ""
    accumulated_thought = "".join(thought_parts) if thought_parts else ""

    return ChatResponse(
        content=accumulated_content or None,
        thought=accumulated_thought or None,
        tool_calls=accumulated_tool_calls or None,
        finish_reason=finish_reason,
        usage=last_usage,
    )


async def dispatch_user_input(
    *,
    user_input: str,
    state: ChatLoopState,
    runtime: ChatRuntime,
    cmd_handler,
    project: str,
    session_name: str | None,
) -> tuple[bool, TurnMetrics | None]:
    """Handle exit/bash/command input before delegating to the agent turn."""
    if user_input.lower() in {"exit", "quit", "/exit"}:
        return True, None

    if user_input.startswith("!"):
        await handle_bash_mode(
            user_input,
            runtime.bash_executor,
            runtime.ctx,
            runtime.cmd_history,
        )
        return False, None

    if user_input.startswith("/"):
        cmd_parts = user_input[1:].split()
        result = await cmd_handler.handle(
            cmd=cmd_parts[0].lower() if cmd_parts else "",
            cmd_args=cmd_parts[1:] if len(cmd_parts) > 1 else [],
            mode=state.mode,
            tool_names=state.tool_names,
            tool_definitions=state.tool_definitions,
            conversation_history=state.conversation_history,
            active_skills_content=state.active_skills_content,
        )
        apply_command_result(result=result, state=state)
        state.session_data = persist_session_state(
            runtime.session_mgr,
            runtime.ctx,
            project,
            state.mode,
            session_name=session_name,
            session_data=state.session_data,
        )
        return False, None

    runtime.cmd_history.add(user_input)
    metrics = await execute_agent_turn(
        user_input=user_input,
        state=state,
        ctx=runtime.ctx,
        checkpoint_mgr=runtime.checkpoint_mgr,
        hook_mgr=runtime.hook_mgr,
        tool_selector=runtime.tool_selector,
        registry=runtime.registry,
        skill_mgr=runtime.skill_mgr,
        model_client=runtime.model_client,
        model_name=runtime.model_name,
        project=project,
        sensitive_tools=runtime.sensitive_tools,
        headless=runtime.headless,
        cost_tracker=runtime.cost_tracker,
        permission_mgr=runtime.permission_mgr,
        session_mgr=runtime.session_mgr,
        session_name=session_name,
    )
    return False, metrics


async def chat_loop(
    project: str = "default",
    resume_session: str | None = None,
    session_name: str | None = None,
    prompt: str | None = None,
    print_mode: bool = False,
    max_turns: int | None = None,
):
    """Main chat loop with automatic context management."""
    configure_root_logger()

    try:
        runtime, state = await initialize_chat_runtime(
            project=project,
            resume_session=resume_session,
            session_name=session_name,
            prompt=prompt,
            print_mode=print_mode,
        )
    except Exception as e:
        console.print(f"[red]Failed to initialize runtime: {e}[/red]")
        return

    # Trigger SessionStart hook
    await runtime.hook_mgr.trigger(
        HookEvent.SESSION_START,
        message_count=len(runtime.ctx.messages),
    )

    # Initialize command handler
    cmd_handler = CommandHandler(
        ctx=runtime.ctx,
        tool_selector=runtime.tool_selector,
        registry=runtime.registry,
        hook_mgr=runtime.hook_mgr,
        project=project,
        permission_mgr=runtime.permission_mgr,
        build_system_prompt=build_system_prompt,
        convert_tools_to_definitions=convert_tools_to_definitions,
    )

    # Setup tab completion for slash commands
    slash_commands = [
        "help",
        "init",
        "mode",
        "clear",
        "compact",
        "reset",
        "memory",
        "commit",
        "review",
        "exit",
    ]
    runtime.cmd_history.setup_completer(slash_commands)

    # Main loop
    _ctrl_c_count = 0
    try:
        while True:
            mode_color = MODE_COLORS.get(state.mode, "yellow")
            show_runtime_status(cost_tracker=runtime.cost_tracker)
            input_result = read_user_input(
                state=state,
                headless=runtime.headless,
                mode_color=mode_color,
                ctrl_c_count=_ctrl_c_count,
            )
            _ctrl_c_count = input_result.ctrl_c_count
            if input_result.should_exit:
                break
            if input_result.user_input is None:
                continue
            should_exit, metrics = await dispatch_user_input(
                user_input=input_result.user_input,
                state=state,
                runtime=runtime,
                cmd_handler=cmd_handler,
                project=project,
                session_name=session_name,
            )
            if should_exit:
                break
            if metrics is None:
                continue
            display_turn_metrics(
                metrics=metrics,
                state=state,
            )

            # Print mode: exit after first response
            if print_mode:
                break

            # Max turns limit
            if max_turns and state.turn_count >= max_turns:
                console.print(f"[yellow]Reached max turns limit ({max_turns})[/yellow]")
                break

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
    except Exception as e:
        console.print(f"[red]Error in chat loop: {e}[/red]")
        import traceback

        traceback.print_exc()
    finally:
        persist_session_state(
            runtime.session_mgr,
            runtime.ctx,
            project,
            state.mode,
            session_name=session_name,
            session_data=state.session_data,
        )
        try:
            await runtime.hook_mgr.trigger(
                HookEvent.SESSION_END,
                message_count=len(runtime.ctx.messages),
            )
        except Exception:
            pass
        try:
            await runtime.model_client.close()
        except Exception:
            pass
        console.print("[dim]Session ended[/dim]")
