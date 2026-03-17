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

from src.core.agent_loop import AgentLoopController, RecommendedShift
from src.core.config import load_config
from src.core.context_manager import ConversationSummary
from src.core.hooks import HookEvent
from src.core.logger import configure_root_logger
from src.core.model_utils import ChatResponse, ClientMode, Message, ToolDefinition
from src.core.parallel_executor import DependencyAnalyzer
from src.core.parallel_executor import ToolCall as PToolCall
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
from src.host.cli.tool_execution import (
    detect_tool_loop,
    execute_tool,
    get_modified_paths,
    is_validation_tool_call,
    parse_tool_arguments,
)

logger = logging.getLogger(__name__)
console = Console()

MAX_TOOL_STEPS = 15  # Prevent infinite tool loops
MAX_INLINE_FILE_CHARS = 50_000
MAX_INLINE_DIR_ENTRIES = 100
_DEPENDENCY_ANALYZER = DependencyAnalyzer()


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

    # Pattern: @ followed by path starting with ./ or / (avoids matching emails/mentions)
    pattern = r'@(\./[\w\-./]+|/[\w\-./]+)'
    cwd = Path.cwd().resolve()

    def _read_preview(path: Path) -> str:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            preview = handle.read(MAX_INLINE_FILE_CHARS + 1)
        if len(preview) > MAX_INLINE_FILE_CHARS:
            return preview[:MAX_INLINE_FILE_CHARS] + "\n... [truncated]"
        return preview

    def replace_reference(match):
        ref_path = match.group(1)
        path = Path(ref_path)

        try:
            # Security: resolve and check path is within cwd
            resolved = path.resolve()
            try:
                resolved.relative_to(cwd)
            except ValueError:
                logger.warning(f"Blocked @reference outside workspace: {ref_path}")
                return match.group(0)

            if path.is_file():
                content = _read_preview(path)
                return f"\n```{path.suffix[1:] if path.suffix else 'text'}\n# File: {path}\n{content}\n```\n"

            elif path.is_dir():
                # List directory
                files = sorted(path.iterdir())
                listing = []
                for f in files[:MAX_INLINE_DIR_ENTRIES]:
                    prefix = "📁 " if f.is_dir() else "📄 "
                    listing.append(f"{prefix}{f.name}")
                if len(files) > MAX_INLINE_DIR_ENTRIES:
                    listing.append(f"... and {len(files) - MAX_INLINE_DIR_ENTRIES} more")
                return f"\n```\n# Directory: {path}/\n" + "\n".join(listing) + "\n```\n"

            else:
                # Path doesn't exist, keep original
                return match.group(0)

        except Exception as e:
            logger.warning(f"Failed to expand @{ref_path}: {e}")
            return match.group(0)

    return re.sub(pattern, replace_reference, text)


def extract_image_references(text: str) -> tuple[str, list[str]]:
    """
    Extract @image references from user input.

    Image files (png, jpg, etc.) are pulled out for multimodal handling
    instead of being expanded as text.

    Returns:
        (text_without_images, image_paths) - cleaned text + valid image file paths
    """
    import re
    from pathlib import Path

    from src.core.model_utils import IMAGE_EXTENSIONS

    pattern = r'@(\.?/?[\w\-./]+)'
    image_paths = []

    def replace_images(match):
        ref_path = match.group(1)
        path = Path(ref_path)

        # Check if it's an image file
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            return match.group(0)  # Not an image, leave for expand_file_references

        try:
            resolved = path.resolve()
            cwd = Path.cwd().resolve()
            resolved.relative_to(cwd)
        except ValueError:
            return match.group(0)

        if path.is_file():
            image_paths.append(str(resolved))
            return ""  # Remove from text

        return match.group(0)

    cleaned = re.sub(pattern, replace_images, text).strip()
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


def resolve_tooling_for_phase(tool_selector, registry, mode: str, phase: str | None):
    """Resolve tool names/definitions with phase-aware ordering."""
    tool_names = tool_selector.get_tools_for_state(mode, phase)
    genai_tools = registry.get_genai_tools(tool_names)
    return tool_names, convert_tools_to_definitions(genai_tools)


def handle_ralph_post_turn(
    ralph_mgr,
    *,
    accumulated_text: str,
    files_modified: list[str],
    loop_controller: AgentLoopController,
) -> None:
    """Record Ralph progress and print a suggested next command."""
    if ralph_mgr is None:
        return

    active_ralph_task = ralph_mgr.get_active_task()
    if active_ralph_task is None:
        return

    suggestion_kind, suggestion_command = ralph_mgr.suggest_outcome(
        active_ralph_task.id,
        files_modified=files_modified,
        verification_performed=loop_controller.state.verification_performed,
        is_stuck=loop_controller.state.is_stuck,
        recent_failures=loop_controller.state.recent_failures,
    )
    summary_note = accumulated_text.strip().replace("\n", " ")[:200] or "turn completed"
    ralph_mgr.record_progress(active_ralph_task.id, summary_note)
    color = {
        "done": "green",
        "blocked": "yellow",
        "progress": "cyan",
    }.get(suggestion_kind, "white")
    console.print(f"[{color}]Ralph suggestion:[/{color}] {suggestion_command}")


def prepare_user_turn(
    user_input: str,
    image_paths: list[str],
    *,
    ctx,
    checkpoint_mgr,
    hook_mgr,
    loop_controller: AgentLoopController,
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
    loop_controller.begin_turn(user_input)
    tool_names, tool_definitions = resolve_tooling_for_phase(
        tool_selector,
        registry,
        mode,
        loop_controller.state.phase.value,
    )

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


def show_runtime_status(*, task_mgr, cost_tracker) -> None:
    """Render background-task and budget warnings before prompting."""
    running_tasks = task_mgr.get_running_tasks()
    if running_tasks:
        console.print(f"[dim cyan]⏳ {len(running_tasks)} background task(s) running[/dim cyan]")

    budget_status = cost_tracker.check_budget()
    if budget_status.get("warning"):
        console.print(f"[yellow]⚠️ {budget_status['warning']}[/yellow]")


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
        console.print(f"\n[bold {mode_color}]> {user_input}[/bold {mode_color}]")
        return InputResult(user_input=user_input, ctrl_c_count=ctrl_c_count)

    if headless:
        return InputResult(user_input=None, ctrl_c_count=ctrl_c_count, should_exit=True)

    try:
        user_input = Prompt.ask(f"\n[bold {mode_color}]You ({state.mode})[/bold {mode_color}]")
        ctrl_c_count = 0

        for delim in ('"""', "'''"):
            stripped = user_input.strip()
            if not stripped.startswith(delim) or stripped.endswith(delim):
                continue

            lines = [user_input]
            console.print("[dim]Multi-line mode (close with matching delimiter)...[/dim]")
            while True:
                if len(lines) >= 1000:
                    console.print(
                        "[yellow]Multi-line input limit (1000 lines) reached, closing automatically.[/yellow]"
                    )
                    break
                line = Prompt.ask("[dim]...[/dim]")
                lines.append(line)
                if delim in line:
                    break
            user_input = "\n".join(lines)
            break

        return InputResult(user_input=user_input, ctrl_c_count=ctrl_c_count)
    except KeyboardInterrupt:
        ctrl_c_count += 1
        if ctrl_c_count >= 2:
            console.print("\n[yellow]Exiting...[/yellow]")
            return InputResult(user_input=None, ctrl_c_count=ctrl_c_count, should_exit=True)
        console.print("\n[dim]Press Ctrl+C again to exit, or type to continue.[/dim]")
        return InputResult(user_input=None, ctrl_c_count=ctrl_c_count)


def apply_command_result(
    *,
    result,
    state: ChatLoopState,
    loop_controller: AgentLoopController,
) -> None:
    """Apply a command result to the mutable chat-loop state."""
    state.mode = result.mode
    loop_controller.update_mode(state.mode)
    state.tool_names = result.tool_names
    state.tool_definitions = result.tool_definitions
    state.active_skills_content = result.active_skills_content
    state.conversation_history = result.conversation_history
    if result.system_prompt:
        state.system_prompt = result.system_prompt
    if result.next_prompt:
        state.initial_prompt = result.next_prompt


def build_shift_reminder(loop_controller: AgentLoopController | None) -> str | None:
    """Build a host-side nudge when the policy recommends a concrete shift."""
    if loop_controller is None:
        return None

    shift = loop_controller.state.recommended_shift
    if shift == RecommendedShift.VERIFY_NOW:
        return (
            "[Verification Reminder]\n"
            "You modified files but have not verified the changes yet. "
            "Before finalizing, run the most relevant checks "
            "(for example lint, typecheck, tests, or build) and then report the results."
        )

    if shift == RecommendedShift.SUMMARIZE_BLOCKER:
        return (
            "[Strategy Reminder]\n"
            "You appear blocked. Stop repeating the same failing actions. "
            "Summarize the blocker, state the missing evidence, and either inspect one new surface or explain why progress is blocked."
        )

    if shift == RecommendedShift.READ_NEW_SURFACE:
        return (
            "[Strategy Reminder]\n"
            "The current approach is not producing progress. Read one different file, run one targeted search, "
            "or perform one validating command before continuing."
        )

    return None


async def send_model_request_with_retry(
    *,
    model_client,
    conversation_history: list[Message],
    user_content,
    system_prompt: str,
    loop_controller: AgentLoopController,
    tool_definitions,
    model_name: str,
    ctx,
    checkpoint_mgr,
) -> ChatResponse | None:
    """Send a model request and retry once after context compaction."""
    conversation_history.append(Message(role="user", content=user_content))
    active_system_prompt = loop_controller.compose_system_prompt(system_prompt)
    messages_for_api = [Message(role="system", content=active_system_prompt)] + conversation_history

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
        active_system_prompt = loop_controller.compose_system_prompt(system_prompt)
        messages_for_api = [Message(role="system", content=active_system_prompt)] + conversation_history
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
    cost_tracker,
    model_name: str,
    loop_controller: AgentLoopController,
    ralph_mgr,
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

    handle_ralph_post_turn(
        ralph_mgr,
        accumulated_text=accumulated_text,
        files_modified=files_modified,
        loop_controller=loop_controller,
    )

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
    loop_controller: AgentLoopController,
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
    ralph_mgr,
    session_mgr,
    session_name: str | None,
) -> TurnMetrics | None:
    """Execute one non-command agent turn and persist resulting session state."""
    user_input, image_paths = extract_image_references(user_input)
    if image_paths:
        console.print(f"[dim cyan]Images: {', '.join(Path(p).name for p in image_paths)}[/dim cyan]")

    user_input = expand_file_references(user_input)
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
        hook_mgr=hook_mgr,
        loop_controller=loop_controller,
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
        loop_controller=loop_controller,
        tool_definitions=state.tool_definitions,
        model_name=model_name,
        ctx=ctx,
        checkpoint_mgr=checkpoint_mgr,
    )
    if response is None:
        return None

    accumulated_text, files_modified, _tool_results_messages = await process_tool_calls(
        response=response,
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
        loop_controller=loop_controller,
        tool_selector=tool_selector,
    )
    state.tool_names, state.tool_definitions = resolve_tooling_for_phase(
        tool_selector,
        registry,
        state.mode,
        loop_controller.state.phase.value,
    )

    finalization = await finalize_turn(
        accumulated_text=accumulated_text,
        files_modified=files_modified,
        response=response,
        conversation_history=state.conversation_history,
        ctx=ctx,
        hook_mgr=hook_mgr,
        checkpoint_mgr=checkpoint_mgr,
        cost_tracker=cost_tracker,
        model_name=model_name,
        loop_controller=loop_controller,
        ralph_mgr=ralph_mgr,
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


def display_turn_metrics(*, metrics: TurnMetrics, state: ChatLoopState, ctx, cost_tracker, model_name: str) -> None:
    """Print turn usage and context stats."""
    if not metrics.usage_available:
        return

    state.turn_count += 1
    stats = ctx.get_context_stats()
    cost = cost_tracker.calculate_cost(
        model_name,
        int(metrics.prompt_tokens or 0),
        int(metrics.completion_tokens or 0),
    )
    console.print(
        f"\n[dim]Turn {state.turn_count} | "
        f"In: {int(metrics.prompt_tokens or 0):,} | "
        f"Out: {int(metrics.completion_tokens or 0):,} | "
        f"Cost: ${cost:.4f} | "
        f"Ctx: {stats['usage_percent']}%[/dim]"
    )


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
            console.print("[dim]Or configure Gateway mode with gateway_url in .agent/config.json[/dim]")
            return False

    return True


def restore_session_history(session_mgr, ctx, resume_session: str | None):
    """Restore session history if resuming a session."""
    if resume_session:
        session_data = session_mgr.resume_session(resume_session)
        if session_data:
            if hasattr(ctx, "clear"):
                ctx.clear(keep_summaries=False)
            ctx.session_id = session_data.metadata.id
            if hasattr(ctx, "summaries"):
                ctx.summaries = [
                    ConversationSummary.from_dict(summary)
                    if isinstance(summary, dict)
                    else summary
                    for summary in session_data.summaries
                ]
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
            return session_data
        else:
            console.print(f"[yellow]Session not found: {resume_session}[/yellow]")
    return None


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
        session = session_data
        if session is None and hasattr(session_mgr, "load_session"):
            session = session_mgr.load_session(ctx.session_id)

        if session is None and hasattr(session_mgr, "create_session"):
            session = session_mgr.create_session(project=project, name=session_name, mode=mode)

        metadata = getattr(session, "metadata", None)
        if metadata is None:
            return session

        session_id = getattr(metadata, "id", None)
        if isinstance(session_id, str) and session_id:
            ctx.session_id = session_id

        metadata.project = project
        metadata.mode = mode
        if session_name and not getattr(metadata, "name", None):
            metadata.name = session_name

        stats = ctx.get_context_stats() if hasattr(ctx, "get_context_stats") else {}
        metadata.message_count = len(getattr(ctx, "messages", []) or [])
        if isinstance(stats, dict):
            metadata.total_tokens = stats.get("estimated_tokens", getattr(metadata, "total_tokens", 0))

        session.messages = _serialize_session_entries(getattr(ctx, "messages", []) or [])
        session.summaries = _serialize_session_entries(getattr(ctx, "summaries", []) or [])

        if hasattr(session_mgr, "save_session"):
            session_mgr.save_session(session)
        return session
    except Exception as e:
        logger.warning(f"Failed to persist session state: {e}")
        return session_data


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
    project: str | None = None,
    loop_controller: AgentLoopController | None = None,
    tool_selector=None,
) -> tuple[str, list[str], list[Message]]:
    """
    Process tool calls from model response with agentic loop.

    After executing tools, sends results back to the model for follow-up
    reasoning. Continues until the model responds with text only (no tool calls).

    Returns:
        tuple of (accumulated_text, files_modified, tool_results_messages)
    """
    project = project or Path.cwd().name
    accumulated_text = ""
    files_modified = []
    tool_steps = 0
    previous_tool_calls = []
    tool_results_messages = []
    last_usage = response.usage
    verification_performed = False
    verification_nudged = False

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
        if loop_controller is not None:
            loop_controller.record_model_response(response)

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

            if loop_controller is not None:
                loop_controller.record_tool_plan(pending_calls, response)

            # Execute tools: parallel for multiple independent calls, sequential for single/sensitive
            if len(pending_calls) > 1:
                pending_call_map = {tc_id: (tn, args) for tn, args, tc_id in pending_calls}
                p_calls = [
                    PToolCall(id=tc_id, name=tn, arguments=a) for tn, a, tc_id in pending_calls
                ]
                stages = _DEPENDENCY_ANALYZER.analyze(p_calls)

                for stage in stages:
                    if len(stage) == 1:
                        pc = stage[0]
                        _, matched_args = pending_call_map[pc.id]
                        r = await execute_tool(
                            tool_name=pc.name,
                            args=matched_args,
                            tool_call_id=pc.id,
                            project=project,
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
                                project=project,
                                registry=registry,
                                sensitive_tools=sensitive_tools,
                                checkpoint_mgr=checkpoint_mgr,
                                hook_mgr=hook_mgr,
                                headless=headless,
                                permission_mgr=permission_mgr,
                            )

                        tasks = []
                        for pc in stage:
                            _, matched_args = pending_call_map[pc.id]
                            tasks.append(_run_tool(pc, matched_args))
                        stage_results = await asyncio.gather(*tasks)
                        tool_results.extend(stage_results)
            else:
                # Single tool call - execute directly
                for tool_name, args, tool_call_id in pending_calls:
                    tool_result_dict = await execute_tool(
                        tool_name=tool_name,
                        args=args,
                        tool_call_id=tool_call_id,
                        project=project,
                        registry=registry,
                        sensitive_tools=sensitive_tools,
                        checkpoint_mgr=checkpoint_mgr,
                        hook_mgr=hook_mgr,
                        headless=headless,
                        permission_mgr=permission_mgr,
                    )
                    tool_results.append(tool_result_dict)

            # Post-process: track modifications
            for tool_name, args, tool_call_id in pending_calls:
                modified_paths = get_modified_paths(tool_name, args)
                files_modified.extend(modified_paths)
                verification_performed = verification_performed or is_validation_tool_call(
                    tool_name, args
                )
                if loop_controller is not None:
                    result_text = next(
                        (
                            tr["result"]
                            for tr in tool_results
                            if tr["tool_call_id"] == tool_call_id
                        ),
                        "",
                    )
                    loop_controller.record_tool_outcome(
                        tool_name=tool_name,
                        args=args,
                        result_text=result_text,
                        modified_paths=modified_paths,
                    )

        # No more tool calls - done with this turn
        if not has_tool_call:
            if loop_controller is not None:
                loop_controller.finalize_response(response)
            shift_reminder = build_shift_reminder(loop_controller)
            should_nudge = (
                shift_reminder is not None
                and not verification_nudged
                and model_client is not None
                and conversation_history is not None
                and tool_definitions is not None
            )
            if should_nudge:
                verification_nudged = True
                if (
                    loop_controller is not None
                    and loop_controller.state.recommended_shift == RecommendedShift.VERIFY_NOW
                ):
                    loop_controller.mark_verification_nudged()
                if loop_controller is not None and tool_selector is not None:
                    _, tool_definitions = resolve_tooling_for_phase(
                        tool_selector,
                        registry,
                        loop_controller.state.mode,
                        loop_controller.state.phase.value,
                    )
                conversation_history.append(Message(role="user", content=shift_reminder))
                active_system_prompt = (
                    loop_controller.compose_system_prompt(system_prompt)
                    if loop_controller is not None
                    else system_prompt
                )
                messages_for_api = [
                    Message(role="system", content=active_system_prompt)
                ] + conversation_history
                try:
                    response = await stream_model_response(
                        model_client,
                        messages_for_api,
                        tool_definitions,
                        model_name,
                    )
                    last_usage = response.usage
                    continue
                except Exception as e:
                    console.print(f"[red]API Error during strategy reminder: {e}[/red]")

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
            if loop_controller is not None and tool_selector is not None:
                _, tool_definitions = resolve_tooling_for_phase(
                    tool_selector,
                    registry,
                    loop_controller.state.mode,
                    loop_controller.state.phase.value,
                )
            active_system_prompt = (
                loop_controller.compose_system_prompt(system_prompt)
                if loop_controller is not None
                else system_prompt
            )
            messages_for_api = [Message(role="system", content=active_system_prompt)] + conversation_history
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
                    active_system_prompt = (
                        loop_controller.compose_system_prompt(system_prompt)
                        if loop_controller is not None
                        else system_prompt
                    )
                    messages_for_api = [
                        Message(role="system", content=active_system_prompt)
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
    configure_root_logger()

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
    from src.host.cli.initialization import initialize_model_client

    try:
        model_client = await initialize_model_client()
    except Exception as e:
        console.print(f"[red]Failed to initialize model client: {e}[/red]")
        return

    if not validate_client_mode(model_client):
        await model_client.close()
        return

    # Initialize all managers
    try:
        managers = await initialize_all_managers(project)
    except Exception as e:
        console.print(f"[red]Failed to initialize managers: {e}[/red]")
        await model_client.close()
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
    ralph_mgr = managers.get("ralph_mgr")

    sensitive_tools = registry.get_sensitive_tools()


    # Handle session resume
    session_data = restore_session_history(session_mgr, ctx, resume_session)

    # Show startup info
    show_startup_info(model_client, project, ctx)

    # State
    mode = getattr(getattr(session_data, "metadata", None), "mode", "build")
    turn_count = 0

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

    # Get tools for current mode
    tool_names, tool_definitions = resolve_tooling_for_phase(
        tool_selector,
        registry,
        mode,
        None,
    )
    console.print(f"[dim]Tools: {len(tool_definitions)} ({mode} mode)[/dim]")

    # Build system prompt
    active_skills_content = ""
    system_prompt = build_system_prompt(mode, active_skills_content)
    loop_controller = AgentLoopController.create(project=project, mode=mode)

    state = ChatLoopState(
        mode=mode,
        tool_names=tool_names,
        tool_definitions=tool_definitions,
        active_skills_content=active_skills_content,
        system_prompt=system_prompt,
        conversation_history=restore_conversation_history(ctx),
        initial_prompt=initial_prompt,
        turn_count=turn_count,
        session_data=session_data,
    )

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
        ralph_mgr=ralph_mgr,
    )

    # Setup tab completion for slash commands
    slash_commands = [
        "help", "init", "mode", "model", "status", "config", "context", "skills",
        "clear", "compact", "reset", "tools", "debug", "doctor", "memory",
        "commit", "review-pr", "review", "sessions", "resume", "rename", "export",
        "fork", "checkpoints", "rewind", "tasks", "task", "plugins", "plugin",
        "ralph", "theme", "vim", "thinking", "workspace", "add-dir", "cost", "agents",
        "history", "exit",
    ]
    cmd_history.setup_completer(slash_commands)

    console.print(
        Panel.fit(
            f"[bold blue]🤖 Code Agent[/bold blue]\n[dim]Type /help for commands. Mode: {mode}[/dim]",
            border_style="blue",
        )
    )

    # Main loop
    _ctrl_c_count = 0
    try:
        while True:
            mode_color = MODE_COLORS.get(state.mode, "yellow")
            show_runtime_status(task_mgr=task_mgr, cost_tracker=cost_tracker)
            input_result = read_user_input(
                state=state,
                headless=headless,
                mode_color=mode_color,
                ctrl_c_count=_ctrl_c_count,
            )
            _ctrl_c_count = input_result.ctrl_c_count
            if input_result.should_exit:
                break
            if input_result.user_input is None:
                continue
            user_input = input_result.user_input

            # Exit
            if user_input.lower() in ["exit", "quit", "/exit"]:
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
                    mode=state.mode,
                    tool_names=state.tool_names,
                    tool_definitions=state.tool_definitions,
                    conversation_history=state.conversation_history,
                    active_skills_content=state.active_skills_content,
                    build_system_prompt=build_system_prompt,
                    convert_tools_to_definitions=convert_tools_to_definitions,
                    sensitive_tools=set(sensitive_tools),
                )

                apply_command_result(
                    result=result,
                    state=state,
                    loop_controller=loop_controller,
                )
                state.session_data = persist_session_state(
                    session_mgr,
                    ctx,
                    project,
                    state.mode,
                    session_name=session_name,
                    session_data=state.session_data,
                )
                continue

            cmd_history.add(user_input)
            metrics = await execute_agent_turn(
                user_input=user_input,
                state=state,
                ctx=ctx,
                checkpoint_mgr=checkpoint_mgr,
                hook_mgr=hook_mgr,
                loop_controller=loop_controller,
                tool_selector=tool_selector,
                registry=registry,
                skill_mgr=skill_mgr,
                model_client=model_client,
                model_name=model_name,
                project=project,
                sensitive_tools=sensitive_tools,
                headless=headless,
                cost_tracker=cost_tracker,
                permission_mgr=permission_mgr,
                ralph_mgr=ralph_mgr,
                session_mgr=session_mgr,
                session_name=session_name,
            )
            if metrics is None:
                continue
            display_turn_metrics(
                metrics=metrics,
                state=state,
                ctx=ctx,
                cost_tracker=cost_tracker,
                model_name=model_name,
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
            session_mgr,
            ctx,
            project,
            state.mode,
            session_name=session_name,
            session_data=state.session_data,
        )
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
