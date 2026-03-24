"""
Tool Call Processing

Refactored from chat_loop.py to reduce complexity.
Handles tool call execution, loop detection, and result processing.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.core.hooks import HookEvent, HookManager
from src.core.model_utils import ChatResponse, Message, ToolDefinition
from src.core.parallel_executor import DependencyAnalyzer
from src.core.parallel_executor import ToolCall as PToolCall
from src.host.cli.tool_execution import (
    detect_tool_loop,
    execute_tool,
    get_modified_paths,
    parse_tool_arguments,
)

logger = logging.getLogger(__name__)
console = Console()

MAX_TOOL_STEPS = 15


@dataclass
class ToolCallContext:
    """Context for processing tool calls."""

    project: str
    registry: Any
    sensitive_tools: set[str]
    checkpoint_mgr: Any
    hook_mgr: HookManager
    ctx: Any
    headless: bool
    model_name: str
    cost_tracker: Any
    model_client: Any | None = None
    conversation_history: list[Message] | None = None
    tool_definitions: list[ToolDefinition] | None = None
    system_prompt: str = ""
    permission_mgr: Any | None = None


@dataclass
class ToolCallResult:
    """Result of processing a single tool call."""

    tool_call_id: str
    name: str
    result: str
    args: dict[str, Any]


@dataclass
class ProcessResult:
    """Result of process_tool_calls."""

    accumulated_text: str
    files_modified: list[str]


def _check_tool_step_limit(tool_steps: int, ctx: Any) -> bool:
    """Check if tool step limit is reached. Returns True if should stop."""
    if tool_steps >= MAX_TOOL_STEPS:
        console.print(
            f"[red]Max tool steps ({MAX_TOOL_STEPS}) reached. Stopping to prevent infinite loop.[/red]"
        )
        ctx.add_system_message(
            f"System: Execution stopped because maximum tool steps ({MAX_TOOL_STEPS}) were exceeded."
        )
        return True
    return False


def _display_thought(thought: str) -> None:
    """Display thought content in a panel."""
    console.print(
        Panel(
            Markdown(thought),
            title="[bold dim]Thinking[/bold dim]",
            border_style="dim white",
            expand=False,
        )
    )


def _display_content(content: str) -> None:
    """Display text content in a panel."""
    console.print(
        Panel(
            Markdown(content),
            title="[bold purple]Response[/bold purple]",
            border_style="purple",
            expand=False,
        )
    )


def _prepare_tool_call(
    tc: dict[str, Any],
    previous_tool_calls: list[str],
) -> tuple[str, dict[str, Any], str] | tuple[str, None, str]:
    """
    Prepare a single tool call.

    Returns:
        Tuple of (tool_name, args, tool_call_id) or (tool_name, None, tool_call_id) if loop detected.
    """
    func = tc.get("function", {})
    tool_name = func.get("name", "")
    tool_call_id = tc.get("id", "")

    args_raw = func.get("arguments", {})
    args = parse_tool_arguments(args_raw)

    is_loop, loop_msg = detect_tool_loop(tool_name, args, previous_tool_calls)
    if is_loop:
        console.print(f"[red]Loop detected: {loop_msg}[/red]")
        return tool_name, None, tool_call_id

    return tool_name, args, tool_call_id


async def _execute_single_tool(
    tool_name: str,
    args: dict[str, Any],
    tool_call_id: str,
    context: ToolCallContext,
) -> dict[str, Any]:
    """Execute a single tool and return result dict."""
    return await execute_tool(
        tool_name=tool_name,
        args=args,
        tool_call_id=tool_call_id,
        project=context.project,
        registry=context.registry,
        sensitive_tools=context.sensitive_tools,
        checkpoint_mgr=context.checkpoint_mgr,
        hook_mgr=context.hook_mgr,
        headless=context.headless,
        permission_mgr=context.permission_mgr,
    )


async def _execute_tool_calls_parallel(
    pending_calls: list[tuple[str, dict[str, Any], str]],
    context: ToolCallContext,
) -> list[dict[str, Any]]:
    """Execute multiple tool calls in parallel based on dependency analysis."""
    results = []
    pending_call_map = {tc_id: (tn, args) for tn, args, tc_id in pending_calls}
    p_calls = [PToolCall(id=tc_id, name=tn, arguments=a) for tn, a, tc_id in pending_calls]
    analyzer = DependencyAnalyzer()
    stages = analyzer.analyze(p_calls)

    for stage in stages:
        if len(stage) == 1:
            pc = stage[0]
            _, matched_args = pending_call_map[pc.id]
            result = await _execute_single_tool(pc.name, matched_args, pc.id, context)
            results.append(result)
        else:
            tasks = []
            for pc in stage:
                _, matched_args = pending_call_map[pc.id]
                tasks.append(_execute_single_tool(pc.name, matched_args, pc.id, context))
            stage_results = await asyncio.gather(*tasks)
            results.extend(stage_results)

    return results


async def _execute_tool_calls(
    pending_calls: list[tuple[str, dict[str, Any], str]],
    context: ToolCallContext,
) -> list[dict[str, Any]]:
    """Execute tool calls (parallel for multiple, sequential for single)."""
    if len(pending_calls) > 1:
        return await _execute_tool_calls_parallel(pending_calls, context)

    results = []
    for tool_name, args, tool_call_id in pending_calls:
        result = await _execute_single_tool(tool_name, args, tool_call_id, context)
        results.append(result)
    return results


def _collect_tool_results(
    prepared_calls: list[tuple[str, dict[str, Any] | None, str]],
    context: ToolCallContext,
) -> tuple[list[dict[str, Any]], list[tuple[str, dict[str, Any], str]]]:
    """
    Collect tool results, handling loop detection.

    Returns:
        Tuple of (tool_results, pending_calls_for_tracking)
    """
    tool_results = []
    pending_calls = []

    for tool_name, args, tool_call_id in prepared_calls:
        if args is None:
            tool_results.append(
                {
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "result": "Error: Loop detected - tool call blocked.",
                }
            )
        else:
            pending_calls.append((tool_name, args, tool_call_id))

    return tool_results, pending_calls


def _track_file_modifications(
    pending_calls: list[tuple[str, dict[str, Any], str]],
) -> list[str]:
    """Track files modified by tool calls."""
    modified = []
    for tool_name, args, _tool_call_id in pending_calls:
        paths = get_modified_paths(tool_name, args)
        modified.extend(paths)
    return modified


async def _request_follow_up(
    response: ChatResponse,
    tool_results: list[dict[str, Any]],
    context: ToolCallContext,
) -> ChatResponse:
    """Request follow-up from model with tool results."""
    conversation_history = context.conversation_history
    if conversation_history is None:
        raise ValueError("conversation_history is required for follow-up")

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

    messages_for_api = [
        Message(role="system", content=context.system_prompt)
    ] + conversation_history

    from src.host.cli.chat_loop import stream_model_response

    try:
        return await stream_model_response(
            context.model_client,
            messages_for_api,
            context.tool_definitions,
            context.model_name,
        )
    except Exception as error:
        msg = str(error).lower()
        if "context" in msg and ("overflow" in msg or "length" in msg or "limit" in msg):
            console.print("[yellow]Context overflow in tool loop, compacting...[/yellow]")
            context.ctx._force_summarize()
            conversation_history.clear()
            from src.host.cli.chat_loop import restore_conversation_history

            conversation_history.extend(restore_conversation_history(context.ctx))
            for tool_result in tool_results:
                conversation_history.append(
                    Message(
                        role="tool",
                        content=tool_result["result"],
                        tool_call_id=tool_result["tool_call_id"],
                        name=tool_result["name"],
                    )
                )
            messages_for_api = [
                Message(role="system", content=context.system_prompt)
            ] + conversation_history
            return await stream_model_response(
                context.model_client,
                messages_for_api,
                context.tool_definitions,
                context.model_name,
            )
        raise


async def process_tool_calls(
    response: ChatResponse,
    context: ToolCallContext,
) -> ProcessResult:
    """
    Process tool calls from model response with agentic loop.

    After executing tools, sends results back to the model for follow-up
    reasoning. Continues until the model responds with text only (no tool calls).

    Returns:
        ProcessResult with accumulated_text and files_modified.
    """
    accumulated_text = ""
    files_modified = []
    tool_steps = 0
    previous_tool_calls: list[str] = []
    last_usage = response.usage

    while True:
        tool_steps += 1

        if _check_tool_step_limit(tool_steps, context.ctx):
            break

        logger.debug(
            f"Tool step {tool_steps}: content={bool(response.content)}, "
            f"tool_calls={bool(response.tool_calls)}, thought={bool(response.thought)}"
        )

        if not response.content and not response.tool_calls:
            if tool_steps == 1:
                console.print("[red]Empty response[/red]")
                logger.warning("Empty response received")
            break

        has_tool_call = response.has_tool_calls

        if response.thought:
            _display_thought(response.thought)

        if response.content:
            accumulated_text += response.content
            _display_content(response.content)

        if response.tool_calls:
            prepared_calls = []
            for tc in response.tool_calls:
                tool_name, args, tool_call_id = _prepare_tool_call(tc, previous_tool_calls)
                prepared_calls.append((tool_name, args, tool_call_id))

            tool_results, pending_calls = _collect_tool_results(prepared_calls, context)

            if pending_calls:
                executed_results = await _execute_tool_calls(pending_calls, context)
                tool_results.extend(executed_results)

                modified = _track_file_modifications(pending_calls)
                files_modified.extend(modified)

        if not has_tool_call:
            logger.info(
                f"Turn complete: accumulated_text={len(accumulated_text)} chars, "
                f"content={bool(response.content)}"
            )
            if not accumulated_text and not response.content:
                console.print("[yellow]No text response generated.[/yellow]")

            if last_usage:
                context.cost_tracker.track(
                    model=context.model_name,
                    input_tokens=last_usage.get("prompt_tokens", 0),
                    output_tokens=last_usage.get("completion_tokens", 0),
                    session_id=context.ctx.session_id,
                )
            break

        if context.model_client and context.conversation_history is not None:
            try:
                response = await _request_follow_up(response, tool_results, context)
                last_usage = response.usage
            except Exception as error:
                console.print(f"[red]API Error during tool loop: {error}[/red]")
                break
        else:
            logger.warning(
                "No model_client provided to process_tool_calls, "
                "cannot send tool results back to model"
            )
            break

    return ProcessResult(accumulated_text=accumulated_text, files_modified=files_modified)
