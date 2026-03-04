"""
Tool Execution and HITL Approval

Handles tool execution, HITL (Human-in-the-Loop) approval for sensitive tools,
loop detection, and tool result processing.
Extracted from main.py for better maintainability.
"""

import json
import logging
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from src.core.diff import print_diff
from src.core.hooks import HookEvent, HookManager
from src.core.permissions import (
    OperationType,
    PermissionLevel,
    PermissionManager,
    PermissionRequest,
)

logger = logging.getLogger(__name__)
console = Console()


async def execute_tool(
    tool_name: str,
    args: dict[str, Any],
    tool_call_id: str,
    registry,
    sensitive_tools: set,
    checkpoint_mgr,
    hook_mgr: HookManager,
    headless: bool = False,
    permission_mgr: PermissionManager | None = None,
) -> dict[str, Any]:
    """
    Execute a tool with HITL approval for sensitive tools.

    Args:
        tool_name: Name of the tool to execute
        args: Tool arguments
        tool_call_id: Tool call ID for tracking
        registry: Tool registry
        sensitive_tools: Set of sensitive tool names
        checkpoint_mgr: Checkpoint manager for file snapshots
        hook_mgr: Hook manager for pre/post tool hooks
        headless: Whether running in headless mode
        permission_mgr: Optional permission manager for fine-grained control

    Returns:
        dict with tool_call_id, name, and result
    """
    # Trigger PreToolUse hook
    pre_hook = await hook_mgr.trigger(
        HookEvent.PRE_TOOL_USE,
        tool_name=tool_name,
        tool_input=args,
    )

    if pre_hook.decision.value == "deny":
        return {
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "result": f"Blocked: {pre_hook.reason}",
        }

    if pre_hook.modified_input:
        args = pre_hook.modified_input

    # ask_user: block in headless mode (it requires interactive terminal)
    if tool_name == "ask_user" and headless:
        return {
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "result": "Headless mode: Cannot ask user for input.",
        }

    # Permission system check (if available)
    if permission_mgr:
        # Determine operation type from tool name
        op_type = permission_mgr.TOOL_OPERATIONS.get(tool_name, OperationType.READ)
        path = args.get("path", args.get("file"))

        perm_request = PermissionRequest(
            tool=tool_name,
            operation=op_type,
            path=path,
            arguments=args,
        )
        perm_result = permission_mgr.check(perm_request)

        if perm_result.level == PermissionLevel.DENY:
            console.print(f"[red]🚫 Permission denied: {perm_result.message}[/red]")
            return {
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "result": f"Permission denied: {perm_result.message}",
            }

        if perm_result.level == PermissionLevel.WARN:
            console.print(f"[yellow]⚠️ Warning: {perm_result.message}[/yellow]")

        if perm_result.level == PermissionLevel.ASK:
            # Route to HITL approval below (same as sensitive_tools)
            if tool_name not in sensitive_tools:
                sensitive_tools = sensitive_tools | {tool_name}

    # Inject project context for memory tools
    if tool_name in ["save_note", "search_notes"]:
        args["collection_name"] = args.get("collection_name", "default")

    # Snapshot file before modification
    if tool_name in ["write_file", "edit_file"] and "path" in args:
        checkpoint_mgr.snapshot_file(args["path"])

    # Show diff for write operations
    if tool_name == "write_file" and "content" in args and "path" in args:
        console.print(f"\n[bold yellow]📝 Proposing changes:[/bold yellow] {args['path']}")
        print_diff(args["path"], args["content"])

    # HITL approval for sensitive tools
    tool_result = None
    if tool_name in sensitive_tools:
        console.print(f"\n[bold red]⚠️ Sensitive:[/bold red] {tool_name}")
        if tool_name != "write_file":
            # Pretty print content/code if present
            display_args = args.copy()
            rendered_field = None

            # Check for content field (save_note, etc.)
            if "content" in display_args and isinstance(display_args["content"], str):
                rendered_field = ("Content Preview", display_args.pop("content"), "markdown")
            # Check for code field (execute_python)
            elif "code" in display_args and isinstance(display_args["code"], str):
                rendered_field = ("Code Preview", display_args.pop("code"), "python")

            if rendered_field:
                title, field_content, lang = rendered_field
                if display_args:
                    console.print(
                        f"[dim]{json.dumps(display_args, indent=2, ensure_ascii=False)}[/dim]"
                    )
                # Render as markdown code block for code, or plain markdown for content
                if lang == "python":
                    console.print(
                        Panel(
                            Markdown(f"```python\n{field_content}\n```"),
                            title=f"[bold]{title}[/bold]",
                            border_style="dim",
                        )
                    )
                else:
                    console.print(
                        Panel(
                            Markdown(field_content),
                            title=f"[bold]{title}[/bold]",
                            border_style="dim",
                        )
                    )
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
            try:
                tool_result = await registry.call_tool(tool_name, args)
            except Exception as e:
                logger.error(f"Sensitive tool {tool_name} failed: {e}", exc_info=True)
                tool_result = f"Error executing {tool_name}: {e}"
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

    return {
        "tool_call_id": tool_call_id,
        "name": tool_name,
        "result": str(tool_result) if tool_result else "",
    }


def detect_tool_loop(
    tool_name: str,
    args: dict[str, Any],
    previous_tool_calls: list[str],
) -> tuple[bool, str]:
    """
    Detect if a tool is being called repeatedly with the same arguments.

    Args:
        tool_name: Name of the tool
        args: Tool arguments
        previous_tool_calls: List of previous tool call signatures

    Returns:
        tuple of (is_loop_detected, error_message)
    """
    try:
        args_str_normalized = json.dumps(args, sort_keys=True)
    except (TypeError, ValueError):
        args_str_normalized = "{}"

    current_call_signature = f"{tool_name}:{args_str_normalized}"
    previous_tool_calls.append(current_call_signature)

    # Check last 3 identical calls
    if len(previous_tool_calls) >= 3:
        last_three = previous_tool_calls[-3:]
        if all(s == current_call_signature for s in last_three):
            return True, f"Loop detected: {tool_name} called repeatedly with same args."

    # Check alternating pattern (A-B-A-B)
    if len(previous_tool_calls) >= 4:
        last_four = previous_tool_calls[-4:]
        if last_four[0] == last_four[2] and last_four[1] == last_four[3] and last_four[0] != last_four[1]:
            return True, "Loop detected: alternating pattern between tools."

    return False, ""


def parse_tool_arguments(args_raw: Any) -> tuple[dict[str, Any], str]:
    """
    Parse tool arguments from raw format.

    Args:
        args_raw: Raw arguments (string or dict)

    Returns:
        tuple of (parsed_args, normalized_json_string)
    """
    try:
        if isinstance(args_raw, str):
            args = json.loads(args_raw)
        else:
            args = args_raw

        args_str_normalized = json.dumps(args, sort_keys=True)
        return args, args_str_normalized
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse tool arguments: {args_raw}, error: {e}")
        return {"_parse_error": f"Invalid JSON arguments: {e}"}, "{}"
