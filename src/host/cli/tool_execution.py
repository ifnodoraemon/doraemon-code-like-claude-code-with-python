"""
Tool Execution and HITL Approval

Handles tool execution, HITL (Human-in-the-Loop) approval for sensitive tools,
loop detection, and tool result processing.
Extracted from main.py for better maintainability.
"""

import json
import logging
from pathlib import Path
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
from src.core.security import validate_path

logger = logging.getLogger(__name__)
console = Console()

VALIDATION_TOOLS = {
    "lsp_diagnostics",
}


def get_modified_paths(tool_name: str, args: dict[str, Any]) -> list[str]:
    """Return paths likely modified by a tool call."""
    if tool_name in {"multi_edit", "notebook_edit", "lsp_rename"}:
        path = args.get("path")
        return [path] if isinstance(path, str) and path else []

    if tool_name not in {"write", "write_file", "edit_file"}:
        return []

    operation = args.get("operation", "create")
    path = args.get("path")
    destination = args.get("destination")
    modified: list[str] = []

    if isinstance(path, str) and path:
        modified.append(path)
    if operation in {"move", "copy"} and isinstance(destination, str) and destination:
        modified.append(destination)

    return modified


def is_validation_tool_call(tool_name: str, args: dict[str, Any]) -> bool:
    """Detect whether a tool call is performing verification work."""
    if tool_name in VALIDATION_TOOLS:
        return True

    if tool_name != "run":
        return False

    command = str(args.get("command") or "").lower()
    verification_markers = [
        "pytest",
        "ruff",
        "mypy",
        "eslint",
        "vitest",
        "jest",
        "npm test",
        "npm run test",
        "npm run build",
        "pnpm test",
        "pnpm lint",
        "pnpm build",
        "yarn test",
        "yarn lint",
        "cargo test",
        "go test",
        "tox",
    ]
    return any(marker in command for marker in verification_markers)


def _build_write_diff_preview(args: dict[str, Any]) -> tuple[str, str] | None:
    """Build a synthetic post-edit file body for diff preview when possible."""
    operation = args.get("operation", "create")
    path = args.get("path")
    if not isinstance(path, str) or not path:
        return None

    if operation == "create" and isinstance(args.get("content"), str):
        return path, args["content"]

    if operation != "edit":
        return None

    old_string = args.get("old_string")
    new_string = args.get("new_string")
    if not isinstance(old_string, str) or not isinstance(new_string, str):
        return None

    try:
        file_path = Path(validate_path(path))
        content = file_path.read_text(encoding="utf-8")
    except Exception:
        return None

    if old_string not in content:
        return None

    count = args.get("count", -1)
    replacements = count if isinstance(count, int) else -1
    new_content = (
        content.replace(old_string, new_string)
        if replacements == -1
        else content.replace(old_string, new_string, replacements)
    )
    return path, new_content


async def execute_tool(
    tool_name: str,
    args: dict[str, Any],
    tool_call_id: str,
    project: str,
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
    if tool_name in ["save_note", "search_notes", "delete_note", "list_notes"]:
        args["collection_name"] = args.get("collection_name", project)

    for path in get_modified_paths(tool_name, args):
        checkpoint_mgr.snapshot_file(path)

    diff_preview = _build_write_diff_preview(args) if tool_name == "write" else None
    if diff_preview:
        diff_path, diff_content = diff_preview
        console.print(f"\n[bold yellow]📝 Proposing changes:[/bold yellow] {diff_path}")
        print_diff(diff_path, diff_content)

    # HITL approval for sensitive tools
    tool_result = None
    if tool_name in sensitive_tools:
        console.print(f"\n[bold red]⚠️ Sensitive:[/bold red] {tool_name}")
        if tool_name != "write" or not diff_preview:
            # Pretty print content/code if present
            display_args = args.copy()
            rendered_field = None

            # Check for content field (save_note, etc.)
            if "content" in display_args and isinstance(display_args["content"], str):
                rendered_field = ("Content Preview", display_args.pop("content"), "markdown")
            # Check for Python code execution preview
            elif (
                tool_name == "run"
                and display_args.get("mode") == "python"
                and isinstance(display_args.get("command"), str)
            ):
                rendered_field = ("Code Preview", display_args.pop("command"), "python")

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


def parse_tool_arguments(args_raw: Any) -> dict[str, Any]:
    """
    Parse tool arguments from raw format.

    Args:
        args_raw: Raw arguments (string or dict)

    Returns:
        Parsed tool arguments as a dictionary
    """
    try:
        if isinstance(args_raw, str):
            return json.loads(args_raw)
        return args_raw
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse tool arguments: {args_raw}, error: {e}")
        return {"_parse_error": f"Invalid JSON arguments: {e}"}
