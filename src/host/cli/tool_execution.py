"""
Tool Execution and HITL Approval

Handles tool execution, HITL (Human-in-the-Loop) approval for sensitive tools,
loop detection, and tool result processing.
"""

import json
import logging
from typing import Any

from src.core.hooks import HookEvent, HookManager
from src.host.cli.tool_execution_helpers import (
    HitlContext,
    build_write_diff_preview,
    check_permission,
    display_diff_preview,
    execute_tool_with_approval,
    inject_project_context,
    snapshot_modified_files,
)

logger = logging.getLogger(__name__)

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


def detect_tool_loop(
    tool_name: str,
    args: dict[str, Any],
    previous_tool_calls: list[str],
) -> tuple[bool, str]:
    """
    Detect if a tool is being called repeatedly with the same arguments.

    Returns:
        tuple of (is_loop_detected, error_message)
    """
    try:
        args_str_normalized = json.dumps(args, sort_keys=True)
    except (TypeError, ValueError):
        args_str_normalized = "{}"

    current_call_signature = f"{tool_name}:{args_str_normalized}"
    previous_tool_calls.append(current_call_signature)

    if len(previous_tool_calls) >= 3:
        last_three = previous_tool_calls[-3:]
        if all(s == current_call_signature for s in last_three):
            return True, f"Loop detected: {tool_name} called repeatedly with same args."

    if len(previous_tool_calls) >= 4:
        last_four = previous_tool_calls[-4:]
        if (
            last_four[0] == last_four[2]
            and last_four[1] == last_four[3]
            and last_four[0] != last_four[1]
        ):
            return True, "Loop detected: alternating pattern between tools."

    return False, ""


def parse_tool_arguments(args_raw: Any) -> dict[str, Any]:
    """Parse tool arguments from raw format."""
    try:
        if isinstance(args_raw, str):
            return json.loads(args_raw)
        return args_raw
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse tool arguments: {args_raw}, error: {e}")
        return {"_parse_error": f"Invalid JSON arguments: {e}"}


async def execute_tool(
    tool_name: str,
    args: dict[str, Any],
    tool_call_id: str,
    project: str,
    registry,
    sensitive_tools: set[str],
    checkpoint_mgr,
    hook_mgr: HookManager,
    headless: bool = False,
    permission_mgr=None,
) -> dict[str, Any]:
    """
    Execute a tool with HITL approval for sensitive tools.

    Args:
        tool_name: Name of the tool to execute
        args: Tool arguments
        tool_call_id: Tool call ID for tracking
        project: Project name for context injection
        registry: Tool registry
        sensitive_tools: Set of sensitive tool names
        checkpoint_mgr: Checkpoint manager for file snapshots
        hook_mgr: Hook manager for pre/post tool hooks
        headless: Whether running in headless mode
        permission_mgr: Optional permission manager for fine-grained control

    Returns:
        dict with tool_call_id, name, and result
    """
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

    if tool_name == "ask_user" and headless:
        return {
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "result": "Headless mode: Cannot ask user for input.",
        }

    perm_result = check_permission(tool_name, args, permission_mgr, sensitive_tools)
    if not perm_result.allowed:
        return {
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "result": f"Permission denied: {perm_result.message}",
        }

    inject_project_context(tool_name, args, project)
    snapshot_modified_files(tool_name, args, checkpoint_mgr, get_modified_paths)

    diff_preview = None
    if tool_name == "write":
        diff_preview = build_write_diff_preview(args)
        if diff_preview:
            display_diff_preview(diff_preview[0], diff_preview[1])

    tool_result: str | None = None
    if perm_result.require_approval:
        hitl_context = HitlContext(
            tool_name=tool_name,
            args=args,
            headless=headless,
            diff_preview=diff_preview,
        )
        tool_result = await execute_tool_with_approval(tool_name, args, registry, hitl_context)
    else:
        tool_result = await execute_tool_with_approval(tool_name, args, registry)

    await hook_mgr.trigger(
        HookEvent.POST_TOOL_USE,
        tool_name=tool_name,
        tool_input=args,
        tool_output=tool_result,
    )

    return {
        "tool_call_id": tool_call_id,
        "name": tool_name,
        "result": tool_result or "",
    }
