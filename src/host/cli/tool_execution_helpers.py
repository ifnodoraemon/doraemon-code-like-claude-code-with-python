"""
Tool Execution Helpers

Extracted from tool_execution.py for better maintainability.
Handles HITL approval, permission checks, and display formatting.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from src.core.diff import print_diff
from src.core.permissions import (
    OperationType,
    PermissionLevel,
    PermissionManager,
    PermissionRequest,
)
from src.core.security import validate_path

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class HitlContext:
    """Context for HITL approval."""

    tool_name: str
    args: dict[str, Any]
    headless: bool
    diff_preview: tuple[str, str] | None = None


@dataclass
class PermissionCheckResult:
    """Result of permission check."""

    allowed: bool
    require_approval: bool
    message: str | None = None


def check_permission(
    tool_name: str,
    args: dict[str, Any],
    permission_mgr: PermissionManager | None,
    sensitive_tools: set[str],
) -> PermissionCheckResult:
    """
    Check if tool execution is permitted.

    Returns:
        PermissionCheckResult with allowed status and approval requirement.
    """
    if not permission_mgr:
        return PermissionCheckResult(allowed=True, require_approval=tool_name in sensitive_tools)

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
        return PermissionCheckResult(
            allowed=False,
            require_approval=False,
            message=perm_result.message,
        )

    if perm_result.level == PermissionLevel.WARN:
        console.print(f"[yellow]⚠️ Warning: {perm_result.message}[/yellow]")

    require_approval = tool_name in sensitive_tools or perm_result.level == PermissionLevel.ASK

    return PermissionCheckResult(allowed=True, require_approval=require_approval)


def build_write_diff_preview(args: dict[str, Any]) -> tuple[str, str] | None:
    """Build a synthetic post-edit file body for diff preview."""
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


def display_diff_preview(path: str, content: str) -> None:
    """Display diff preview for file changes."""
    console.print(f"\n[bold yellow]📝 Proposing changes:[/bold yellow] {path}")
    print_diff(path, content)


def _render_preview_field(title: str, content: str, lang: str) -> None:
    """Render a field with syntax highlighting."""
    if lang == "python":
        console.print(
            Panel(
                Markdown(f"```python\n{content}\n```"),
                title=f"[bold]{title}[/bold]",
                border_style="dim",
            )
        )
    else:
        console.print(
            Panel(
                Markdown(content),
                title=f"[bold]{title}[/bold]",
                border_style="dim",
            )
        )


def _display_sensitive_tool_args(tool_name: str, args: dict[str, Any]) -> None:
    """Display arguments for a sensitive tool."""
    display_args = args.copy()
    rendered_field = None

    if "content" in display_args and isinstance(display_args["content"], str):
        rendered_field = ("Content Preview", display_args.pop("content"), "markdown")
    elif (
        tool_name == "run"
        and display_args.get("mode") == "python"
        and isinstance(display_args.get("command"), str)
    ):
        rendered_field = ("Code Preview", display_args.pop("command"), "python")

    if rendered_field:
        title, field_content, lang = rendered_field
        if display_args:
            console.print(f"[dim]{json.dumps(display_args, indent=2, ensure_ascii=False)}[/dim]")
        _render_preview_field(title, field_content, lang)
    else:
        console.print(f"[dim]{json.dumps(args, indent=2, ensure_ascii=False)}[/dim]")


async def request_hitl_approval(context: HitlContext) -> tuple[bool, str | None]:
    """
    Request HITL approval for a sensitive tool.

    Returns:
        Tuple of (approved, error_message).
    """
    console.print(f"\n[bold red]⚠️ Sensitive:[/bold red] {context.tool_name}")

    if context.tool_name != "write" or not context.diff_preview:
        _display_sensitive_tool_args(context.tool_name, context.args)

    if context.headless:
        console.print("[red]Headless mode: Sensitive tool denied[/red]")
        return False, "Headless mode: Sensitive tool execution denied."

    if Prompt.ask("Execute?", choices=["y", "n"], default="n") != "y":
        console.print("[red]Cancelled[/red]")
        return False, "User denied the operation."

    return True, None


async def execute_tool_with_approval(
    tool_name: str,
    args: dict[str, Any],
    registry: Any,
    hitl_context: HitlContext | None = None,
) -> str:
    """
    Execute a tool with optional HITL approval.

    Returns:
        Tool result as string.
    """
    if hitl_context:
        approved, error_msg = await request_hitl_approval(hitl_context)
        if not approved:
            return error_msg or "Operation denied."

    console.print(f"[cyan]Running {tool_name}...[/cyan]")
    try:
        result = await registry.call_tool(tool_name, args)
        return str(result) if result else ""
    except Exception as e:
        logger.error(f"Tool {tool_name} failed: {e}", exc_info=True)
        return f"Error executing {tool_name}: {e}"


def inject_project_context(tool_name: str, args: dict[str, Any], project: str) -> None:
    """Inject project context into tool arguments if needed."""
    if tool_name in ["save_note", "search_notes", "delete_note", "list_notes"]:
        args["collection_name"] = args.get("collection_name", project)


def snapshot_modified_files(
    tool_name: str,
    args: dict[str, Any],
    checkpoint_mgr: Any,
    get_modified_paths_func: Any,
) -> None:
    """Create snapshots of files that will be modified."""
    for path in get_modified_paths_func(tool_name, args):
        checkpoint_mgr.snapshot_file(path)
