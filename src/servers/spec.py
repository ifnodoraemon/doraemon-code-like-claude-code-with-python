"""
Spec Server - Tools for LLM to update spec progress during execution.

Provides 3 tools:
  spec_update_task  - Mark a task as done/in_progress/skipped
  spec_check_item   - Check off a verification item
  spec_progress     - Get formatted progress report
"""

import logging

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("DoraemonSpecServer")

# Module-level reference, set by chat_loop at startup
_spec_mgr = None


def set_spec_manager(mgr) -> None:
    """Inject the SpecManager instance (called once at startup)."""
    global _spec_mgr
    _spec_mgr = mgr


def _require_active() -> str | None:
    """Return error string if no active session, else None."""
    if not _spec_mgr or not _spec_mgr.is_active:
        return "Error: No active spec session"
    return None


def _progress_suffix() -> str:
    """Format a one-line progress summary for tool responses."""
    p = _spec_mgr.get_progress()
    return (
        f"Tasks {p['tasks_done']}/{p['tasks_total']}, "
        f"checks {p['checks_done']}/{p['checks_total']}, "
        f"{p['percent']}% overall"
    )


@mcp.tool()
def spec_update_task(task_id: str, status: str) -> str:
    """Update a task's status in the current spec session.

    Args:
        task_id: Task identifier (e.g. "T1", "T2")
        status: New status - one of "pending", "in_progress", "done", "skipped"

    Returns:
        Progress summary or error message
    """
    if err := _require_active():
        return err
    if _spec_mgr.update_task_status(task_id, status):
        return f"Updated {task_id} → {status}. {_progress_suffix()}"
    return f"Error: Task {task_id} not found or invalid status '{status}'"


@mcp.tool()
def spec_check_item(item_id: str, checked: bool = True) -> str:
    """Check or uncheck a verification item in the current spec session.

    Args:
        item_id: Checklist item identifier (e.g. "C1", "C2")
        checked: True to check, False to uncheck

    Returns:
        Progress summary or error message
    """
    if err := _require_active():
        return err
    if _spec_mgr.check_item(item_id, checked):
        mark = "checked" if checked else "unchecked"
        return f"{item_id} {mark}. {_progress_suffix()}"
    return f"Error: Checklist item {item_id} not found"


@mcp.tool()
def spec_progress() -> str:
    """Get a formatted progress report for the current spec session.

    Returns:
        Formatted progress report with task and checklist status
    """
    if err := _require_active():
        return err

    session = _spec_mgr.session
    lines = [
        f"# Spec Progress: {session.name}",
        f"Phase: {session.phase.value}",
        f"Overall: {_progress_suffix()}",
        "",
        f"## Tasks",
    ]

    status_icons = {"done": "✅", "in_progress": "🔄", "pending": "⬜", "skipped": "⏭️"}
    for task in session.tasks:
        icon = status_icons.get(task.status, "⬜")
        lines.append(f"  {icon} {task.id}: {task.title}")

    lines.append("")
    lines.append("## Checklist")
    for item in session.checklist:
        icon = "✅" if item.checked else "⬜"
        lines.append(f"  {icon} {item.id}: {item.description}")

    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
