"""
Task Management MCP Server.

Provides task planning and tracking tools via MCP protocol.
Uses the unified TaskManager from src.core.tasks.
"""

import logging

from mcp.server.fastmcp import FastMCP

from src.core.tasks import TaskManager, TaskPriority, TaskStatus

# Setup logging
logger = logging.getLogger(__name__)

mcp = FastMCP("DoraemonTaskServer")
manager = TaskManager()


@mcp.tool()
def add_task(
    title: str, description: str = "", parent_id: str | None = None, priority: str = "medium"
) -> str:
    """
    Add a new task to the project plan.

    Args:
        title: Short title of the task
        description: Detailed description
        parent_id: Optional ID of a parent task to create a sub-task
        priority: Task priority ('low', 'medium', 'high', 'critical')

    Returns:
        Success message with task ID or error message
    """
    try:
        # Convert priority string to enum
        task_priority = TaskPriority(priority.lower())
    except ValueError:
        task_priority = TaskPriority.MEDIUM
        logger.warning(f"Invalid priority '{priority}', using MEDIUM")

    try:
        task = manager.create_task(
            title=title, description=description, priority=task_priority, parent_id=parent_id
        )
        logger.info(f"Created task: {task.id} - {title}")
        return f"Task added. ID: {task.id}"
    except Exception as e:
        logger.error(f"Failed to create task: {e}")
        return f"Error creating task: {e}"


@mcp.tool()
def add_subtask(parent_id: str, title: str, description: str = "") -> str:
    """
    Dynamically add a SUB-TASK to a running Main Task.
    Use this WHENEVER you discover a specific step that needs to be done to complete the Main Task.

    Args:
        parent_id: The ID of the Main Task this belongs to.
        title: Short title (e.g., "Fix logic in auth.py")
        description: Details

    Returns:
        Success message with task ID or error message
    """
    # Verify parent exists
    parent = manager.get_task(parent_id)
    if not parent:
        return f"Error: Parent task '{parent_id}' not found."

    return add_task(title, description, parent_id)


@mcp.tool()
def update_task_status(task_id: str, status: str) -> str:
    """
    Update the status of a task.

    Args:
        task_id: The ID of the task
        status: 'pending', 'in_progress', 'completed', 'blocked', or 'cancelled'
              (Legacy: 'todo' -> 'pending', 'done' -> 'completed')

    Returns:
        Success message or error message
    """
    # Map legacy status values
    status_map = {"todo": "pending", "done": "completed"}
    normalized_status = status_map.get(status.lower(), status.lower())

    try:
        task_status = TaskStatus(normalized_status)
    except ValueError:
        valid_statuses = ", ".join([s.value for s in TaskStatus])
        return f"Error: Invalid status '{status}'. Use one of: {valid_statuses}"

    if manager.update_task_status(task_id, task_status):
        logger.info(f"Updated task {task_id} to {task_status.value}")
        return f"Task {task_id} updated to {task_status.value}."
    else:
        return f"Error: Task '{task_id}' not found."


@mcp.tool()
def list_tasks(status_filter: str | None = None) -> str:
    """
    Show the current project task tree.
    Use this to review progress and decide what to do next.

    Args:
        status_filter: Optional filter by status ('pending', 'in_progress', 'completed', etc.)

    Returns:
        Formatted task tree as text
    """
    # Parse status filter
    filter_status = None
    if status_filter:
        try:
            filter_status = TaskStatus(status_filter.lower())
        except ValueError:
            logger.warning(f"Invalid status filter '{status_filter}', showing all tasks")

    tasks = manager.list_tasks(status=filter_status)

    if not tasks:
        return "No tasks found."

    # Build tree structure
    tree_data = manager.get_task_tree()

    output = ["=== Project Tasks ==="]

    def render_node(node: dict, depth: int = 0):
        status = node.get("status", "pending")
        icon = {
            "pending": "☐",
            "in_progress": "⏳",
            "completed": "✅",
            "blocked": "✗",
            "cancelled": "⊘",
        }.get(status, "○")

        indent = "  " * depth
        priority = node.get("priority", "medium")
        priority_marker = {"critical": "🔴", "high": "🟠", "medium": "", "low": "⚪"}.get(
            priority, ""
        )

        output.append(f"{indent}{icon} [{node['id']}] {priority_marker}{node['title']}")

        if node.get("description"):
            output.append(f"{indent}    ↳ {node['description'][:60]}...")

        for child in node.get("children", []):
            render_node(child, depth + 1)

    for root_task in tree_data:
        render_node(root_task)

    return "\n".join(output)


@mcp.tool()
def delete_task(task_id: str, delete_subtasks: bool = False) -> str:
    """
    Delete a task.

    Args:
        task_id: The ID of the task to delete
        delete_subtasks: If True, also delete all subtasks recursively

    Returns:
        Success message or error message
    """
    if manager.delete_task(task_id, delete_subtasks=delete_subtasks):
        logger.info(f"Deleted task {task_id}")
        return f"Task {task_id} deleted."
    else:
        return f"Error: Task '{task_id}' not found."


if __name__ == "__main__":
    mcp.run()
