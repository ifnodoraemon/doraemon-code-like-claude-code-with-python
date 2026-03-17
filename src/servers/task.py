"""Unified task tool for lightweight task persistence."""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from src.core.tasks import TaskManager, TaskStatus

mcp = FastMCP("AgentTaskServer")
manager = TaskManager()


def _normalize_status(value: str | None) -> TaskStatus | None:
    if value is None:
        return None
    normalized = {"todo": "pending", "done": "completed"}.get(value.lower(), value.lower())
    try:
        return TaskStatus(normalized)
    except ValueError:
        return None


def _dump(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)


@mcp.tool()
def task(
    operation: str = "list",
    title: str | None = None,
    description: str | None = None,
    task_id: str | None = None,
    status: str | None = None,
    parent_id: str | None = None,
    delete_subtasks: bool = False,
) -> str:
    """
    Minimal task store for decomposition and progress tracking.

    Operations:
      - create: requires title
      - list: optional status and parent_id filters
      - get: requires task_id
      - update: requires task_id and status
      - delete: requires task_id
      - clear: delete all tasks
    """
    operation = operation.lower().strip()

    if operation == "create":
        if not title:
            return _dump({"ok": False, "error": "title is required"})
        created = manager.create_task(
            title=title,
            description=description or "",
            parent_id=parent_id,
        )
        return _dump({"ok": True, "task": created.to_dict()})

    if operation == "list":
        parsed_status = _normalize_status(status)
        if status is not None and parsed_status is None:
            return _dump({"ok": False, "error": f"invalid status: {status}"})
        tasks = [
            task.to_dict()
            for task in manager.list_tasks(status=parsed_status, parent_id=parent_id)
        ]
        return _dump({"ok": True, "tasks": tasks, "tree": manager.get_task_tree()})

    if operation == "get":
        if not task_id:
            return _dump({"ok": False, "error": "task_id is required"})
        found = manager.get_task(task_id)
        if found is None:
            return _dump({"ok": False, "error": f"task not found: {task_id}"})
        return _dump({"ok": True, "task": found.to_dict()})

    if operation == "update":
        if not task_id:
            return _dump({"ok": False, "error": "task_id is required"})
        parsed_status = _normalize_status(status)
        if parsed_status is None:
            return _dump({"ok": False, "error": "valid status is required"})
        updated = manager.update_task_status(task_id, parsed_status)
        if not updated:
            return _dump({"ok": False, "error": f"task not found: {task_id}"})
        return _dump({"ok": True, "task": manager.get_task(task_id).to_dict()})

    if operation == "delete":
        if not task_id:
            return _dump({"ok": False, "error": "task_id is required"})
        deleted = manager.delete_task(task_id, delete_subtasks=delete_subtasks)
        if not deleted:
            return _dump({"ok": False, "error": f"task not found: {task_id}"})
        return _dump({"ok": True, "deleted": task_id})

    if operation == "clear":
        manager.clear_all_tasks()
        return _dump({"ok": True, "cleared": True})

    return _dump({"ok": False, "error": f"unknown operation: {operation}"})


if __name__ == "__main__":
    mcp.run()
