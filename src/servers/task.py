"""Unified task tool for lightweight task persistence."""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from src.core.tasks import TaskClaimError, TaskManager, TaskStatus

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


def _normalize_dependencies(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    return [part for part in parts if part]


@mcp.tool()
def task(
    operation: str = "list",
    title: str | None = None,
    description: str | None = None,
    task_id: str | None = None,
    status: str | None = None,
    parent_id: str | None = None,
    dependencies: str | None = None,
    agent_id: str | None = None,
    priority: int | None = None,
    workspace_id: str | None = None,
    delete_subtasks: bool = False,
) -> str:
    """
    Minimal task store for decomposition and progress tracking.

    Operations:
      - create: requires title
      - list: optional status and parent_id filters
      - get: requires task_id
      - update: requires task_id and status
      - ready: list tasks whose dependencies are satisfied
      - claim: requires task_id and agent_id
      - release: requires task_id
      - delete: requires task_id
      - clear: delete all tasks
    """
    operation = operation.lower().strip()
    parsed_dependencies = _normalize_dependencies(dependencies)

    if operation == "create":
        if not title:
            return _dump({"ok": False, "error": "title is required"})
        try:
            created = manager.create_task(
                title=title,
                description=description or "",
                parent_id=parent_id,
                dependencies=parsed_dependencies,
                priority=priority or 0,
                workspace_id=workspace_id,
            )
        except ValueError as exc:
            return _dump({"ok": False, "error": str(exc)})
        return _dump({"ok": True, "task": created.to_dict()})

    if operation == "list":
        parsed_status = _normalize_status(status)
        if status is not None and parsed_status is None:
            return _dump({"ok": False, "error": f"invalid status: {status}"})
        tasks = [
            task.to_dict() for task in manager.list_tasks(status=parsed_status, parent_id=parent_id)
        ]
        return _dump({"ok": True, "tasks": tasks, "tree": manager.get_task_tree()})

    if operation == "ready":
        tasks = [task.to_dict() for task in manager.list_ready_tasks()]
        return _dump({"ok": True, "tasks": tasks})

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
        parsed_status = _normalize_status(status) if status is not None else None
        if status is not None and parsed_status is None:
            return _dump({"ok": False, "error": "valid status is required"})
        update_kwargs = {
            "status": parsed_status,
            "dependencies": parsed_dependencies,
            "priority": priority,
            "workspace_id": workspace_id,
        }
        if agent_id is not None:
            update_kwargs["assigned_agent"] = agent_id
        try:
            updated = manager.update_task(
                task_id,
                **update_kwargs,
            )
        except ValueError as exc:
            return _dump({"ok": False, "error": str(exc)})
        if not updated:
            return _dump({"ok": False, "error": f"task not found: {task_id}"})
        return _dump({"ok": True, "task": updated.to_dict()})

    if operation == "claim":
        if not task_id:
            return _dump({"ok": False, "error": "task_id is required"})
        if not agent_id:
            return _dump({"ok": False, "error": "agent_id is required"})
        try:
            claimed = manager.claim_task(task_id, agent_id)
        except TaskClaimError as exc:
            return _dump({"ok": False, "error": str(exc)})
        return _dump({"ok": True, "task": claimed.to_dict()})

    if operation == "release":
        if not task_id:
            return _dump({"ok": False, "error": "task_id is required"})
        try:
            released = manager.release_task(task_id, agent_id=agent_id)
        except TaskClaimError as exc:
            return _dump({"ok": False, "error": str(exc)})
        return _dump({"ok": True, "task": released.to_dict()})

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
