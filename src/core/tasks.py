"""Minimal persistent task store for the agent."""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from src.core.paths import tasks_path

logger = logging.getLogger(__name__)

_UNSET = object()


class TaskStatus(Enum):
    """Task status values exposed to the model."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class Task:
    """A small persistent task record."""

    id: str
    title: str
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    parent_id: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "parent_id": self.parent_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        status = data.get("status", TaskStatus.PENDING.value)
        if status == "todo":
            status = TaskStatus.PENDING.value
        elif status == "done":
            status = TaskStatus.COMPLETED.value
        return cls(
            id=data["id"],
            title=data["title"],
            description=data.get("description", ""),
            status=TaskStatus(status),
            parent_id=data.get("parent_id"),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", data.get("created_at", time.time())),
        )


class TaskManager:
    """Persistent task storage with just enough structure for agent work."""

    def __init__(self, storage_path: Path | None = None):
        self.storage_path = storage_path or tasks_path()
        self._tasks: dict[str, Task] = {}
        self._load()

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            with self.storage_path.open(encoding="utf-8") as handle:
                raw = json.load(handle)
        except Exception as exc:
            logger.error(f"Failed to load tasks: {exc}")
            return

        items: list[dict[str, Any]]
        if isinstance(raw, list):
            items = raw
        elif isinstance(raw, dict):
            items = list(raw.values())
        else:
            logger.error("Unsupported task storage format")
            return

        for item in items:
            try:
                task = Task.from_dict(item)
            except Exception as exc:
                logger.error(f"Skipping invalid task entry: {exc}")
                continue
            self._tasks[task.id] = task

    def _save(self) -> None:
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {task_id: task.to_dict() for task_id, task in self._tasks.items()}
            with self.storage_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
        except Exception as exc:
            logger.error(f"Failed to save tasks: {exc}")

    def create_task(
        self,
        title: str,
        description: str = "",
        parent_id: str | None = None,
    ) -> Task:
        task_id = uuid.uuid4().hex[:8]
        task = Task(
            id=task_id,
            title=title,
            description=description,
            parent_id=parent_id if parent_id in self._tasks else None,
        )
        self._tasks[task.id] = task
        self._save()
        return task

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def update_task_status(self, task_id: str, status: TaskStatus) -> bool:
        task = self._tasks.get(task_id)
        if task is None:
            return False
        task.status = status
        task.updated_at = time.time()
        self._save()
        return True

    def list_tasks(
        self,
        status: TaskStatus | None = None,
        parent_id: str | None | object = _UNSET,
    ) -> list[Task]:
        tasks = list(self._tasks.values())
        if status is not None:
            tasks = [task for task in tasks if task.status == status]
        if parent_id is not _UNSET:
            tasks = [task for task in tasks if task.parent_id == parent_id]
        return sorted(tasks, key=lambda task: (task.created_at, task.id))

    def get_task_tree(self) -> list[dict[str, Any]]:
        def build(parent_id: str | None) -> list[dict[str, Any]]:
            nodes = []
            for task in self.list_tasks(parent_id=parent_id):
                node = task.to_dict()
                children = build(task.id)
                if children:
                    node["children"] = children
                nodes.append(node)
            return nodes

        return build(None)

    def delete_task(self, task_id: str, delete_subtasks: bool = False) -> bool:
        task = self._tasks.get(task_id)
        if task is None:
            return False

        children = [child.id for child in self.list_tasks(parent_id=task_id)]
        if delete_subtasks:
            for child_id in children:
                self.delete_task(child_id, delete_subtasks=True)
        else:
            for child_id in children:
                child = self._tasks[child_id]
                child.parent_id = None
                child.updated_at = time.time()

        del self._tasks[task_id]
        self._save()
        return True

    def clear_all_tasks(self) -> None:
        self._tasks.clear()
        self._save()
