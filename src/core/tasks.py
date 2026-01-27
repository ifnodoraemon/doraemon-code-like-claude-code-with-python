"""
Unified Task Manager for Doraemon.
Provides task planning, tracking, and persistence.
"""

import json
import logging
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Task:
    def __init__(
        self,
        id: str,
        title: str,
        description: str = "",
        status: TaskStatus = TaskStatus.PENDING,
        priority: TaskPriority = TaskPriority.MEDIUM,
        parent_id: str | None = None,
        tags: list[str] | None = None,
        created_at: float | None = None,
        updated_at: float | None = None,
    ):
        self.id = id
        self.title = title
        self.description = description
        self.status = status
        self.priority = priority
        self.parent_id = parent_id
        self.tags = tags or []
        self.created_at = created_at or time.time()
        self.updated_at = updated_at or self.created_at
        self.subtasks: list[str] = []  # List of IDs

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "parent_id": self.parent_id,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "subtasks": self.subtasks,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        task = cls(
            id=data["id"],
            title=data["title"],
            description=data.get("description", ""),
            status=TaskStatus(data.get("status", "pending")),
            priority=TaskPriority(data.get("priority", "medium")),
            parent_id=data.get("parent_id"),
            tags=data.get("tags", []),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )
        task.subtasks = data.get("subtasks", [])
        return task

class TaskManager:
    def __init__(self, storage_path: Path | None = None):
        self.storage_path = storage_path or Path.cwd() / ".doraemon" / "tasks.json"
        self._tasks: dict[str, Task] = {}
        self._load()

    def _load(self):
        if not self.storage_path.exists():
            return
        try:
            with open(self.storage_path, encoding="utf-8") as f:
                data = json.load(f)
                
                # Handle list format (legacy)
                if isinstance(data, list):
                    for item in data:
                        # Map legacy status
                        if item.get("status") == "todo":
                            item["status"] = "pending"
                        elif item.get("status") == "done":
                            item["status"] = "completed"
                        
                        task = Task.from_dict(item)
                        self._tasks[task.id] = task
                # Handle dict format (new)
                elif isinstance(data, dict):
                    for task_id, task_data in data.items():
                        self._tasks[task_id] = Task.from_dict(task_data)
        except Exception as e:
            logger.error(f"Failed to load tasks: {e}")

    def _save(self):
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump({k: v.to_dict() for k, v in self._tasks.items()}, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save tasks: {e}")

    def create_task(
        self,
        title: str,
        description: str = "",
        priority: TaskPriority = TaskPriority.MEDIUM,
        parent_id: str | None = None,
        tags: list[str] | None = None,
    ) -> Task:
        task_id = str(uuid.uuid4())[:8]
        task = Task(
            id=task_id,
            title=title,
            description=description,
            priority=priority,
            parent_id=parent_id,
            tags=tags,
        )
        self._tasks[task_id] = task
        if parent_id and parent_id in self._tasks:
            self._tasks[parent_id].subtasks.append(task_id)
        self._save()
        return task

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def update_task_status(self, task_id: str, status: TaskStatus) -> bool:
        if task_id not in self._tasks:
            return False
        self._tasks[task_id].status = status
        self._tasks[task_id].updated_at = time.time()
        self._save()
        return True

    def list_tasks(
        self, status: TaskStatus | None = None, parent_id: str | None = None
    ) -> list[Task]:
        tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        if parent_id:
            tasks = [t for t in tasks if t.parent_id == parent_id]
        elif parent_id is None and any(t.parent_id is not None for t in tasks):
            # If parent_id is explicitly None, show only root tasks
            tasks = [t for t in tasks if t.parent_id is None]
        return sorted(tasks, key=lambda t: t.created_at)

    def get_task_tree(self) -> list[dict[str, Any]]:
        root_tasks = [t for t in self._tasks.values() if t.parent_id is None]
        return [self._build_tree_node(t) for t in sorted(root_tasks, key=lambda t: t.created_at)]

    def _build_tree_node(self, task: Task) -> dict[str, Any]:
        node = task.to_dict()
        node["children"] = [
            self._build_tree_node(self._tasks[sid])
            for sid in task.subtasks
            if sid in self._tasks
        ]
        return node

    def delete_task(self, task_id: str, delete_subtasks: bool = False) -> bool:
        if task_id not in self._tasks:
            return False
        task = self._tasks[task_id]
        
        # Remove from parent's subtasks list
        if task.parent_id and task.parent_id in self._tasks:
            if task_id in self._tasks[task.parent_id].subtasks:
                self._tasks[task.parent_id].subtasks.remove(task_id)

        # Delete subtasks if requested
        if delete_subtasks:
            for sid in list(task.subtasks):
                self.delete_task(sid, delete_subtasks=True)
        else:
            # Orphan subtasks
            for sid in task.subtasks:
                if sid in self._tasks:
                    self._tasks[sid].parent_id = None

        del self._tasks[task_id]
        self._save()
        return True

    def clear_all_tasks(self):
        self._tasks.clear()
        self._save()
