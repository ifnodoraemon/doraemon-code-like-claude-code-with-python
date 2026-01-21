"""
Task Management System for Polymath

Provides task planning, tracking, and persistence for plan/build workflow.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class TaskStatus(Enum):
    """Task status enumeration"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Task:
    """Represents a single task"""

    id: str
    title: str
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    parent_id: str | None = None
    subtasks: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data["status"] = self.status.value
        data["priority"] = self.priority.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        """Create from dictionary"""
        data["status"] = TaskStatus(data["status"])
        data["priority"] = TaskPriority(data["priority"])
        return cls(**data)


class TaskManager:
    """Manages tasks with persistence"""

    def __init__(self, workspace_path: Path | None = None):
        """
        Initialize task manager

        Args:
            workspace_path: Path to workspace directory (defaults to .polymath/)
        """
        if workspace_path is None:
            workspace_path = Path.cwd() / ".polymath"

        self.workspace_path = Path(workspace_path)
        self.workspace_path.mkdir(parents=True, exist_ok=True)

        self.tasks_file = self.workspace_path / "tasks.json"
        self.tasks: dict[str, Task] = {}
        self._load_tasks()

    def _load_tasks(self):
        """Load tasks from disk"""
        if self.tasks_file.exists():
            try:
                with open(self.tasks_file, encoding="utf-8") as f:
                    data = json.load(f)

                    # Handle both list format (legacy) and dict format (new)
                    if isinstance(data, list):
                        # Legacy format: list of task dicts
                        for task_data in data:
                            task_id = task_data.get("id", "")
                            if task_id:
                                # Convert legacy format to new format
                                converted = self._convert_legacy_task(task_data)
                                self.tasks[task_id] = Task.from_dict(converted)
                    elif isinstance(data, dict):
                        # New format: dict keyed by task_id
                        self.tasks = {
                            task_id: Task.from_dict(task_data)
                            for task_id, task_data in data.items()
                        }
            except Exception as e:
                print(f"Warning: Failed to load tasks: {e}")
                self.tasks = {}

    def _convert_legacy_task(self, legacy_data: dict) -> dict:
        """Convert legacy task format to new format"""
        # Map old status values to new enum values
        status_map = {
            "todo": "pending",
            "in_progress": "in_progress",
            "done": "completed",
            "blocked": "blocked",
            "cancelled": "cancelled",
        }

        old_status = legacy_data.get("status", "pending")
        new_status = status_map.get(old_status, old_status)

        return {
            "id": legacy_data.get("id", ""),
            "title": legacy_data.get("title", ""),
            "description": legacy_data.get("description", ""),
            "status": new_status,
            "priority": legacy_data.get("priority", "medium"),
            "parent_id": legacy_data.get("parent_id"),
            "subtasks": legacy_data.get("subtasks", []),
            "created_at": self._convert_timestamp(legacy_data.get("created_at")),
            "updated_at": self._convert_timestamp(
                legacy_data.get("updated_at", legacy_data.get("created_at"))
            ),
            "tags": legacy_data.get("tags", []),
            "metadata": legacy_data.get("metadata", {}),
        }

    def _convert_timestamp(self, ts) -> str:
        """Convert timestamp to ISO format string"""
        if ts is None:
            return datetime.now().isoformat()
        if isinstance(ts, str):
            return ts
        if isinstance(ts, int | float):
            return datetime.fromtimestamp(ts).isoformat()
        return datetime.now().isoformat()

    def _save_tasks(self):
        """Save tasks to disk"""
        try:
            data = {task_id: task.to_dict() for task_id, task in self.tasks.items()}
            with open(self.tasks_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to save tasks: {e}")

    def create_task(
        self,
        title: str,
        description: str = "",
        priority: TaskPriority = TaskPriority.MEDIUM,
        parent_id: str | None = None,
        tags: list[str] | None = None,
    ) -> Task:
        """Create a new task"""
        # Generate unique ID
        task_id = f"task_{len(self.tasks) + 1}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        task = Task(
            id=task_id,
            title=title,
            description=description,
            priority=priority,
            parent_id=parent_id,
            tags=tags if tags is not None else [],
        )

        self.tasks[task_id] = task

        # Add to parent's subtasks if parent exists
        if parent_id and parent_id in self.tasks:
            self.tasks[parent_id].subtasks.append(task_id)

        self._save_tasks()
        return task

    def get_task(self, task_id: str) -> Task | None:
        """Get task by ID"""
        return self.tasks.get(task_id)

    def update_task_status(self, task_id: str, status: TaskStatus) -> bool:
        """Update task status"""
        if task_id in self.tasks:
            self.tasks[task_id].status = status
            self.tasks[task_id].updated_at = datetime.now().isoformat()
            self._save_tasks()
            return True
        return False

    def list_tasks(
        self,
        status: TaskStatus | None = None,
        parent_id: str | None = None,
        include_subtasks: bool = False,
    ) -> list[Task]:
        """
        List tasks with optional filtering

        Args:
            status: Filter by status
            parent_id: Filter by parent (None = root tasks only)
            include_subtasks: Include all subtasks recursively
        """
        tasks = list(self.tasks.values())

        # Filter by parent
        if parent_id is None and not include_subtasks:
            # Root tasks only (no parent)
            tasks = [t for t in tasks if t.parent_id is None]
        elif parent_id is not None:
            # Direct children of parent
            tasks = [t for t in tasks if t.parent_id == parent_id]

        # Filter by status
        if status is not None:
            tasks = [t for t in tasks if t.status == status]

        return tasks

    def delete_task(self, task_id: str, delete_subtasks: bool = False) -> bool:
        """
        Delete a task

        Args:
            task_id: Task to delete
            delete_subtasks: Also delete all subtasks recursively
        """
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]

        # Handle subtasks
        if delete_subtasks:
            for subtask_id in task.subtasks:
                self.delete_task(subtask_id, delete_subtasks=True)
        else:
            # Re-parent subtasks to this task's parent
            for subtask_id in task.subtasks:
                if subtask_id in self.tasks:
                    self.tasks[subtask_id].parent_id = task.parent_id

        # Remove from parent's subtask list
        if task.parent_id and task.parent_id in self.tasks:
            parent = self.tasks[task.parent_id]
            if task_id in parent.subtasks:
                parent.subtasks.remove(task_id)

        # Delete the task
        del self.tasks[task_id]
        self._save_tasks()
        return True

    def clear_all_tasks(self):
        """Clear all tasks"""
        self.tasks = {}
        self._save_tasks()

    def get_task_tree(self, root_id: str | None = None) -> list[dict[str, Any]]:
        """
        Get task hierarchy as nested structure

        Args:
            root_id: Root task ID (None = all root tasks)

        Returns:
            List of task dictionaries with 'children' key
        """

        def build_tree(task: Task) -> dict[str, Any]:
            task_dict = task.to_dict()
            task_dict["children"] = [
                build_tree(self.tasks[subtask_id])
                for subtask_id in task.subtasks
                if subtask_id in self.tasks
            ]
            return task_dict

        if root_id:
            if root_id in self.tasks:
                return [build_tree(self.tasks[root_id])]
            return []
        else:
            # Get all root tasks
            root_tasks = [t for t in self.tasks.values() if t.parent_id is None]
            return [build_tree(task) for task in root_tasks]
