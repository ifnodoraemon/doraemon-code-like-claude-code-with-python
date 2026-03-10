"""
Unified Task Manager for Doraemon.
Provides task planning, tracking, and persistence with advanced features:
- Task priority levels (low/medium/high/critical)
- Task dependencies (blocks/blockedBy relationships)
- Progress tracking (0-100%)
- Task status management (pending/in_progress/completed/blocked)
- Advanced querying and filtering
- Task metadata and ownership
"""

import json
import logging
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

from src.core.paths import tasks_path

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Task:
    """
    Represents a task with comprehensive tracking capabilities.

    Attributes:
        id: Unique task identifier
        title: Task title
        description: Detailed task description
        status: Current task status
        priority: Task priority level
        progress: Progress percentage (0-100)
        owner: Task owner identifier
        parent_id: Parent task ID for hierarchical tasks
        tags: List of tags for categorization
        blocks: List of task IDs that this task blocks
        blockedBy: List of task IDs that block this task
        metadata: Custom metadata dictionary
        created_at: Creation timestamp
        updated_at: Last update timestamp
        subtasks: List of subtask IDs
    """

    def __init__(
        self,
        id: str,
        title: str,
        description: str = "",
        status: TaskStatus = TaskStatus.PENDING,
        priority: TaskPriority = TaskPriority.MEDIUM,
        progress: int = 0,
        owner: str | None = None,
        parent_id: str | None = None,
        tags: list[str] | None = None,
        blocks: list[str] | None = None,
        blockedBy: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        created_at: float | None = None,
        updated_at: float | None = None,
    ):
        self.id = id
        self.title = title
        self.description = description
        self.status = status
        self.priority = priority
        self.progress = max(0, min(100, progress))  # Clamp to 0-100
        self.owner = owner
        self.parent_id = parent_id
        self.tags = tags or []
        self.blocks = blocks or []
        self.blockedBy = blockedBy or []
        self.metadata = metadata or {}
        self.created_at = created_at or time.time()
        self.updated_at = updated_at or self.created_at
        self.subtasks: list[str] = []  # List of IDs

    def to_dict(self) -> dict[str, Any]:
        """Convert task to dictionary representation."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "progress": self.progress,
            "owner": self.owner,
            "parent_id": self.parent_id,
            "tags": self.tags,
            "blocks": self.blocks,
            "blockedBy": self.blockedBy,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "subtasks": self.subtasks,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        """Create task from dictionary representation."""
        task = cls(
            id=data["id"],
            title=data["title"],
            description=data.get("description", ""),
            status=TaskStatus(data.get("status", "pending")),
            priority=TaskPriority(data.get("priority", "medium")),
            progress=data.get("progress", 0),
            owner=data.get("owner"),
            parent_id=data.get("parent_id"),
            tags=data.get("tags", []),
            blocks=data.get("blocks", []),
            blockedBy=data.get("blockedBy", []),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )
        task.subtasks = data.get("subtasks", [])
        return task

class TaskManager:
    """
    Manages tasks with support for hierarchies, dependencies, and advanced querying.

    Features:
    - Task creation and management
    - Hierarchical task structures (parent-child relationships)
    - Task dependencies (blocks/blockedBy)
    - Progress tracking
    - Advanced filtering and querying
    - Persistence to JSON
    """

    def __init__(self, storage_path: Path | None = None):
        self.storage_path = storage_path or tasks_path()
        self._tasks: dict[str, Task] = {}
        self._load()

    def _load(self) -> None:
        """Load tasks from storage file."""
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

    def _save(self) -> None:
        """Save tasks to storage file."""
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
        progress: int = 0,
        owner: str | None = None,
        parent_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Task:
        """
        Create a new task.

        Args:
            title: Task title
            description: Task description
            priority: Task priority level
            progress: Initial progress (0-100)
            owner: Task owner identifier
            parent_id: Parent task ID for hierarchical tasks
            tags: List of tags
            metadata: Custom metadata dictionary

        Returns:
            Created Task object
        """
        task_id = str(uuid.uuid4())[:8]
        task = Task(
            id=task_id,
            title=title,
            description=description,
            priority=priority,
            progress=progress,
            owner=owner,
            parent_id=parent_id,
            tags=tags,
            metadata=metadata,
        )
        self._tasks[task_id] = task
        if parent_id and parent_id in self._tasks:
            self._tasks[parent_id].subtasks.append(task_id)
        self._save()
        return task

    def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def update_task_status(self, task_id: str, status: TaskStatus) -> bool:
        """Update task status."""
        if task_id not in self._tasks:
            return False
        self._tasks[task_id].status = status
        self._tasks[task_id].updated_at = time.time()
        self._save()
        return True

    def update_task_progress(self, task_id: str, progress: int) -> bool:
        """
        Update task progress.

        Args:
            task_id: Task ID
            progress: Progress percentage (0-100)

        Returns:
            True if successful, False if task not found
        """
        if task_id not in self._tasks:
            return False
        self._tasks[task_id].progress = max(0, min(100, progress))
        self._tasks[task_id].updated_at = time.time()
        self._save()
        return True

    def update_task_owner(self, task_id: str, owner: str | None) -> bool:
        """
        Update task owner.

        Args:
            task_id: Task ID
            owner: Owner identifier or None

        Returns:
            True if successful, False if task not found
        """
        if task_id not in self._tasks:
            return False
        self._tasks[task_id].owner = owner
        self._tasks[task_id].updated_at = time.time()
        self._save()
        return True

    def update_task_metadata(self, task_id: str, metadata: dict[str, Any]) -> bool:
        """
        Update task metadata.

        Args:
            task_id: Task ID
            metadata: Metadata dictionary to merge

        Returns:
            True if successful, False if task not found
        """
        if task_id not in self._tasks:
            return False
        self._tasks[task_id].metadata.update(metadata)
        self._tasks[task_id].updated_at = time.time()
        self._save()
        return True

    def add_task_dependency(self, task_id: str, blocks_task_id: str) -> bool:
        """
        Add a dependency: this task blocks another task.

        Args:
            task_id: Task that blocks
            blocks_task_id: Task that is blocked

        Returns:
            True if successful, False if either task not found
        """
        if task_id not in self._tasks or blocks_task_id not in self._tasks:
            return False
        if blocks_task_id not in self._tasks[task_id].blocks:
            self._tasks[task_id].blocks.append(blocks_task_id)
        if task_id not in self._tasks[blocks_task_id].blockedBy:
            self._tasks[blocks_task_id].blockedBy.append(task_id)
        self._save()
        return True

    def remove_task_dependency(self, task_id: str, blocks_task_id: str) -> bool:
        """
        Remove a dependency.

        Args:
            task_id: Task that blocks
            blocks_task_id: Task that is blocked

        Returns:
            True if successful, False if either task not found
        """
        if task_id not in self._tasks or blocks_task_id not in self._tasks:
            return False
        if blocks_task_id in self._tasks[task_id].blocks:
            self._tasks[task_id].blocks.remove(blocks_task_id)
        if task_id in self._tasks[blocks_task_id].blockedBy:
            self._tasks[blocks_task_id].blockedBy.remove(task_id)
        self._save()
        return True

    def list_tasks(
        self,
        status: TaskStatus | None = None,
        priority: TaskPriority | None = None,
        owner: str | None = None,
        parent_id: str | None = None,
        tags: list[str] | None = None,
        sort_by: str = "created_at",
    ) -> list[Task]:
        """
        List tasks with advanced filtering.

        Args:
            status: Filter by status
            priority: Filter by priority
            owner: Filter by owner
            parent_id: Filter by parent task
            tags: Filter by tags (all tags must match)
            sort_by: Sort field (created_at, priority, progress, updated_at)

        Returns:
            Filtered and sorted list of tasks
        """
        tasks = list(self._tasks.values())

        # Apply filters
        if status:
            tasks = [t for t in tasks if t.status == status]
        if priority:
            tasks = [t for t in tasks if t.priority == priority]
        if owner:
            tasks = [t for t in tasks if t.owner == owner]
        if tags:
            tasks = [t for t in tasks if all(tag in t.tags for tag in tags)]

        # Handle parent_id filter
        if parent_id:
            tasks = [t for t in tasks if t.parent_id == parent_id]
        elif parent_id is None and any(t.parent_id is not None for t in tasks):
            # If parent_id is explicitly None, show only root tasks
            tasks = [t for t in tasks if t.parent_id is None]

        # Sort tasks
        return self._sort_tasks(tasks, sort_by)

    def _sort_tasks(self, tasks: list[Task], sort_by: str) -> list[Task]:
        """Sort tasks by specified field."""
        sort_map = {
            "created_at": lambda t: t.created_at,
            "updated_at": lambda t: t.updated_at,
            "priority": lambda t: self._priority_order(t.priority),
            "progress": lambda t: t.progress,
            "title": lambda t: t.title,
        }
        key_func = sort_map.get(sort_by, lambda t: t.created_at)
        return sorted(tasks, key=key_func)

    @staticmethod
    def _priority_order(priority: TaskPriority) -> int:
        """Get numeric order for priority (lower is higher priority)."""
        order = {
            TaskPriority.CRITICAL: 0,
            TaskPriority.HIGH: 1,
            TaskPriority.MEDIUM: 2,
            TaskPriority.LOW: 3,
        }
        return order.get(priority, 2)

    def get_blocked_tasks(self, task_id: str) -> list[Task]:
        """Get all tasks blocked by the given task."""
        if task_id not in self._tasks:
            return []
        task = self._tasks[task_id]
        return [self._tasks[bid] for bid in task.blocks if bid in self._tasks]

    def get_blocking_tasks(self, task_id: str) -> list[Task]:
        """Get all tasks that block the given task."""
        if task_id not in self._tasks:
            return []
        task = self._tasks[task_id]
        return [self._tasks[bid] for bid in task.blockedBy if bid in self._tasks]

    def is_task_blocked(self, task_id: str) -> bool:
        """Check if a task is currently blocked."""
        if task_id not in self._tasks:
            return False
        task = self._tasks[task_id]
        # Task is blocked if it has blocking tasks that are not completed
        return any(
            self._tasks[bid].status != TaskStatus.COMPLETED
            for bid in task.blockedBy
            if bid in self._tasks
        )

    def get_task_tree(self) -> list[dict[str, Any]]:
        """Get hierarchical task tree."""
        root_tasks = [t for t in self._tasks.values() if t.parent_id is None]
        return [self._build_tree_node(t) for t in sorted(root_tasks, key=lambda t: t.created_at)]

    def _build_tree_node(self, task: Task) -> dict[str, Any]:
        """Build a tree node for a task and its subtasks."""
        node = task.to_dict()
        node["children"] = [
            self._build_tree_node(self._tasks[sid])
            for sid in task.subtasks
            if sid in self._tasks
        ]
        return node

    def delete_task(self, task_id: str, delete_subtasks: bool = False) -> bool:
        """
        Delete a task.

        Args:
            task_id: Task ID to delete
            delete_subtasks: If True, delete all subtasks; if False, orphan them

        Returns:
            True if successful, False if task not found
        """
        if task_id not in self._tasks:
            return False
        task = self._tasks[task_id]

        # Remove from parent's subtasks list
        if task.parent_id and task.parent_id in self._tasks:
            if task_id in self._tasks[task.parent_id].subtasks:
                self._tasks[task.parent_id].subtasks.remove(task_id)

        # Clean up dependencies
        for blocked_id in list(task.blocks):
            if blocked_id in self._tasks:
                self._tasks[blocked_id].blockedBy.remove(task_id)
        for blocker_id in list(task.blockedBy):
            if blocker_id in self._tasks:
                self._tasks[blocker_id].blocks.remove(task_id)

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

    def clear_all_tasks(self) -> None:
        """Clear all tasks."""
        self._tasks.clear()
        self._save()

    def get_tasks_by_owner(self, owner: str) -> list[Task]:
        """Get all tasks owned by a specific owner."""
        return [t for t in self._tasks.values() if t.owner == owner]

    def get_tasks_by_status(self, status: TaskStatus) -> list[Task]:
        """Get all tasks with a specific status."""
        return [t for t in self._tasks.values() if t.status == status]

    def get_tasks_by_priority(self, priority: TaskPriority) -> list[Task]:
        """Get all tasks with a specific priority."""
        return [t for t in self._tasks.values() if t.priority == priority]

    def get_high_priority_tasks(self) -> list[Task]:
        """Get all high and critical priority tasks."""
        return [
            t for t in self._tasks.values()
            if t.priority in (TaskPriority.HIGH, TaskPriority.CRITICAL)
        ]

    def get_in_progress_tasks(self) -> list[Task]:
        """Get all in-progress tasks."""
        return self.get_tasks_by_status(TaskStatus.IN_PROGRESS)

    def get_blocked_tasks_list(self) -> list[Task]:
        """Get all blocked tasks."""
        return self.get_tasks_by_status(TaskStatus.BLOCKED)

    def get_completed_tasks(self) -> list[Task]:
        """Get all completed tasks."""
        return self.get_tasks_by_status(TaskStatus.COMPLETED)

    def get_pending_tasks(self) -> list[Task]:
        """Get all pending tasks."""
        return self.get_tasks_by_status(TaskStatus.PENDING)

    def get_tasks_with_tag(self, tag: str) -> list[Task]:
        """Get all tasks with a specific tag."""
        return [t for t in self._tasks.values() if tag in t.tags]

    def get_incomplete_tasks(self) -> list[Task]:
        """Get all incomplete tasks (not completed or cancelled)."""
        return [
            t for t in self._tasks.values()
            if t.status not in (TaskStatus.COMPLETED, TaskStatus.CANCELLED)
        ]

    def get_task_statistics(self) -> dict[str, Any]:
        """Get statistics about all tasks."""
        tasks = list(self._tasks.values())
        if not tasks:
            return {
                "total": 0,
                "by_status": {},
                "by_priority": {},
                "by_owner": {},
                "average_progress": 0,
                "blocked_count": 0,
            }

        return {
            "total": len(tasks),
            "by_status": self._count_by_status(tasks),
            "by_priority": self._count_by_priority(tasks),
            "by_owner": self._count_by_owner(tasks),
            "average_progress": sum(t.progress for t in tasks) / len(tasks),
            "blocked_count": sum(1 for t in tasks if self.is_task_blocked(t.id)),
        }

    @staticmethod
    def _count_by_status(tasks: list[Task]) -> dict[str, int]:
        """Count tasks by status."""
        counts = {}
        for task in tasks:
            status = task.status.value
            counts[status] = counts.get(status, 0) + 1
        return counts

    @staticmethod
    def _count_by_priority(tasks: list[Task]) -> dict[str, int]:
        """Count tasks by priority."""
        counts = {}
        for task in tasks:
            priority = task.priority.value
            counts[priority] = counts.get(priority, 0) + 1
        return counts

    @staticmethod
    def _count_by_owner(tasks: list[Task]) -> dict[str, int]:
        """Count tasks by owner."""
        counts = {}
        for task in tasks:
            owner = task.owner or "unassigned"
            counts[owner] = counts.get(owner, 0) + 1
        return counts
