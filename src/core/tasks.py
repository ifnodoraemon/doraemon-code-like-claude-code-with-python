"""Persistent task graph for coordination, claiming, and workspace isolation."""

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


class TaskClaimError(ValueError):
    """Raised when a task cannot be claimed or released."""


@dataclass(slots=True)
class Task:
    """A persistent task record with lightweight graph metadata."""

    id: str
    title: str
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    parent_id: str | None = None
    dependencies: list[str] = field(default_factory=list)
    assigned_agent: str | None = None
    claimed_at: float | None = None
    priority: int = 0
    workspace_id: str | None = None
    workspace_path: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "parent_id": self.parent_id,
            "dependencies": list(self.dependencies),
            "assigned_agent": self.assigned_agent,
            "claimed_at": self.claimed_at,
            "priority": self.priority,
            "workspace_id": self.workspace_id,
            "workspace_path": self.workspace_path,
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
            dependencies=list(dict.fromkeys(data.get("dependencies", []))),
            assigned_agent=data.get("assigned_agent"),
            claimed_at=data.get("claimed_at"),
            priority=int(data.get("priority", 0)),
            workspace_id=data.get("workspace_id"),
            workspace_path=data.get("workspace_path"),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", data.get("created_at", time.time())),
        )


class TaskManager:
    """Persistent task storage with dependency and claiming support."""

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
        dependencies: list[str] | None = None,
        priority: int = 0,
        workspace_id: str | None = None,
    ) -> Task:
        task_id = uuid.uuid4().hex[:8]
        normalized_dependencies = self._normalize_dependencies(dependencies or [], task_id=task_id)
        workspace_meta = self._build_workspace_metadata(task_id, workspace_id=workspace_id)
        task = Task(
            id=task_id,
            title=title,
            description=description,
            parent_id=parent_id if parent_id in self._tasks else None,
            dependencies=normalized_dependencies,
            priority=priority,
            workspace_id=workspace_meta["workspace_id"],
            workspace_path=workspace_meta["workspace_path"],
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
        if status in {TaskStatus.COMPLETED, TaskStatus.CANCELLED}:
            task.assigned_agent = None
            task.claimed_at = None
        task.updated_at = time.time()
        self._save()
        return True

    def update_task(
        self,
        task_id: str,
        *,
        status: TaskStatus | None = None,
        dependencies: list[str] | None = None,
        priority: int | None = None,
        assigned_agent: str | None | object = _UNSET,
        workspace_id: str | None = None,
    ) -> Task | None:
        task = self._tasks.get(task_id)
        if task is None:
            return None

        if status is not None:
            task.status = status
            if status in {TaskStatus.COMPLETED, TaskStatus.CANCELLED}:
                task.assigned_agent = None
                task.claimed_at = None

        if dependencies is not None:
            task.dependencies = self._normalize_dependencies(dependencies, task_id=task.id)

        if priority is not None:
            task.priority = priority

        if assigned_agent is not _UNSET:
            task.assigned_agent = assigned_agent
            task.claimed_at = time.time() if assigned_agent else None

        if workspace_id is not None:
            workspace_meta = self._build_workspace_metadata(task.id, workspace_id=workspace_id)
            task.workspace_id = workspace_meta["workspace_id"]
            task.workspace_path = workspace_meta["workspace_path"]

        task.updated_at = time.time()
        self._save()
        return task

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

    def list_ready_tasks(self, include_claimed: bool = False) -> list[Task]:
        ready = []
        for task in self.list_tasks():
            if task.status != TaskStatus.PENDING:
                continue
            if task.assigned_agent and not include_claimed:
                continue
            if self.are_dependencies_satisfied(task.id):
                ready.append(task)
        return sorted(ready, key=lambda task: (-task.priority, task.created_at, task.id))

    def are_dependencies_satisfied(self, task_id: str) -> bool:
        task = self.get_task(task_id)
        if task is None:
            return False
        for dependency_id in task.dependencies:
            dependency = self.get_task(dependency_id)
            if dependency is None or dependency.status != TaskStatus.COMPLETED:
                return False
        return True

    def claim_task(self, task_id: str, agent_id: str) -> Task:
        task = self.get_task(task_id)
        if task is None:
            raise TaskClaimError(f"task not found: {task_id}")
        if task.assigned_agent == agent_id and task.status == TaskStatus.IN_PROGRESS:
            return task
        if task.status != TaskStatus.PENDING:
            raise TaskClaimError(f"task {task_id} is not pending")
        if not self.are_dependencies_satisfied(task_id):
            raise TaskClaimError(f"task {task_id} has unresolved dependencies")
        if task.assigned_agent and task.assigned_agent != agent_id:
            raise TaskClaimError(f"task {task_id} already claimed by {task.assigned_agent}")

        task.assigned_agent = agent_id
        task.claimed_at = time.time()
        task.status = TaskStatus.IN_PROGRESS
        task.updated_at = task.claimed_at
        self._save()
        return task

    def release_task(self, task_id: str, agent_id: str | None = None) -> Task:
        task = self.get_task(task_id)
        if task is None:
            raise TaskClaimError(f"task not found: {task_id}")
        if task.assigned_agent is None:
            raise TaskClaimError(f"task {task_id} is not claimed")
        if agent_id and task.assigned_agent != agent_id:
            raise TaskClaimError(f"task {task_id} is claimed by {task.assigned_agent}")

        task.assigned_agent = None
        task.claimed_at = None
        if task.status == TaskStatus.IN_PROGRESS:
            task.status = TaskStatus.PENDING
        task.updated_at = time.time()
        self._save()
        return task

    def get_task_tree(self) -> list[dict[str, Any]]:
        def build(parent_id: str | None) -> list[dict[str, Any]]:
            nodes = []
            for task in self.list_tasks(parent_id=parent_id):
                node = task.to_dict()
                node["ready"] = self.are_dependencies_satisfied(task.id)
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
        for other in self._tasks.values():
            if task_id in other.dependencies:
                other.dependencies = [dep for dep in other.dependencies if dep != task_id]
                other.updated_at = time.time()
        self._save()
        return True

    def clear_all_tasks(self) -> None:
        self._tasks.clear()
        self._save()

    def _normalize_dependencies(self, dependencies: list[str], task_id: str) -> list[str]:
        normalized: list[str] = []
        for dependency_id in dependencies:
            if not dependency_id or dependency_id == task_id:
                continue
            if dependency_id not in self._tasks and dependency_id != task_id:
                raise ValueError(f"unknown dependency: {dependency_id}")
            if self._has_dependency_path(dependency_id, task_id):
                raise ValueError(f"dependency cycle detected: {task_id} -> {dependency_id}")
            if dependency_id not in normalized:
                normalized.append(dependency_id)
        return normalized

    def _has_dependency_path(self, start_id: str, target_id: str) -> bool:
        if start_id == target_id:
            return True

        stack = [start_id]
        seen: set[str] = set()
        while stack:
            current_id = stack.pop()
            if current_id in seen:
                continue
            seen.add(current_id)
            if current_id == target_id:
                return True
            current = self._tasks.get(current_id)
            if current:
                stack.extend(current.dependencies)
        return False

    def _build_workspace_metadata(
        self,
        task_id: str,
        *,
        workspace_id: str | None = None,
    ) -> dict[str, str]:
        workspace_value = workspace_id or f"task-{task_id}"
        workspace_root = self.storage_path.parent / "workspaces"
        workspace_path = workspace_root / workspace_value
        workspace_path.mkdir(parents=True, exist_ok=True)
        return {
            "workspace_id": workspace_value,
            "workspace_path": str(workspace_path),
        }
