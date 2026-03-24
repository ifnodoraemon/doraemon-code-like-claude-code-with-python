"""
Background Task System

Enables running long-running tasks in the background while continuing
to interact with the agent.

Features:
- Move tasks to background with Ctrl+B
- Track task progress and status
- Retrieve task output
- Cancel running tasks
"""

import asyncio
import logging
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BackgroundTask:
    """A background task."""

    id: str
    name: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    output: str = ""
    error: str | None = None
    progress: int = 0  # 0-100
    token_usage: int = 0
    _task: asyncio.Task | None = field(default=None, repr=False)
    _output_buffer: list[str] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "progress": self.progress,
            "token_usage": self.token_usage,
            "duration": self._get_duration(),
            "output_preview": self.output[:500] if self.output else None,
            "error": self.error,
        }

    def _get_duration(self) -> float | None:
        """Get task duration in seconds."""
        if not self.started_at:
            return None
        end_time = self.completed_at or time.time()
        return round(end_time - self.started_at, 2)

    def append_output(self, text: str):
        """Append to output buffer."""
        self._output_buffer.append(text)
        self.output = "".join(self._output_buffer)


class BackgroundTaskManager:
    """
    Manages background tasks.

    Usage:
        mgr = BackgroundTaskManager()

        # Start a background task
        task_id = await mgr.start_task(
            name="Code Review",
            description="Reviewing all Python files",
            coroutine=review_code()
        )

        # Check status
        status = mgr.get_task_status(task_id)

        # Get output
        output = mgr.get_task_output(task_id)

        # List all tasks
        tasks = mgr.list_tasks()

        # Cancel a task
        mgr.cancel_task(task_id)
    """

    def __init__(self, max_concurrent: int = 5):
        """
        Initialize task manager.

        Args:
            max_concurrent: Maximum concurrent background tasks
        """
        self.max_concurrent = max_concurrent
        self._tasks: dict[str, BackgroundTask] = {}
        self._running_count: int = 0
        self._max_completed: int = 100  # Auto-cleanup threshold

    async def start_task(
        self,
        name: str,
        description: str,
        coroutine: Coroutine,
        on_complete: Callable[[BackgroundTask], None] | None = None,
    ) -> str:
        """
        Start a new background task.

        Args:
            name: Task name
            description: Task description
            coroutine: Async coroutine to run
            on_complete: Optional callback when task completes

        Returns:
            Task ID

        Raises:
            RuntimeError: If max concurrent tasks reached
        """
        if self._running_count >= self.max_concurrent:
            raise RuntimeError(
                f"Maximum concurrent tasks ({self.max_concurrent}) reached. "
                "Wait for a task to complete or cancel one."
            )

        task_id = str(uuid.uuid4())[:8]
        task = BackgroundTask(
            id=task_id,
            name=name,
            description=description,
        )

        self._tasks[task_id] = task

        # Auto-cleanup if too many completed tasks accumulated
        completed_count = sum(
            1
            for t in self._tasks.values()
            if t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
        )
        if completed_count > self._max_completed:
            self.cleanup_completed(max_age=1800)

        # Create wrapper that updates task status
        async def run_task():
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            self._running_count += 1

            try:
                result = await coroutine
                task.output = str(result) if result else ""
                task.status = TaskStatus.COMPLETED
                task.progress = 100
                logger.info(f"Background task {task_id} completed: {name}")

            except asyncio.CancelledError:
                task.status = TaskStatus.CANCELLED
                task.error = "Task was cancelled"
                logger.info(f"Background task {task_id} cancelled: {name}")

            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                logger.error(f"Background task {task_id} failed: {e}")

            finally:
                task.completed_at = time.time()
                self._running_count -= 1

                if on_complete:
                    try:
                        on_complete(task)
                    except Exception as e:
                        logger.error(f"on_complete callback failed: {e}")

        # Start the task
        task._task = asyncio.create_task(run_task())
        logger.info(f"Started background task {task_id}: {name}")

        return task_id

    def background_current(
        self,
        name: str,
        description: str,
        task: asyncio.Task,
    ) -> str:
        """
        Move a currently running task to background.

        This is used when the user presses Ctrl+B to background
        a long-running operation.

        Args:
            name: Task name
            description: Task description
            task: The asyncio Task to background

        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())[:8]
        bg_task = BackgroundTask(
            id=task_id,
            name=name,
            description=description,
            status=TaskStatus.RUNNING,
            started_at=time.time(),
            _task=task,
        )

        self._tasks[task_id] = bg_task
        self._running_count += 1

        # Add completion callback
        def on_done(future):
            self._running_count -= 1
            bg_task.completed_at = time.time()

            try:
                result = future.result()
                bg_task.output = str(result) if result else ""
                bg_task.status = TaskStatus.COMPLETED
                bg_task.progress = 100
            except asyncio.CancelledError:
                bg_task.status = TaskStatus.CANCELLED
                bg_task.error = "Task was cancelled"
            except Exception as e:
                bg_task.status = TaskStatus.FAILED
                bg_task.error = str(e)

        task.add_done_callback(on_done)
        logger.info(f"Backgrounded task {task_id}: {name}")

        return task_id

    def get_task(self, task_id: str) -> BackgroundTask | None:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def get_task_status(self, task_id: str) -> dict[str, Any] | None:
        """Get task status."""
        task = self._tasks.get(task_id)
        if not task:
            return None
        return task.to_dict()

    def get_task_output(self, task_id: str, tail: int | None = None) -> str | None:
        """
        Get task output.

        Args:
            task_id: Task ID
            tail: If provided, return only last N characters

        Returns:
            Task output or None if task not found
        """
        task = self._tasks.get(task_id)
        if not task:
            return None

        output = task.output
        if tail and len(output) > tail:
            return f"... (truncated)\n{output[-tail:]}"
        return output

    def update_task_progress(self, task_id: str, progress: int, output: str = ""):
        """
        Update task progress.

        Args:
            task_id: Task ID
            progress: Progress percentage (0-100)
            output: Optional output to append
        """
        task = self._tasks.get(task_id)
        if task:
            task.progress = min(100, max(0, progress))
            if output:
                task.append_output(output)

    def update_task_tokens(self, task_id: str, tokens: int):
        """Update task token usage."""
        task = self._tasks.get(task_id)
        if task:
            task.token_usage += tokens

    def list_tasks(self, status: TaskStatus | None = None, limit: int = 20) -> list[dict[str, Any]]:
        """
        List tasks.

        Args:
            status: Filter by status
            limit: Maximum number to return

        Returns:
            List of task summaries
        """
        tasks = list(self._tasks.values())

        if status:
            tasks = [t for t in tasks if t.status == status]

        # Sort by created_at descending
        tasks.sort(key=lambda t: t.created_at, reverse=True)

        return [t.to_dict() for t in tasks[:limit]]

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task.

        Args:
            task_id: Task ID

        Returns:
            True if task was cancelled
        """
        task = self._tasks.get(task_id)
        if not task:
            return False

        if task.status != TaskStatus.RUNNING:
            return False

        if task._task:
            task._task.cancel()
            logger.info(f"Cancelled task {task_id}")
            return True

        return False

    def get_running_tasks(self) -> list[BackgroundTask]:
        """Get all running tasks."""
        return [t for t in self._tasks.values() if t.status == TaskStatus.RUNNING]

    def get_running_count(self) -> int:
        """Get number of running tasks."""
        return self._running_count

    def cleanup_completed(self, max_age: float = 3600):
        """
        Remove completed tasks older than max_age seconds.

        Args:
            max_age: Maximum age in seconds (default: 1 hour)
        """
        cutoff = time.time() - max_age
        to_remove = []

        for task_id, task in self._tasks.items():
            if task.status in (
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
            ):
                if task.completed_at and task.completed_at < cutoff:
                    to_remove.append(task_id)

        for task_id in to_remove:
            del self._tasks[task_id]

        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} old tasks")


# Global instance
_task_manager: BackgroundTaskManager | None = None


def get_task_manager() -> BackgroundTaskManager:
    """Get the global task manager."""
    global _task_manager
    if _task_manager is None:
        _task_manager = BackgroundTaskManager()
    return _task_manager
