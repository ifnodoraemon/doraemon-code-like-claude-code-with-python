"""
Task Recovery System

Automatic task interruption detection and recovery.

Features:
- Auto-save task state before interruption
- Resume interrupted tasks
- Graceful shutdown handling
- Progress persistence
- Recovery strategies
"""

import asyncio
import atexit
import json
import logging
import signal
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TaskState(Enum):
    """Task execution state."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class InterruptionType(Enum):
    """Type of interruption."""

    USER_CANCEL = "user_cancel"  # Ctrl+C
    SYSTEM_SHUTDOWN = "system_shutdown"  # SIGTERM
    ERROR = "error"  # Unhandled error
    TIMEOUT = "timeout"  # Task timeout
    RESOURCE_LIMIT = "resource_limit"  # Memory/CPU limit


@dataclass
class TaskProgress:
    """Progress of a task."""

    current_step: int = 0
    total_steps: int = 0
    current_action: str = ""
    completed_actions: list[str] = field(default_factory=list)
    pending_actions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "current_action": self.current_action,
            "completed_actions": self.completed_actions,
            "pending_actions": self.pending_actions,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskProgress":
        return cls(
            current_step=data.get("current_step", 0),
            total_steps=data.get("total_steps", 0),
            current_action=data.get("current_action", ""),
            completed_actions=data.get("completed_actions", []),
            pending_actions=data.get("pending_actions", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RecoverableTask:
    """A task that can be recovered after interruption."""

    id: str
    name: str
    description: str
    state: TaskState
    progress: TaskProgress
    context: dict[str, Any]  # Task-specific context
    created_at: float
    updated_at: float
    interruption_type: InterruptionType | None = None
    error_message: str | None = None
    conversation_id: str | None = None
    checkpoint_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "state": self.state.value,
            "progress": self.progress.to_dict(),
            "context": self.context,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "interruption_type": self.interruption_type.value if self.interruption_type else None,
            "error_message": self.error_message,
            "conversation_id": self.conversation_id,
            "checkpoint_id": self.checkpoint_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RecoverableTask":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            state=TaskState(data["state"]),
            progress=TaskProgress.from_dict(data.get("progress", {})),
            context=data.get("context", {}),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            interruption_type=InterruptionType(data["interruption_type"]) if data.get("interruption_type") else None,
            error_message=data.get("error_message"),
            conversation_id=data.get("conversation_id"),
            checkpoint_id=data.get("checkpoint_id"),
        )


class TaskRecoveryManager:
    """
    Manages task interruption and recovery.

    Usage:
        recovery = TaskRecoveryManager()

        # Register signal handlers
        recovery.setup_signal_handlers()

        # Start a recoverable task
        task_id = recovery.start_task("Build feature X", context={"files": [...]})

        # Update progress
        recovery.update_progress(task_id, step=1, action="Reading files")

        # Mark completed
        recovery.complete_task(task_id)

        # On restart, check for interrupted tasks
        interrupted = recovery.get_interrupted_tasks()
        for task in interrupted:
            if should_resume(task):
                recovery.resume_task(task.id)
    """

    def __init__(
        self,
        storage_dir: Path | None = None,
        auto_save_interval: float = 5.0,
    ):
        """
        Initialize task recovery manager.

        Args:
            storage_dir: Directory for task state storage
            auto_save_interval: Auto-save interval in seconds
        """
        self._storage_dir = storage_dir or Path.home() / ".doraemon" / "recovery"
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        self._tasks: dict[str, RecoverableTask] = {}
        self._current_task: str | None = None
        self._auto_save_interval = auto_save_interval
        self._auto_save_task: asyncio.Task | None = None
        self._shutdown_callbacks: list[Callable] = []
        self._signal_handlers_setup = False

        # Load existing tasks
        self._load_tasks()

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        if self._signal_handlers_setup:
            return

        def handle_interrupt(signum, frame):
            """Handle Ctrl+C."""
            logger.info("Received interrupt signal, saving state...")
            self._handle_interruption(InterruptionType.USER_CANCEL)
            # Re-raise to allow normal exit
            raise KeyboardInterrupt()

        def handle_terminate(signum, frame):
            """Handle SIGTERM."""
            logger.info("Received terminate signal, saving state...")
            self._handle_interruption(InterruptionType.SYSTEM_SHUTDOWN)

        # Register signal handlers
        signal.signal(signal.SIGINT, handle_interrupt)
        signal.signal(signal.SIGTERM, handle_terminate)

        # Register atexit handler
        atexit.register(self._on_exit)

        self._signal_handlers_setup = True
        logger.debug("Signal handlers registered")

    def _on_exit(self):
        """Called on normal exit."""
        self._save_all_tasks()

    def _handle_interruption(self, interruption_type: InterruptionType):
        """Handle task interruption."""
        if self._current_task and self._current_task in self._tasks:
            task = self._tasks[self._current_task]
            task.state = TaskState.INTERRUPTED
            task.interruption_type = interruption_type
            task.updated_at = time.time()
            self._save_task(task)
            logger.info(f"Task interrupted: {task.name} ({interruption_type.value})")

        # Call shutdown callbacks
        for callback in self._shutdown_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Shutdown callback error: {e}")

    def on_shutdown(self, callback: Callable):
        """Register a shutdown callback."""
        self._shutdown_callbacks.append(callback)

    def start_task(
        self,
        name: str,
        description: str = "",
        context: dict | None = None,
        conversation_id: str | None = None,
        checkpoint_id: str | None = None,
        total_steps: int = 0,
    ) -> str:
        """
        Start a new recoverable task.

        Args:
            name: Task name
            description: Task description
            context: Task context data
            conversation_id: Associated conversation ID
            checkpoint_id: Associated checkpoint ID
            total_steps: Total number of steps (0 if unknown)

        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())

        task = RecoverableTask(
            id=task_id,
            name=name,
            description=description,
            state=TaskState.RUNNING,
            progress=TaskProgress(total_steps=total_steps),
            context=context or {},
            created_at=time.time(),
            updated_at=time.time(),
            conversation_id=conversation_id,
            checkpoint_id=checkpoint_id,
        )

        self._tasks[task_id] = task
        self._current_task = task_id
        self._save_task(task)

        logger.info(f"Started recoverable task: {name} ({task_id})")
        return task_id

    def update_progress(
        self,
        task_id: str,
        step: int | None = None,
        action: str | None = None,
        completed_action: str | None = None,
        pending_actions: list[str] | None = None,
        metadata: dict | None = None,
    ):
        """
        Update task progress.

        Args:
            task_id: Task ID
            step: Current step number
            action: Current action description
            completed_action: Action just completed
            pending_actions: List of pending actions
            metadata: Additional metadata
        """
        if task_id not in self._tasks:
            return

        task = self._tasks[task_id]
        progress = task.progress

        if step is not None:
            progress.current_step = step
        if action is not None:
            progress.current_action = action
        if completed_action:
            progress.completed_actions.append(completed_action)
        if pending_actions is not None:
            progress.pending_actions = pending_actions
        if metadata:
            progress.metadata.update(metadata)

        task.updated_at = time.time()
        self._save_task(task)

    def update_context(self, task_id: str, context: dict):
        """Update task context."""
        if task_id not in self._tasks:
            return

        task = self._tasks[task_id]
        task.context.update(context)
        task.updated_at = time.time()
        self._save_task(task)

    def complete_task(self, task_id: str):
        """Mark a task as completed."""
        if task_id not in self._tasks:
            return

        task = self._tasks[task_id]
        task.state = TaskState.COMPLETED
        task.updated_at = time.time()
        self._save_task(task)

        if self._current_task == task_id:
            self._current_task = None

        logger.info(f"Task completed: {task.name}")

    def fail_task(self, task_id: str, error_message: str):
        """Mark a task as failed."""
        if task_id not in self._tasks:
            return

        task = self._tasks[task_id]
        task.state = TaskState.FAILED
        task.error_message = error_message
        task.interruption_type = InterruptionType.ERROR
        task.updated_at = time.time()
        self._save_task(task)

        if self._current_task == task_id:
            self._current_task = None

        logger.error(f"Task failed: {task.name} - {error_message}")

    def pause_task(self, task_id: str):
        """Pause a task."""
        if task_id not in self._tasks:
            return

        task = self._tasks[task_id]
        task.state = TaskState.PAUSED
        task.updated_at = time.time()
        self._save_task(task)

        logger.info(f"Task paused: {task.name}")

    def resume_task(self, task_id: str) -> RecoverableTask | None:
        """
        Resume an interrupted task.

        Args:
            task_id: Task ID

        Returns:
            Task if found and resumable
        """
        if task_id not in self._tasks:
            # Try to load from storage
            self._load_task(task_id)

        if task_id not in self._tasks:
            logger.error(f"Task not found: {task_id}")
            return None

        task = self._tasks[task_id]

        if task.state not in (TaskState.INTERRUPTED, TaskState.PAUSED):
            logger.warning(f"Task not resumable: {task.name} ({task.state.value})")
            return None

        task.state = TaskState.RUNNING
        task.updated_at = time.time()
        self._current_task = task_id
        self._save_task(task)

        logger.info(f"Task resumed: {task.name}")
        return task

    def cancel_task(self, task_id: str):
        """Cancel a task."""
        if task_id not in self._tasks:
            return

        task = self._tasks[task_id]
        task.state = TaskState.CANCELLED
        task.updated_at = time.time()
        self._save_task(task)

        if self._current_task == task_id:
            self._current_task = None

        logger.info(f"Task cancelled: {task.name}")

    def get_task(self, task_id: str) -> RecoverableTask | None:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def get_current_task(self) -> RecoverableTask | None:
        """Get the current running task."""
        if self._current_task:
            return self._tasks.get(self._current_task)
        return None

    def get_interrupted_tasks(self) -> list[RecoverableTask]:
        """Get all interrupted tasks."""
        return [
            t for t in self._tasks.values()
            if t.state in (TaskState.INTERRUPTED, TaskState.PAUSED)
        ]

    def get_recent_tasks(self, limit: int = 10) -> list[RecoverableTask]:
        """Get recent tasks."""
        tasks = sorted(
            self._tasks.values(),
            key=lambda t: t.updated_at,
            reverse=True,
        )
        return tasks[:limit]

    def delete_task(self, task_id: str):
        """Delete a task."""
        if task_id in self._tasks:
            del self._tasks[task_id]

        # Delete from storage
        task_file = self._storage_dir / f"{task_id}.json"
        task_file.unlink(missing_ok=True)

    def cleanup_old_tasks(self, max_age_days: int = 7):
        """Clean up old completed/cancelled tasks."""
        cutoff = time.time() - (max_age_days * 24 * 3600)

        to_delete = []
        for task_id, task in self._tasks.items():
            if task.state in (TaskState.COMPLETED, TaskState.CANCELLED, TaskState.FAILED):
                if task.updated_at < cutoff:
                    to_delete.append(task_id)

        for task_id in to_delete:
            self.delete_task(task_id)

        if to_delete:
            logger.info(f"Cleaned up {len(to_delete)} old tasks")

    def _save_task(self, task: RecoverableTask):
        """Save a task to storage."""
        try:
            task_file = self._storage_dir / f"{task.id}.json"
            task_file.write_text(
                json.dumps(task.to_dict(), indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error(f"Failed to save task: {e}")

    def _save_all_tasks(self):
        """Save all tasks to storage."""
        for task in self._tasks.values():
            self._save_task(task)

    def _load_task(self, task_id: str) -> bool:
        """Load a single task from storage."""
        task_file = self._storage_dir / f"{task_id}.json"
        if not task_file.exists():
            return False

        try:
            data = json.loads(task_file.read_text(encoding="utf-8"))
            task = RecoverableTask.from_dict(data)
            self._tasks[task_id] = task
            return True
        except Exception as e:
            logger.error(f"Failed to load task {task_id}: {e}")
            return False

    def _load_tasks(self):
        """Load all tasks from storage."""
        for task_file in self._storage_dir.glob("*.json"):
            try:
                data = json.loads(task_file.read_text(encoding="utf-8"))
                task = RecoverableTask.from_dict(data)
                self._tasks[task.id] = task
            except Exception as e:
                logger.warning(f"Failed to load task file {task_file}: {e}")

        # Find any that were running when we crashed
        for task in self._tasks.values():
            if task.state == TaskState.RUNNING:
                task.state = TaskState.INTERRUPTED
                task.interruption_type = InterruptionType.SYSTEM_SHUTDOWN
                self._save_task(task)

        interrupted = self.get_interrupted_tasks()
        if interrupted:
            logger.info(f"Found {len(interrupted)} interrupted tasks")

    def get_recovery_prompt(self, task: RecoverableTask) -> str:
        """
        Generate a prompt to help resume a task.

        Args:
            task: The interrupted task

        Returns:
            Recovery prompt string
        """
        progress = task.progress

        prompt_parts = [
            f"## Task Recovery: {task.name}",
            "",
            f"**Description:** {task.description}",
            f"**Status:** Interrupted ({task.interruption_type.value if task.interruption_type else 'unknown'})",
            "",
        ]

        if progress.completed_actions:
            prompt_parts.append("### Completed Actions:")
            for action in progress.completed_actions:
                prompt_parts.append(f"- ✅ {action}")
            prompt_parts.append("")

        if progress.current_action:
            prompt_parts.append("### Was working on:")
            prompt_parts.append(f"- ⏸️ {progress.current_action}")
            prompt_parts.append("")

        if progress.pending_actions:
            prompt_parts.append("### Remaining Actions:")
            for action in progress.pending_actions:
                prompt_parts.append(f"- ⏳ {action}")
            prompt_parts.append("")

        if task.context:
            prompt_parts.append("### Context:")
            for key, value in task.context.items():
                if isinstance(value, list):
                    prompt_parts.append(f"- {key}: {len(value)} items")
                else:
                    prompt_parts.append(f"- {key}: {value}")

        prompt_parts.append("")
        prompt_parts.append("Please continue from where we left off.")

        return "\n".join(prompt_parts)

    def get_summary(self) -> dict[str, Any]:
        """Get recovery manager summary."""
        by_state = {}
        for task in self._tasks.values():
            state = task.state.value
            by_state[state] = by_state.get(state, 0) + 1

        return {
            "total_tasks": len(self._tasks),
            "by_state": by_state,
            "current_task": self._current_task,
            "interrupted_count": len(self.get_interrupted_tasks()),
        }


# Global instance
_recovery_manager: TaskRecoveryManager | None = None


def get_recovery_manager() -> TaskRecoveryManager:
    """Get the global task recovery manager."""
    global _recovery_manager
    if _recovery_manager is None:
        _recovery_manager = TaskRecoveryManager()
    return _recovery_manager
