"""Minimal Ralph-style outer task orchestrator.

The goal is not to replace the interactive agent loop. This module adds a
fresh-context task layer above it:

- persistent backlog stored on disk
- next-task selection
- generated prompt for a fresh inner-agent run
- explicit done/blocked transitions
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.paths import ralph_active_path, ralph_runs_dir, ralph_tasks_path


@dataclass
class RalphTask:
    """A single outer-loop task."""

    id: str
    title: str
    acceptance_criteria: str = ""
    status: str = "pending"
    attempts: int = 0
    notes: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_prompt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "acceptance_criteria": self.acceptance_criteria,
            "status": self.status,
            "attempts": self.attempts,
            "notes": self.notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_prompt": self.last_prompt,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RalphTask:
        return cls(
            id=data["id"],
            title=data["title"],
            acceptance_criteria=data.get("acceptance_criteria", ""),
            status=data.get("status", "pending"),
            attempts=data.get("attempts", 0),
            notes=data.get("notes", []),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            last_prompt=data.get("last_prompt", ""),
        )


class RalphLoopManager:
    """Persistent task orchestrator for fresh-context runs."""

    def __init__(self, project_dir: Path | None = None):
        self.project_dir = (project_dir or Path.cwd()).resolve()
        self.tasks_path = ralph_tasks_path(self.project_dir)
        self.runs_dir = ralph_runs_dir(self.project_dir)
        self.active_path = ralph_active_path(self.project_dir)
        self._tasks: dict[str, RalphTask] = {}
        self._load()

    def init_storage(self) -> Path:
        """Ensure Ralph storage exists on disk."""
        self.tasks_path.parent.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        if not self.tasks_path.exists():
            self._save()
        return self.tasks_path.parent

    def list_tasks(self) -> list[RalphTask]:
        """Return all tasks sorted by recency."""
        return sorted(
            self._tasks.values(),
            key=lambda task: (task.status != "in_progress", task.created_at, task.id),
        )

    def add_task(self, title: str, acceptance_criteria: str = "") -> RalphTask:
        """Create a new Ralph task."""
        self.init_storage()
        task_id = uuid.uuid4().hex[:8]
        task = RalphTask(
            id=task_id,
            title=title.strip(),
            acceptance_criteria=acceptance_criteria.strip(),
        )
        self._tasks[task.id] = task
        self._save()
        return task

    def get_task(self, task_id: str) -> RalphTask | None:
        return self._tasks.get(task_id)

    def choose_next_task(self) -> RalphTask | None:
        """Pick the next unfinished task."""
        candidates = [
            task for task in self._tasks.values() if task.status in {"pending", "in_progress", "blocked"}
        ]
        if not candidates:
            return None

        candidates.sort(
            key=lambda task: (
                task.status == "completed",
                task.status == "blocked",
                task.attempts,
                task.created_at,
            )
        )
        task = candidates[0]
        task.status = "in_progress"
        task.attempts += 1
        task.updated_at = time.time()
        task.last_prompt = self.build_prompt(task)
        self._save()
        self._set_active_task(task.id)
        self._persist_run_record(task)
        return task

    def mark_done(self, task_id: str, note: str = "") -> bool:
        """Mark a task complete."""
        task = self._tasks.get(task_id)
        if task is None:
            return False
        task.status = "completed"
        if note.strip():
            task.notes.append(f"done: {note.strip()}")
        task.updated_at = time.time()
        self._clear_active_task(task_id)
        self._save()
        return True

    def mark_blocked(self, task_id: str, reason: str) -> bool:
        """Mark a task blocked with a reason."""
        task = self._tasks.get(task_id)
        if task is None:
            return False
        task.status = "blocked"
        if reason.strip():
            task.notes.append(f"blocked: {reason.strip()}")
        task.updated_at = time.time()
        self._clear_active_task(task_id)
        self._save()
        return True

    def get_active_task(self) -> RalphTask | None:
        """Return the currently active Ralph task, if any."""
        if not self.active_path.exists():
            return None
        try:
            payload = json.loads(self.active_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        task_id = payload.get("task_id")
        if not isinstance(task_id, str):
            return None
        return self.get_task(task_id)

    def record_progress(self, task_id: str, note: str) -> bool:
        """Append an in-progress note to a task."""
        task = self._tasks.get(task_id)
        if task is None:
            return False
        clean_note = note.strip()
        if clean_note:
            task.notes.append(f"progress: {clean_note}")
            task.updated_at = time.time()
            self._save()
        return True

    def build_prompt(self, task: RalphTask) -> str:
        """Create a fresh-run prompt for the inner coding agent."""
        acceptance = task.acceptance_criteria or "Not specified"
        notes = "\n".join(f"- {note}" for note in task.notes[-3:]) or "- none"
        return (
            f"[Ralph Task {task.id}]\n"
            f"Task: {task.title}\n"
            f"Acceptance criteria: {acceptance}\n"
            f"Previous notes:\n{notes}\n\n"
            "Use a fresh-context inner agent run for this task.\n"
            "Choose the next best action dynamically: inspect, modify, verify, or summarize based on evidence.\n"
            "Do not follow a rigid workflow; gather only enough context to act safely.\n"
            "If you modify files, run relevant checks before declaring the task done.\n"
            "When finished, summarize what changed and whether the acceptance criteria are met."
        )

    def resume_active_prompt(self) -> str | None:
        """Return the active task prompt for a fresh continuation run."""
        task = self.get_active_task()
        if task is None:
            return None
        if not task.last_prompt:
            task.last_prompt = self.build_prompt(task)
            task.updated_at = time.time()
            self._save()
        return task.last_prompt

    def _load(self) -> None:
        if not self.tasks_path.exists():
            return
        try:
            data = json.loads(self.tasks_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(data, list):
            return
        self._tasks = {
            item["id"]: RalphTask.from_dict(item)
            for item in data
            if isinstance(item, dict) and "id" in item
        }

    def _save(self) -> None:
        self.tasks_path.parent.mkdir(parents=True, exist_ok=True)
        self.tasks_path.write_text(
            json.dumps([task.to_dict() for task in self.list_tasks()], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _set_active_task(self, task_id: str) -> None:
        self.active_path.parent.mkdir(parents=True, exist_ok=True)
        self.active_path.write_text(
            json.dumps({"task_id": task_id, "updated_at": time.time()}, indent=2),
            encoding="utf-8",
        )

    def _clear_active_task(self, task_id: str) -> None:
        active = self.get_active_task()
        if active and active.id == task_id and self.active_path.exists():
            self.active_path.unlink()

    def _persist_run_record(self, task: RalphTask) -> None:
        """Write a simple run record for observability."""
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        run_path = self.runs_dir / f"{int(time.time())}_{task.id}.json"
        payload = {
            "task_id": task.id,
            "title": task.title,
            "attempt": task.attempts,
            "prompt": task.last_prompt,
            "created_at": time.time(),
        }
        run_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
