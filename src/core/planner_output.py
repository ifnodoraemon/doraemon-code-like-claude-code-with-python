"""
Task Planning Output and Data Models

Provides data structures for task planning:
- Task status, priority, and risk enums
- Task and ExecutionPlan data classes
- Serialization (to_dict, from_dict, to_markdown)
- Risk assessment and dependency tracking
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"


class TaskPriority(Enum):
    """Task priority level."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RiskLevel(Enum):
    """Risk level for a task."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class TaskDependency:
    """A dependency between tasks."""

    task_id: str
    dependency_type: str  # "requires", "blocked_by", "related_to"
    reason: str | None = None


@dataclass
class RiskAssessment:
    """Risk assessment for a task."""

    level: RiskLevel
    factors: list[str] = field(default_factory=list)
    mitigations: list[str] = field(default_factory=list)


@dataclass
class Task:
    """A decomposed task with metadata."""

    id: str
    title: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    complexity: int = 1  # 1-5 scale
    estimated_minutes: int = 0
    dependencies: list[TaskDependency] = field(default_factory=list)
    subtasks: list["Task"] = field(default_factory=list)
    risk: RiskAssessment | None = None
    files_affected: list[str] = field(default_factory=list)
    checkpoint_recommended: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "complexity": self.complexity,
            "estimated_minutes": self.estimated_minutes,
            "dependencies": [
                {"task_id": d.task_id, "type": d.dependency_type, "reason": d.reason}
                for d in self.dependencies
            ],
            "subtasks": [s.to_dict() for s in self.subtasks],
            "risk": {
                "level": self.risk.level.value,
                "factors": self.risk.factors,
                "mitigations": self.risk.mitigations,
            }
            if self.risk
            else None,
            "files_affected": self.files_affected,
            "checkpoint_recommended": self.checkpoint_recommended,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        """Create from dictionary."""
        task = cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            status=TaskStatus(data.get("status", "pending")),
            priority=TaskPriority(data.get("priority", "medium")),
            complexity=data.get("complexity", 1),
            estimated_minutes=data.get("estimated_minutes", 0),
            files_affected=data.get("files_affected", []),
            checkpoint_recommended=data.get("checkpoint_recommended", False),
        )

        # Parse dependencies
        for dep in data.get("dependencies", []):
            task.dependencies.append(
                TaskDependency(
                    task_id=dep["task_id"],
                    dependency_type=dep["type"],
                    reason=dep.get("reason"),
                )
            )

        # Parse subtasks recursively
        for subtask in data.get("subtasks", []):
            task.subtasks.append(Task.from_dict(subtask))

        # Parse risk
        if data.get("risk"):
            task.risk = RiskAssessment(
                level=RiskLevel(data["risk"]["level"]),
                factors=data["risk"].get("factors", []),
                mitigations=data["risk"].get("mitigations", []),
            )

        return task


@dataclass
class ExecutionPlan:
    """A complete execution plan for a task."""

    id: str
    goal: str
    tasks: list[Task]
    total_estimated_minutes: int
    total_complexity: int
    high_risk_count: int
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "tasks": [t.to_dict() for t in self.tasks],
            "total_estimated_minutes": self.total_estimated_minutes,
            "total_complexity": self.total_complexity,
            "high_risk_count": self.high_risk_count,
            "created_at": self.created_at.isoformat(),
        }

    def to_markdown(self) -> str:
        """Convert plan to markdown format."""
        lines = [
            "# Execution Plan",
            "",
            f"**Goal:** {self.goal}",
            "",
            "**Estimates:**",
            f"- Total Time: ~{self.total_estimated_minutes} minutes",
            f"- Overall Complexity: {self.total_complexity}/5",
            f"- High Risk Tasks: {self.high_risk_count}",
            "",
            "---",
            "",
            "## Tasks",
            "",
        ]

        for i, task in enumerate(self.tasks, 1):
            status_icon = {
                TaskStatus.PENDING: "⬜",
                TaskStatus.IN_PROGRESS: "🔄",
                TaskStatus.COMPLETED: "✅",
                TaskStatus.BLOCKED: "🚫",
                TaskStatus.FAILED: "❌",
            }.get(task.status, "⬜")

            priority_badge = {
                TaskPriority.CRITICAL: "🔴",
                TaskPriority.HIGH: "🟠",
                TaskPriority.MEDIUM: "🟡",
                TaskPriority.LOW: "🟢",
            }.get(task.priority, "")

            lines.append(f"### {i}. {status_icon} {task.title} {priority_badge}")
            lines.append("")
            lines.append(f"{task.description}")
            lines.append("")
            lines.append(f"- **Complexity:** {task.complexity}/5")
            lines.append(f"- **Est. Time:** {task.estimated_minutes} min")

            if task.dependencies:
                deps = ", ".join(d.task_id for d in task.dependencies)
                lines.append(f"- **Depends on:** {deps}")

            if task.files_affected:
                files = ", ".join(f"`{f}`" for f in task.files_affected[:5])
                lines.append(f"- **Files:** {files}")

            if task.risk and task.risk.level in [RiskLevel.MEDIUM, RiskLevel.HIGH]:
                lines.append(f"- **Risk:** {task.risk.level.value.upper()}")
                for factor in task.risk.factors[:3]:
                    lines.append(f"  - {factor}")

            if task.checkpoint_recommended:
                lines.append("- **Checkpoint recommended**")

            # Subtasks
            if task.subtasks:
                lines.append("")
                lines.append("**Subtasks:**")
                for st in task.subtasks:
                    st_icon = "✅" if st.status == TaskStatus.COMPLETED else "⬜"
                    lines.append(f"  - {st_icon} {st.title}")

            lines.append("")

        return "\n".join(lines)
