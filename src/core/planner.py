"""
Intelligent Task Planner

Provides AI-powered task planning and decomposition:
- Automatic task breakdown
- Dependency analysis
- Complexity estimation
- Risk assessment
- Checkpoint suggestions

This enables smarter planning mode in Doraemon.
"""

import hashlib
import json
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
            } if self.risk else None,
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
            task.dependencies.append(TaskDependency(
                task_id=dep["task_id"],
                dependency_type=dep["type"],
                reason=dep.get("reason"),
            ))

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
            f"# Execution Plan",
            f"",
            f"**Goal:** {self.goal}",
            f"",
            f"**Estimates:**",
            f"- Total Time: ~{self.total_estimated_minutes} minutes",
            f"- Overall Complexity: {self.total_complexity}/5",
            f"- High Risk Tasks: {self.high_risk_count}",
            f"",
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
            lines.append(f"")
            lines.append(f"{task.description}")
            lines.append(f"")
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
                    lines.append(f"  - ⚠️ {factor}")

            if task.checkpoint_recommended:
                lines.append(f"- **💾 Checkpoint recommended**")

            # Subtasks
            if task.subtasks:
                lines.append(f"")
                lines.append(f"**Subtasks:**")
                for st in task.subtasks:
                    st_icon = "✅" if st.status == TaskStatus.COMPLETED else "⬜"
                    lines.append(f"  - {st_icon} {st.title}")

            lines.append("")

        return "\n".join(lines)


class TaskPlanner:
    """
    Intelligent task planner for decomposing and analyzing tasks.

    Usage:
        planner = TaskPlanner()

        # Analyze and decompose a task
        plan = planner.generate_plan("Implement user authentication")

        # Get markdown representation
        print(plan.to_markdown())
    """

    # Complexity keywords for estimation
    COMPLEXITY_KEYWORDS = {
        5: ["refactor", "redesign", "migration", "architecture", "security"],
        4: ["integrate", "optimize", "test suite", "api", "database"],
        3: ["implement", "feature", "endpoint", "component"],
        2: ["update", "modify", "fix", "bug"],
        1: ["typo", "comment", "rename", "config", "doc"],
    }

    # Risk keywords
    RISK_KEYWORDS = {
        "high": ["delete", "drop", "remove", "migration", "production", "credentials", "security"],
        "medium": ["refactor", "update", "database", "api", "config"],
        "low": ["add", "create", "doc", "test", "comment"],
    }

    def __init__(self):
        self._id_counter = 0

    def _generate_id(self, prefix: str = "task") -> str:
        """Generate a unique task ID."""
        self._id_counter += 1
        hash_part = hashlib.md5(f"{datetime.now().isoformat()}{self._id_counter}".encode()).hexdigest()[:6]
        return f"{prefix}_{hash_part}"

    def generate_plan(self, goal: str, context: dict[str, Any] | None = None) -> ExecutionPlan:
        """
        Generate an execution plan for a goal.

        Args:
            goal: The high-level goal description
            context: Optional context (files, codebase info, etc.)

        Returns:
            An ExecutionPlan with decomposed tasks
        """
        context = context or {}

        # Analyze goal complexity
        goal_complexity = self._estimate_complexity(goal)

        # Decompose into tasks based on goal type
        tasks = self._decompose_goal(goal, context)

        # Analyze dependencies
        self._analyze_dependencies(tasks)

        # Assess risks
        for task in tasks:
            task.risk = self._assess_risk(task)

        # Recommend checkpoints
        self._recommend_checkpoints(tasks)

        # Calculate totals
        total_time = sum(t.estimated_minutes for t in tasks)
        total_complexity = min(5, sum(t.complexity for t in tasks) // max(1, len(tasks)))
        high_risk = sum(1 for t in tasks if t.risk and t.risk.level == RiskLevel.HIGH)

        return ExecutionPlan(
            id=self._generate_id("plan"),
            goal=goal,
            tasks=tasks,
            total_estimated_minutes=total_time,
            total_complexity=total_complexity,
            high_risk_count=high_risk,
        )

    def _estimate_complexity(self, text: str) -> int:
        """Estimate complexity from text."""
        text_lower = text.lower()
        for complexity, keywords in self.COMPLEXITY_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return complexity
        return 2  # Default medium

    def _estimate_time(self, complexity: int) -> int:
        """Estimate time in minutes based on complexity."""
        time_map = {1: 5, 2: 15, 3: 30, 4: 60, 5: 120}
        return time_map.get(complexity, 30)

    def _decompose_goal(self, goal: str, context: dict) -> list[Task]:
        """Decompose a goal into tasks."""
        tasks = []
        goal_lower = goal.lower()

        # Pattern-based decomposition
        if "implement" in goal_lower or "create" in goal_lower or "add" in goal_lower:
            # Implementation task pattern
            tasks.extend(self._create_implementation_tasks(goal, context))
        elif "fix" in goal_lower or "bug" in goal_lower or "debug" in goal_lower:
            # Bug fix pattern
            tasks.extend(self._create_bugfix_tasks(goal, context))
        elif "refactor" in goal_lower or "improve" in goal_lower:
            # Refactoring pattern
            tasks.extend(self._create_refactor_tasks(goal, context))
        elif "test" in goal_lower:
            # Testing pattern
            tasks.extend(self._create_testing_tasks(goal, context))
        else:
            # Generic task
            tasks.append(self._create_generic_task(goal, context))

        return tasks

    def _create_implementation_tasks(self, goal: str, context: dict) -> list[Task]:
        """Create tasks for an implementation goal."""
        planning_task = Task(
            id=self._generate_id(),
            title="Analyze Requirements",
            description=f"Understand what needs to be built for: {goal}",
            complexity=2,
            estimated_minutes=15,
            priority=TaskPriority.HIGH,
        )

        design_task = Task(
            id=self._generate_id(),
            title="Design Solution",
            description="Plan the implementation approach and identify components",
            complexity=3,
            estimated_minutes=20,
            priority=TaskPriority.HIGH,
            dependencies=[TaskDependency(planning_task.id, "requires")],
        )

        implement_task = Task(
            id=self._generate_id(),
            title="Implement Solution",
            description=f"Write the code to: {goal}",
            complexity=4,
            estimated_minutes=60,
            priority=TaskPriority.MEDIUM,
            dependencies=[TaskDependency(design_task.id, "requires")],
            checkpoint_recommended=True,
        )

        test_task = Task(
            id=self._generate_id(),
            title="Test Implementation",
            description="Write and run tests to verify the implementation",
            complexity=2,
            estimated_minutes=20,
            priority=TaskPriority.MEDIUM,
            dependencies=[TaskDependency(implement_task.id, "requires")],
        )

        return [planning_task, design_task, implement_task, test_task]

    def _create_bugfix_tasks(self, goal: str, context: dict) -> list[Task]:
        """Create tasks for a bug fix goal."""
        reproduce_task = Task(
            id=self._generate_id(),
            title="Reproduce Issue",
            description="Identify steps to reproduce the bug",
            complexity=2,
            estimated_minutes=15,
            priority=TaskPriority.HIGH,
        )

        investigate_task = Task(
            id=self._generate_id(),
            title="Investigate Root Cause",
            description="Find the source of the bug in the code",
            complexity=3,
            estimated_minutes=30,
            priority=TaskPriority.HIGH,
            dependencies=[TaskDependency(reproduce_task.id, "requires")],
        )

        fix_task = Task(
            id=self._generate_id(),
            title="Implement Fix",
            description=f"Fix: {goal}",
            complexity=3,
            estimated_minutes=30,
            priority=TaskPriority.MEDIUM,
            dependencies=[TaskDependency(investigate_task.id, "requires")],
            checkpoint_recommended=True,
        )

        verify_task = Task(
            id=self._generate_id(),
            title="Verify Fix",
            description="Confirm the fix works and doesn't break other functionality",
            complexity=2,
            estimated_minutes=15,
            priority=TaskPriority.MEDIUM,
            dependencies=[TaskDependency(fix_task.id, "requires")],
        )

        return [reproduce_task, investigate_task, fix_task, verify_task]

    def _create_refactor_tasks(self, goal: str, context: dict) -> list[Task]:
        """Create tasks for a refactoring goal."""
        analyze_task = Task(
            id=self._generate_id(),
            title="Analyze Current Code",
            description="Understand the current implementation and its issues",
            complexity=2,
            estimated_minutes=20,
            priority=TaskPriority.HIGH,
        )

        plan_task = Task(
            id=self._generate_id(),
            title="Plan Refactoring",
            description="Design the target architecture and migration path",
            complexity=4,
            estimated_minutes=30,
            priority=TaskPriority.HIGH,
            dependencies=[TaskDependency(analyze_task.id, "requires")],
        )

        refactor_task = Task(
            id=self._generate_id(),
            title="Refactor Code",
            description=f"Refactor: {goal}",
            complexity=5,
            estimated_minutes=90,
            priority=TaskPriority.MEDIUM,
            dependencies=[TaskDependency(plan_task.id, "requires")],
            checkpoint_recommended=True,
        )

        test_task = Task(
            id=self._generate_id(),
            title="Run Tests",
            description="Ensure all existing tests pass after refactoring",
            complexity=2,
            estimated_minutes=20,
            priority=TaskPriority.MEDIUM,
            dependencies=[TaskDependency(refactor_task.id, "requires")],
        )

        return [analyze_task, plan_task, refactor_task, test_task]

    def _create_testing_tasks(self, goal: str, context: dict) -> list[Task]:
        """Create tasks for a testing goal."""
        analyze_task = Task(
            id=self._generate_id(),
            title="Identify Test Cases",
            description="Determine what needs to be tested",
            complexity=2,
            estimated_minutes=15,
            priority=TaskPriority.HIGH,
        )

        write_task = Task(
            id=self._generate_id(),
            title="Write Tests",
            description=f"Write tests for: {goal}",
            complexity=3,
            estimated_minutes=45,
            priority=TaskPriority.MEDIUM,
            dependencies=[TaskDependency(analyze_task.id, "requires")],
        )

        run_task = Task(
            id=self._generate_id(),
            title="Run and Validate",
            description="Run tests and ensure they pass",
            complexity=1,
            estimated_minutes=10,
            priority=TaskPriority.MEDIUM,
            dependencies=[TaskDependency(write_task.id, "requires")],
        )

        return [analyze_task, write_task, run_task]

    def _create_generic_task(self, goal: str, context: dict) -> Task:
        """Create a generic task."""
        complexity = self._estimate_complexity(goal)
        return Task(
            id=self._generate_id(),
            title=goal[:50] + ("..." if len(goal) > 50 else ""),
            description=goal,
            complexity=complexity,
            estimated_minutes=self._estimate_time(complexity),
        )

    def _analyze_dependencies(self, tasks: list[Task]):
        """Analyze and enrich task dependencies."""
        # Dependencies are set during task creation
        # This method can add cross-task dependencies if needed
        pass

    def _assess_risk(self, task: Task) -> RiskAssessment:
        """Assess risk for a task."""
        text = f"{task.title} {task.description}".lower()

        # Check for high risk keywords
        for level, keywords in self.RISK_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                factors = []
                mitigations = []

                if "delete" in text or "remove" in text:
                    factors.append("Destructive operation")
                    mitigations.append("Create backup before proceeding")

                if "production" in text:
                    factors.append("Affects production")
                    mitigations.append("Test in staging first")

                if "database" in text or "migration" in text:
                    factors.append("Database changes")
                    mitigations.append("Create database backup")

                if "security" in text or "credentials" in text:
                    factors.append("Security-sensitive")
                    mitigations.append("Review security implications")

                return RiskAssessment(
                    level=RiskLevel(level),
                    factors=factors,
                    mitigations=mitigations,
                )

        return RiskAssessment(level=RiskLevel.LOW)

    def _recommend_checkpoints(self, tasks: list[Task]):
        """Recommend checkpoints for tasks."""
        for task in tasks:
            # Recommend checkpoint for high complexity or high risk
            if task.complexity >= 4:
                task.checkpoint_recommended = True
            if task.risk and task.risk.level in [RiskLevel.MEDIUM, RiskLevel.HIGH]:
                task.checkpoint_recommended = True
            # Files affected
            if len(task.files_affected) >= 3:
                task.checkpoint_recommended = True

    def update_task_status(
        self, plan: ExecutionPlan, task_id: str, status: TaskStatus
    ) -> bool:
        """Update a task's status."""
        for task in plan.tasks:
            if task.id == task_id:
                task.status = status
                if status == TaskStatus.COMPLETED:
                    task.completed_at = datetime.now()
                return True
            # Check subtasks
            for subtask in task.subtasks:
                if subtask.id == task_id:
                    subtask.status = status
                    if status == TaskStatus.COMPLETED:
                        subtask.completed_at = datetime.now()
                    return True
        return False

    def get_next_tasks(self, plan: ExecutionPlan) -> list[Task]:
        """Get the next tasks that can be executed (dependencies satisfied)."""
        completed_ids = set()
        for task in plan.tasks:
            if task.status == TaskStatus.COMPLETED:
                completed_ids.add(task.id)

        ready = []
        for task in plan.tasks:
            if task.status != TaskStatus.PENDING:
                continue
            # Check if all dependencies are completed
            deps_satisfied = all(
                d.task_id in completed_ids
                for d in task.dependencies
                if d.dependency_type == "requires"
            )
            if deps_satisfied:
                ready.append(task)

        # Sort by priority
        priority_order = {
            TaskPriority.CRITICAL: 0,
            TaskPriority.HIGH: 1,
            TaskPriority.MEDIUM: 2,
            TaskPriority.LOW: 3,
        }
        ready.sort(key=lambda t: priority_order.get(t.priority, 2))

        return ready
