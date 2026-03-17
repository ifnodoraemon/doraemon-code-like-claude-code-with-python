"""
Core Task Planning Logic

Provides intelligent task planning and decomposition:
- Automatic task breakdown by goal type
- Plan execution and status tracking
- Dependency-aware task sequencing
"""

import hashlib
import logging
from datetime import datetime
from typing import Any

from .planner_analysis import TaskAnalyzer
from .planner_decompose import TaskDecomposer
from .planner_output import ExecutionPlan, RiskLevel, Task, TaskPriority, TaskStatus

logger = logging.getLogger(__name__)


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

    def __init__(self):
        self._id_counter = 0
        self._analyzer = TaskAnalyzer()
        self._decomposer = TaskDecomposer(self._generate_id)

    def _generate_id(self, prefix: str = "task") -> str:
        """Generate a unique task ID."""
        self._id_counter += 1
        hash_part = hashlib.md5(
            f"{datetime.now().isoformat()}{self._id_counter}".encode()
        ).hexdigest()[:6]
        return f"{prefix}_{hash_part}"

    def generate_plan(
        self, goal: str, context: dict[str, Any] | None = None
    ) -> ExecutionPlan:
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
        self._analyzer.estimate_complexity(goal)

        # Decompose into tasks based on goal type
        tasks = self._decompose_goal(goal, context)

        # Analyze dependencies
        self._analyzer.analyze_dependencies(tasks)

        # Assess risks
        for task in tasks:
            task.risk = self._analyzer.assess_risk(task)

        # Recommend checkpoints
        self._analyzer.recommend_checkpoints(tasks)

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

    def _decompose_goal(self, goal: str, context: dict) -> list[Task]:
        """Decompose a goal into tasks."""
        tasks = []
        goal_lower = goal.lower()

        # Pattern-based decomposition
        if "implement" in goal_lower or "create" in goal_lower or "add" in goal_lower:
            tasks.extend(self._decomposer.create_implementation_tasks(goal, context))
        elif "fix" in goal_lower or "bug" in goal_lower or "debug" in goal_lower:
            tasks.extend(self._decomposer.create_bugfix_tasks(goal, context))
        elif "refactor" in goal_lower or "improve" in goal_lower:
            tasks.extend(self._decomposer.create_refactor_tasks(goal, context))
        elif "test" in goal_lower:
            tasks.extend(self._decomposer.create_testing_tasks(goal, context))
        else:
            complexity = self._analyzer.estimate_complexity(goal)
            time = self._analyzer.estimate_time(complexity)
            tasks.append(self._decomposer.create_generic_task(goal, context, complexity, time))

        return tasks

    def update_task_status(
        self, plan: ExecutionPlan, task_id: str, status: TaskStatus
    ) -> bool:
        """
        Update a task's status.

        Args:
            plan: The execution plan
            task_id: ID of task to update
            status: New status

        Returns:
            True if task was found and updated, False otherwise
        """
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
        """
        Get the next tasks that can be executed (dependencies satisfied).

        Args:
            plan: The execution plan

        Returns:
            List of ready tasks sorted by priority
        """
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
