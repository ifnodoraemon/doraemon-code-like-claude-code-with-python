"""Thin orchestration facade over AgentSession and the persistent task graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.core.planner import TaskPlanner
from src.core.tasks import TaskManager, TaskStatus

if TYPE_CHECKING:
    from src.agent.adapter import AgentSession, AgentTurnResult
    from src.core.planner import ExecutionPlan


@dataclass(slots=True)
class LeadExecutionResult:
    """Result from a serial lead-runtime execution."""

    root_task_id: str
    plan_id: str
    executed_task_ids: list[str] = field(default_factory=list)
    completed_task_ids: list[str] = field(default_factory=list)
    blocked_task_id: str | None = None
    success: bool = True
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_task_id": self.root_task_id,
            "plan_id": self.plan_id,
            "executed_task_ids": list(self.executed_task_ids),
            "completed_task_ids": list(self.completed_task_ids),
            "blocked_task_id": self.blocked_task_id,
            "success": self.success,
            "summary": self.summary,
        }


class LeadAgentRuntime:
    """Serial orchestration scaffold for future lead/worker execution."""

    def __init__(
        self,
        session: "AgentSession",
        planner: TaskPlanner | None = None,
    ):
        self.session = session
        self.planner = planner or TaskPlanner()

    async def execute(
        self,
        goal: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> LeadExecutionResult:
        """Plan a goal into persistent tasks and execute ready tasks serially."""
        if not self.session._agent:
            await self.session.initialize()

        task_manager = self.session.get_task_manager()
        if task_manager is None:
            raise RuntimeError("Task manager is not available for orchestration")

        root_task = task_manager.create_task(
            title=self._trim_title(goal),
            description=goal,
            priority=100,
        )
        task_manager.update_task(
            root_task.id,
            status=TaskStatus.IN_PROGRESS,
            assigned_agent=self.session.session_id,
        )

        plan = self.planner.generate_plan(goal, context=context or {})
        persistent_ids = self._materialize_plan(task_manager, root_task.id, plan)

        result = LeadExecutionResult(
            root_task_id=root_task.id,
            plan_id=plan.id,
        )

        for planner_task in plan.tasks:
            persistent_task_id = persistent_ids[planner_task.id]
            task_manager.claim_task(persistent_task_id, self.session.session_id)
            turn_result = await self.session.turn(
                planner_task.description,
                create_runtime_task=False,
            )
            result.executed_task_ids.append(persistent_task_id)

            if turn_result.success:
                task_manager.update_task(
                    persistent_task_id,
                    status=TaskStatus.COMPLETED,
                    assigned_agent=None,
                )
                result.completed_task_ids.append(persistent_task_id)
                continue

            task_manager.update_task(
                persistent_task_id,
                status=TaskStatus.BLOCKED,
                assigned_agent=None,
            )
            task_manager.update_task(
                root_task.id,
                status=TaskStatus.BLOCKED,
                assigned_agent=None,
            )
            result.success = False
            result.blocked_task_id = persistent_task_id
            result.summary = turn_result.error or turn_result.response or "Task execution failed"
            return result

        task_manager.update_task(
            root_task.id,
            status=TaskStatus.COMPLETED,
            assigned_agent=None,
        )
        result.summary = f"Completed {len(result.completed_task_ids)} planned task(s)"
        return result

    def _materialize_plan(
        self,
        task_manager: TaskManager,
        root_task_id: str,
        plan: "ExecutionPlan",
    ) -> dict[str, str]:
        """Persist planner tasks into the shared task graph."""
        persistent_ids: dict[str, str] = {}

        for planner_task in plan.tasks:
            created = task_manager.create_task(
                title=planner_task.title,
                description=planner_task.description,
                parent_id=root_task_id,
                priority=self._priority_value(planner_task.priority.value),
            )
            persistent_ids[planner_task.id] = created.id

        for planner_task in plan.tasks:
            dependency_ids = [
                persistent_ids[dependency.task_id]
                for dependency in planner_task.dependencies
                if dependency.task_id in persistent_ids and dependency.dependency_type == "requires"
            ]
            if dependency_ids:
                task_manager.update_task(
                    persistent_ids[planner_task.id],
                    dependencies=dependency_ids,
                )

        return persistent_ids

    def _trim_title(self, goal: str) -> str:
        compact = " ".join(goal.strip().split()) or "Orchestrated task"
        if len(compact) <= 80:
            return compact
        return f"{compact[:77]}..."

    def _priority_value(self, value: str) -> int:
        mapping = {
            "critical": 100,
            "high": 75,
            "medium": 50,
            "low": 25,
        }
        return mapping.get(value, 0)
