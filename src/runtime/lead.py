"""Thin orchestration facade over AgentSession and the persistent task graph."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.core.planner import TaskPlanner
from src.core.tasks import TaskManager, TaskStatus

if TYPE_CHECKING:
    from src.agent.adapter import AgentSession, AgentTurnResult
    from src.core.planner import ExecutionPlan


@dataclass(slots=True)
class LeadExecutionResult:
    """Result from a lead-runtime execution."""

    root_task_id: str
    plan_id: str
    executed_task_ids: list[str] = field(default_factory=list)
    completed_task_ids: list[str] = field(default_factory=list)
    failed_task_ids: list[str] = field(default_factory=list)
    blocked_task_id: str | None = None
    success: bool = True
    summary: str = ""
    task_summaries: dict[str, str] = field(default_factory=dict)
    worker_assignments: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_task_id": self.root_task_id,
            "plan_id": self.plan_id,
            "executed_task_ids": list(self.executed_task_ids),
            "completed_task_ids": list(self.completed_task_ids),
            "failed_task_ids": list(self.failed_task_ids),
            "blocked_task_id": self.blocked_task_id,
            "success": self.success,
            "summary": self.summary,
            "task_summaries": dict(self.task_summaries),
            "worker_assignments": dict(self.worker_assignments),
        }


@dataclass(slots=True)
class _TaskExecutionOutcome:
    planner_task_id: str
    persistent_task_id: str
    worker_session_id: str
    success: bool
    summary: str
    error: str | None = None


@dataclass(slots=True)
class WorkerProfile:
    role: str
    allowed_tool_names: list[str]
    instruction: str


class LeadAgentRuntime:
    """Dependency-aware orchestration scaffold for lead/worker execution."""

    def __init__(
        self,
        session: "AgentSession",
        planner: TaskPlanner | None = None,
        *,
        max_workers: int = 2,
    ):
        self.session = session
        self.planner = planner or TaskPlanner()
        self.max_workers = max(1, max_workers)

    async def execute(
        self,
        goal: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> LeadExecutionResult:
        """Plan a goal into persistent tasks and execute ready tasks in parallel batches."""
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

        pending_planner_ids = {task.id for task in plan.tasks}
        while pending_planner_ids:
            ready_batch = self._collect_ready_batch(
                task_manager,
                root_task.id,
                plan,
                persistent_ids,
                pending_planner_ids,
            )
            if not ready_batch:
                task_manager.update_task(
                    root_task.id,
                    status=TaskStatus.BLOCKED,
                    assigned_agent=None,
                )
                result.success = False
                result.summary = "Task graph stalled before all planned tasks became ready"
                return result

            outcomes = await asyncio.gather(
                *[
                    self._execute_planner_task(
                        planner_task=planner_task,
                        persistent_task_id=persistent_ids[planner_task.id],
                        task_manager=task_manager,
                        result=result,
                    )
                    for planner_task in ready_batch
                ]
            )

            for outcome in outcomes:
                pending_planner_ids.discard(outcome.planner_task_id)
                result.executed_task_ids.append(outcome.persistent_task_id)
                result.task_summaries[outcome.persistent_task_id] = outcome.summary

                if outcome.success:
                    result.completed_task_ids.append(outcome.persistent_task_id)
                    continue

                result.failed_task_ids.append(outcome.persistent_task_id)
                if result.blocked_task_id is None:
                    result.blocked_task_id = outcome.persistent_task_id

            if result.failed_task_ids:
                task_manager.update_task(
                    root_task.id,
                    status=TaskStatus.BLOCKED,
                    assigned_agent=None,
                )
                result.success = False
                result.summary = self._build_failure_summary(result)
                return result

        task_manager.update_task(
            root_task.id,
            status=TaskStatus.COMPLETED,
            assigned_agent=None,
        )
        result.summary = (
            f"Completed {len(result.completed_task_ids)} planned task(s) "
            f"with up to {self.max_workers} worker(s)"
        )
        return result

    async def _execute_planner_task(
        self,
        *,
        planner_task: Any,
        persistent_task_id: str,
        task_manager: TaskManager,
        result: LeadExecutionResult,
    ) -> _TaskExecutionOutcome:
        """Claim and execute a planner task through an isolated worker session."""
        worker_profile = self._select_worker_profile(planner_task)
        worker_session = await self.session.spawn_worker_session(
            enable_trace=False,
            worker_role=worker_profile.role,
            allowed_tool_names=worker_profile.allowed_tool_names,
        )
        worker_session_id = worker_session.session_id
        result.worker_assignments[persistent_task_id] = {
            "planner_task_id": planner_task.id,
            "role": worker_profile.role,
            "worker_session_id": worker_session_id,
            "allowed_tool_names": list(worker_profile.allowed_tool_names),
        }
        task_manager.claim_task(persistent_task_id, worker_session_id)

        try:
            try:
                turn_result = await worker_session.turn(
                    self._build_worker_input(planner_task, worker_profile),
                    create_runtime_task=False,
                )
            except Exception as exc:
                task_manager.update_task(
                    persistent_task_id,
                    status=TaskStatus.BLOCKED,
                    assigned_agent=None,
                )
                return _TaskExecutionOutcome(
                    planner_task_id=planner_task.id,
                    persistent_task_id=persistent_task_id,
                    worker_session_id=worker_session_id,
                    success=False,
                    summary=str(exc),
                    error=str(exc),
                )
        finally:
            await worker_session.aclose()

        if turn_result.success:
            task_manager.update_task(
                persistent_task_id,
                status=TaskStatus.COMPLETED,
                assigned_agent=None,
            )
            return _TaskExecutionOutcome(
                planner_task_id=planner_task.id,
                persistent_task_id=persistent_task_id,
                worker_session_id=worker_session_id,
                success=True,
                summary=turn_result.response or "Task completed",
            )

        task_manager.update_task(
            persistent_task_id,
            status=TaskStatus.BLOCKED,
            assigned_agent=None,
        )
        return _TaskExecutionOutcome(
            planner_task_id=planner_task.id,
            persistent_task_id=persistent_task_id,
            worker_session_id=worker_session_id,
            success=False,
            summary=turn_result.error or turn_result.response or "Task execution failed",
            error=turn_result.error,
        )

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

    def _collect_ready_batch(
        self,
        task_manager: TaskManager,
        root_task_id: str,
        plan: "ExecutionPlan",
        persistent_ids: dict[str, str],
        pending_planner_ids: set[str],
    ) -> list[Any]:
        """Collect the next dependency-satisfied batch for parallel execution."""
        planner_task_by_id = {task.id: task for task in plan.tasks}
        ready_tasks: list[Any] = []

        for persistent_task in task_manager.list_ready_tasks():
            if persistent_task.parent_id != root_task_id:
                continue

            planner_task_id = next(
                (
                    planner_id
                    for planner_id, mapped_task_id in persistent_ids.items()
                    if mapped_task_id == persistent_task.id
                ),
                None,
            )
            if planner_task_id is None or planner_task_id not in pending_planner_ids:
                continue

            ready_tasks.append(planner_task_by_id[planner_task_id])
            if len(ready_tasks) >= self.max_workers:
                break

        return ready_tasks

    def _build_failure_summary(self, result: LeadExecutionResult) -> str:
        """Build a compact failure summary from failed task results."""
        if result.blocked_task_id:
            detail = result.task_summaries.get(result.blocked_task_id, "Task execution failed")
        else:
            detail = "Task execution failed"
        return (
            f"Blocked after {len(result.completed_task_ids)} completed task(s); "
            f"{detail}"
        )

    def _select_worker_profile(self, planner_task: Any) -> WorkerProfile:
        """Choose a worker role and tool scope for a planner task."""
        text = f"{planner_task.title} {planner_task.description}".lower()
        available = set(getattr(self.session.registry, "get_tool_names", lambda: [])())

        role_tools = {
            "researcher": [
                "read",
                "search",
                "web_search",
                "web_fetch",
                "lsp_hover",
                "lsp_definition",
                "lsp_references",
                "memory_get",
                "memory_search",
                "memory_list",
                "task",
            ],
            "verifier": [
                "read",
                "search",
                "run",
                "lsp_diagnostics",
                "lsp_hover",
                "lsp_definition",
                "memory_get",
                "memory_search",
                "memory_list",
                "task",
            ],
            "implementer": [
                "read",
                "search",
                "write",
                "multi_edit",
                "notebook_edit",
                "run",
                "lsp_diagnostics",
                "lsp_completions",
                "lsp_hover",
                "lsp_definition",
                "lsp_references",
                "lsp_rename",
                "memory_get",
                "memory_put",
                "memory_search",
                "memory_list",
                "task",
            ],
        }
        role_instructions = {
            "researcher": "Investigate the codebase, gather facts, and avoid speculative edits.",
            "verifier": "Validate behavior, run checks, and report concrete pass/fail evidence.",
            "implementer": "Make the required code changes and verify the subtask before returning.",
        }

        if any(keyword in text for keyword in ("verify", "validation", "test", "check", "diagnostic", "integration")):
            role = "verifier"
        elif any(keyword in text for keyword in ("analyze", "inspect", "research", "explore", "read", "investigate", "design")):
            role = "researcher"
        else:
            role = "implementer"

        allowed_tool_names = [name for name in role_tools[role] if name in available]
        if "task" not in allowed_tool_names and "task" in available:
            allowed_tool_names.append("task")

        return WorkerProfile(
            role=role,
            allowed_tool_names=allowed_tool_names,
            instruction=role_instructions[role],
        )

    def _build_worker_input(self, planner_task: Any, worker_profile: WorkerProfile) -> str:
        """Wrap a planner task in role-specific worker instructions."""
        return (
            f"Worker role: {worker_profile.role}\n"
            f"Role instruction: {worker_profile.instruction}\n"
            f"Subtask: {planner_task.description}\n"
            "Return a concise subtask result for the lead agent."
        )

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
