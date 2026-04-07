import asyncio
from dataclasses import dataclass

import pytest

from src.core.planner import ExecutionPlan, Task, TaskDependency, TaskPriority
from src.core.tasks import TaskManager, TaskStatus
from src.runtime.lead import LeadAgentRuntime


@dataclass
class StubTurnResult:
    success: bool
    response: str = ""
    error: str | None = None


class StubWorkerSession:
    def __init__(self, parent: "StubSession", session_id: str):
        self._parent = parent
        self.session_id = session_id

    async def turn(self, user_input: str, **kwargs):
        return await self._parent.run_worker_turn(
            self.session_id,
            user_input,
            create_runtime_task=kwargs.get("create_runtime_task", True),
        )

    async def aclose(self) -> None:
        self._parent.closed_workers.append(self.session_id)


class StubSession:
    def __init__(
        self,
        task_manager: TaskManager,
        results_by_input: dict[str, StubTurnResult],
        *,
        barrier_inputs: set[str] | None = None,
    ):
        self._agent = object()
        self._task_manager = task_manager
        self.registry = SimpleRegistry()
        self.session_id = "session-test"
        self.results_by_input = results_by_input
        self.barrier_inputs = barrier_inputs or set()
        self.turn_calls: list[tuple[str, str, bool]] = []
        self.started_inputs: list[str] = []
        self.closed_workers: list[str] = []
        self.spawned_workers: list[dict[str, object]] = []
        self._worker_count = 0
        self._barrier_event = asyncio.Event()

    async def initialize(self) -> None:
        self._agent = object()

    def get_task_manager(self) -> TaskManager:
        return self._task_manager

    async def spawn_worker_session(
        self,
        *,
        enable_trace: bool | None = None,
        worker_role: str | None = None,
        allowed_tool_names: list[str] | None = None,
    ):
        self._worker_count += 1
        self.spawned_workers.append(
            {
                "worker_role": worker_role,
                "allowed_tool_names": list(allowed_tool_names or []),
            }
        )
        return StubWorkerSession(self, f"worker-{self._worker_count}")

    async def run_worker_turn(
        self,
        worker_session_id: str,
        user_input: str,
        *,
        create_runtime_task: bool,
    ) -> StubTurnResult:
        self.turn_calls.append((worker_session_id, user_input, create_runtime_task))
        task_line = next(
            line for line in user_input.splitlines() if line.startswith("Subtask: ")
        )
        subtask = task_line.removeprefix("Subtask: ")
        self.started_inputs.append(subtask)

        if subtask in self.barrier_inputs:
            started = sum(1 for item in self.started_inputs if item in self.barrier_inputs)
            if started >= len(self.barrier_inputs):
                self._barrier_event.set()
            await asyncio.wait_for(self._barrier_event.wait(), timeout=1)

        return self.results_by_input[subtask]


class RaisingStubSession(StubSession):
    async def run_worker_turn(
        self,
        worker_session_id: str,
        user_input: str,
        *,
        create_runtime_task: bool,
    ) -> StubTurnResult:
        self.turn_calls.append((worker_session_id, user_input, create_runtime_task))
        raise RuntimeError(f"worker crashed for {user_input}")


class StubPlanner:
    def __init__(self, plan: ExecutionPlan):
        self.plan = plan

    def generate_plan(self, goal: str, context=None) -> ExecutionPlan:
        return self.plan


class RaisingPlanner:
    def generate_plan(self, goal: str, context=None) -> ExecutionPlan:
        raise RuntimeError("planner boom")


class SimpleRegistry:
    def get_tool_names(self) -> list[str]:
        return [
            "read",
            "search",
            "write",
            "run",
            "web_search",
            "web_fetch",
            "task",
            "memory_get",
            "memory_put",
            "memory_search",
            "memory_list",
            "lsp_diagnostics",
            "lsp_hover",
            "lsp_definition",
            "lsp_references",
            "lsp_completions",
            "lsp_rename",
        ]


def build_parallel_plan() -> ExecutionPlan:
    task_a = Task(
        id="task-a",
        title="Inspect auth flow",
        description="Inspect auth flow",
        priority=TaskPriority.HIGH,
    )
    task_b = Task(
        id="task-b",
        title="Implement backend changes",
        description="Implement backend changes",
        priority=TaskPriority.HIGH,
    )
    task_c = Task(
        id="task-c",
        title="Run integration verification",
        description="Run integration verification",
        priority=TaskPriority.MEDIUM,
        dependencies=[
            TaskDependency(task_id="task-a", dependency_type="requires"),
            TaskDependency(task_id="task-b", dependency_type="requires"),
        ],
    )
    return ExecutionPlan(
        id="plan-auth",
        goal="Implement authentication",
        tasks=[task_a, task_b, task_c],
        total_estimated_minutes=25,
        total_complexity=3,
        high_risk_count=0,
    )


@pytest.mark.asyncio
async def test_lead_runtime_executes_ready_tasks_in_parallel_batches(tmp_path):
    task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
    session = StubSession(
        task_manager=task_manager,
        results_by_input={
            "Inspect auth flow": StubTurnResult(success=True, response="inspection done"),
            "Implement backend changes": StubTurnResult(success=True, response="backend done"),
            "Run integration verification": StubTurnResult(success=True, response="verification done"),
        },
        barrier_inputs={"Inspect auth flow", "Implement backend changes"},
    )

    runtime = LeadAgentRuntime(session, planner=StubPlanner(build_parallel_plan()), max_workers=2)
    result = await runtime.execute("Implement authentication")

    root = task_manager.get_task(result.root_task_id)
    child_tasks = [task for task in task_manager.list_tasks() if task.parent_id == result.root_task_id]

    assert result.success is True
    assert root is not None
    assert root.status == TaskStatus.COMPLETED
    assert len(result.completed_task_ids) == 3
    assert result.failed_task_ids == []
    assert set(session.started_inputs[:2]) == {"Inspect auth flow", "Implement backend changes"}
    assert session.started_inputs[2] == "Run integration verification"
    assert all(create_runtime_task is False for _, _, create_runtime_task in session.turn_calls)
    assert len({worker_id for worker_id, _, _ in session.turn_calls}) == 3
    assert set(session.closed_workers) == {worker_id for worker_id, _, _ in session.turn_calls}
    assert all(task.status == TaskStatus.COMPLETED for task in child_tasks)
    assert {assignment["role"] for assignment in result.worker_assignments.values()} == {
        "inspect",
        "change",
        "validate",
    }
    assert any(worker["worker_role"] == "inspect" for worker in session.spawned_workers)
    assert any("write" in worker["allowed_tool_names"] for worker in session.spawned_workers)
    assert any("run" in worker["allowed_tool_names"] for worker in session.spawned_workers)
    assert any(
        assignment["capability_groups"] == ["read", "edit", "memory", "task"]
        for assignment in result.worker_assignments.values()
    )


@pytest.mark.asyncio
async def test_lead_runtime_blocks_root_when_parallel_subtask_fails(tmp_path):
    task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
    session = StubSession(
        task_manager=task_manager,
        results_by_input={
            "Inspect auth flow": StubTurnResult(success=True, response="inspection done"),
            "Implement backend changes": StubTurnResult(success=False, error="backend failed"),
            "Run integration verification": StubTurnResult(success=True, response="verification done"),
        },
        barrier_inputs={"Inspect auth flow", "Implement backend changes"},
    )

    runtime = LeadAgentRuntime(session, planner=StubPlanner(build_parallel_plan()), max_workers=2)
    result = await runtime.execute("Implement authentication")

    root = task_manager.get_task(result.root_task_id)
    child_statuses = {task.title: task.status for task in task_manager.list_tasks() if task.parent_id == result.root_task_id}

    assert result.success is False
    assert result.blocked_task_id is not None
    assert len(result.completed_task_ids) == 1
    assert len(result.failed_task_ids) == 1
    assert root is not None
    assert root.status == TaskStatus.BLOCKED
    assert child_statuses["Implement backend changes"] == TaskStatus.BLOCKED
    assert child_statuses["Run integration verification"] == TaskStatus.PENDING
    assert "backend failed" in result.summary


@pytest.mark.asyncio
async def test_lead_runtime_blocks_root_when_worker_raises(tmp_path):
    task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
    session = RaisingStubSession(
        task_manager=task_manager,
        results_by_input={},
    )

    runtime = LeadAgentRuntime(session, planner=StubPlanner(build_parallel_plan()), max_workers=1)
    result = await runtime.execute("Implement authentication")

    root = task_manager.get_task(result.root_task_id)

    assert result.success is False
    assert result.blocked_task_id is not None
    assert root is not None
    assert root.status == TaskStatus.BLOCKED
    assert "worker crashed" in result.summary


@pytest.mark.asyncio
async def test_lead_runtime_blocks_root_when_planner_raises(tmp_path):
    task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
    session = StubSession(
        task_manager=task_manager,
        results_by_input={},
    )

    runtime = LeadAgentRuntime(session, planner=RaisingPlanner(), max_workers=1)

    with pytest.raises(RuntimeError, match="planner boom"):
        await runtime.execute("Implement authentication")

    tasks = task_manager.list_tasks()
    assert len(tasks) == 1
    assert tasks[0].status == TaskStatus.BLOCKED
    assert tasks[0].assigned_agent is None
