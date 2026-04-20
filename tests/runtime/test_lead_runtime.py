import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from src.core.home import Trace
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
        enable_trace: bool = False,
        trace: Trace | None = None,
    ):
        self._agent = object()
        self._task_manager = task_manager
        self.registry = SimpleRegistry()
        self.session_id = "session-test"
        self.enable_trace = enable_trace
        self._trace = trace
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

    def get_trace(self) -> Trace | None:
        return self._trace

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


class MalformedStubSession(StubSession):
    async def run_worker_turn(
        self,
        worker_session_id: str,
        user_input: str,
        *,
        create_runtime_task: bool,
    ):
        self.turn_calls.append((worker_session_id, user_input, create_runtime_task))
        return object()


class ClaimFailingTaskManager(TaskManager):
    def __init__(self, storage_path):
        super().__init__(storage_path=storage_path)
        self.claim_attempts: list[tuple[str, str]] = []

    def claim_task(self, task_id: str, agent_id: str):
        self.claim_attempts.append((task_id, agent_id))
        raise RuntimeError(f"claim failed for {task_id}")


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
async def test_lead_runtime_records_orchestration_trace_events(tmp_path):
    task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
    trace = Trace("session_test")
    session = StubSession(
        task_manager=task_manager,
        results_by_input={
            "Inspect auth flow": StubTurnResult(success=True, response="inspection done"),
            "Implement backend changes": StubTurnResult(success=True, response="backend done"),
            "Run integration verification": StubTurnResult(success=True, response="verification done"),
        },
        barrier_inputs={"Inspect auth flow", "Implement backend changes"},
        enable_trace=True,
        trace=trace,
    )

    runtime = LeadAgentRuntime(session, planner=StubPlanner(build_parallel_plan()), max_workers=2)
    result = await runtime.execute("Implement authentication", trace_run_id="run-1")

    orchestration_events = [event for event in trace.events if event.type == "orchestration"]
    event_names = [event.name for event in orchestration_events]

    assert result.success is True
    assert event_names[:3] == ["execution_started", "planning_started", "plan_generated"]
    assert "ready_batch_started" in event_names
    assert event_names[-1] == "completed"
    assert all(event.data["run_id"] == "run-1" for event in orchestration_events)


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
async def test_lead_runtime_can_resume_blocked_run(tmp_path):
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
    first = await runtime.execute("Implement authentication")

    session.results_by_input["Implement backend changes"] = StubTurnResult(
        success=True, response="backend fixed"
    )
    resumed = await runtime.resume(
        first.root_task_id,
        prior_state={
            "plan_id": first.plan_id,
            "executed_task_ids": first.executed_task_ids,
            "completed_task_ids": first.completed_task_ids,
            "task_summaries": first.task_summaries,
        },
    )

    child_statuses = {
        task.title: task.status for task in task_manager.list_tasks() if task.parent_id == first.root_task_id
    }
    assert first.success is False
    assert resumed.success is True
    assert child_statuses["Inspect auth flow"] == TaskStatus.COMPLETED
    assert child_statuses["Implement backend changes"] == TaskStatus.COMPLETED
    assert child_statuses["Run integration verification"] == TaskStatus.COMPLETED
    assert "Resumed orchestration" in resumed.summary


@pytest.mark.asyncio
async def test_lead_runtime_rejects_resume_for_completed_root(tmp_path):
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
    first = await runtime.execute("Implement authentication")

    with pytest.raises(ValueError, match="not resumable"):
        await runtime.resume(first.root_task_id, prior_state={"plan_id": first.plan_id})


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
    result = await runtime.execute("Implement authentication")

    tasks = task_manager.list_tasks()
    assert result.success is False
    assert result.root_task_id == tasks[0].id
    assert result.blocked_task_id == tasks[0].id
    assert "planner boom" in result.summary
    assert len(tasks) == 1
    assert tasks[0].status == TaskStatus.BLOCKED
    assert tasks[0].assigned_agent is None


@pytest.mark.asyncio
async def test_lead_execution_result_to_dict():
    from src.runtime.lead import LeadExecutionResult

    result = LeadExecutionResult(
        root_task_id="root-1",
        plan_id="plan-1",
        executed_task_ids=["t1"],
        completed_task_ids=["t1"],
        failed_task_ids=[],
        blocked_task_id=None,
        success=True,
        summary="done",
        task_summaries={"t1": "ok"},
        worker_assignments={"t1": {"role": "inspect"}},
    )
    d = result.to_dict()
    assert d["root_task_id"] == "root-1"
    assert d["plan_id"] == "plan-1"
    assert d["success"] is True
    assert d["worker_assignments"]["t1"]["role"] == "inspect"


@pytest.mark.asyncio
async def test_lead_runtime_resumes_completed_root_raises(tmp_path):
    task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
    session = StubSession(
        task_manager=task_manager,
        results_by_input={
            "Inspect auth flow": StubTurnResult(success=True, response="ok"),
            "Implement backend changes": StubTurnResult(success=True, response="ok"),
            "Run integration verification": StubTurnResult(success=True, response="ok"),
        },
        barrier_inputs={"Inspect auth flow", "Implement backend changes"},
    )

    runtime = LeadAgentRuntime(session, planner=StubPlanner(build_parallel_plan()), max_workers=2)
    first = await runtime.execute("Implement authentication")

    with pytest.raises(ValueError, match="not resumable"):
        await runtime.resume(first.root_task_id, prior_state={"plan_id": first.plan_id})


@pytest.mark.asyncio
async def test_lead_runtime_resume_no_resumable_subtasks(tmp_path):
    task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
    runtime = LeadAgentRuntime(
        StubSession(task_manager=task_manager, results_by_input={}),
        planner=StubPlanner(build_parallel_plan()),
        max_workers=1,
    )
    root = task_manager.create_task(title="goal", description="goal", priority=100)
    task_manager.update_task(root.id, status=TaskStatus.BLOCKED, assigned_agent=None)
    child = task_manager.create_task(title="child", description="child", parent_id=root.id, priority=50)
    task_manager.update_task(child.id, status=TaskStatus.COMPLETED, assigned_agent=None)

    with pytest.raises(ValueError, match="no blocked or pending subtasks"):
        await runtime.resume(root.id, prior_state={"plan_id": "p1"})


@pytest.mark.asyncio
async def test_lead_runtime_resume_unknown_root(tmp_path):
    task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
    runtime = LeadAgentRuntime(
        StubSession(task_manager=task_manager, results_by_input={}),
        max_workers=1,
    )
    with pytest.raises(ValueError, match="unknown orchestration root"):
        await runtime.resume("nonexistent", prior_state={})


@pytest.mark.asyncio
async def test_lead_runtime_trim_title(tmp_path):
    task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
    session = StubSession(task_manager=task_manager, results_by_input={})
    runtime = LeadAgentRuntime(session, max_workers=1)

    short = runtime._trim_title("short goal")
    assert short == "short goal"

    long = runtime._trim_title("x" * 100)
    assert len(long) <= 80


@pytest.mark.asyncio
async def test_lead_runtime_priority_value(tmp_path):
    task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
    session = StubSession(task_manager=task_manager, results_by_input={})
    runtime = LeadAgentRuntime(session, max_workers=1)

    assert runtime._priority_value("critical") == 100
    assert runtime._priority_value("high") == 75
    assert runtime._priority_value("medium") == 50
    assert runtime._priority_value("low") == 25
    assert runtime._priority_value("unknown") == 0


@pytest.mark.asyncio
async def test_lead_runtime_select_worker_profile_validate(tmp_path):
    task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
    session = StubSession(task_manager=task_manager, results_by_input={})
    runtime = LeadAgentRuntime(session, max_workers=1)

    task = SimpleNamespace(title="Verify the integration test", description="run tests")
    profile = runtime._select_worker_profile(task)
    assert profile.role == "validate"


@pytest.mark.asyncio
async def test_lead_runtime_select_worker_profile_inspect(tmp_path):
    task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
    session = StubSession(task_manager=task_manager, results_by_input={})
    runtime = LeadAgentRuntime(session, max_workers=1)

    task = SimpleNamespace(title="Analyze the auth flow", description="inspect code")
    profile = runtime._select_worker_profile(task)
    assert profile.role == "inspect"


@pytest.mark.asyncio
async def test_lead_runtime_select_worker_profile_change(tmp_path):
    task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
    session = StubSession(task_manager=task_manager, results_by_input={})
    runtime = LeadAgentRuntime(session, max_workers=1)

    task = SimpleNamespace(title="Implement the feature", description="write code")
    profile = runtime._select_worker_profile(task)
    assert profile.role == "change"


@pytest.mark.asyncio
async def test_lead_runtime_build_worker_input(tmp_path):
    task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
    session = StubSession(task_manager=task_manager, results_by_input={})
    runtime = LeadAgentRuntime(session, max_workers=1)

    from src.runtime.lead import WorkerProfile
    profile = WorkerProfile(
        role="inspect",
        capability_groups=["read"],
        allowed_tool_names=["read"],
        instruction="Inspect only",
    )
    task = SimpleNamespace(title="Check auth", description="Read auth.py")
    result = runtime._build_worker_input(task, profile)
    assert "inspect" in result.lower()
    assert "Read auth.py" in result


@pytest.mark.asyncio
async def test_lead_runtime_blocks_child_when_worker_returns_malformed_result(tmp_path):
    task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
    session = MalformedStubSession(
        task_manager=task_manager,
        results_by_input={},
    )

    runtime = LeadAgentRuntime(session, planner=StubPlanner(build_parallel_plan()), max_workers=1)
    result = await runtime.execute("Implement authentication")

    root = task_manager.get_task(result.root_task_id)
    child_statuses = {
        task.title: task.status for task in task_manager.list_tasks() if task.parent_id == result.root_task_id
    }

    assert result.success is False
    assert root is not None
    assert root.status == TaskStatus.BLOCKED
    assert child_statuses["Inspect auth flow"] == TaskStatus.BLOCKED
    assert "invalid turn result" in result.summary


@pytest.mark.asyncio
async def test_lead_runtime_closes_worker_when_claim_fails(tmp_path):
    task_manager = ClaimFailingTaskManager(storage_path=tmp_path / "tasks.json")
    session = StubSession(
        task_manager=task_manager,
        results_by_input={},
    )

    runtime = LeadAgentRuntime(session, planner=StubPlanner(build_parallel_plan()), max_workers=1)
    result = await runtime.execute("Implement authentication")

    root = task_manager.get_task(result.root_task_id)
    child_statuses = {
        task.title: task.status for task in task_manager.list_tasks() if task.parent_id == result.root_task_id
    }

    assert result.success is False
    assert root is not None
    assert root.status == TaskStatus.BLOCKED
    assert result.worker_assignments == {}
    assert len(session.closed_workers) == 1
    assert len(task_manager.claim_attempts) == 1
    assert child_statuses["Inspect auth flow"] == TaskStatus.BLOCKED
    assert "claim failed" in result.summary
