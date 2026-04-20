"""Additional coverage tests for runtime.lead - execute with parallel subtasks, resume with prior state, worker assignment, priority mapping."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.core.planner import ExecutionPlan, Task, TaskDependency, TaskPriority
from src.core.tasks import TaskManager, TaskStatus
from src.runtime.lead import LeadAgentRuntime, WorkerProfile, _TaskExecutionOutcome


class StubTurnResult:
    def __init__(self, success=True, response="", error=None):
        self.success = success
        self.response = response
        self.error = error


class StubWorkerSession:
    def __init__(self, parent, session_id):
        self._parent = parent
        self.session_id = session_id

    async def turn(self, user_input, **kwargs):
        return await self._parent.run_worker_turn(
            self.session_id, user_input, create_runtime_task=kwargs.get("create_runtime_task", True),
        )

    async def aclose(self):
        self._parent.closed_workers.append(self.session_id)


class SimpleRegistry:
    def get_tool_names(self):
        return ["read", "search", "write", "run", "task", "web_search", "web_fetch",
                "memory_get", "memory_put", "memory_search", "memory_list",
                "lsp_diagnostics", "lsp_hover", "lsp_definition", "lsp_references",
                "lsp_completions", "lsp_rename", "multi_edit", "notebook_edit"]


class StubSession:
    def __init__(self, task_manager, results_by_input, *, enable_trace=False):
        self._agent = object()
        self._task_manager = task_manager
        self.registry = SimpleRegistry()
        self.session_id = "session-test"
        self.enable_trace = enable_trace
        self._trace = None
        self.results_by_input = results_by_input
        self.closed_workers = []
        self._worker_count = 0
        self._mcp_extensions = []
        self.mode = "build"

    async def initialize(self):
        self._agent = object()

    def get_task_manager(self):
        return self._task_manager

    def get_trace(self):
        return None

    async def spawn_worker_session(self, *, enable_trace=None, worker_role=None, allowed_tool_names=None):
        self._worker_count += 1
        return StubWorkerSession(self, f"worker-{self._worker_count}")

    async def run_worker_turn(self, worker_session_id, user_input, *, create_runtime_task=True):
        task_line = next(line for line in user_input.splitlines() if line.startswith("Subtask: "))
        subtask = task_line.removeprefix("Subtask: ")
        return self.results_by_input.get(subtask, StubTurnResult(success=True, response="ok"))


class StubPlanner:
    def __init__(self, plan):
        self.plan = plan

    def generate_plan(self, goal, context=None):
        return self.plan


def build_parallel_plan():
    task_a = Task(id="task-a", title="Analyze code", description="Analyze code", priority=TaskPriority.HIGH)
    task_b = Task(id="task-b", title="Write tests", description="Write tests", priority=TaskPriority.HIGH)
    return ExecutionPlan(id="plan-parallel", goal="Parallel execution", tasks=[task_a, task_b],
                         total_estimated_minutes=10, total_complexity=2, high_risk_count=0)


class TestExecuteParallel:
    @pytest.mark.asyncio
    async def test_parallel_execution_both_succeed(self, tmp_path):
        task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
        session = StubSession(
            task_manager=task_manager,
            results_by_input={
                "Analyze code": StubTurnResult(success=True, response="analysis done"),
                "Write tests": StubTurnResult(success=True, response="tests written"),
            },
        )
        runtime = LeadAgentRuntime(session, planner=StubPlanner(build_parallel_plan()), max_workers=2)
        result = await runtime.execute("Parallel execution")
        assert result.success is True
        assert len(result.completed_task_ids) == 2

    @pytest.mark.asyncio
    async def test_parallel_execution_one_fails(self, tmp_path):
        task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
        session = StubSession(
            task_manager=task_manager,
            results_by_input={
                "Analyze code": StubTurnResult(success=True, response="ok"),
                "Write tests": StubTurnResult(success=False, error="test fail"),
            },
        )
        runtime = LeadAgentRuntime(session, planner=StubPlanner(build_parallel_plan()), max_workers=2)
        result = await runtime.execute("Parallel execution")
        assert result.success is False
        assert len(result.failed_task_ids) == 1
        assert result.blocked_task_id is not None


class TestExecuteNoTaskManager:
    @pytest.mark.asyncio
    async def test_execute_no_task_manager_raises(self, tmp_path):
        session = StubSession(task_manager=TaskManager(storage_path=tmp_path / "tasks.json"), results_by_input={})
        session._task_manager = None
        runtime = LeadAgentRuntime(session, max_workers=1)
        with pytest.raises(RuntimeError, match="Task manager"):
            await runtime.execute("goal")

    @pytest.mark.asyncio
    async def test_resume_no_task_manager_raises(self, tmp_path):
        session = StubSession(task_manager=TaskManager(storage_path=tmp_path / "tasks.json"), results_by_input={})
        session._task_manager = None
        runtime = LeadAgentRuntime(session, max_workers=1)
        with pytest.raises(RuntimeError, match="Task manager"):
            await runtime.resume("root-1")


class TestResumeWithPriorState:
    @pytest.mark.asyncio
    async def test_resume_completed_root(self, tmp_path):
        task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
        runtime = LeadAgentRuntime(
            StubSession(task_manager=task_manager, results_by_input={}),
            max_workers=1,
        )
        root = task_manager.create_task(title="goal", description="goal", priority=100)
        task_manager.update_task(root.id, status=TaskStatus.COMPLETED, assigned_agent=None)
        with pytest.raises(ValueError, match="not resumable"):
            await runtime.resume(root.id)

    @pytest.mark.asyncio
    async def test_resume_non_root_task(self, tmp_path):
        task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
        runtime = LeadAgentRuntime(
            StubSession(task_manager=task_manager, results_by_input={}),
            max_workers=1,
        )
        root = task_manager.create_task(title="goal", description="goal", priority=100)
        child = task_manager.create_task(title="child", description="child", parent_id=root.id, priority=50)
        with pytest.raises(ValueError, match="unknown orchestration root"):
            await runtime.resume(child.id)

    @pytest.mark.asyncio
    async def test_resume_no_resumable_children(self, tmp_path):
        task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
        session = StubSession(task_manager=task_manager, results_by_input={})
        runtime = LeadAgentRuntime(session, max_workers=1)
        root = task_manager.create_task(title="goal", description="goal", priority=100)
        task_manager.update_task(root.id, status=TaskStatus.BLOCKED, assigned_agent=None)
        child = task_manager.create_task(title="child", description="child", parent_id=root.id, priority=50)
        task_manager.update_task(child.id, status=TaskStatus.COMPLETED, assigned_agent=None)
        with pytest.raises(ValueError, match="no blocked or pending subtasks"):
            await runtime.resume(root.id, prior_state={"plan_id": "p1"})


class TestPriorityValue:
    def test_critical(self, tmp_path):
        task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
        session = StubSession(task_manager=task_manager, results_by_input={})
        runtime = LeadAgentRuntime(session, max_workers=1)
        assert runtime._priority_value("critical") == 100

    def test_high(self, tmp_path):
        task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
        session = StubSession(task_manager=task_manager, results_by_input={})
        runtime = LeadAgentRuntime(session, max_workers=1)
        assert runtime._priority_value("high") == 75

    def test_medium(self, tmp_path):
        task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
        session = StubSession(task_manager=task_manager, results_by_input={})
        runtime = LeadAgentRuntime(session, max_workers=1)
        assert runtime._priority_value("medium") == 50

    def test_low(self, tmp_path):
        task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
        session = StubSession(task_manager=task_manager, results_by_input={})
        runtime = LeadAgentRuntime(session, max_workers=1)
        assert runtime._priority_value("low") == 25

    def test_unknown(self, tmp_path):
        task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
        session = StubSession(task_manager=task_manager, results_by_input={})
        runtime = LeadAgentRuntime(session, max_workers=1)
        assert runtime._priority_value("bogus") == 0


class TestBuildFailureSummary:
    def test_with_blocked_task_id(self, tmp_path):
        task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
        session = StubSession(task_manager=task_manager, results_by_input={})
        runtime = LeadAgentRuntime(session, max_workers=1)
        from src.runtime.lead import LeadExecutionResult
        result = LeadExecutionResult(
            root_task_id="root",
            completed_task_ids=["t1"],
            failed_task_ids=["t2"],
            blocked_task_id="t2",
        )
        result.task_summaries = {"t2": "test fail"}
        summary = runtime._build_failure_summary(result)
        assert "t2" not in summary
        assert "1 completed" in summary

    def test_without_blocked_task_id(self, tmp_path):
        task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
        session = StubSession(task_manager=task_manager, results_by_input={})
        runtime = LeadAgentRuntime(session, max_workers=1)
        from src.runtime.lead import LeadExecutionResult
        result = LeadExecutionResult(root_task_id="root")
        summary = runtime._build_failure_summary(result)
        assert "Task execution failed" in summary


class TestToolIsVisibleEdge:
    def test_non_dict_policy(self, tmp_path):
        task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
        session = StubSession(task_manager=task_manager, results_by_input={})
        session.registry = SimpleNamespace(
            get_tool_policy=MagicMock(return_value="not a dict"),
        )
        runtime = LeadAgentRuntime(session, max_workers=1)
        assert runtime._tool_is_visible("read") is True

    def test_invisible_policy(self, tmp_path):
        task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
        session = StubSession(task_manager=task_manager, results_by_input={})
        session.registry = SimpleNamespace(
            get_tool_policy=MagicMock(return_value={"visible": False}),
        )
        runtime = LeadAgentRuntime(session, max_workers=1)
        assert runtime._tool_is_visible("write") is False


class TestSelectWorkerProfileEdge:
    def test_no_visible_tools_falls_back(self, tmp_path):
        task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
        session = StubSession(task_manager=task_manager, results_by_input={})
        session.registry = SimpleNamespace(
            get_tool_names=lambda: ["read", "task"],
            get_tool_policy=lambda *a, **kw: {"visible": False},
        )
        runtime = LeadAgentRuntime(session, max_workers=1)
        task = SimpleNamespace(title="Fix bug", description="edit code")
        profile = runtime._select_worker_profile(task)
        assert isinstance(profile, WorkerProfile)


class TestTaskToPlannerTask:
    def test_converts_task(self, tmp_path):
        task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
        session = StubSession(task_manager=task_manager, results_by_input={})
        runtime = LeadAgentRuntime(session, max_workers=1)
        task = SimpleNamespace(id="t1", title="My Task", description="Do stuff")
        result = runtime._task_to_planner_task(task)
        assert result.id == "t1"
        assert result.title == "My Task"
