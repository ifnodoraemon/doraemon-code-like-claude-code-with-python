"""Targeted coverage tests for runtime.lead - execute with subtasks, resume, worker selection."""

from types import SimpleNamespace

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


class SimpleRegistry:
    def get_tool_names(self):
        return ["read", "search", "write", "run", "task", "web_search", "web_fetch",
                "memory_get", "memory_put", "memory_search", "memory_list",
                "lsp_diagnostics", "lsp_hover", "lsp_definition", "lsp_references",
                "lsp_completions", "lsp_rename", "multi_edit", "notebook_edit"]


def build_serial_plan():
    task_a = Task(id="task-a", title="Analyze code", description="Analyze code", priority=TaskPriority.HIGH)
    task_b = Task(
        id="task-b",
        title="Implement feature",
        description="Implement feature",
        priority=TaskPriority.HIGH,
        dependencies=[TaskDependency(task_id="task-a", dependency_type="requires")],
    )
    return ExecutionPlan(id="plan-serial", goal="Serial execution", tasks=[task_a, task_b],
                         total_estimated_minutes=10, total_complexity=2, high_risk_count=0)


class StubPlanner:
    def __init__(self, plan):
        self.plan = plan

    def generate_plan(self, goal, context=None):
        return self.plan


class TestLeadRuntimeExecuteSerial:
    @pytest.mark.asyncio
    async def test_serial_execution_respects_dependencies(self, tmp_path):
        task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
        session = StubSession(
            task_manager=task_manager,
            results_by_input={
                "Analyze code": StubTurnResult(success=True, response="analysis done"),
                "Implement feature": StubTurnResult(success=True, response="feature done"),
            },
        )
        runtime = LeadAgentRuntime(session, planner=StubPlanner(build_serial_plan()), max_workers=2)
        result = await runtime.execute("Serial execution")

        assert result.success is True
        assert len(result.completed_task_ids) == 2
        assert result.failed_task_ids == []


class TestLeadRuntimeResumeNoChildren:
    @pytest.mark.asyncio
    async def test_resume_root_with_no_children_raises(self, tmp_path):
        task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
        runtime = LeadAgentRuntime(
            StubSession(task_manager=task_manager, results_by_input={}),
            max_workers=1,
        )
        root = task_manager.create_task(title="goal", description="goal", priority=100)
        task_manager.update_task(root.id, status=TaskStatus.BLOCKED, assigned_agent=None)

        with pytest.raises(ValueError, match="no resumable subtasks"):
            await runtime.resume(root.id, prior_state={"plan_id": "p1"})


class TestWorkerProfileSelection:
    def test_validate_profile_for_verification_task(self, tmp_path):
        task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
        session = StubSession(task_manager=task_manager, results_by_input={})
        runtime = LeadAgentRuntime(session, max_workers=1)
        task = SimpleNamespace(title="Verify the tests pass", description="run pytest")
        profile = runtime._select_worker_profile(task)
        assert profile.role == "validate"
        assert "run" in profile.allowed_tool_names

    def test_inspect_profile_for_research(self, tmp_path):
        task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
        session = StubSession(task_manager=task_manager, results_by_input={})
        runtime = LeadAgentRuntime(session, max_workers=1)
        task = SimpleNamespace(title="Research the architecture", description="read code")
        profile = runtime._select_worker_profile(task)
        assert profile.role == "inspect"

    def test_change_profile_default(self, tmp_path):
        task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
        session = StubSession(task_manager=task_manager, results_by_input={})
        runtime = LeadAgentRuntime(session, max_workers=1)
        task = SimpleNamespace(title="Fix the bug", description="edit and run")
        profile = runtime._select_worker_profile(task)
        assert profile.role == "change"


class TestBuildWorkerInput:
    def test_input_contains_profile_info(self, tmp_path):
        task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
        session = StubSession(task_manager=task_manager, results_by_input={})
        runtime = LeadAgentRuntime(session, max_workers=1)
        profile = WorkerProfile(
            role="inspect", capability_groups=["read"], allowed_tool_names=["read"],
            instruction="Inspect only",
        )
        task = SimpleNamespace(title="Check auth", description="Read auth.py")
        result = runtime._build_worker_input(task, profile)
        assert "inspect" in result
        assert "Read auth.py" in result


class TestTrimTitle:
    def test_empty_string(self, tmp_path):
        task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
        session = StubSession(task_manager=task_manager, results_by_input={})
        runtime = LeadAgentRuntime(session, max_workers=1)
        result = runtime._trim_title("   ")
        assert result == "Orchestrated task"

    def test_exactly_80(self, tmp_path):
        task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
        session = StubSession(task_manager=task_manager, results_by_input={})
        runtime = LeadAgentRuntime(session, max_workers=1)
        result = runtime._trim_title("x" * 80)
        assert len(result) == 80


class TestToolIsVisible:
    def test_no_get_tool_policy(self, tmp_path):
        task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
        session = StubSession(task_manager=task_manager, results_by_input={})
        session.registry = SimpleNamespace()
        runtime = LeadAgentRuntime(session, max_workers=1)
        assert runtime._tool_is_visible("read") is True
