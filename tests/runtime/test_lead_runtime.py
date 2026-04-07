from dataclasses import dataclass
from pathlib import Path

import pytest

from src.core.tasks import TaskManager, TaskStatus
from src.runtime.lead import LeadAgentRuntime


@dataclass
class StubTurnResult:
    success: bool
    response: str = ""
    error: str | None = None


class StubSession:
    def __init__(self, task_manager: TaskManager, results: list[StubTurnResult]):
        self._agent = object()
        self._task_manager = task_manager
        self.session_id = "session-test"
        self.results = results
        self.turn_calls: list[tuple[str, bool]] = []

    async def initialize(self) -> None:
        self._agent = object()

    def get_task_manager(self) -> TaskManager:
        return self._task_manager

    async def turn(self, user_input: str, **kwargs):
        self.turn_calls.append((user_input, kwargs.get("create_runtime_task", True)))
        return self.results.pop(0)


@pytest.mark.asyncio
async def test_lead_runtime_materializes_plan_and_executes_serially(tmp_path):
    task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
    session = StubSession(
        task_manager=task_manager,
        results=[
            StubTurnResult(success=True, response="ok-1"),
            StubTurnResult(success=True, response="ok-2"),
            StubTurnResult(success=True, response="ok-3"),
            StubTurnResult(success=True, response="ok-4"),
        ],
    )

    runtime = LeadAgentRuntime(session)
    result = await runtime.execute("Implement user authentication")

    tasks = task_manager.list_tasks()
    root = task_manager.get_task(result.root_task_id)

    assert result.success is True
    assert root is not None
    assert root.status == TaskStatus.COMPLETED
    assert len(result.completed_task_ids) == 4
    assert len(tasks) == 5
    assert all(create_runtime_task is False for _, create_runtime_task in session.turn_calls)


@pytest.mark.asyncio
async def test_lead_runtime_blocks_root_when_subtask_fails(tmp_path):
    task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
    session = StubSession(
        task_manager=task_manager,
        results=[
            StubTurnResult(success=True, response="ok-1"),
            StubTurnResult(success=False, error="subtask failed"),
        ],
    )

    runtime = LeadAgentRuntime(session)
    result = await runtime.execute("Fix login bug")

    root = task_manager.get_task(result.root_task_id)
    child_statuses = {task.id: task.status for task in task_manager.list_tasks() if task.parent_id == root.id}

    assert result.success is False
    assert result.blocked_task_id is not None
    assert root is not None
    assert root.status == TaskStatus.BLOCKED
    assert TaskStatus.BLOCKED in child_statuses.values()
