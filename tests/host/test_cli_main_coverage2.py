"""Additional coverage tests for host.cli.main - handle_command commands, interactive loop helpers."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.host.cli.main import (
    _parse_orchestrate_args,
    _parse_resume_args,
    _format_task_tree,
    _find_orchestration_run,
    _resolve_task_root,
    handle_command,
    run_chat_loop,
)


class TestHandleCommandClear:
    @pytest.mark.asyncio
    async def test_clear_with_state(self, monkeypatch):
        printed = []
        monkeypatch.setattr("src.host.cli.main.console.print", lambda m: printed.append(m))
        state = MagicMock()
        session = SimpleNamespace(_state=state)
        await handle_command("/clear", session)
        state.clear_history.assert_called_once()
        assert any("cleared" in str(p).lower() for p in printed)


class TestHandleCommandReset:
    @pytest.mark.asyncio
    async def test_reset_command(self, monkeypatch):
        printed = []
        monkeypatch.setattr("src.host.cli.main.console.print", lambda m: printed.append(m))
        session = SimpleNamespace(reset=MagicMock())
        await handle_command("/reset", session)
        session.reset.assert_called_once()
        assert any("reset" in str(p).lower() for p in printed)


class TestHandleCommandTrace:
    @pytest.mark.asyncio
    async def test_trace_with_trace(self, monkeypatch):
        printed = []
        monkeypatch.setattr("src.host.cli.main.console.print", lambda m: printed.append(m))
        trace = MagicMock()
        trace.session_id = "sess-1"
        trace.events = [SimpleNamespace(type="tool_call"), SimpleNamespace(type="llm_call"), SimpleNamespace(type="error")]
        session = SimpleNamespace(get_trace=lambda: trace)
        await handle_command("/trace", session)
        assert any("Session ID" in str(p) for p in printed)
        assert any("Tool calls" in str(p) for p in printed)

    @pytest.mark.asyncio
    async def test_trace_no_trace(self, monkeypatch):
        printed = []
        monkeypatch.setattr("src.host.cli.main.console.print", lambda m: printed.append(m))
        session = SimpleNamespace(get_trace=lambda: None)
        await handle_command("/trace", session)
        assert any("No trace" in str(p) for p in printed)


class TestHandleCommandSession:
    @pytest.mark.asyncio
    async def test_session_command(self, monkeypatch):
        printed = []
        monkeypatch.setattr("src.host.cli.main.console.print", lambda m: printed.append(m))
        session = SimpleNamespace(session_id="abc", project_dir="/tmp")
        await handle_command("/session", session)
        assert any("abc" in str(p) for p in printed)


class TestHandleCommandExit:
    @pytest.mark.asyncio
    async def test_exit_command(self, monkeypatch):
        session = SimpleNamespace()
        result = await handle_command("/exit", session)
        assert result == "exit"

    @pytest.mark.asyncio
    async def test_quit_command(self, monkeypatch):
        session = SimpleNamespace()
        result = await handle_command("/quit", session)
        assert result == "exit"


class TestHandleCommandUnknown:
    @pytest.mark.asyncio
    async def test_unknown_command(self, monkeypatch):
        printed = []
        monkeypatch.setattr("src.host.cli.main.console.print", lambda m: printed.append(m))
        session = SimpleNamespace()
        result = await handle_command("/unknown", session)
        assert result is None
        assert any("Unknown" in str(p) for p in printed)


class TestHandleCommandOrchestrate:
    @pytest.mark.asyncio
    async def test_orchestrate_failed(self, monkeypatch):
        printed = []
        monkeypatch.setattr("src.host.cli.main.console.print", lambda m: printed.append(m))

        async def fail_orchestrate(goal, **kw):
            raise RuntimeError("boom")

        session = SimpleNamespace(orchestrate=fail_orchestrate)
        result = await handle_command("/orchestrate goal", session)
        assert result is None
        assert any("failed" in str(p).lower() for p in printed)


class TestHandleCommandRuns:
    @pytest.mark.asyncio
    async def test_runs_with_no_runs(self, monkeypatch):
        printed = []
        monkeypatch.setattr("src.host.cli.main.console.print", lambda m: printed.append(m))
        session = SimpleNamespace(get_orchestration_runs=lambda: [])
        await handle_command("/runs", session)
        assert any("No orchestration runs" in str(p) for p in printed)

    @pytest.mark.asyncio
    async def test_runs_with_runs(self, monkeypatch):
        printed = []
        monkeypatch.setattr("src.host.cli.main.console.print", lambda m: printed.append(m))
        session = SimpleNamespace(
            get_orchestration_runs=lambda: [{"run_id": "r1", "success": True, "goal": "test"}],
            get_active_orchestration_run_id=lambda: "r1",
        )
        await handle_command("/runs", session)
        assert any("r1" in str(p) for p in printed)


class TestHandleCommandResume:
    @pytest.mark.asyncio
    async def test_resume_success(self, monkeypatch):
        printed = []
        monkeypatch.setattr("src.host.cli.main.console.print", lambda m: printed.append(m))
        result_obj = SimpleNamespace(
            success=True, summary="done", root_task_id="root-1",
            completed_task_ids=["t1"], failed_task_ids=[],
        )

        async def do_resume(goal, **kw):
            return result_obj

        session = SimpleNamespace(orchestrate=do_resume)
        await handle_command("/resume run-1", session)
        assert any("Summary" in str(p) for p in printed)

    @pytest.mark.asyncio
    async def test_resume_failed(self, monkeypatch):
        printed = []
        monkeypatch.setattr("src.host.cli.main.console.print", lambda m: printed.append(m))

        async def fail_resume(goal, **kw):
            raise RuntimeError("resume fail")

        session = SimpleNamespace(orchestrate=fail_resume)
        await handle_command("/resume run-1", session)
        assert any("failed" in str(p).lower() for p in printed)


class TestHandleCommandTasks:
    @pytest.mark.asyncio
    async def test_tasks_no_task_manager(self, monkeypatch):
        printed = []
        monkeypatch.setattr("src.host.cli.main.console.print", lambda m: printed.append(m))
        session = SimpleNamespace(get_task_manager=lambda: None)
        await handle_command("/tasks", session)
        assert any("No task manager" in str(p) for p in printed)

    @pytest.mark.asyncio
    async def test_tasks_tree_output(self, monkeypatch):
        printed = []
        monkeypatch.setattr("src.host.cli.main.console.print", lambda m: printed.append(m))
        task_manager = SimpleNamespace(
            get_task_tree=lambda root=None: [
                {"id": "t1", "title": "Task 1", "status": "completed", "ready": True, "children": []},
            ],
            list_ready_tasks=lambda: [],
        )
        session = SimpleNamespace(
            get_task_manager=lambda: task_manager,
            get_orchestration_state=lambda: {"root_task_id": "root-1"},
        )
        await handle_command("/tasks", session)
        assert any("Task graph" in str(p) for p in printed)

    @pytest.mark.asyncio
    async def test_tasks_no_tasks(self, monkeypatch):
        printed = []
        monkeypatch.setattr("src.host.cli.main.console.print", lambda m: printed.append(m))
        task_manager = SimpleNamespace(
            get_task_tree=lambda root=None: [],
            list_ready_tasks=lambda: [],
        )
        session = SimpleNamespace(
            get_task_manager=lambda: task_manager,
            get_orchestration_state=lambda: {},
        )
        await handle_command("/tasks", session)
        assert any("No tasks" in str(p) for p in printed)


class TestHandleCommandMode:
    @pytest.mark.asyncio
    async def test_mode_switch(self, monkeypatch):
        printed = []
        monkeypatch.setattr("src.host.cli.main.console.print", lambda m: printed.append(m))
        session = SimpleNamespace(mode="build", set_mode=AsyncMock())
        await handle_command("/mode plan", session)
        session.set_mode.assert_called_once_with("plan")
        assert any("Switched" in str(p) for p in printed)

    @pytest.mark.asyncio
    async def test_mode_no_args(self, monkeypatch):
        printed = []
        monkeypatch.setattr("src.host.cli.main.console.print", lambda m: printed.append(m))
        session = SimpleNamespace(mode="build", set_mode=AsyncMock())
        await handle_command("/mode", session)
        assert any("Current mode" in str(p) or "build" in str(p) for p in printed)


class TestParseOrchestrateArgsEdge:
    def test_empty_args(self):
        assert _parse_orchestrate_args([]) is None

    def test_workers_non_int(self):
        assert _parse_orchestrate_args(["--workers", "abc", "goal"]) is None

    def test_empty_goal(self):
        assert _parse_orchestrate_args(["--workers", "2"]) is None


class TestParseResumeArgsEdge:
    def test_empty_args(self):
        assert _parse_resume_args([]) is None

    def test_empty_run_id(self):
        assert _parse_resume_args(["  "]) is None

    def test_valid(self):
        result = _parse_resume_args(["run-1", "--workers", "3"])
        assert result == ("run-1", 3)


class TestResolveTaskRootEdge:
    def test_with_run_id_not_found(self):
        session = SimpleNamespace(get_orchestration_runs=lambda: [])
        root, err = _resolve_task_root(session, "missing")
        assert root is None
        assert "Unknown" in err

    def test_with_run_id_found(self):
        session = SimpleNamespace(
            get_orchestration_runs=lambda: [{"run_id": "r1", "root_task_id": "root-1"}],
        )
        root, err = _resolve_task_root(session, "r1")
        assert root == "root-1"
        assert err is None
