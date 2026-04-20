"""Targeted coverage for host/cli/main.py uncovered lines: 135-212,331-332,397-422,428,433,437."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.host.cli.main import (
    _format_task_tree,
    _find_orchestration_run,
    _parse_orchestrate_args,
    _parse_resume_args,
    _resolve_task_root,
    handle_command,
)


class TestParseOrchestrateArgs:
    def test_no_args(self):
        assert _parse_orchestrate_args([]) is None

    def test_workers_flag(self):
        result = _parse_orchestrate_args(["--workers", "3", "my", "goal"])
        assert result == (3, "my goal")

    def test_workers_short_flag(self):
        result = _parse_orchestrate_args(["-w", "4", "build", "it"])
        assert result == (4, "build it")

    def test_invalid_workers(self):
        result = _parse_orchestrate_args(["--workers", "abc", "goal"])
        assert result is None

    def test_goal_only(self):
        result = _parse_orchestrate_args(["deploy", "stuff"])
        assert result == (2, "deploy stuff")

    def test_empty_goal(self):
        result = _parse_orchestrate_args(["--workers", "2"])
        assert result is None


class TestParseResumeArgs:
    def test_no_args(self):
        assert _parse_resume_args([]) is None

    def test_basic(self):
        result = _parse_resume_args(["run-1"])
        assert result == ("run-1", 2)

    def test_with_workers(self):
        result = _parse_resume_args(["run-1", "--workers", "3"])
        assert result == ("run-1", 3)

    def test_invalid_workers(self):
        result = _parse_resume_args(["run-1", "--workers", "bad"])
        assert result is None

    def test_extra_args(self):
        result = _parse_resume_args(["run-1", "extra"])
        assert result is None

    def test_empty_run_id(self):
        result = _parse_resume_args(["  "])
        assert result is None


class TestFormatTaskTree:
    def test_basic(self):
        nodes = [{"id": "t1", "title": "Task 1", "status": "completed", "ready": True}]
        lines = _format_task_tree(nodes)
        assert len(lines) == 1
        assert "Task 1" in lines[0]
        assert "completed" in lines[0]

    def test_with_children(self):
        nodes = [
            {
                "id": "t1",
                "title": "Parent",
                "status": "in_progress",
                "children": [
                    {"id": "t2", "title": "Child", "status": "pending"},
                ],
            }
        ]
        lines = _format_task_tree(nodes)
        assert len(lines) == 2
        assert "Parent" in lines[0]
        assert "Child" in lines[1]

    def test_with_assigned_agent(self):
        nodes = [{"id": "t1", "title": "Task", "status": "running", "assigned_agent": "worker-1"}]
        lines = _format_task_tree(nodes)
        assert "worker-1" in lines[0]


class TestFindOrchestrationRun:
    def test_found(self):
        session = MagicMock()
        session.get_orchestration_runs = MagicMock(return_value=[
            {"run_id": "r1"}, {"run_id": "r2"}
        ])
        result = _find_orchestration_run(session, "r2")
        assert result["run_id"] == "r2"

    def test_not_found(self):
        session = MagicMock()
        session.get_orchestration_runs = MagicMock(return_value=[])
        result = _find_orchestration_run(session, "r1")
        assert result is None

    def test_no_method(self):
        session = MagicMock(spec=[])
        result = _find_orchestration_run(session, "r1")
        assert result is None


class TestResolveTaskRoot:
    def test_with_run_id_found(self):
        session = MagicMock()
        session.get_orchestration_runs = MagicMock(return_value=[
            {"run_id": "r1", "root_task_id": "root-1"}
        ])
        root_id, err = _resolve_task_root(session, "r1")
        assert root_id == "root-1"
        assert err is None

    def test_with_run_id_not_found(self):
        session = MagicMock()
        session.get_orchestration_runs = MagicMock(return_value=[])
        root_id, err = _resolve_task_root(session, "missing")
        assert root_id is None
        assert "Unknown" in err

    def test_no_run_id_from_state(self):
        session = MagicMock()
        session.get_orchestration_state = MagicMock(return_value={"root_task_id": "root-2"})
        root_id, err = _resolve_task_root(session)
        assert root_id == "root-2"

    def test_no_run_id_no_state(self):
        session = MagicMock()
        session.get_orchestration_state = MagicMock(return_value={})
        root_id, err = _resolve_task_root(session)
        assert root_id is None


class TestHandleCommand:
    @pytest.mark.asyncio
    async def test_exit_command(self):
        session = MagicMock()
        result = await handle_command("/exit", session)
        assert result == "exit"

    @pytest.mark.asyncio
    async def test_quit_command(self):
        session = MagicMock()
        result = await handle_command("/quit", session)
        assert result == "exit"

    @pytest.mark.asyncio
    async def test_unknown_command(self):
        session = MagicMock()
        with patch("src.host.cli.main.console") as mock_console:
            result = await handle_command("/badcmd", session)
            assert result is None
            mock_console.print.assert_called()

    @pytest.mark.asyncio
    async def test_help_command(self):
        session = MagicMock()
        with patch("src.host.cli.main.console") as mock_console:
            result = await handle_command("/help", session)
            assert result is None
            mock_console.print.assert_called()

    @pytest.mark.asyncio
    async def test_reset_command(self):
        session = MagicMock()
        with patch("src.host.cli.main.console") as mock_console:
            result = await handle_command("/reset", session)
            session.reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_command(self):
        session = MagicMock()
        session.session_id = "sess-1"
        session.project_dir = "/tmp"
        with patch("src.host.cli.main.console") as mock_console:
            result = await handle_command("/session", session)
            assert result is None

    @pytest.mark.asyncio
    async def test_mode_command_switch(self):
        session = MagicMock()
        session.mode = "build"
        session.set_mode = AsyncMock()
        with patch("src.host.cli.main.console") as mock_console:
            result = await handle_command("/mode plan", session)
            session.set_mode.assert_called_once_with("plan")

    @pytest.mark.asyncio
    async def test_mode_command_no_arg(self):
        session = MagicMock()
        session.mode = "build"
        with patch("src.host.cli.main.console") as mock_console:
            result = await handle_command("/mode", session)
            assert result is None

    @pytest.mark.asyncio
    async def test_clear_command(self):
        session = MagicMock()
        session._state = MagicMock()
        with patch("src.host.cli.main.console") as mock_console:
            result = await handle_command("/clear", session)
            session._state.clear_history.assert_called_once()

    @pytest.mark.asyncio
    async def test_trace_command(self):
        session = MagicMock()
        trace = MagicMock()
        trace.session_id = "s1"
        trace.events = [MagicMock(type="tool_call"), MagicMock(type="error")]
        session.get_trace = MagicMock(return_value=trace)
        with patch("src.host.cli.main.console") as mock_console:
            result = await handle_command("/trace", session)
            assert result is None

    @pytest.mark.asyncio
    async def test_trace_no_trace(self):
        session = MagicMock()
        session.get_trace = MagicMock(return_value=None)
        with patch("src.host.cli.main.console") as mock_console:
            result = await handle_command("/trace", session)
            assert result is None

    @pytest.mark.asyncio
    async def test_orchestrate_no_args(self):
        session = MagicMock()
        with patch("src.host.cli.main.console") as mock_console:
            result = await handle_command("/orchestrate", session)
            assert result is None

    @pytest.mark.asyncio
    async def test_runs_command_no_runs(self):
        session = MagicMock()
        session.get_orchestration_runs = MagicMock(return_value=[])
        with patch("src.host.cli.main.console") as mock_console:
            result = await handle_command("/runs", session)
            assert result is None

    @pytest.mark.asyncio
    async def test_runs_command_with_runs(self):
        session = MagicMock()
        session.get_orchestration_runs = MagicMock(return_value=[
            {"run_id": "r1", "success": True, "goal": "test goal"}
        ])
        session.get_active_orchestration_run_id = MagicMock(return_value="r1")
        with patch("src.host.cli.main.console") as mock_console:
            result = await handle_command("/runs", session)
            assert result is None

    @pytest.mark.asyncio
    async def test_resume_no_args(self):
        session = MagicMock()
        with patch("src.host.cli.main.console") as mock_console:
            result = await handle_command("/resume", session)
            assert result is None

    @pytest.mark.asyncio
    async def test_tasks_no_task_manager(self):
        session = MagicMock()
        session.get_task_manager = MagicMock(return_value=None)
        with patch("src.host.cli.main.console") as mock_console:
            result = await handle_command("/tasks", session)
            assert result is None

    @pytest.mark.asyncio
    async def test_tasks_with_ready_filter(self):
        session = MagicMock()
        tm = MagicMock()
        tm.list_ready_tasks = MagicMock(return_value=[])
        session.get_task_manager = MagicMock(return_value=tm)
        session.get_orchestration_state = MagicMock(return_value={"root_task_id": "root-1"})
        with patch("src.host.cli.main.console") as mock_console:
            result = await handle_command("/tasks ready", session)
            assert result is None
