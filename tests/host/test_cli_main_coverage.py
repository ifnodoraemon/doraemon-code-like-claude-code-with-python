"""Targeted coverage tests for host.cli.main."""

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
)


class TestParseOrchestrateArgsEdge:
    def test_workers_zero_clamped_to_one(self):
        parsed = _parse_orchestrate_args(["--workers", "0", "do", "stuff"])
        assert parsed == (1, "do stuff")

    def test_workers_negative_clamped(self):
        parsed = _parse_orchestrate_args(["--workers", "-5", "goal"])
        assert parsed == (1, "goal")

    def test_w_shorthand(self):
        parsed = _parse_orchestrate_args(["-w", "4", "run", "tests"])
        assert parsed == (4, "run tests")


class TestParseResumeArgsEdge:
    def test_w_shorthand(self):
        parsed = _parse_resume_args(["run-1", "-w", "5"])
        assert parsed == ("run-1", 5)

    def test_extra_positional_args(self):
        assert _parse_resume_args(["run-1", "unexpected"]) is None

    def test_invalid_workers_value(self):
        assert _parse_resume_args(["run-1", "--workers", "abc"]) is None


class TestFormatTaskTreeEdge:
    def test_node_without_id(self):
        lines = _format_task_tree([{"title": "No ID", "status": "pending"}])
        assert len(lines) == 1
        assert "No ID" in lines[0]

    def test_deep_nesting(self):
        nodes = [
            {
                "id": "root",
                "title": "Root",
                "status": "in_progress",
                "children": [
                    {
                        "id": "c1",
                        "title": "C1",
                        "status": "pending",
                        "children": [
                            {"id": "gc1", "title": "GC1", "status": "completed"}
                        ],
                    }
                ],
            }
        ]
        lines = _format_task_tree(nodes)
        assert len(lines) == 3


class TestFindOrchestrationRunEdge:
    def test_empty_runs(self):
        session = SimpleNamespace(get_orchestration_runs=lambda: [])
        assert _find_orchestration_run(session, "any") is None

    def test_returns_most_recent_match(self):
        runs = [
            {"run_id": "r1", "goal": "first"},
            {"run_id": "r1", "goal": "duplicate"},
        ]
        session = SimpleNamespace(get_orchestration_runs=lambda: runs)
        result = _find_orchestration_run(session, "r1")
        assert result["goal"] == "duplicate"


class TestResolveTaskRootEdge:
    def test_state_without_root_task_id(self):
        session = SimpleNamespace(get_orchestration_state=lambda: {})
        root, err = _resolve_task_root(session, None)
        assert root is None
        assert err is None

    def test_state_callable_not_dict(self):
        session = SimpleNamespace(get_orchestration_state=lambda: None)
        root, err = _resolve_task_root(session, None)
        assert root is None
        assert err is None


class TestHandleCommandAdditional:
    @pytest.mark.asyncio
    async def test_orchestrate_success_no_worker_assignments(self, monkeypatch):
        printed = []

        async def orchestrate(goal, **kw):
            return SimpleNamespace(
                success=True, summary="done", root_task_id="root-1",
                completed_task_ids=["t1"], failed_task_ids=[],
                worker_assignments={},
            )

        session = SimpleNamespace(orchestrate=orchestrate)
        monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))
        result = await handle_command("/orchestrate --workers 2 goal", session)
        assert result is None
        assert any("Summary:" in p for p in printed)

    @pytest.mark.asyncio
    async def test_mode_same_mode_prints_current(self, monkeypatch):
        printed = []
        session = SimpleNamespace(mode="build", set_mode=AsyncMock())
        monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))
        await handle_command("/mode build", session)
        assert any("Current mode" in p or "build" in p for p in printed)

    @pytest.mark.asyncio
    async def test_help_shorthand_h(self, monkeypatch):
        printed = []
        session = SimpleNamespace()
        monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))
        result = await handle_command("/h", session)
        assert any("Commands" in str(p) for p in printed)

    @pytest.mark.asyncio
    async def test_help_question_mark(self, monkeypatch):
        printed = []
        session = SimpleNamespace()
        monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))
        result = await handle_command("/?", session)
        assert any("Commands" in str(p) for p in printed)

    @pytest.mark.asyncio
    async def test_tasks_ready_with_run_id(self, monkeypatch):
        printed = []
        task_manager = SimpleNamespace(
            list_ready_tasks=lambda: [
                SimpleNamespace(title="Ready task", id="t1", status=SimpleNamespace(value="pending"), parent_id="root-1")
            ],
            get_task_tree=lambda root=None: [],
        )
        session = SimpleNamespace(
            get_task_manager=lambda: task_manager,
            get_orchestration_runs=lambda: [
                {"run_id": "run-1", "root_task_id": "root-1", "success": True},
            ],
        )
        monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))
        result = await handle_command("/tasks ready run-1", session)
        assert result is None
        assert any("Ready task" in p for p in printed)

    @pytest.mark.asyncio
    async def test_resume_invalid_extra_arg(self, monkeypatch):
        printed = []
        session = SimpleNamespace()
        monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))
        result = await handle_command("/resume run-1 extra", session)
        assert any("Usage" in p for p in printed)
