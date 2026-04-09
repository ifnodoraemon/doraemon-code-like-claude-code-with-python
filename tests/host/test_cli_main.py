from types import SimpleNamespace

import pytest

from src.host.cli.main import (
    _format_task_tree,
    _parse_orchestrate_args,
    _parse_resume_args,
    handle_command,
)


def test_parse_orchestrate_args_supports_worker_flag():
    parsed = _parse_orchestrate_args(["--workers", "3", "implement", "auth", "flow"])

    assert parsed == (3, "implement auth flow")


def test_parse_orchestrate_args_rejects_missing_goal():
    assert _parse_orchestrate_args(["--workers", "2"]) is None
    assert _parse_orchestrate_args([]) is None


def test_parse_resume_args_supports_worker_flag():
    parsed = _parse_resume_args(["run-1", "--workers", "3"])

    assert parsed == ("run-1", 3)


def test_parse_resume_args_rejects_invalid_shape():
    assert _parse_resume_args([]) is None
    assert _parse_resume_args(["run-1", "extra"]) is None


def test_format_task_tree_renders_nested_assignments():
    lines = _format_task_tree(
        [
            {
                "id": "root",
                "title": "Root",
                "status": "in_progress",
                "ready": True,
                "children": [
                    {
                        "id": "child",
                        "title": "Child",
                        "status": "pending",
                        "ready": False,
                        "assigned_agent": "worker-1",
                    }
                ],
            }
        ]
    )

    assert lines[0] == "- [in_progress] Root (root) ready"
    assert lines[1] == "  - [pending] Child (child) @worker-1"


@pytest.mark.asyncio
async def test_handle_command_mode_uses_session_state(monkeypatch):
    printed: list[str] = []

    async def set_mode(mode: str) -> None:
        session.mode = mode

    session = SimpleNamespace(mode="build", set_mode=set_mode)
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))

    await handle_command("/mode plan", session)
    await handle_command("/mode", session)

    assert session.mode == "plan"
    assert any("Switched to plan mode" in message for message in printed)
    assert any("Current mode: plan" in message for message in printed)


@pytest.mark.asyncio
async def test_handle_command_clear_uses_state_clear_history(monkeypatch):
    printed: list[str] = []
    cleared = {"count": 0}

    class State:
        def clear_history(self):
            cleared["count"] += 1

    session = SimpleNamespace(_state=State())
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))

    await handle_command("/clear", session)

    assert cleared["count"] == 1
    assert any("Conversation cleared" in message for message in printed)


@pytest.mark.asyncio
async def test_handle_command_orchestrate_reports_failure_without_raising(monkeypatch):
    printed: list[str] = []

    async def orchestrate(goal: str, *, max_workers: int | None = None):
        raise RuntimeError(f"planner boom for {goal}")

    session = SimpleNamespace(orchestrate=orchestrate)
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))

    result = await handle_command("/orchestrate broken goal", session)

    assert result is None
    assert any("Running orchestration" in message for message in printed)
    assert any("Orchestration failed:" in message for message in printed)


@pytest.mark.asyncio
async def test_handle_command_runs_lists_history(monkeypatch):
    printed: list[str] = []
    session = SimpleNamespace(
        get_orchestration_runs=lambda: [
            {"run_id": "run-1", "goal": "First goal", "success": True},
            {"run_id": "run-2", "goal": "Second goal", "success": False},
        ],
        get_active_orchestration_run_id=lambda: "run-2",
    )
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))

    await handle_command("/runs", session)

    assert any("Runs:" in message for message in printed)
    assert any("* [blocked] run-2 Second goal" in message for message in printed)
    assert any("- [completed] run-1 First goal" in message for message in printed)


@pytest.mark.asyncio
async def test_handle_command_resume_forwards_run_id(monkeypatch):
    printed: list[str] = []
    captured: dict[str, object] = {}

    async def orchestrate(goal: str, *, max_workers: int | None = None, resume_run_id: str | None = None):
        captured["goal"] = goal
        captured["max_workers"] = max_workers
        captured["resume_run_id"] = resume_run_id
        return SimpleNamespace(
            success=True,
            summary="resumed",
            root_task_id="root-1",
        )

    session = SimpleNamespace(orchestrate=orchestrate)
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))

    result = await handle_command("/resume run-1 --workers 3", session)

    assert result is None
    assert captured == {"goal": "", "max_workers": 3, "resume_run_id": "run-1"}
    assert any("Resuming run" in message for message in printed)
    assert any("Summary:" in message for message in printed)


@pytest.mark.asyncio
async def test_handle_command_tasks_accepts_run_id(monkeypatch):
    printed: list[str] = []
    task_manager = SimpleNamespace(
        list_ready_tasks=lambda: [],
        get_task_tree=lambda root_task_id=None: [
            {"id": root_task_id or "root", "title": "Root", "status": "completed", "children": []}
        ],
    )
    session = SimpleNamespace(
        get_task_manager=lambda: task_manager,
        get_orchestration_runs=lambda: [
            {"run_id": "run-1", "root_task_id": "root-1", "success": True},
        ],
        get_orchestration_state=lambda: {},
    )
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))

    result = await handle_command("/tasks run-1", session)

    assert result is None
    assert any("Task graph:" in message for message in printed)
    assert any("(root-1)" in message for message in printed)
