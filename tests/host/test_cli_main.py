from types import SimpleNamespace

import pytest

from src.host.cli.main import (
    _find_orchestration_run,
    _format_task_tree,
    _parse_orchestrate_args,
    _parse_resume_args,
    _resolve_task_root,
    handle_command,
)


def test_parse_orchestrate_args_supports_worker_flag():
    parsed = _parse_orchestrate_args(["--workers", "3", "implement", "auth", "flow"])

    assert parsed == (3, "implement auth flow")


def test_parse_orchestrate_args_rejects_missing_goal():
    assert _parse_orchestrate_args(["--workers", "2"]) is None
    assert _parse_orchestrate_args([]) is None


def test_parse_orchestrate_args_rejects_invalid_workers():
    assert _parse_orchestrate_args(["--workers", "abc", "goal"]) is None


def test_parse_orchestrate_args_default_workers():
    parsed = _parse_orchestrate_args(["do", "something"])
    assert parsed == (2, "do something")


def test_parse_resume_args_supports_worker_flag():
    parsed = _parse_resume_args(["run-1", "--workers", "3"])

    assert parsed == ("run-1", 3)


def test_parse_resume_args_rejects_invalid_shape():
    assert _parse_resume_args([]) is None
    assert _parse_resume_args(["run-1", "extra"]) is None


def test_parse_resume_args_rejects_invalid_workers():
    assert _parse_resume_args(["run-1", "--workers", "abc"]) is None


def test_parse_resume_args_default_workers():
    parsed = _parse_resume_args(["run-1"])
    assert parsed == ("run-1", 2)


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


def test_format_task_tree_empty_nodes():
    assert _format_task_tree([]) == []


def test_find_orchestration_run_found():
    runs = [
        {"run_id": "run-1", "goal": "first"},
        {"run_id": "run-2", "goal": "second"},
    ]
    session = SimpleNamespace(get_orchestration_runs=lambda: runs)
    result = _find_orchestration_run(session, "run-2")
    assert result["goal"] == "second"


def test_find_orchestration_run_not_found():
    runs = [{"run_id": "run-1"}]
    session = SimpleNamespace(get_orchestration_runs=lambda: runs)
    result = _find_orchestration_run(session, "run-99")
    assert result is None


def test_find_orchestration_run_no_method():
    session = SimpleNamespace()
    result = _find_orchestration_run(session, "run-1")
    assert result is None


def test_resolve_task_root_with_run_id():
    runs = [{"run_id": "run-1", "root_task_id": "root-1"}]
    session = SimpleNamespace(get_orchestration_runs=lambda: runs)
    root, err = _resolve_task_root(session, "run-1")
    assert root == "root-1"
    assert err is None


def test_resolve_task_root_unknown_run_id():
    session = SimpleNamespace(get_orchestration_runs=lambda: [])
    root, err = _resolve_task_root(session, "run-99")
    assert root is None
    assert "Unknown" in err


def test_resolve_task_root_no_run_id():
    session = SimpleNamespace(get_orchestration_state=lambda: {"root_task_id": "root-1"})
    root, err = _resolve_task_root(session, None)
    assert root == "root-1"
    assert err is None


def test_resolve_task_root_no_state():
    session = SimpleNamespace(get_orchestration_state=lambda: None)
    root, err = _resolve_task_root(session, None)
    assert root is None
    assert err is None


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


@pytest.mark.asyncio
async def test_handle_command_help(monkeypatch):
    printed: list[str] = []
    session = SimpleNamespace()
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))
    result = await handle_command("/help", session)
    assert result is None
    assert any("Commands" in str(p) for p in printed)


@pytest.mark.asyncio
async def test_handle_command_exit(monkeypatch):
    session = SimpleNamespace()
    result = await handle_command("/exit", session)
    assert result == "exit"


@pytest.mark.asyncio
async def test_handle_command_quit(monkeypatch):
    session = SimpleNamespace()
    result = await handle_command("/quit", session)
    assert result == "exit"


@pytest.mark.asyncio
async def test_handle_command_unknown(monkeypatch):
    printed: list[str] = []
    session = SimpleNamespace()
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))
    result = await handle_command("/unknown_cmd", session)
    assert result is None
    assert any("Unknown command" in p for p in printed)


@pytest.mark.asyncio
async def test_handle_command_reset(monkeypatch):
    printed: list[str] = []
    session = SimpleNamespace(reset=lambda: None)
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))
    result = await handle_command("/reset", session)
    assert result is None
    assert any("reset" in p.lower() for p in printed)


@pytest.mark.asyncio
async def test_handle_command_trace_with_trace(monkeypatch):
    printed: list[str] = []

    class FakeEvent:
        type = "tool_call"

    trace_obj = SimpleNamespace(session_id="sid", events=[FakeEvent()])
    session = SimpleNamespace(get_trace=lambda: trace_obj)
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))
    result = await handle_command("/trace", session)
    assert result is None
    assert any("sid" in p for p in printed)


@pytest.mark.asyncio
async def test_handle_command_trace_no_trace(monkeypatch):
    printed: list[str] = []
    session = SimpleNamespace(get_trace=lambda: None)
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))
    result = await handle_command("/trace", session)
    assert result is None
    assert any("No trace" in p for p in printed)


@pytest.mark.asyncio
async def test_handle_command_session(monkeypatch):
    printed: list[str] = []
    session = SimpleNamespace(session_id="abc123", project_dir="/tmp/proj")
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))
    result = await handle_command("/session", session)
    assert result is None
    assert any("abc123" in p for p in printed)


@pytest.mark.asyncio
async def test_handle_command_orchestrate_no_args(monkeypatch):
    printed: list[str] = []
    session = SimpleNamespace()
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))
    result = await handle_command("/orchestrate", session)
    assert result is None
    assert any("Usage" in p for p in printed)


@pytest.mark.asyncio
async def test_handle_command_resume_no_args(monkeypatch):
    printed: list[str] = []
    session = SimpleNamespace()
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))
    result = await handle_command("/resume", session)
    assert result is None
    assert any("Usage" in p for p in printed)


@pytest.mark.asyncio
async def test_handle_command_runs_no_method(monkeypatch):
    printed: list[str] = []
    session = SimpleNamespace(get_orchestration_runs=SimpleNamespace())
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))
    result = await handle_command("/runs", session)
    assert result is None
    assert any("not available" in p or "No orchestration" in p for p in printed)


@pytest.mark.asyncio
async def test_handle_command_runs_empty(monkeypatch):
    printed: list[str] = []
    session = SimpleNamespace(
        get_orchestration_runs=lambda: [],
        get_active_orchestration_run_id=lambda: None,
    )
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))
    result = await handle_command("/runs", session)
    assert result is None
    assert any("No orchestration runs" in p for p in printed)


@pytest.mark.asyncio
async def test_handle_command_resume_failure(monkeypatch):
    printed: list[str] = []

    async def orchestrate(goal, **kw):
        raise RuntimeError("resume boom")

    session = SimpleNamespace(orchestrate=orchestrate)
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))
    result = await handle_command("/resume run-1", session)
    assert result is None
    assert any("Resume failed" in p for p in printed)


@pytest.mark.asyncio
async def test_handle_command_tasks_no_task_manager(monkeypatch):
    printed: list[str] = []
    session = SimpleNamespace(get_task_manager=lambda: None)
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))
    result = await handle_command("/tasks", session)
    assert result is None
    assert any("No task manager" in p for p in printed)


@pytest.mark.asyncio
async def test_handle_command_tasks_ready_empty(monkeypatch):
    printed: list[str] = []
    task_manager = SimpleNamespace(list_ready_tasks=lambda: [])
    session = SimpleNamespace(
        get_task_manager=lambda: task_manager,
        get_orchestration_state=lambda: {"root_task_id": "root-1"},
    )
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))
    result = await handle_command("/tasks ready", session)
    assert result is None
    assert any("No ready tasks" in p for p in printed)


@pytest.mark.asyncio
async def test_handle_command_tasks_empty_tree(monkeypatch):
    printed: list[str] = []
    task_manager = SimpleNamespace(
        list_ready_tasks=lambda: [],
        get_task_tree=lambda root=None: [],
    )
    session = SimpleNamespace(
        get_task_manager=lambda: task_manager,
        get_orchestration_state=lambda: {},
    )
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))
    result = await handle_command("/tasks", session)
    assert result is None
    assert any("No tasks" in p for p in printed)


@pytest.mark.asyncio
async def test_handle_command_orchestrate_success(monkeypatch):
    printed: list[str] = []

    async def orchestrate(goal, **kw):
        return SimpleNamespace(
            success=True, summary="done", root_task_id="root-1",
            completed_task_ids=["t1"], failed_task_ids=[],
            worker_assignments={"t1": {"role": "inspect", "worker_session_id": "w1"}},
        )

    session = SimpleNamespace(orchestrate=orchestrate)
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))
    result = await handle_command("/orchestrate --workers 2 do stuff", session)
    assert result is None
    assert any("Summary:" in p for p in printed)


@pytest.mark.asyncio
async def test_handle_command_orchestrate_with_failed_tasks(monkeypatch):
    printed: list[str] = []

    async def orchestrate(goal, **kw):
        return SimpleNamespace(
            success=False, summary="partial", root_task_id="root-1",
            completed_task_ids=[], failed_task_ids=["t1"],
            worker_assignments={},
        )

    session = SimpleNamespace(orchestrate=orchestrate)
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))
    result = await handle_command("/orchestrate goal", session)
    assert result is None
    assert any("Failed:" in p for p in printed)


@pytest.mark.asyncio
async def test_handle_command_mode_invalid(monkeypatch):
    printed: list[str] = []
    session = SimpleNamespace(mode="build")
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))
    await handle_command("/mode invalid", session)
    assert any("Usage" in p or "Current" in p for p in printed)


@pytest.mark.asyncio
async def test_handle_command_clear_no_state(monkeypatch):
    printed: list[str] = []
    session = SimpleNamespace(_state=None)
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))
    result = await handle_command("/clear", session)
    assert result is None
