from types import SimpleNamespace

import pytest

from src.host.cli.main import _format_task_tree, _parse_orchestrate_args, handle_command


def test_parse_orchestrate_args_supports_worker_flag():
    parsed = _parse_orchestrate_args(["--workers", "3", "implement", "auth", "flow"])

    assert parsed == (3, "implement auth flow")


def test_parse_orchestrate_args_rejects_missing_goal():
    assert _parse_orchestrate_args(["--workers", "2"]) is None
    assert _parse_orchestrate_args([]) is None


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
