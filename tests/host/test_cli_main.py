from types import SimpleNamespace

import pytest

from src.core.commands import CommandArgument, CommandDefinition
from src.core.session import SessionManager
from src.host.cli.main import (
    _argv_with_default_start,
    _find_orchestration_run,
    _format_task_tree,
    _handle_bang_command,
    _handle_initial_prompt,
    _handle_pre_session_initial_prompt,
    _parse_custom_command_args,
    _parse_orchestrate_args,
    _parse_resume_args,
    _print_session_list,
    _resolve_resume_session_id,
    _resolve_task_root,
    handle_command,
)


def test_argv_with_default_start_for_bare_cli():
    assert _argv_with_default_start(["doraemon"]) == ["doraemon", "start"]


def test_argv_with_default_start_for_start_options():
    assert _argv_with_default_start(["doraemon", "--prompt", "hi"]) == [
        "doraemon",
        "start",
        "--prompt",
        "hi",
    ]
    assert _argv_with_default_start(["doraemon", "-P", "hi"]) == [
        "doraemon",
        "start",
        "-P",
        "hi",
    ]


def test_argv_with_default_start_preserves_top_level_commands_and_help():
    assert _argv_with_default_start(["doraemon", "start", "--prompt", "hi"]) == [
        "doraemon",
        "start",
        "--prompt",
        "hi",
    ]
    assert _argv_with_default_start(["doraemon", "version"]) == ["doraemon", "version"]
    assert _argv_with_default_start(["doraemon", "--help"]) == ["doraemon", "--help"]


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


def test_parse_custom_command_args_supports_positional_and_flags():
    command = CommandDefinition(
        name="greet",
        description="",
        arguments=[CommandArgument("NAME"), CommandArgument("PUNCT", required=False)],
    )

    parsed = _parse_custom_command_args(command, ["Alice", "--PUNCT", "!"])

    assert parsed == {"NAME": "Alice", "PUNCT": "!"}


def test_parse_custom_command_args_rejects_extra_positionals():
    command = CommandDefinition(name="one", description="", arguments=[CommandArgument("NAME")])

    assert _parse_custom_command_args(command, ["Alice", "extra"]) is None


def test_parse_custom_command_args_supports_arguments_remainder():
    command = CommandDefinition(
        name="review",
        description="",
        arguments=[CommandArgument("ARGUMENTS", required=False)],
    )

    parsed = _parse_custom_command_args(command, ["src/app.py", "tests/test_app.py"])

    assert parsed == {"ARGUMENTS": "src/app.py tests/test_app.py"}


def test_parse_custom_command_args_defaults_to_arguments_for_prompt_commands():
    command = CommandDefinition(name="review", description="")

    parsed = _parse_custom_command_args(command, ["src/app.py", "tests/test_app.py"])

    assert parsed == {"ARGUMENTS": "src/app.py tests/test_app.py"}


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
async def test_handle_bang_command_runs_shell_tool(tmp_path, monkeypatch):
    printed: list[str] = []
    captured: dict[str, object] = {}

    def fake_run(command: str, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return "shell output"

    monkeypatch.setattr("src.servers.run.run", fake_run)
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))

    await _handle_bang_command("!echo hi", tmp_path)

    assert captured == {
        "command": "echo hi",
        "mode": "shell",
        "working_dir": str(tmp_path),
    }
    assert printed == ["shell output"]


@pytest.mark.asyncio
async def test_handle_initial_prompt_routes_slash_command(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    async def fake_handle_command(command: str, session):
        captured["command"] = command
        captured["session"] = session
        return None

    session = SimpleNamespace()
    monkeypatch.setattr("src.host.cli.main.handle_command", fake_handle_command)

    await _handle_initial_prompt("/commands", session, tmp_path)

    assert captured == {"command": "/commands", "session": session}


@pytest.mark.asyncio
async def test_handle_initial_prompt_routes_bang_command(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    async def fake_bang_command(command: str, project_dir):
        captured["command"] = command
        captured["project_dir"] = project_dir

    session = SimpleNamespace()
    monkeypatch.setattr("src.host.cli.main._handle_bang_command", fake_bang_command)

    await _handle_initial_prompt("!echo hi", session, tmp_path)

    assert captured == {"command": "!echo hi", "project_dir": tmp_path}


@pytest.mark.asyncio
async def test_handle_initial_prompt_runs_normal_turn(tmp_path, monkeypatch):
    printed: list[object] = []
    captured: dict[str, str] = {}

    async def turn(prompt: str):
        captured["prompt"] = prompt
        return SimpleNamespace(success=True, response="hello", error=None)

    session = SimpleNamespace(turn=turn)
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))

    await _handle_initial_prompt("hello agent", session, tmp_path)

    assert captured["prompt"] == "hello agent"
    assert printed


@pytest.mark.asyncio
async def test_handle_pre_session_initial_prompt_handles_help(tmp_path, monkeypatch):
    printed: list[object] = []
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))

    handled = await _handle_pre_session_initial_prompt("/help", tmp_path)

    assert handled is True
    assert any("Commands" in str(message) for message in printed)


@pytest.mark.asyncio
async def test_handle_pre_session_initial_prompt_handles_commands(tmp_path, monkeypatch):
    printed: list[str] = []
    cmd_dir = tmp_path / ".agent" / "commands"
    cmd_dir.mkdir(parents=True)
    (cmd_dir / "review.md").write_text(
        "---\nname: Review\ndescription: Review files\n---\nReview $ARGUMENTS",
        encoding="utf-8",
    )
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))

    handled = await _handle_pre_session_initial_prompt("/commands", tmp_path)

    assert handled is True
    assert any("/review" in message for message in printed)


@pytest.mark.asyncio
async def test_handle_pre_session_initial_prompt_defers_agent_commands(tmp_path):
    handled = await _handle_pre_session_initial_prompt("/review src/app.py", tmp_path)

    assert handled is False


@pytest.mark.asyncio
async def test_handle_pre_session_initial_prompt_handles_bang(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    async def fake_bang_command(command: str, project_dir):
        captured["command"] = command
        captured["project_dir"] = project_dir

    monkeypatch.setattr("src.host.cli.main._handle_bang_command", fake_bang_command)

    handled = await _handle_pre_session_initial_prompt("!echo hi", tmp_path)

    assert handled is True
    assert captured == {"command": "!echo hi", "project_dir": tmp_path}


def test_print_session_list_shows_recent_sessions(tmp_path, monkeypatch):
    printed: list[str] = []
    manager = SessionManager(tmp_path / ".agent" / "sessions")
    session = manager.create_session(project="demo", name="Demo Session")
    session.metadata.message_count = 2
    manager.save_session(session)
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))

    _print_session_list(tmp_path, "demo")

    assert any("Recent sessions for demo" in message for message in printed)
    assert any(session.metadata.id in message for message in printed)
    assert any("Demo Session" in message for message in printed)


def test_print_session_list_handles_empty_project(tmp_path, monkeypatch):
    printed: list[str] = []
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))

    _print_session_list(tmp_path, "demo")

    assert any("No sessions found" in message for message in printed)


def test_resolve_resume_session_id_by_id_or_name(tmp_path, monkeypatch):
    printed: list[str] = []
    manager = SessionManager(tmp_path / ".agent" / "sessions")
    session = manager.create_session(project="demo", name="Named Session")
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))

    by_id = _resolve_resume_session_id(
        tmp_path,
        "demo",
        resume=session.metadata.id,
        continue_last=False,
    )
    by_name = _resolve_resume_session_id(
        tmp_path,
        "demo",
        resume="Named Session",
        continue_last=False,
    )

    assert by_id == session.metadata.id
    assert by_name == session.metadata.id


def test_resolve_resume_session_id_name_is_project_scoped(tmp_path, monkeypatch):
    printed: list[str] = []
    manager = SessionManager(tmp_path / ".agent" / "sessions")
    other = manager.create_session(project="other", name="Shared Name")
    current = manager.create_session(project="demo", name="Shared Name")
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))

    resolved = _resolve_resume_session_id(
        tmp_path,
        "demo",
        resume="Shared Name",
        continue_last=False,
    )

    assert resolved == current.metadata.id
    assert resolved != other.metadata.id


def test_resolve_resume_session_id_latest_by_project(tmp_path, monkeypatch):
    printed: list[str] = []
    manager = SessionManager(tmp_path / ".agent" / "sessions")
    manager.create_session(project="other", name="Other")
    latest = manager.create_session(project="demo", name="Latest")
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))

    resolved = _resolve_resume_session_id(
        tmp_path,
        "demo",
        resume=None,
        continue_last=True,
    )

    assert resolved == latest.metadata.id
    assert any("Resuming latest session" in message for message in printed)


def test_resolve_resume_session_id_missing_returns_none(tmp_path, monkeypatch):
    printed: list[str] = []
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))

    resolved = _resolve_resume_session_id(
        tmp_path,
        "demo",
        resume="missing",
        continue_last=False,
    )

    assert resolved is None
    assert any("Session not found" in message for message in printed)


@pytest.mark.asyncio
async def test_handle_command_lists_custom_commands(tmp_path, monkeypatch):
    printed: list[str] = []
    cmd_dir = tmp_path / ".agent" / "commands"
    cmd_dir.mkdir(parents=True)
    (cmd_dir / "greet.md").write_text(
        "---\nname: greet\ndescription: Say hello\narguments:\n  - NAME\n---\nRUN echo hello $NAME\n",
        encoding="utf-8",
    )
    session = SimpleNamespace(project_dir=tmp_path)
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))

    result = await handle_command("/commands", session)

    assert result is None
    assert any("/greet $NAME" in message for message in printed)


@pytest.mark.asyncio
async def test_handle_command_runs_custom_command(tmp_path, monkeypatch):
    printed: list[str] = []
    cmd_dir = tmp_path / ".agent" / "commands"
    cmd_dir.mkdir(parents=True)
    (cmd_dir / "greet.md").write_text(
        "---\nname: greet\ndescription: Say hello\narguments:\n  - NAME\n---\nRUN echo hello $NAME\n",
        encoding="utf-8",
    )
    session = SimpleNamespace(project_dir=tmp_path)
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))

    result = await handle_command("/greet Alice", session)

    assert result is None
    assert any("Command completed" in message for message in printed)
    assert "hello Alice" in printed


@pytest.mark.asyncio
async def test_handle_command_runs_prompt_only_custom_command(tmp_path, monkeypatch):
    printed: list[object] = []
    captured: dict[str, str] = {}
    cmd_dir = tmp_path / ".agent" / "commands"
    cmd_dir.mkdir(parents=True)
    (cmd_dir / "review.md").write_text(
        "---\nname: Review\ndescription: Review files\narguments:\n  - ARGUMENTS (optional)\n---\nReview $ARGUMENTS",
        encoding="utf-8",
    )

    async def turn(prompt: str):
        captured["prompt"] = prompt
        return SimpleNamespace(success=True, response="reviewed", error=None)

    session = SimpleNamespace(project_dir=tmp_path, turn=turn)
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))

    result = await handle_command("/review src/app.py tests", session)

    assert result is None
    assert captured["prompt"] == "Review src/app.py tests"
    assert any(message.__class__.__name__ == "Panel" for message in printed)


@pytest.mark.asyncio
async def test_handle_command_custom_command_help(tmp_path, monkeypatch):
    printed: list[str] = []
    cmd_dir = tmp_path / ".agent" / "commands"
    cmd_dir.mkdir(parents=True)
    (cmd_dir / "greet.md").write_text(
        "---\nname: greet\ndescription: Say hello\narguments:\n  - NAME\n---\nRUN echo hello $NAME\n",
        encoding="utf-8",
    )
    session = SimpleNamespace(project_dir=tmp_path)
    monkeypatch.setattr("src.host.cli.main.console.print", lambda message: printed.append(message))

    result = await handle_command("/commands help greet", session)

    assert result is None
    assert any("Say hello" in message for message in printed)


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
