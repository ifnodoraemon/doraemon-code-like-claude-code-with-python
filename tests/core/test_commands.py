"""Tests for src/core/commands.py"""

import asyncio
from pathlib import Path

import pytest

from src.core.commands import (
    CommandArgument,
    CommandDefinition,
    CommandExecutor,
    CommandLoader,
    CommandManager,
    CommandResult,
)


class TestCommandArgument:
    def test_defaults(self):
        arg = CommandArgument(name="FOO")
        assert arg.required is True
        assert arg.default is None
        assert arg.description == ""

    def test_optional(self):
        arg = CommandArgument(name="BAR", required=False, default="x")
        assert arg.required is False
        assert arg.default == "x"


class TestCommandDefinition:
    def test_to_dict(self):
        cmd = CommandDefinition(
            name="test-cmd",
            description="A test",
            arguments=[CommandArgument(name="A"), CommandArgument(name="B", required=False)],
            steps=[{"type": "run", "command": "echo $A"}],
        )
        d = cmd.to_dict()
        assert d["name"] == "test-cmd"
        assert d["description"] == "A test"
        assert len(d["arguments"]) == 2
        assert d["arguments"][0]["required"] is True
        assert d["arguments"][1]["required"] is False
        assert len(d["steps"]) == 1


class TestCommandResult:
    def test_defaults(self):
        r = CommandResult(command_name="cmd", success=True)
        assert r.outputs == []
        assert r.errors == []
        assert r.duration == 0.0
        assert r.step_results == []


class TestCommandLoader:
    def test_discover_empty(self, tmp_path):
        loader = CommandLoader(tmp_path)
        assert loader.discover_commands() == []

    def test_discover_finds_md_files(self, tmp_path):
        cmd_dir = tmp_path / ".agent" / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "hello.md").write_text("---\nname: hello\n---\nRUN echo hi")
        loader = CommandLoader(tmp_path)
        names = loader.discover_commands()
        assert "hello" in names

    def test_load_command_missing(self, tmp_path):
        loader = CommandLoader(tmp_path)
        assert loader.load_command("nonexistent") is None

    def test_load_command_parses(self, tmp_path):
        cmd_dir = tmp_path / ".agent" / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "greet.md").write_text(
            "---\nname: greet\ndescription: Say hi\narguments:\n  - NAME\n---\nRUN echo $NAME\nREAD README.md"
        )
        loader = CommandLoader(tmp_path)
        cmd = loader.load_command("greet")
        assert cmd is not None
        assert cmd.name == "greet"
        assert cmd.description == "Say hi"
        assert len(cmd.arguments) == 1
        assert cmd.arguments[0].name == "NAME"
        assert len(cmd.steps) == 2
        assert cmd.steps[0]["type"] == "run"
        assert cmd.steps[1]["type"] == "read"

    def test_load_all_commands(self, tmp_path):
        cmd_dir = tmp_path / ".agent" / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "a.md").write_text("---\nname: a\n---\nRUN echo a")
        (cmd_dir / "b.md").write_text("---\nname: b\n---\nRUN echo b")
        loader = CommandLoader(tmp_path)
        result = loader.load_all_commands()
        assert "a" in result
        assert "b" in result

    def test_load_all_caches(self, tmp_path):
        cmd_dir = tmp_path / ".agent" / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "a.md").write_text("---\nname: a\n---\nRUN echo a")
        loader = CommandLoader(tmp_path)
        first = loader.load_all_commands()
        second = loader.load_all_commands()
        assert first is second

    def test_parse_frontmatter_invalid(self):
        loader = CommandLoader()
        meta, body = loader._parse_frontmatter("no frontmatter")
        assert meta == {}
        assert "no frontmatter" in body

    def test_parse_frontmatter_valid(self):
        loader = CommandLoader()
        meta, body = loader._parse_frontmatter("---\nname: test\n---\nbody content")
        assert meta["name"] == "test"
        assert "body content" in body

    def test_parse_steps(self):
        loader = CommandLoader()
        steps = loader._parse_steps("RUN echo hi\nREAD foo.txt\nASYNC sleep 1\nIF $VAR echo $VAR")
        assert len(steps) == 4
        assert steps[0]["type"] == "run"
        assert steps[1]["type"] == "read"
        assert steps[2]["type"] == "async"
        assert steps[3]["type"] == "condition"

    def test_parse_prompt_keeps_non_directive_markdown(self):
        loader = CommandLoader()
        prompt = loader._parse_prompt("# Review\n\nRUN echo ignored\nPlease inspect $ARGUMENTS")

        assert prompt == "# Review\n\nPlease inspect $ARGUMENTS"

    def test_load_prompt_only_command(self, tmp_path):
        cmd_dir = tmp_path / ".agent" / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "review.md").write_text(
            "---\nname: Review\ndescription: Review files\n---\nReview $ARGUMENTS",
            encoding="utf-8",
        )

        command = CommandLoader(tmp_path).load_command("review")

        assert command is not None
        assert command.steps == []
        assert command.prompt == "Review $ARGUMENTS"

    def test_parse_steps_dict_args(self):
        loader = CommandLoader()
        content = "---\nname: x\narguments:\n  - name: FOO\n    required: false\n    default: bar\n---\nRUN echo $FOO"
        cmd = loader._parse_command_file(content, Path("x.md"))
        assert cmd.arguments[0].name == "FOO"
        assert cmd.arguments[0].required is False
        assert cmd.arguments[0].default == "bar"

    def test_get_command(self, tmp_path):
        cmd_dir = tmp_path / ".agent" / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "hi.md").write_text("---\nname: hi\n---\nRUN echo hi")
        loader = CommandLoader(tmp_path)
        cmd = loader.get_command("hi")
        assert cmd is not None
        assert cmd.name == "hi"

    def test_list_commands(self, tmp_path):
        cmd_dir = tmp_path / ".agent" / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "hi.md").write_text("---\nname: hi\n---\nRUN echo hi")
        loader = CommandLoader(tmp_path)
        cmds = loader.list_commands()
        assert len(cmds) >= 1


class TestCommandExecutor:
    def test_validate_arguments_missing_required(self):
        executor = CommandExecutor()
        cmd = CommandDefinition(
            name="test", description="", arguments=[CommandArgument(name="REQ")]
        )
        err = executor._validate_arguments(cmd, {})
        assert "REQ" in err

    def test_validate_arguments_ok(self):
        executor = CommandExecutor()
        cmd = CommandDefinition(
            name="test", description="", arguments=[CommandArgument(name="REQ")]
        )
        err = executor._validate_arguments(cmd, {"REQ": "val"})
        assert err is None

    def test_validate_arguments_optional(self):
        executor = CommandExecutor()
        cmd = CommandDefinition(
            name="test", description="", arguments=[CommandArgument(name="OPT", required=False)]
        )
        err = executor._validate_arguments(cmd, {})
        assert err is None

    def test_prepare_arguments(self):
        executor = CommandExecutor()
        cmd = CommandDefinition(
            name="test",
            description="",
            arguments=[
                CommandArgument(name="A"),
                CommandArgument(name="B", default="def"),
            ],
        )
        args = executor._prepare_arguments(cmd, {"A": "val_a"})
        assert args["A"] == "val_a"
        assert args["B"] == "def"

    def test_substitute(self):
        executor = CommandExecutor()
        result = executor._substitute("echo $NAME in ${DIR}", {"NAME": "world", "DIR": "/tmp"})
        assert result == "echo world in /tmp"

    def test_render_prompt_uses_unquoted_arguments(self, tmp_path):
        executor = CommandExecutor(project_dir=tmp_path)
        command = CommandDefinition(
            name="review",
            description="",
            arguments=[CommandArgument("ARGUMENTS", required=False)],
            prompt="Review $ARGUMENTS in $PROJECT_DIR",
        )

        prompt, error = executor.render_prompt(command, {"ARGUMENTS": "src/app.py tests"})

        assert error is None
        assert prompt == f"Review src/app.py tests in {tmp_path}"

    @pytest.mark.asyncio
    async def test_execute_missing_required_arg(self, tmp_path):
        executor = CommandExecutor(project_dir=tmp_path)
        cmd = CommandDefinition(
            name="test", description="", arguments=[CommandArgument(name="REQ")], steps=[]
        )
        validation_error = executor._validate_arguments(cmd, {})
        assert validation_error is not None
        assert "REQ" in validation_error

    @pytest.mark.asyncio
    async def test_run_step_success(self, tmp_path):
        executor = CommandExecutor(project_dir=tmp_path)
        step = {"type": "run", "command": "echo hello"}
        result = await executor._execute_run(step, {}, True)
        assert result["success"] is True
        assert "hello" in result["output"]

    @pytest.mark.asyncio
    async def test_run_step_failure(self, tmp_path):
        executor = CommandExecutor(project_dir=tmp_path)
        step = {"type": "run", "command": "false"}
        result = await executor._execute_run(step, {}, True)
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_read_step_missing_file(self, tmp_path):
        executor = CommandExecutor(project_dir=tmp_path)
        step = {"type": "read", "path": "missing.txt"}
        result = await executor._execute_read(step, {})
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_read_step_existing_file(self, tmp_path):
        (tmp_path / "hello.txt").write_text("content here")
        executor = CommandExecutor(project_dir=tmp_path)
        step = {"type": "read", "path": "hello.txt"}
        result = await executor._execute_read(step, {})
        assert result["success"] is True
        assert "content here" in result["output"]

    @pytest.mark.asyncio
    async def test_condition_met(self, tmp_path):
        executor = CommandExecutor(project_dir=tmp_path)
        step = {"type": "condition", "variable": "FLAG", "command": "echo yes"}
        result = await executor._execute_condition(step, {"FLAG": "1"})
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_condition_not_met(self, tmp_path):
        executor = CommandExecutor(project_dir=tmp_path)
        step = {"type": "condition", "variable": "FLAG", "command": "echo yes"}
        result = await executor._execute_condition(step, {})
        assert "Skipped" in result["output"]

    @pytest.mark.asyncio
    async def test_async_step(self, tmp_path):
        executor = CommandExecutor(project_dir=tmp_path)
        step = {"type": "async", "command": "echo async"}
        result = await executor._execute_async(step, {})
        assert result["success"] is True
        executor.terminate_background_processes()

    @pytest.mark.asyncio
    async def test_async_step_tracks_and_terminates_background_processes(self, tmp_path):
        executor = CommandExecutor(project_dir=tmp_path)
        step = {"type": "async", "command": "sleep 10"}
        result = await executor._execute_async(step, {})
        assert result["success"] is True
        assert len(executor._background_processes) == 1

        executor.terminate_background_processes()
        assert executor._background_processes == []

    @pytest.mark.asyncio
    async def test_unknown_step(self, tmp_path):
        executor = CommandExecutor(project_dir=tmp_path)
        step = {"type": "unknown"}
        result = await executor._execute_step(step, {}, True)
        assert result["success"] is False

    def test_clear_cache(self, tmp_path):
        executor = CommandExecutor(project_dir=tmp_path)
        executor._cache["key"] = "val"
        executor.clear_cache()
        assert executor._cache == {}


class TestCommandManager:
    def test_list_commands(self, tmp_path):
        cmd_dir = tmp_path / ".agent" / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "hi.md").write_text(
            "---\nname: hi\ndescription: greet\narguments:\n  - NAME\n---\nRUN echo $NAME"
        )
        mgr = CommandManager(tmp_path)
        cmds = mgr.list_commands()
        assert len(cmds) >= 1
        assert cmds[0]["name"] == "hi"
        assert "NAME" in cmds[0]["arguments"]

    def test_list_commands_uses_file_stem_as_invocation_name(self, tmp_path):
        cmd_dir = tmp_path / ".agent" / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "fetch-issue.md").write_text(
            "---\nname: Fetch Issue Context\ndescription: fetch\n---\nRUN echo ok",
            encoding="utf-8",
        )
        mgr = CommandManager(tmp_path)

        cmds = mgr.list_commands()
        help_text = mgr.get_command_help("fetch-issue")

        assert cmds[0]["name"] == "fetch-issue"
        assert cmds[0]["title"] == "Fetch Issue Context"
        assert help_text.startswith("/fetch-issue")

    @pytest.mark.asyncio
    async def test_run_command_missing(self, tmp_path):
        mgr = CommandManager(tmp_path)
        result = await mgr.run_command("missing")
        assert result.success is False

    def test_get_command_help(self, tmp_path):
        cmd_dir = tmp_path / ".agent" / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "hi.md").write_text(
            "---\nname: hi\ndescription: greet\narguments:\n  - NAME\n---\nRUN echo $NAME"
        )
        mgr = CommandManager(tmp_path)
        help_text = mgr.get_command_help("hi")
        assert help_text is not None
        assert "hi" in help_text
        assert "NAME" in help_text

    def test_get_command_help_missing(self, tmp_path):
        mgr = CommandManager(tmp_path)
        assert mgr.get_command_help("nope") is None

    def test_create_command_template(self, tmp_path):
        mgr = CommandManager(tmp_path)
        path = mgr.create_command_template("mycmd", description="desc", arguments=["ARG1"])
        assert path.exists()
        content = path.read_text()
        assert "mycmd" in content
        assert "ARG1" in content

    def test_create_command_template_no_args(self, tmp_path):
        mgr = CommandManager(tmp_path)
        path = mgr.create_command_template("simple")
        assert path.exists()

    @pytest.mark.asyncio
    async def test_execute_with_missing_required_arg(self, tmp_path):
        cmd_dir = tmp_path / ".agent" / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "need.md").write_text(
            "---\nname: need\narguments:\n  - REQ\n---\nRUN echo $REQ"
        )
        mgr = CommandManager(tmp_path)
        result = await mgr.run_command("need", {})
        assert result.success is False
        assert any("REQ" in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_execute_with_default_arg(self, tmp_path):
        cmd_dir = tmp_path / ".agent" / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "defcmd.md").write_text(
            "---\nname: defcmd\narguments:\n  - name: OPT\n    required: false\n    default: hello\n---\nRUN echo $OPT"
        )
        mgr = CommandManager(tmp_path)
        result = await mgr.run_command("defcmd", {})
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_run_step_cached(self, tmp_path):
        executor = CommandExecutor(project_dir=tmp_path)
        step = {"type": "run", "command": "echo cached"}
        first = await executor._execute_run(step, {}, True)
        assert first.get("cached") is None
        second = await executor._execute_run(step, {}, True)
        assert second.get("cached") is True

    @pytest.mark.asyncio
    async def test_execute_run_step_timeout(self, tmp_path):
        executor = CommandExecutor(project_dir=tmp_path, timeout=0.1)
        step = {"type": "run", "command": "sleep 10"}
        result = await executor._execute_run(step, {}, False)
        assert result["success"] is False
        assert "timed out" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_run_step_timeout_terminates_process_group(self, tmp_path):
        executor = CommandExecutor(project_dir=tmp_path, timeout=0.05)
        step = {
            "type": "run",
            "command": "sh -c 'sleep 0.3; echo late > late.txt'",
        }
        result = await executor._execute_run(step, {}, False)
        await asyncio.sleep(0.5)

        assert result["success"] is False
        assert not (tmp_path / "late.txt").exists()

    @pytest.mark.asyncio
    async def test_execute_run_step_exception(self, tmp_path):
        executor = CommandExecutor(project_dir=tmp_path)
        step = {"type": "run", "command": "echo hi"}
        result = await executor._execute_run(step, {}, False)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_read_step_exception(self, tmp_path):
        executor = CommandExecutor(project_dir=tmp_path)
        step = {"type": "read", "path": "nonexistent.txt"}
        result = await executor._execute_read(step, {})
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_execute_async_step_exception(self, tmp_path):
        executor = CommandExecutor(project_dir=tmp_path)
        step = {"type": "async", "command": "echo async_test"}
        result = await executor._execute_async(step, {})
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_full_command_workflow(self, tmp_path):
        cmd_dir = tmp_path / ".agent" / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "flow.md").write_text(
            "---\nname: flow\ndescription: test flow\narguments:\n  - NAME\n---\nRUN echo hello $NAME\nIF $EXTRA echo extra"
        )
        mgr = CommandManager(tmp_path)
        result = await mgr.run_command("flow", {"NAME": "world"})
        assert result.success is True
        assert len(result.outputs) >= 1

    @pytest.mark.asyncio
    async def test_execute_full_command_missing_command(self, tmp_path):
        mgr = CommandManager(tmp_path)
        result = await mgr.run_command("nope")
        assert result.success is False
        assert "not found" in result.errors[0]

    def test_get_command_help_with_default_arg(self, tmp_path):
        cmd_dir = tmp_path / ".agent" / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "helpcmd.md").write_text(
            "---\nname: helpcmd\ndescription: test\narguments:\n  - name: OPT\n    required: false\n    default: defval\n---\nRUN echo"
        )
        mgr = CommandManager(tmp_path)
        help_text = mgr.get_command_help("helpcmd")
        assert "OPT" in help_text
        assert "defval" in help_text

    @pytest.mark.asyncio
    async def test_run_command_with_empty_args(self, tmp_path):
        cmd_dir = tmp_path / ".agent" / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "noargs.md").write_text("---\nname: noargs\ndescription: no args\n---\nRUN echo hi")
        mgr = CommandManager(tmp_path)
        result = await mgr.run_command("noargs")
        assert result.success is True
