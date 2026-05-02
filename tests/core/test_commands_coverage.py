"""Targeted coverage tests for core.commands - CommandExecutor.execute, missing args, caching."""

import pytest

from src.core.commands import (
    CommandDefinition,
    CommandExecutor,
    CommandLoader,
    CommandManager,
)


class TestCommandExecutorExecute:
    @pytest.mark.asyncio
    async def test_execute_with_all_step_types(self, tmp_path):
        (tmp_path / "hello.txt").write_text("hello content")
        executor = CommandExecutor(project_dir=tmp_path)
        cmd = CommandDefinition(
            name="multi",
            description="multi step",
            steps=[
                {"type": "run", "command": "echo step1"},
                {"type": "read", "path": "hello.txt"},
                {"type": "unknown"},
            ],
        )
        result = await executor.execute(cmd, {}, use_cache=False)
        assert len(result.step_results) == 3
        assert result.step_results[2]["success"] is False
        assert "Unknown step type" in result.step_results[2].get("error", "")

    @pytest.mark.asyncio
    async def test_execute_step_exception_caught(self, tmp_path):
        executor = CommandExecutor(project_dir=tmp_path)
        cmd = CommandDefinition(
            name="fail",
            description="failing",
            steps=[{"type": "run", "command": "echo ok"}, {"type": "read", "path": "missing.txt"}],
        )
        result = await executor.execute(cmd, {})
        assert result.success is False
        assert len(result.errors) > 0


class TestCommandLoaderParseInvalidYaml:
    def test_invalid_yaml_frontmatter(self, tmp_path):
        cmd_dir = tmp_path / ".agent" / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "bad.md").write_text(
            "---\n: invalid: yaml: [\n---\nRUN echo hi"
        )
        loader = CommandLoader(tmp_path)
        cmd = loader.load_command("bad")
        assert cmd is not None
        assert cmd.name == "bad"


class TestCommandLoaderNoDir:
    def test_commands_dir_missing(self, tmp_path):
        loader = CommandLoader(tmp_path / "nonexistent")
        result = loader.discover_commands()
        assert result == []


class TestCommandManagerRunCommand:
    @pytest.mark.asyncio
    async def test_run_command_success(self, tmp_path):
        cmd_dir = tmp_path / ".agent" / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "greet.md").write_text(
            "---\nname: greet\ndescription: hi\n---\nRUN echo hello"
        )
        mgr = CommandManager(tmp_path)
        result = await mgr.run_command("greet", {})
        assert result.success is True

    def test_list_commands_empty(self, tmp_path):
        mgr = CommandManager(tmp_path)
        result = mgr.list_commands()
        assert result == []


class TestCommandDefinitionToDict:
    def test_to_dict_empty_steps(self):
        cmd = CommandDefinition(name="x", description="d")
        d = cmd.to_dict()
        assert d["steps"] == []
        assert d["arguments"] == []
