from pathlib import Path

import pytest

from src.core.paths import (
    STATE_DIRNAME,
    checkpoints_dir,
    chroma_dir,
    config_path,
    conversations_dir,
    hooks_path,
    history_dir,
    layered_memory_dir,
    local_plugins_dir,
    logs_dir,
    mailboxes_dir,
    memory_path,
    permissions_path,
    persona_path,
    plugins_dir,
    project_root,
    recovery_dir,
    sessions_dir,
    skills_dir,
    state_dir,
    tasks_path,
    theme_path,
    usage_dir,
    workspaces_dir,
)


class TestProjectRoot:
    def test_with_project_dir(self, tmp_path):
        result = project_root(tmp_path)
        assert result == tmp_path.resolve()

    def test_without_project_dir(self):
        result = project_root()
        assert result == Path.cwd().resolve()

    def test_resolves_path(self, tmp_path):
        result = project_root(tmp_path / "nonexistent")
        assert result.is_absolute()


class TestStateDir:
    def test_with_project_dir(self, tmp_path):
        result = state_dir(tmp_path)
        assert result == tmp_path.resolve() / STATE_DIRNAME

    def test_without_project_dir(self):
        result = state_dir()
        expected = Path.cwd().resolve() / STATE_DIRNAME
        assert result == expected


class TestConfigPath:
    def test_with_project_dir(self, tmp_path):
        result = config_path(tmp_path)
        assert result == state_dir(tmp_path) / "config.json"

    def test_without_project_dir(self):
        result = config_path()
        assert result == state_dir() / "config.json"


class TestHooksPath:
    def test_with_project_dir(self, tmp_path):
        result = hooks_path(tmp_path)
        assert result == state_dir(tmp_path) / "hooks.json"

    def test_without_project_dir(self):
        result = hooks_path()
        assert result == state_dir() / "hooks.json"


class TestPermissionsPath:
    def test_with_project_dir(self, tmp_path):
        result = permissions_path(tmp_path)
        assert result == state_dir(tmp_path) / "permissions.json"

    def test_without_project_dir(self):
        result = permissions_path()
        assert result == state_dir() / "permissions.json"


class TestMemoryPath:
    def test_with_project_dir(self, tmp_path):
        result = memory_path(tmp_path)
        assert result == state_dir(tmp_path) / "MEMORY.md"

    def test_without_project_dir(self):
        result = memory_path()
        assert result == state_dir() / "MEMORY.md"


class TestSessionsDir:
    def test_with_project_dir(self, tmp_path):
        result = sessions_dir(tmp_path)
        assert result == state_dir(tmp_path) / "sessions"

    def test_without_project_dir(self):
        result = sessions_dir()
        assert result == state_dir() / "sessions"


class TestConversationsDir:
    def test_with_project_dir(self, tmp_path):
        result = conversations_dir(tmp_path)
        assert result == state_dir(tmp_path) / "conversations"

    def test_without_project_dir(self):
        result = conversations_dir()
        assert result == state_dir() / "conversations"


class TestCheckpointsDir:
    def test_with_project_dir(self, tmp_path):
        result = checkpoints_dir(tmp_path)
        assert result == state_dir(tmp_path) / "checkpoints"

    def test_without_project_dir(self):
        result = checkpoints_dir()
        assert result == state_dir() / "checkpoints"


class TestUsageDir:
    def test_with_project_dir(self, tmp_path):
        result = usage_dir(tmp_path)
        assert result == state_dir(tmp_path) / "usage"

    def test_without_project_dir(self):
        result = usage_dir()
        assert result == state_dir() / "usage"


class TestHistoryDir:
    def test_with_project_dir(self, tmp_path):
        result = history_dir(tmp_path)
        assert result == state_dir(tmp_path) / "history"

    def test_without_project_dir(self):
        result = history_dir()
        assert result == state_dir() / "history"


class TestRecoveryDir:
    def test_with_project_dir(self, tmp_path):
        result = recovery_dir(tmp_path)
        assert result == state_dir(tmp_path) / "recovery"

    def test_without_project_dir(self):
        result = recovery_dir()
        assert result == state_dir() / "recovery"


class TestTasksPath:
    def test_with_project_dir(self, tmp_path):
        result = tasks_path(tmp_path)
        assert result == state_dir(tmp_path) / "tasks.json"

    def test_without_project_dir(self):
        result = tasks_path()
        assert result == state_dir() / "tasks.json"


class TestMailboxesDir:
    def test_with_project_dir(self, tmp_path):
        result = mailboxes_dir(tmp_path)
        assert result == state_dir(tmp_path) / "mailboxes"

    def test_without_project_dir(self):
        result = mailboxes_dir()
        assert result == state_dir() / "mailboxes"


class TestWorkspacesDir:
    def test_with_project_dir(self, tmp_path):
        result = workspaces_dir(tmp_path)
        assert result == state_dir(tmp_path) / "workspaces"

    def test_without_project_dir(self):
        result = workspaces_dir()
        assert result == state_dir() / "workspaces"


class TestThemePath:
    def test_with_project_dir(self, tmp_path):
        result = theme_path(tmp_path)
        assert result == state_dir(tmp_path) / "theme.json"

    def test_without_project_dir(self):
        result = theme_path()
        assert result == state_dir() / "theme.json"


class TestLogsDir:
    def test_with_project_dir(self, tmp_path):
        result = logs_dir(tmp_path)
        assert result == state_dir(tmp_path) / "logs"

    def test_without_project_dir(self):
        result = logs_dir()
        assert result == state_dir() / "logs"


class TestChromaDir:
    def test_with_project_dir(self, tmp_path):
        result = chroma_dir(tmp_path)
        assert result == state_dir(tmp_path) / "chroma_db"

    def test_without_project_dir(self):
        result = chroma_dir()
        assert result == state_dir() / "chroma_db"


class TestPersonaPath:
    def test_with_project_dir(self, tmp_path):
        result = persona_path(tmp_path)
        assert result == state_dir(tmp_path) / "memory.json"

    def test_without_project_dir(self):
        result = persona_path()
        assert result == state_dir() / "memory.json"


class TestLayeredMemoryDir:
    def test_with_project_dir(self, tmp_path):
        result = layered_memory_dir(tmp_path)
        assert result == state_dir(tmp_path) / "memory"

    def test_without_project_dir(self):
        result = layered_memory_dir()
        assert result == state_dir() / "memory"


class TestSkillsDir:
    def test_with_project_dir(self, tmp_path):
        result = skills_dir(tmp_path)
        assert result == state_dir(tmp_path) / "skills"

    def test_without_project_dir(self):
        result = skills_dir()
        assert result == state_dir() / "skills"


class TestPluginsDir:
    def test_with_project_dir(self, tmp_path):
        result = plugins_dir(tmp_path)
        assert result == state_dir(tmp_path) / "plugins"

    def test_without_project_dir(self):
        result = plugins_dir()
        assert result == state_dir() / "plugins"


class TestLocalPluginsDir:
    def test_with_project_dir(self, tmp_path):
        result = local_plugins_dir(tmp_path)
        assert result == state_dir(tmp_path) / "plugins.local"

    def test_without_project_dir(self):
        result = local_plugins_dir()
        assert result == state_dir() / "plugins.local"


class TestPathConsistency:
    def test_all_paths_under_state_dir(self, tmp_path):
        sd = state_dir(tmp_path)
        path_fns = [
            config_path,
            hooks_path,
            permissions_path,
            memory_path,
            sessions_dir,
            conversations_dir,
            checkpoints_dir,
            usage_dir,
            history_dir,
            recovery_dir,
            tasks_path,
            mailboxes_dir,
            workspaces_dir,
            theme_path,
            logs_dir,
            chroma_dir,
            persona_path,
            layered_memory_dir,
            skills_dir,
            plugins_dir,
            local_plugins_dir,
        ]
        for fn in path_fns:
            result = fn(tmp_path)
            assert str(result).startswith(str(sd)), f"{fn.__name__} not under state_dir"

    def test_state_dirname_constant(self):
        assert STATE_DIRNAME == ".agent"

    def test_file_vs_dir_paths(self, tmp_path):
        sd = state_dir(tmp_path)
        file_paths = [
            (config_path, "config.json"),
            (hooks_path, "hooks.json"),
            (permissions_path, "permissions.json"),
            (memory_path, "MEMORY.md"),
            (tasks_path, "tasks.json"),
            (theme_path, "theme.json"),
            (persona_path, "memory.json"),
        ]
        for fn, name in file_paths:
            assert fn(tmp_path) == sd / name

        dir_paths = [
            (sessions_dir, "sessions"),
            (conversations_dir, "conversations"),
            (checkpoints_dir, "checkpoints"),
            (usage_dir, "usage"),
            (history_dir, "history"),
            (recovery_dir, "recovery"),
            (mailboxes_dir, "mailboxes"),
            (workspaces_dir, "workspaces"),
            (logs_dir, "logs"),
            (chroma_dir, "chroma_db"),
            (layered_memory_dir, "memory"),
            (skills_dir, "skills"),
            (plugins_dir, "plugins"),
            (local_plugins_dir, "plugins.local"),
        ]
        for fn, name in dir_paths:
            assert fn(tmp_path) == sd / name
