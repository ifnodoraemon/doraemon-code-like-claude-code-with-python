"""Tests for src.core.home."""

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

from src.core.home import (
    DEFAULT_USER_SETTINGS,
    Trace,
    TraceEvent,
    append_history,
    backup_file,
    get_agent_dir,
    get_checkpoints_dir,
    get_conversations_dir,
    get_memory_dir,
    get_project_config_path,
    get_project_dir,
    get_project_skills_dir,
    get_traces_dir,
    get_usage_data_dir,
    get_user_cache_dir,
    get_user_skills_dir,
    list_backups,
    list_sessions,
    load_session,
    load_user_settings,
    read_history,
    restore_backup,
    save_session,
    save_user_settings,
    set_project_dir,
)


class TestProjectDir:
    def test_set_and_get(self, tmp_path):
        set_project_dir(tmp_path)
        assert get_project_dir() == tmp_path
        set_project_dir(None)

    def test_defaults_to_cwd(self):
        set_project_dir(None)
        assert get_project_dir() == Path.cwd()

    def test_get_agent_dir(self, tmp_path):
        set_project_dir(tmp_path)
        assert get_agent_dir() == tmp_path / ".agent"
        set_project_dir(None)


class TestUserSettings:
    def test_load_defaults(self, tmp_path):
        with (
            patch("src.core.home.USER_HOME", tmp_path),
            patch("src.core.home.get_user_settings_path", return_value=tmp_path / "settings.json"),
        ):
            settings = load_user_settings()
            assert settings == DEFAULT_USER_SETTINGS

    def test_save_and_load(self, tmp_path):
        settings_path = tmp_path / "settings.json"
        with (
            patch("src.core.home.USER_HOME", tmp_path),
            patch("src.core.home.get_user_settings_path", return_value=settings_path),
            patch("src.core.home._ensure_user_home", return_value=tmp_path),
        ):
            save_user_settings({"mode": "plan"})
            result = load_user_settings()
            assert result["mode"] == "plan"

    def test_load_corrupt_file(self, tmp_path):
        settings_path = tmp_path / "settings.json"
        settings_path.write_text("not json")
        with patch("src.core.home.get_user_settings_path", return_value=settings_path):
            result = load_user_settings()
            assert result == DEFAULT_USER_SETTINGS


class TestHistory:
    def test_append_and_read(self, tmp_path):
        history_path = tmp_path / "history.jsonl"
        with (
            patch("src.core.home.get_history_path", return_value=history_path),
            patch("src.core.home._ensure_user_home"),
        ):
            append_history({"cmd": "test"})
            entries = read_history()
            assert len(entries) == 1
            assert entries[0]["cmd"] == "test"
            assert "timestamp" in entries[0]

    def test_read_nonexistent(self, tmp_path):
        with patch("src.core.home.get_history_path", return_value=tmp_path / "nope.jsonl"):
            assert read_history() == []

    def test_read_with_limit(self, tmp_path):
        history_path = tmp_path / "history.jsonl"
        with (
            patch("src.core.home.get_history_path", return_value=history_path),
            patch("src.core.home._ensure_user_home"),
        ):
            for i in range(5):
                append_history({"cmd": f"cmd{i}"})
            entries = read_history(limit=3)
            assert len(entries) <= 3


class TestTraceEvent:
    def test_to_dict(self):
        e = TraceEvent(type="tool_call", name="read", data={"k": "v"}, timestamp=1.0, duration=0.5)
        d = e.to_dict()
        assert d["type"] == "tool_call"
        assert d["name"] == "read"
        assert d["data"] == {"k": "v"}
        assert d["duration"] == 0.5


class TestTrace:
    def test_start_and_end_turn(self, tmp_path):
        trace = Trace("sess1", project_dir=tmp_path)
        turn_id = trace.start_turn("hello")
        assert turn_id.startswith("sess1")
        trace.end_turn(success=True)
        assert len(trace.events) == 2

    def test_tool_call(self, tmp_path):
        trace = Trace("sess1", project_dir=tmp_path)
        trace.start_turn("hi")
        span = trace.tool_call("read", {"path": "f.py"}, "content", 0.05)
        assert span is not None
        assert trace.events[-1].type == "tool_call"

    def test_llm_call(self, tmp_path):
        trace = Trace("sess1", project_dir=tmp_path)
        trace.start_turn("hi")
        span = trace.llm_call("gemini", [{"role": "user", "content": "hi"}], {}, 100, 50, 1.5)
        assert span is not None
        assert trace.events[-1].type == "llm_call"

    def test_error_event(self, tmp_path):
        trace = Trace("sess1", project_dir=tmp_path)
        eid = trace.error("something broke")
        assert eid is not None
        assert trace.events[-1].type == "error"

    def test_generic_event(self, tmp_path):
        trace = Trace("sess1", project_dir=tmp_path)
        trace.event("custom", "test_event", {"key": "val"})
        assert trace.events[-1].type == "custom"

    def test_save_and_load(self, tmp_path):
        trace = Trace("sess_save", project_dir=tmp_path)
        trace.start_turn("hello")
        trace.tool_call("read", {"path": "f"}, "result", 0.1)
        trace.end_turn()
        path = trace.save()
        assert path.exists()

        loaded = Trace.load("sess_save", project_dir=tmp_path)
        assert loaded is not None
        assert len(loaded.events) == 3

    def test_load_nonexistent(self, tmp_path):
        assert Trace.load("nonexistent", project_dir=tmp_path) is None

    def test_empty_session_id(self, tmp_path):
        trace = Trace("", project_dir=tmp_path)
        assert len(trace.session_id) > 0

    def test_cleanup_old_traces(self, tmp_path):
        traces_dir = tmp_path / ".agent" / "traces"
        traces_dir.mkdir(parents=True)
        old_file = traces_dir / "old.json"
        old_file.write_text("{}")
        import os

        os.utime(old_file, (time.time() - 31 * 86400, time.time() - 31 * 86400))
        Trace._cleanup_old_traces(traces_dir, max_files=200, max_age_days=30)
        assert not old_file.exists()


class TestBackupSystem:
    def test_backup_file(self, tmp_path):
        src = tmp_path / "test.py"
        src.write_text("original")
        with patch("src.core.home.get_checkpoints_dir", return_value=tmp_path / "backups"):
            (tmp_path / "backups").mkdir()
            result = backup_file(src)
            assert result is not None
            assert result.exists()

    def test_backup_nonexistent(self, tmp_path):
        with patch("src.core.home.get_checkpoints_dir", return_value=tmp_path):
            result = backup_file(tmp_path / "nope.py")
            assert result is None

    def test_list_backups(self, tmp_path):
        bak_dir = tmp_path / "backups"
        bak_dir.mkdir()
        (bak_dir / "test.py.20240101_000000.edit.bak").write_text("bak")
        with patch("src.core.home.get_checkpoints_dir", return_value=bak_dir):
            backups = list_backups()
            assert len(backups) >= 1

    def test_restore_backup(self, tmp_path):
        bak_dir = tmp_path / "backups"
        bak_dir.mkdir()
        bak = bak_dir / "test.py.bak"
        bak.write_text("backup content")
        target = tmp_path / "test.py"
        with patch("src.core.home.get_checkpoints_dir", return_value=bak_dir):
            assert restore_backup(bak, target) is True
            assert target.read_text() == "backup content"

    def test_restore_nonexistent_backup(self, tmp_path):
        assert restore_backup(tmp_path / "nope.bak", tmp_path / "target") is False


class TestSessionManagement:
    def test_save_and_load(self, tmp_path):
        conv_dir = tmp_path / "conv"
        conv_dir.mkdir()
        with patch("src.core.home.get_conversations_dir", return_value=conv_dir):
            save_session("s1", {"data": "test"})
            loaded = load_session("s1")
            assert loaded is not None
            assert loaded["data"] == "test"

    def test_load_nonexistent(self, tmp_path):
        conv_dir = tmp_path / "conv"
        conv_dir.mkdir()
        with patch("src.core.home.get_conversations_dir", return_value=conv_dir):
            assert load_session("nope") is None

    def test_list_sessions(self, tmp_path):
        conv_dir = tmp_path / "conv"
        conv_dir.mkdir()
        with patch("src.core.home.get_conversations_dir", return_value=conv_dir):
            save_session("s1", {"data": "a"})
            save_session("s2", {"data": "b"})
            sessions = list_sessions()
            assert len(sessions) >= 2


class TestDirectoryCreation:
    def test_ensure_agent_dir(self, tmp_path):
        set_project_dir(tmp_path)
        agent_dir = get_agent_dir()
        assert agent_dir == tmp_path / ".agent"
        set_project_dir(None)

    def test_ensure_agent_dir_creates_subdirs(self, tmp_path):
        set_project_dir(tmp_path)
        _ = get_conversations_dir()
        assert (tmp_path / ".agent" / "conversations").exists()
        _ = get_checkpoints_dir()
        assert (tmp_path / ".agent" / "checkpoints").exists()
        _ = get_traces_dir()
        assert (tmp_path / ".agent" / "traces").exists()
        _ = get_memory_dir()
        assert (tmp_path / ".agent" / "memory").exists()
        skills_path = get_project_skills_dir()
        assert skills_path == tmp_path / ".agent" / "skills"
        set_project_dir(None)

    def test_get_project_config_path(self, tmp_path):
        set_project_dir(tmp_path)
        path = get_project_config_path()
        assert path == tmp_path / ".agent" / "config.json"
        set_project_dir(None)

    def test_get_user_skills_dir(self, tmp_path):
        with patch("src.core.home.USER_HOME", tmp_path):
            d = get_user_skills_dir()
            assert d == tmp_path / "skills"
            assert d.exists()

    def test_get_user_cache_dir(self, tmp_path):
        with patch("src.core.home.USER_HOME", tmp_path):
            d = get_user_cache_dir()
            assert d == tmp_path / "cache"
            assert d.exists()

    def test_get_usage_data_dir(self, tmp_path):
        with patch("src.core.home.USER_HOME", tmp_path):
            d = get_usage_data_dir()
            assert d == tmp_path / "usage-data"
            assert d.exists()


class TestTraceSaveLoad:
    def test_save_uses_project_dir(self, tmp_path):
        trace = Trace("save_proj", project_dir=tmp_path)
        trace.start_turn("input")
        trace.end_turn()
        path = trace.save()
        assert path.exists()
        assert tmp_path / ".agent" / "traces" in path.parents

    def test_load_corrupt_trace(self, tmp_path):
        traces_dir = tmp_path / ".agent" / "traces"
        traces_dir.mkdir(parents=True)
        bad = traces_dir / "bad.json"
        bad.write_text("not json")
        result = Trace.load("bad", project_dir=tmp_path)
        assert result is None

    def test_trace_start_turn_with_metadata(self, tmp_path):
        trace = Trace("meta", project_dir=tmp_path)
        turn_id = trace.start_turn("input", metadata={"key": "val"})
        assert turn_id is not None
        assert any(e.data.get("key") == "val" for e in trace.events)

    def test_trace_end_turn_without_current(self, tmp_path):
        trace = Trace("no_turn", project_dir=tmp_path)
        trace.end_turn()
        assert len(trace.events) == 0


class TestReadHistoryEdgeCases:
    def test_read_history_with_corrupt_line(self, tmp_path):
        history_path = tmp_path / "history.jsonl"
        history_path.write_text('{"cmd": "good"}\nnot json\n{"cmd": "also_good"}\n')
        with patch("src.core.home.get_history_path", return_value=history_path):
            entries = read_history()
            assert len(entries) == 2

    def test_list_backups_with_path(self, tmp_path):
        bak_dir = tmp_path / "baks"
        bak_dir.mkdir()
        (bak_dir / "main.py.20240101.edit.bak").write_text("bak1")
        (bak_dir / "other.py.20240101.edit.bak").write_text("bak2")
        with patch("src.core.home.get_checkpoints_dir", return_value=bak_dir):
            filtered = list_backups(Path("main.py"))
            assert len(filtered) == 1
            assert "main.py" in str(filtered[0])

    def test_load_session_corrupt(self, tmp_path):
        conv_dir = tmp_path / "conv"
        conv_dir.mkdir()
        (conv_dir / "bad.json").write_text("not json")
        with patch("src.core.home.get_conversations_dir", return_value=conv_dir):
            result = load_session("bad")
            assert result is None
