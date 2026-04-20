"""Targeted coverage tests for servers.filesystem - edit_file, multi_edit, create_directory, move/copy."""

import os
import tempfile

import pytest

from src.servers.filesystem import (
    _apply_path_edits,
    _copy_path,
    _create_path_directory,
    _move_path,
    _replace_path_content,
    _write_path_content,
    multi_edit,
    write,
)


class TestEditFile:
    def test_edit_with_count(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        f = tmp_path / "test.txt"
        f.write_text("aaa bbb aaa bbb aaa")
        result = _replace_path_content(str(f), "aaa", "zzz", count=1)
        assert "1 replacement" in result
        content = f.read_text()
        assert content.startswith("zzz bbb aaa")

    def test_edit_string_not_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        f = tmp_path / "test.txt"
        f.write_text("hello")
        result = _replace_path_content(str(f), "missing", "replacement")
        assert "not found" in result

    def test_edit_file_not_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = _replace_path_content(str(tmp_path / "nope.txt"), "a", "b")
        assert "not found" in result


class TestMultiEdit:
    def test_multi_edit_success(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        f = tmp_path / "test.txt"
        f.write_text("foo bar baz")
        result = multi_edit(str(f), [
            {"old_string": "foo", "new_string": "one"},
            {"old_string": "baz", "new_string": "three"},
        ])
        assert "2 edits" in result
        assert f.read_text() == "one bar three"

    def test_multi_edit_rollback_on_failure(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        f = tmp_path / "test.txt"
        original = "foo bar baz"
        f.write_text(original)
        result = multi_edit(str(f), [
            {"old_string": "foo", "new_string": "one"},
            {"old_string": "missing", "new_string": "two"},
        ])
        assert "Error" in result or "Rolled back" in result
        assert f.read_text() == original

    def test_multi_edit_duplicate_match(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        f = tmp_path / "test.txt"
        f.write_text("dup dup")
        result = multi_edit(str(f), [
            {"old_string": "dup", "new_string": "one"},
        ])
        assert "appears 2 times" in result or "unique" in result

    def test_multi_edit_missing_old_string(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        f = tmp_path / "test.txt"
        f.write_text("hello")
        result = multi_edit(str(f), [{"old_string": "", "new_string": "x"}])
        assert "missing" in result.lower() or "Error" in result

    def test_multi_edit_file_not_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = multi_edit(str(tmp_path / "missing.txt"), [{"old_string": "a", "new_string": "b"}])
        assert "not found" in result.lower() or "Error" in result


class TestCreateDirectory:
    def test_create_directory_success(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = _create_path_directory(str(tmp_path / "newdir" / "sub"))
        assert "Successfully" in result
        assert (tmp_path / "newdir" / "sub").is_dir()

    def test_create_directory_via_write(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = write(str(tmp_path / "mydir"), content=None)
        assert "Successfully" in result
        assert (tmp_path / "mydir").is_dir()


class TestMoveOperation:
    def test_move_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        src = tmp_path / "src.txt"
        src.write_text("data")
        result = _move_path(str(src), str(tmp_path / "dst.txt"))
        assert "Successfully" in result
        assert not src.exists()
        assert (tmp_path / "dst.txt").read_text() == "data"

    def test_move_source_not_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = _move_path(str(tmp_path / "nope.txt"), str(tmp_path / "dst.txt"))
        assert "not found" in result

    def test_move_via_write(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        src = tmp_path / "orig.txt"
        src.write_text("move me")
        result = write(str(src), operation="move", destination=str(tmp_path / "moved.txt"))
        assert "Successfully" in result

    def test_move_missing_destination(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = write(str(tmp_path / "f.txt"), operation="move")
        assert "destination" in result.lower()


class TestCopyOperation:
    def test_copy_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        src = tmp_path / "src.txt"
        src.write_text("copy me")
        result = _copy_path(str(src), str(tmp_path / "copy.txt"))
        assert "Successfully" in result
        assert (tmp_path / "copy.txt").read_text() == "copy me"

    def test_copy_source_not_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = _copy_path(str(tmp_path / "nope.txt"), str(tmp_path / "dst.txt"))
        assert "not found" in result

    def test_copy_dest_exists_no_overwrite(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        src = tmp_path / "src.txt"
        src.write_text("src")
        dst = tmp_path / "dst.txt"
        dst.write_text("existing")
        result = _copy_path(str(src), str(dst), overwrite=False)
        assert "already exists" in result

    def test_copy_overwrite(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        src = tmp_path / "src.txt"
        src.write_text("new")
        dst = tmp_path / "dst.txt"
        dst.write_text("old")
        result = _copy_path(str(src), str(dst), overwrite=True)
        assert "Successfully" in result
        assert dst.read_text() == "new"

    def test_copy_missing_destination(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = write(str(tmp_path / "f.txt"), operation="copy")
        assert "destination" in result.lower()


class TestWriteEditMissingStrings:
    def test_edit_missing_old_or_new(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = write(str(tmp_path / "f.txt"), operation="edit")
        assert "requires" in result.lower()

    def test_invalid_operation(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = write(str(tmp_path / "f.txt"), operation="invalid")
        assert "Invalid" in result


class TestApplyPathEdits:
    def test_missing_keys_in_edit(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        f = tmp_path / "test.txt"
        f.write_text("hello")
        result = _apply_path_edits(str(f), [{"old_string": None, "new_string": "x"}])
        assert "Error" in result or "missing" in result.lower()
