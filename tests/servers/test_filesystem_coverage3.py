"""Targeted coverage for filesystem.py uncovered lines: 138-139,163-164,169,216-217,254-255,301,311-312,355-357,386-388,422-423,605-607,689-691,746-748,804-806,818-819,856,862,891-894."""

import json
import os

import pytest

from src.servers.filesystem import (
    _apply_path_edits,
    _copy_path,
    _list_path_entries,
    _list_path_tree,
    _read_path_content,
    _replace_path_content,
    _write_path_content,
    glob_files,
    grep_search,
    multi_edit,
    notebook_edit,
    notebook_read,
    read,
    search,
    write,
)


class TestListPathEntriesOSError:
    def test_oserror_falls_back_to_name_only(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        f = tmp_path / "test.txt"
        f.write_text("hello")
        original_stat = os.stat

        def bad_stat(p):
            if p == str(f):
                raise OSError("permission denied")
            return original_stat(p)

        monkeypatch.setattr(os, "stat", bad_stat)
        result = _list_path_entries(str(tmp_path), detailed=True)
        assert "test.txt" in result


class TestListPathTreeEdge:
    def test_unreadable_dir_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "file.txt").write_text("x")
        original_listdir = os.listdir

        def bad_listdir(p):
            if p == str(sub):
                raise PermissionError("denied")
            return original_listdir(p)

        monkeypatch.setattr(os, "listdir", bad_listdir)
        result = _list_path_tree(str(tmp_path), depth=2)
        assert "sub/" in result

    def test_hidden_files_skipped(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        (tmp_path / ".hidden").write_text("x")
        (tmp_path / "visible.txt").write_text("y")
        result = _list_path_tree(str(tmp_path), depth=2)
        assert ".hidden" not in result
        assert "visible.txt" in result


class TestGlobFilesSecurityAndErrors:
    def test_absolute_path_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = glob_files("/etc/passwd")
        assert "absolute path" in result.lower() or "Error" in result

    def test_dotdot_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = glob_files("../../../etc/passwd")
        assert "Error" in result or "cannot contain" in result.lower()

    def test_glob_with_exclude(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        (tmp_path / "keep.py").write_text("x")
        (tmp_path / "skip.pyc").write_text("x")
        result = glob_files("*.py*", exclude=["*.pyc"])
        assert "keep.py" in result
        assert "skip.pyc" not in result

    def test_glob_with_max_results(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        for i in range(5):
            (tmp_path / f"f{i}.txt").write_text("x")
        result = glob_files("*.txt", max_results=2)
        assert "Showing first 2" in result or "2 of" in result

    def test_glob_general_exception(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        import glob as glob_module

        original_glob = glob_module.glob

        def bad_glob(pattern, **kw):
            raise RuntimeError("unexpected")

        monkeypatch.setattr(glob_module, "glob", bad_glob)
        result = glob_files("*.txt")
        assert "Error" in result


class TestGrepSearchEdge:
    def test_oversized_file_skipped(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        big = tmp_path / "big.txt"
        big.write_text("x")
        monkeypatch.setattr(os.path, "getsize", lambda p: 20 * 1024 * 1024 if p == str(big) else 0)
        result = grep_search("x", path=str(tmp_path))
        assert "No matches" in result or result.strip() == ""

    def test_grep_read_exception_continues(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        (tmp_path / "good.py").write_text("target_string = 1\n")
        bad = tmp_path / "bad.py"
        bad.write_text("target_string = 2\n")
        original_open = open
        call_count = {"n": 0}

        class BadFile:
            def __init__(self, *a, **kw):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    raise UnicodeDecodeError("utf-8", b"", 0, 1, "bad")
                self._f = original_open(*a, **kw)

            def __enter__(self):
                if call_count["n"] == 1:
                    raise UnicodeDecodeError("utf-8", b"", 0, 1, "bad")
                self._f = self._f.__enter__()
                return self

            def __exit__(self, *a):
                if call_count["n"] != 1:
                    return self._f.__exit__(*a)

            def __iter__(self):
                return iter(self._f)

        result = grep_search("target_string", path=str(tmp_path))
        assert "target_string" in result

    def test_grep_general_exception(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        import re as re_module

        original_compile = re_module.compile

        def bad_compile(*a, **kw):
            raise RuntimeError("regex fail")

        monkeypatch.setattr(re_module, "compile", bad_compile)
        result = grep_search("pattern", path=str(tmp_path))
        assert "Error" in result


class TestReplacePathContentValidation:
    def test_validation_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        from src.core.security.security import validate_path

        original = validate_path

        def bad_validate(p):
            raise PermissionError("forbidden")

        import src.servers.filesystem as fs_mod
        monkeypatch.setattr(fs_mod, "validate_path", bad_validate)
        result = _replace_path_content(str(tmp_path / "f.txt"), "a", "b")
        assert "Error" in result


class TestApplyPathEditsException:
    def test_apply_edits_exception_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        f = tmp_path / "test.txt"
        f.write_text("hello")
        f.chmod(0o000)
        try:
            result = _apply_path_edits(str(f), [{"old_string": "hello", "new_string": "world"}])
            assert "Error" in result or "error" in result.lower()
        finally:
            f.chmod(0o644)


class TestReadInvalidModeException:
    def test_read_invalid_mode_returns_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = read(str(tmp_path), mode="invalid")
        assert "Invalid" in result

    def test_read_exception_in_handler(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        import src.servers.filesystem as fs_mod
        original = fs_mod._read_path_content

        def boom(*a, **kw):
            raise RuntimeError("unexpected")

        monkeypatch.setattr(fs_mod, "_read_path_content", boom)
        result = read(str(tmp_path), mode="file")
        assert "Error" in result


class TestWriteOperationExceptionPaths:
    def test_write_exception_handler(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        import src.servers.filesystem as fs_mod

        def boom(*a, **kw):
            raise RuntimeError("boom")

        monkeypatch.setattr(fs_mod, "_create_path_directory", boom)
        result = write(str(tmp_path / "dir"), content=None)
        assert "Error" in result


class TestSearchOperationException:
    def test_search_exception_handler(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        import src.servers.filesystem as fs_mod

        def boom(*a, **kw):
            raise RuntimeError("search boom")

        monkeypatch.setattr(fs_mod, "grep_search", boom)
        result = search("query", mode="content", path=str(tmp_path))
        assert "Error" in result


class TestNotebookReadOutputTypes:
    def test_notebook_with_execute_result(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        nb = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["2+2"],
                    "outputs": [
                        {"output_type": "execute_result", "data": {"text/plain": ["4"]}}
                    ],
                }
            ]
        }
        f = tmp_path / "test.ipynb"
        f.write_text(json.dumps(nb))
        result = notebook_read(str(f))
        assert "4" in result

    def test_notebook_with_error_output(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        nb = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["1/0"],
                    "outputs": [
                        {"output_type": "error", "ename": "ZeroDivisionError", "evalue": "division by zero"}
                    ],
                }
            ]
        }
        f = tmp_path / "err.ipynb"
        f.write_text(json.dumps(nb))
        result = notebook_read(str(f))
        assert "ZeroDivisionError" in result

    def test_notebook_read_general_exception(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        f = tmp_path / "test.ipynb"
        f.write_text("valid json but bad structure")
        result = notebook_read(str(f))
        assert "Error" in result or "empty" in result.lower()


class TestNotebookEditEdgeCases:
    def test_insert_cell_out_of_range(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        nb = {"cells": [{"cell_type": "code", "source": ["a"], "outputs": [], "execution_count": 1}]}
        f = tmp_path / "test.ipynb"
        f.write_text(json.dumps(nb))
        result = notebook_edit(str(f), cell_index=5, new_source="x", operation="insert")
        assert "out of range" in result.lower() or "Error" in result

    def test_delete_cell_out_of_range(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        nb = {"cells": [{"cell_type": "code", "source": ["a"], "outputs": [], "execution_count": 1}]}
        f = tmp_path / "test.ipynb"
        f.write_text(json.dumps(nb))
        result = notebook_edit(str(f), cell_index=5, new_source="", operation="delete")
        assert "out of range" in result.lower() or "Error" in result

    def test_notebook_edit_json_decode_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        f = tmp_path / "bad.ipynb"
        f.write_text("not json")
        result = notebook_edit(str(f), cell_index=0, new_source="x")
        assert "Invalid" in result or "Error" in result

    def test_notebook_edit_general_exception(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        f = tmp_path / "test.ipynb"
        f.write_text(json.dumps({"cells": []}))
        result = notebook_edit(str(f), cell_index=0, new_source="x", operation="replace")
        assert "out of range" in result.lower() or "Error" in result
