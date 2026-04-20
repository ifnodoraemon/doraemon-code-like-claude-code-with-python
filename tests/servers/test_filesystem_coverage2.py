"""Additional coverage tests for servers.filesystem - edit edge cases, create_directory, move, copy, search, notebook ops."""

import json
import os

import pytest

from src.servers.filesystem import (
    _apply_path_edits,
    _copy_path,
    _create_path_directory,
    _delete_path,
    _move_path,
    _read_path_content,
    _replace_path_content,
    _write_path_content,
    multi_edit,
    notebook_edit,
    notebook_read,
    read,
    search,
    write,
)


class TestReadPathContentEdgeCases:
    def test_file_not_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = _read_path_content(str(tmp_path / "nope.txt"))
        assert "not found" in result.lower() or "Error" in result

    def test_read_with_offset_and_limit(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        f = tmp_path / "lines.txt"
        f.write_text("line1\nline2\nline3\nline4\nline5\n")
        result = _read_path_content(str(f), offset=1, limit=2)
        assert "line2" in result
        assert "line3" in result

    def test_read_offset_beyond_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        f = tmp_path / "short.txt"
        f.write_text("only\n")
        result = _read_path_content(str(f), offset=100, limit=10)
        assert "No lines" in result

    def test_read_invalid_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = _read_path_content(str(tmp_path / "nope.txt"))
        assert "not found" in result.lower() or "Error" in result


class TestCreateDirectoryEdge:
    def test_create_existing_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        (tmp_path / "existing").mkdir()
        result = _create_path_directory(str(tmp_path / "existing"))
        assert "Successfully" in result

    def test_create_nested_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = _create_path_directory(str(tmp_path / "a" / "b" / "c"))
        assert "Successfully" in result
        assert (tmp_path / "a" / "b" / "c").is_dir()


class TestMoveEdge:
    def test_move_path_validation_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = _move_path("../../../etc/passwd", str(tmp_path / "dst.txt"))
        assert "Error" in result

    def test_move_creates_destination_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        src = tmp_path / "src.txt"
        src.write_text("data")
        result = _move_path(str(src), str(tmp_path / "subdir" / "dst.txt"))
        assert "Successfully" in result
        assert (tmp_path / "subdir" / "dst.txt").read_text() == "data"


class TestCopyEdge:
    def test_copy_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        src_dir = tmp_path / "srcdir"
        src_dir.mkdir()
        (src_dir / "file.txt").write_text("content")
        result = _copy_path(str(src_dir), str(tmp_path / "dstdir"))
        assert "Successfully" in result
        assert (tmp_path / "dstdir" / "file.txt").read_text() == "content"

    def test_copy_directory_overwrite(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        src_dir = tmp_path / "srcdir"
        src_dir.mkdir()
        (src_dir / "file.txt").write_text("new")
        dst_dir = tmp_path / "dstdir"
        dst_dir.mkdir()
        (dst_dir / "old.txt").write_text("old")
        result = _copy_path(str(src_dir), str(dst_dir), overwrite=True)
        assert "Successfully" in result


class TestDeleteEdge:
    def test_delete_directory_not_recursive(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        d = tmp_path / "mydir"
        d.mkdir()
        result = _delete_path(str(d), recursive=False)
        assert "directory" in result.lower() or "Error" in result

    def test_delete_directory_recursive(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        d = tmp_path / "mydir"
        d.mkdir()
        (d / "f.txt").write_text("x")
        result = _delete_path(str(d), recursive=True)
        assert "Successfully" in result
        assert not d.exists()

    def test_delete_nonexistent(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = _delete_path(str(tmp_path / "nope.txt"))
        assert "not found" in result.lower() or "Error" in result

    def test_delete_path_validation_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = _delete_path("../../../etc/passwd")
        assert "Error" in result


class TestWritePathContentEdge:
    def test_write_creates_parent_dirs(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = _write_path_content(str(tmp_path / "deep" / "nested" / "file.txt"), "hello")
        assert "Successfully" in result
        assert (tmp_path / "deep" / "nested" / "file.txt").read_text() == "hello"

    def test_write_path_validation_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = _write_path_content("../../../etc/evil.txt", "bad")
        assert "Error" in result


class TestApplyPathEditsEdge:
    def test_apply_edits_file_not_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = _apply_path_edits(str(tmp_path / "missing.txt"), [{"old_string": "a", "new_string": "b"}])
        assert "not found" in result.lower() or "Error" in result

    def test_apply_edits_string_not_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        f = tmp_path / "test.txt"
        f.write_text("hello")
        result = _apply_path_edits(str(f), [{"old_string": "missing", "new_string": "b"}])
        assert "not found" in result.lower() or "Error" in result


class TestSearchModes:
    def test_search_content(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        f = tmp_path / "test.py"
        f.write_text("def hello():\n    pass\n")
        result = search("def hello", mode="content", path=str(tmp_path))
        assert "hello" in result

    def test_search_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        (tmp_path / "test.py").write_text("x")
        result = search("*.py", mode="files", path=str(tmp_path))
        assert "test.py" in result

    def test_search_invalid_mode(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = search("query", mode="invalid")
        assert "Invalid" in result


class TestReadModeEdge:
    def test_read_invalid_mode(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = read(str(tmp_path), mode="invalid")
        assert "Invalid" in result


class TestWriteOperationEdge:
    def test_write_invalid_operation(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = write(str(tmp_path / "f.txt"), operation="bad_op")
        assert "Invalid" in result


class TestNotebookRead:
    def test_notebook_not_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = notebook_read(str(tmp_path / "missing.ipynb"))
        assert "not found" in result.lower() or "Error" in result

    def test_notebook_not_ipynb(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        f = tmp_path / "test.txt"
        f.write_text("not a notebook")
        result = notebook_read(str(f))
        assert ".ipynb" in result or "Error" in result

    def test_notebook_read_valid(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        nb = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["print('hello')"],
                    "outputs": [{"output_type": "stream", "text": ["hello\n"]}],
                    "execution_count": 1,
                },
                {
                    "cell_type": "markdown",
                    "source": ["# Title"],
                },
            ],
        }
        f = tmp_path / "test.ipynb"
        f.write_text(json.dumps(nb))
        result = notebook_read(str(f))
        assert "print" in result
        assert "hello" in result
        assert "Title" in result

    def test_notebook_empty(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        f = tmp_path / "empty.ipynb"
        f.write_text(json.dumps({"cells": []}))
        result = notebook_read(str(f))
        assert "empty" in result.lower()

    def test_notebook_invalid_json(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        f = tmp_path / "bad.ipynb"
        f.write_text("not json at all")
        result = notebook_read(str(f))
        assert "Invalid" in result or "Error" in result


class TestNotebookEdit:
    def test_replace_cell(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        nb = {"cells": [{"cell_type": "code", "source": ["old"], "outputs": [], "execution_count": 1}]}
        f = tmp_path / "test.ipynb"
        f.write_text(json.dumps(nb))
        result = notebook_edit(str(f), cell_index=0, new_source="new code")
        assert "Successfully" in result
        data = json.loads(f.read_text())
        assert "new code" in str(data["cells"][0]["source"])

    def test_insert_cell(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        nb = {"cells": [{"cell_type": "code", "source": ["a"], "outputs": [], "execution_count": 1}]}
        f = tmp_path / "test.ipynb"
        f.write_text(json.dumps(nb))
        result = notebook_edit(str(f), cell_index=0, new_source="inserted", operation="insert", cell_type="markdown")
        assert "Successfully" in result

    def test_delete_cell(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        nb = {"cells": [
            {"cell_type": "code", "source": ["a"], "outputs": [], "execution_count": 1},
            {"cell_type": "code", "source": ["b"], "outputs": [], "execution_count": 2},
        ]}
        f = tmp_path / "test.ipynb"
        f.write_text(json.dumps(nb))
        result = notebook_edit(str(f), cell_index=0, new_source="", operation="delete")
        assert "Successfully" in result

    def test_edit_cell_out_of_range(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        nb = {"cells": [{"cell_type": "code", "source": ["a"], "outputs": [], "execution_count": 1}]}
        f = tmp_path / "test.ipynb"
        f.write_text(json.dumps(nb))
        result = notebook_edit(str(f), cell_index=5, new_source="x", operation="replace")
        assert "out of range" in result.lower() or "Error" in result

    def test_edit_not_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = notebook_edit(str(tmp_path / "missing.ipynb"), cell_index=0, new_source="x")
        assert "not found" in result.lower() or "Error" in result

    def test_edit_invalid_operation(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        nb = {"cells": [{"cell_type": "code", "source": ["a"], "outputs": [], "execution_count": 1}]}
        f = tmp_path / "test.ipynb"
        f.write_text(json.dumps(nb))
        result = notebook_edit(str(f), cell_index=0, new_source="x", operation="bad")
        assert "Invalid" in result


class TestMultiEditEdgeCases:
    def test_multi_edit_exception(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        f = tmp_path / "test.txt"
        f.write_text("hello")
        f.chmod(0o444)
        try:
            result = multi_edit(str(f), [{"old_string": "hello", "new_string": "world"}])
        finally:
            f.chmod(0o644)

    def test_multi_edit_non_unique_old_string(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        f = tmp_path / "test.txt"
        f.write_text("dup dup")
        result = multi_edit(str(f), [{"old_string": "dup", "new_string": "one"}])
        assert "2 times" in result or "unique" in result.lower() or "Error" in result
