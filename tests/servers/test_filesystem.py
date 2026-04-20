"""
Unit tests for the Unified Filesystem Server.

Tests file reading, directory listing, and code navigation functionality.
"""

import os
import tempfile

import pytest

from src.core.security.security import validate_path
from src.servers.filesystem import (
    _list_path_entries,
    _list_path_tree,
    _read_path_content,
    _read_path_outline,
    glob_files,
    grep_search,
)

# ========================================
# Fixtures
# ========================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory with test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        test_file = os.path.join(tmpdir, "test.txt")
        with open(test_file, "w") as f:
            f.write("Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n")

        # Create a Python file for outline testing
        python_file = os.path.join(tmpdir, "test_module.py")
        with open(python_file, "w") as f:
            f.write('''
class TestClass:
    """A test class."""

    def method_one(self):
        pass

    def method_two(self, arg):
        return arg

def standalone_function():
    """A standalone function."""
    pass
''')

        # Create subdirectory
        subdir = os.path.join(tmpdir, "subdir")
        os.makedirs(subdir)

        sub_file = os.path.join(subdir, "sub.txt")
        with open(sub_file, "w") as f:
            f.write("Subdirectory file content")

        yield tmpdir


# ========================================
# Path Validation Tests
# ========================================


class TestPathValidation:
    """Tests for path validation."""

    def test_validate_path_relative(self, temp_dir):
        """Test validating a relative path."""
        os.chdir(temp_dir)
        result = validate_path("test.txt")
        assert result == os.path.join(temp_dir, "test.txt")

    def test_validate_path_absolute(self, temp_dir):
        """Test validating an absolute path within sandbox."""
        os.chdir(temp_dir)
        abs_path = os.path.join(temp_dir, "test.txt")
        result = validate_path(abs_path)
        assert result == abs_path

    def test_validate_path_traversal_blocked(self, temp_dir):
        """Test that path traversal is blocked."""
        os.chdir(temp_dir)
        with pytest.raises(PermissionError):
            validate_path("../../../etc/passwd")

    def test_validate_path_empty(self):
        """Test that empty path raises error."""
        with pytest.raises(ValueError):
            validate_path("")

    def test_validate_path_whitespace_only(self):
        """Test that whitespace-only path raises error."""
        with pytest.raises(ValueError):
            validate_path("   ")


# ========================================
# Read Path Tests
# ========================================


class TestReadPath:
    """Tests for _read_path_content function."""

    def test_read_path_full(self, temp_dir):
        """Test reading entire file."""
        os.chdir(temp_dir)
        content = _read_path_content("test.txt")
        assert "Line 1" in content
        assert "Line 5" in content

    def test_read_path_with_offset(self, temp_dir):
        """Test reading file with offset."""
        os.chdir(temp_dir)
        content = _read_path_content("test.txt", offset=2, limit=2)
        assert "Line 3" in content
        assert "Line 4" in content
        assert "Line 1" not in content

    def test_read_path_not_found(self, temp_dir):
        """Test reading non-existent file."""
        os.chdir(temp_dir)
        result = _read_path_content("nonexistent.txt")
        assert "Error" in result or "not found" in result.lower()

    def test_read_path_encoding(self, temp_dir):
        """Test reading file with specific encoding."""
        os.chdir(temp_dir)
        content = _read_path_content("test.txt", encoding="utf-8")
        assert "Line 1" in content

    def test_read_path_does_not_truncate_large_file_by_default(self, temp_dir):
        """Full file reads should not be implicitly truncated."""
        os.chdir(temp_dir)
        large_content = "".join(f"Line {i}\n" for i in range(2505))
        with open("large.txt", "w") as handle:
            handle.write(large_content)

        content = _read_path_content("large.txt")

        assert content == large_content


# ========================================
# List Path Tests
# ========================================


class TestListPathEntries:
    """Tests for _list_path_entries function."""

    def test_list_path_entries_basic(self, temp_dir):
        """Test basic directory listing."""
        os.chdir(temp_dir)
        result = _list_path_entries(".")
        assert "test.txt" in result
        assert "test_module.py" in result
        assert "subdir" in result

    def test_list_path_entries_detailed(self, temp_dir):
        """Test detailed directory listing."""
        os.chdir(temp_dir)
        result = _list_path_entries(".", detailed=True)
        assert "[file]" in result or "[dir]" in result

    def test_list_path_entries_not_found(self, temp_dir):
        """Test listing non-existent directory."""
        os.chdir(temp_dir)
        result = _list_path_entries("nonexistent_dir")
        assert "Error" in result or "not found" in result.lower()


# ========================================
# Read Path Outline Tests
# ========================================


class TestReadPathOutline:
    """Tests for _read_path_outline function."""

    def test_read_outline_python(self, temp_dir):
        """Test reading Python file outline."""
        os.chdir(temp_dir)
        result = _read_path_outline("test_module.py")
        assert "TestClass" in result
        assert "method_one" in result
        assert "standalone_function" in result

    def test_read_outline_not_found(self, temp_dir):
        """Test outline of non-existent file."""
        os.chdir(temp_dir)
        result = _read_path_outline("nonexistent.py")
        assert "Error" in result or "not found" in result.lower()


# ========================================
# Glob Files Tests
# ========================================


class TestGlobFiles:
    """Tests for glob_files function."""

    def test_glob_txt_files(self, temp_dir):
        """Test globbing .txt files."""
        os.chdir(temp_dir)
        result = glob_files("**/*.txt")
        assert "test.txt" in result

    def test_glob_py_files(self, temp_dir):
        """Test globbing .py files."""
        os.chdir(temp_dir)
        result = glob_files("*.py")
        assert "test_module.py" in result

    def test_glob_no_matches(self, temp_dir):
        """Test glob with no matches."""
        os.chdir(temp_dir)
        result = glob_files("*.xyz")
        assert "No files found" in result


# ========================================
# Grep Search Tests
# ========================================


class TestGrepSearch:
    """Tests for grep_search function."""

    def test_grep_simple_pattern(self, temp_dir):
        """Test grep with simple pattern."""
        os.chdir(temp_dir)
        result = grep_search("Line", include="*.txt")
        assert "test.txt" in result
        assert "Line" in result

    def test_grep_regex_pattern(self, temp_dir):
        """Test grep with regex pattern."""
        os.chdir(temp_dir)
        result = grep_search(r"Line \d", include="*.txt")
        assert "test.txt" in result

    def test_grep_no_matches(self, temp_dir):
        """Test grep with no matches."""
        os.chdir(temp_dir)
        result = grep_search("nonexistent_pattern_xyz")
        assert "No matches" in result

    def test_grep_search_does_not_limit_results_by_default(self, temp_dir):
        """Content search should return all matches unless a limit is requested."""
        os.chdir(temp_dir)
        with open("many.txt", "w") as handle:
            handle.write("".join(f"match {i}\n" for i in range(120)))

        result = grep_search("match", include="*.txt")

        assert "... (limit reached)" not in result
        assert result.count("many.txt:") == 120


# ========================================
# Directory Tree Tests
# ========================================


class TestDirectoryTree:
    """Tests for _list_path_tree function."""

    def test_directory_tree_basic(self, temp_dir):
        """Test basic directory tree."""
        os.chdir(temp_dir)
        result = _list_path_tree(".", depth=2)
        assert "test.txt" in result
        assert "subdir" in result

    def test_directory_tree_depth_limit(self, temp_dir):
        """Test directory tree respects depth limit."""
        os.chdir(temp_dir)
        result = _list_path_tree(".", depth=1)
        assert "Project Tree" in result


class TestNotebookRead:
    def test_read_notebook(self, tmp_path):
        import json

        from src.servers.filesystem import notebook_read

        nb = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["print('hello')\n"],
                    "outputs": [{"output_type": "stream", "text": ["hello\n"]}],
                },
                {"cell_type": "markdown", "source": ["# Title\n"]},
            ],
            "metadata": {},
            "nbformat": 4,
        }
        nb_path = tmp_path / "test.ipynb"
        nb_path.write_text(json.dumps(nb), encoding="utf-8")

        os.chdir(tmp_path)
        result = notebook_read("test.ipynb")
        assert "Cell 0 (code)" in result
        assert "Cell 1 (markdown)" in result

    def test_read_notebook_not_found(self, tmp_path):
        from src.servers.filesystem import notebook_read

        os.chdir(tmp_path)
        result = notebook_read("missing.ipynb")
        assert "not found" in result.lower()

    def test_read_non_notebook(self, tmp_path):
        from src.servers.filesystem import notebook_read

        (tmp_path / "test.txt").write_text("hi")
        os.chdir(tmp_path)
        result = notebook_read("test.txt")
        assert "must be a Jupyter notebook" in result

    def test_read_notebook_with_error_output(self, tmp_path):
        import json

        from src.servers.filesystem import notebook_read

        nb = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["1/0"],
                    "outputs": [
                        {
                            "output_type": "error",
                            "ename": "ZeroDivisionError",
                            "evalue": "division by zero",
                        }
                    ],
                }
            ],
            "metadata": {},
            "nbformat": 4,
        }
        nb_path = tmp_path / "err.ipynb"
        nb_path.write_text(json.dumps(nb), encoding="utf-8")

        os.chdir(tmp_path)
        result = notebook_read("err.ipynb")
        assert "ZeroDivisionError" in result


class TestNotebookEdit:
    def test_replace_cell(self, tmp_path):
        import json

        from src.servers.filesystem import notebook_edit

        nb = {
            "cells": [{"cell_type": "code", "source": ["old"], "outputs": []}],
            "metadata": {},
            "nbformat": 4,
        }
        nb_path = tmp_path / "edit.ipynb"
        nb_path.write_text(json.dumps(nb), encoding="utf-8")

        os.chdir(tmp_path)
        result = notebook_edit("edit.ipynb", cell_index=0, new_source="new code")
        assert "Successfully" in result

    def test_insert_cell(self, tmp_path):
        import json

        from src.servers.filesystem import notebook_edit

        nb = {"cells": [], "metadata": {}, "nbformat": 4}
        nb_path = tmp_path / "insert.ipynb"
        nb_path.write_text(json.dumps(nb), encoding="utf-8")

        os.chdir(tmp_path)
        result = notebook_edit("insert.ipynb", cell_index=0, new_source="x=1", operation="insert")
        assert "Successfully" in result

    def test_delete_cell(self, tmp_path):
        import json

        from src.servers.filesystem import notebook_edit

        nb = {"cells": [{"cell_type": "code", "source": ["x"]}], "metadata": {}, "nbformat": 4}
        nb_path = tmp_path / "del.ipynb"
        nb_path.write_text(json.dumps(nb), encoding="utf-8")

        os.chdir(tmp_path)
        result = notebook_edit("del.ipynb", cell_index=0, new_source="", operation="delete")
        assert "Successfully" in result

    def test_invalid_operation(self, tmp_path):
        import json

        from src.servers.filesystem import notebook_edit

        nb = {"cells": [{"cell_type": "code", "source": ["x"]}], "metadata": {}, "nbformat": 4}
        nb_path = tmp_path / "inv.ipynb"
        nb_path.write_text(json.dumps(nb), encoding="utf-8")

        os.chdir(tmp_path)
        result = notebook_edit("inv.ipynb", cell_index=0, new_source="", operation="invalid")
        assert "Invalid operation" in result


class TestMultiEdit:
    def test_multi_edit_success(self, tmp_path):
        from src.servers.filesystem import multi_edit

        f = tmp_path / "me.txt"
        f.write_text("alpha beta gamma")

        os.chdir(tmp_path)
        result = multi_edit(
            "me.txt",
            [
                {"old_string": "alpha", "new_string": "ALPHA"},
                {"old_string": "beta", "new_string": "BETA"},
            ],
        )
        assert "Successfully applied 2 edits" in result

    def test_multi_edit_duplicate_old_string(self, tmp_path):
        from src.servers.filesystem import multi_edit

        f = tmp_path / "dup.txt"
        f.write_text("x x y")

        os.chdir(tmp_path)
        result = multi_edit("dup.txt", [{"old_string": "x", "new_string": "z"}])
        assert "appears 2 times" in result

    def test_multi_edit_missing_old_string(self, tmp_path):
        from src.servers.filesystem import multi_edit

        f = tmp_path / "m.txt"
        f.write_text("hello")

        os.chdir(tmp_path)
        result = multi_edit("m.txt", [{"old_string": "", "new_string": "x"}])
        assert "Error" in result


class TestReadPathSpecialFormats:
    def test_read_pdf(self, tmp_path, monkeypatch):
        from src.servers.filesystem import _read_path_content
        from src.servers._services import document

        (tmp_path / "test.pdf").write_bytes(b"%PDF-1.4")
        os.chdir(tmp_path)
        monkeypatch.setattr(document, "parse_pdf", lambda p: "PDF content")
        result = _read_path_content("test.pdf")
        assert result == "PDF content"

    def test_read_docx(self, tmp_path, monkeypatch):
        from src.servers.filesystem import _read_path_content
        from src.servers._services import document

        (tmp_path / "test.docx").write_bytes(b"PK")
        os.chdir(tmp_path)
        monkeypatch.setattr(document, "parse_docx", lambda p: "DOCX content")
        result = _read_path_content("test.docx")
        assert result == "DOCX content"

    def test_read_pptx(self, tmp_path, monkeypatch):
        from src.servers.filesystem import _read_path_content
        from src.servers._services import document

        (tmp_path / "test.pptx").write_bytes(b"PK")
        os.chdir(tmp_path)
        monkeypatch.setattr(document, "parse_pptx", lambda p: "PPTX content")
        result = _read_path_content("test.pptx")
        assert result == "PPTX content"

    def test_read_xlsx(self, tmp_path, monkeypatch):
        from src.servers.filesystem import _read_path_content
        from src.servers._services import document

        (tmp_path / "test.xlsx").write_bytes(b"PK")
        os.chdir(tmp_path)
        monkeypatch.setattr(document, "parse_xlsx", lambda p: "XLSX content")
        result = _read_path_content("test.xlsx")
        assert result == "XLSX content"

    def test_read_image(self, tmp_path, monkeypatch):
        from src.servers.filesystem import _read_path_content
        from src.servers._services import vision

        (tmp_path / "test.png").write_bytes(b"\x89PNG")
        os.chdir(tmp_path)
        monkeypatch.setattr(vision, "process_image", lambda p: "Image: test.png")
        result = _read_path_content("test.png")
        assert result == "Image: test.png"


class TestReadPathOffsetAndLimit:
    def test_read_with_limit(self, temp_dir):
        os.chdir(temp_dir)
        result = _read_path_content("test.txt", offset=0, limit=2)
        assert "Line 1" in result
        assert "Line 2" in result

    def test_read_with_high_offset(self, temp_dir):
        os.chdir(temp_dir)
        result = _read_path_content("test.txt", offset=100, limit=5)
        assert "No lines found" in result

    def test_read_with_limit_zero(self, temp_dir):
        os.chdir(temp_dir)
        result = _read_path_content("test.txt", offset=0, limit=0)
        assert "Line 1" in result


class TestReadPathOutlineInvalid:
    def test_outline_invalid_path(self, tmp_path, monkeypatch):
        from src.servers.filesystem import _read_path_outline

        os.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = _read_path_outline("../../../etc/passwd")
        assert "Error" in result


class TestGlobFilesSecurity:
    def test_glob_absolute_path(self, tmp_path):
        os.chdir(tmp_path)
        result = glob_files("/etc/passwd")
        assert "Error" in result

    def test_glob_traversal(self, tmp_path):
        os.chdir(tmp_path)
        result = glob_files("..")
        assert "Error" in result


class TestGlobFilesExclude:
    def test_glob_with_exclude(self, temp_dir):
        os.chdir(temp_dir)
        with open(os.path.join(temp_dir, "test.py"), "w") as f:
            f.write("x")
        result = glob_files("*.py", exclude=["test.py"])
        assert "test.py" not in result or "No files" in result


class TestGrepSearchWithMaxResults:
    def test_grep_max_results(self, temp_dir):
        os.chdir(temp_dir)
        with open(os.path.join(temp_dir, "many.txt"), "w") as handle:
            handle.write("".join(f"match {i}\n" for i in range(200)))
        result = grep_search("match", include="*.txt", max_results=5)
        lines = [l for l in result.strip().split("\n") if l.strip()]
        assert len(lines) <= 5


class TestHumanSize:
    def test_bytes(self):
        from src.servers.filesystem import _human_size
        assert _human_size(500) == "500.0B"

    def test_kilobytes(self):
        from src.servers.filesystem import _human_size
        assert "KB" in _human_size(2048)

    def test_megabytes(self):
        from src.servers.filesystem import _human_size
        assert "MB" in _human_size(5 * 1024 * 1024)


class TestListPathEntriesNonDetailed:
    def test_non_detailed(self, temp_dir):
        os.chdir(temp_dir)
        result = _list_path_entries(".", detailed=False)
        assert "test.txt" in result


class TestListPathEntriesEmpty:
    def test_empty_dir(self, tmp_path):
        empty = tmp_path / "empty_dir"
        empty.mkdir()
        os.chdir(tmp_path)
        result = _list_path_entries("empty_dir")
        assert "empty" in result.lower() or result.strip() == ""


class TestListPathTreeEmpty:
    def test_empty_tree(self, tmp_path):
        empty = tmp_path / "empty_tree"
        empty.mkdir()
        os.chdir(tmp_path)
        result = _list_path_tree("empty_tree")
        assert "Empty" in result or result.strip() == ""


class TestFindSymbol:
    def test_find_symbol(self, temp_dir, monkeypatch):
        from src.servers.filesystem import find_symbol
        from src.servers._services import code_nav

        os.chdir(temp_dir)
        monkeypatch.setattr(code_nav, "find_definition", lambda p, s: f"Found: {s}")
        result = find_symbol("TestClass", ".")
        assert "Found: TestClass" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
