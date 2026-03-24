"""
Tests for Unified Filesystem Tools

Tests the 3 unified tools (read, write, search) that replace 15 scattered tools.
"""

import os
from unittest.mock import patch

import pytest

from src.servers.filesystem_unified import read, search, write


# Mock validate_path to allow test paths
@pytest.fixture(autouse=True)
def mock_validate_path():
    """Mock validate_path to allow test paths outside workspace"""
    with patch("src.servers.filesystem_unified.validate_path") as mock:
        mock.side_effect = lambda p: p  # Return path as-is
        yield mock


# ========================================
# Test Read Tool
# ========================================


class TestReadTool:
    """Test the unified read tool"""

    def test_read_file_mode(self, tmp_path):
        """Test reading a file"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello World\nLine 2\nLine 3")

        result = read(str(test_file), mode="file")
        assert "Hello World" in result
        assert "Line 2" in result

    def test_read_file_with_offset_limit(self, tmp_path):
        """Test reading file with offset and limit"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Line 1\nLine 2\nLine 3\nLine 4\nLine 5")

        result = read(str(test_file), mode="file", offset=1, limit=2)
        assert "Line 2" in result
        assert "Line 3" in result
        assert "Line 1" not in result
        assert "Line 4" not in result

    def test_read_outline_mode(self, tmp_path):
        """Test reading file outline"""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def function1():
    pass

class MyClass:
    def method1(self):
        pass
""")

        result = read(str(test_file), mode="outline")
        # Should contain structure information
        assert result  # Basic check that it returns something

    def test_read_directory_mode(self, tmp_path):
        """Test listing directory"""
        (tmp_path / "file1.txt").write_text("content1")
        (tmp_path / "file2.txt").write_text("content2")
        (tmp_path / "subdir").mkdir()

        result = read(str(tmp_path), mode="directory")
        assert "file1.txt" in result
        assert "file2.txt" in result
        assert "subdir" in result

    def test_read_directory_show_hidden(self, tmp_path):
        """Test listing directory with hidden files"""
        (tmp_path / "visible.txt").write_text("content")
        (tmp_path / ".hidden").write_text("secret")

        # Without show_hidden
        result = read(str(tmp_path), mode="directory", show_hidden=False)
        assert "visible.txt" in result
        assert ".hidden" not in result

        # With show_hidden
        result = read(str(tmp_path), mode="directory", show_hidden=True)
        assert "visible.txt" in result
        assert ".hidden" in result

    def test_read_tree_mode(self, tmp_path):
        """Test directory tree"""
        (tmp_path / "file1.txt").write_text("content")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "file2.txt").write_text("content")

        result = read(str(tmp_path), mode="tree", depth=2)
        assert "file1.txt" in result
        assert "subdir" in result

    def test_read_nonexistent_file(self, tmp_path):
        """Test reading non-existent file"""
        result = read(str(tmp_path / "nonexistent.txt"), mode="file")
        assert "Error" in result or "not found" in result.lower()

    def test_read_invalid_mode(self, tmp_path):
        """Test invalid mode"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        result = read(str(test_file), mode="invalid")  # type: ignore
        assert "Error" in result
        assert "Invalid mode" in result


# ========================================
# Test Write Tool
# ========================================


class TestWriteTool:
    """Test the unified write tool"""

    def test_write_create_file(self, tmp_path):
        """Test creating a file"""
        test_file = tmp_path / "new.txt"

        result = write(str(test_file), content="Hello World", operation="create")
        assert "Success" in result or "wrote" in result.lower()
        assert test_file.exists()
        assert test_file.read_text() == "Hello World"

    def test_write_create_directory(self, tmp_path):
        """Test creating a directory"""
        test_dir = tmp_path / "newdir"

        result = write(str(test_dir), content=None, operation="create")
        assert "Success" in result or "created" in result.lower()
        assert test_dir.exists()
        assert test_dir.is_dir()

    def test_write_edit_file(self, tmp_path):
        """Test editing a file"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello World")

        result = write(
            str(test_file),
            operation="edit",
            old_string="World",
            new_string="Python",
        )
        assert "Success" in result or "edited" in result.lower()
        assert test_file.read_text() == "Hello Python"

    def test_write_edit_with_count(self, tmp_path):
        """Test editing with replacement count"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("foo foo foo")

        result = write(
            str(test_file),
            operation="edit",
            old_string="foo",
            new_string="bar",
            count=2,
        )
        assert "Success" in result or "edited" in result.lower()
        assert test_file.read_text() == "bar bar foo"

    def test_write_edit_missing_params(self, tmp_path):
        """Test edit without required parameters"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        result = write(str(test_file), operation="edit")
        assert "Error" in result
        assert "old_string" in result or "new_string" in result

    def test_write_delete_file(self, tmp_path):
        """Test deleting a file"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        result = write(str(test_file), operation="delete")
        assert "Success" in result or "deleted" in result.lower()
        assert not test_file.exists()

    def test_write_delete_directory_recursive(self, tmp_path):
        """Test deleting directory recursively"""
        test_dir = tmp_path / "testdir"
        test_dir.mkdir()
        (test_dir / "file.txt").write_text("content")

        result = write(str(test_dir), operation="delete", recursive=True)
        assert "Success" in result or "deleted" in result.lower()
        assert not test_dir.exists()

    def test_write_move_file(self, tmp_path):
        """Test moving/renaming a file"""
        src = tmp_path / "src.txt"
        dst = tmp_path / "dst.txt"
        src.write_text("content")

        result = write(str(src), operation="move", destination=str(dst))
        assert "Success" in result or "moved" in result.lower()
        assert not src.exists()
        assert dst.exists()
        assert dst.read_text() == "content"

    def test_write_move_missing_destination(self, tmp_path):
        """Test move without destination"""
        src = tmp_path / "src.txt"
        src.write_text("content")

        result = write(str(src), operation="move")
        assert "Error" in result
        assert "destination" in result.lower()

    def test_write_copy_file(self, tmp_path):
        """Test copying a file"""
        src = tmp_path / "src.txt"
        dst = tmp_path / "dst.txt"
        src.write_text("content")

        result = write(str(src), operation="copy", destination=str(dst))
        assert "Success" in result or "copied" in result.lower()
        assert src.exists()
        assert dst.exists()
        assert dst.read_text() == "content"

    def test_write_copy_with_overwrite(self, tmp_path):
        """Test copying with overwrite"""
        src = tmp_path / "src.txt"
        dst = tmp_path / "dst.txt"
        src.write_text("new content")
        dst.write_text("old content")

        result = write(str(src), operation="copy", destination=str(dst), overwrite=True)
        assert "Success" in result or "copied" in result.lower()
        assert dst.read_text() == "new content"

    def test_write_invalid_operation(self, tmp_path):
        """Test invalid operation"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        result = write(str(test_file), operation="invalid")  # type: ignore
        assert "Error" in result
        assert "Invalid operation" in result


# ========================================
# Test Search Tool
# ========================================


class TestSearchTool:
    """Test the unified search tool"""

    def test_search_content_mode(self, tmp_path):
        """Test searching file contents"""
        (tmp_path / "file1.py").write_text("def main():\n    pass")
        (tmp_path / "file2.py").write_text("def helper():\n    pass")

        result = search("def main", mode="content", path=str(tmp_path))
        assert "file1.py" in result
        assert "def main" in result

    def test_search_files_mode(self, tmp_path):
        """Test searching file names"""
        (tmp_path / "test1.py").write_text("content")
        (tmp_path / "test2.py").write_text("content")
        (tmp_path / "other.txt").write_text("content")

        # Search from tmp_path
        os.chdir(tmp_path)
        result = search("*.py", mode="files")
        assert "test1.py" in result
        assert "test2.py" in result
        assert "other.txt" not in result

    def test_search_files_with_exclude(self, tmp_path):
        """Test searching files with exclusion"""
        (tmp_path / "include.py").write_text("content")
        (tmp_path / "exclude.py").write_text("content")

        os.chdir(tmp_path)
        result = search("*.py", mode="files", exclude=["*exclude*"])
        assert "include.py" in result
        assert "exclude.py" not in result

    def test_search_symbol_mode(self, tmp_path):
        """Test searching for symbols"""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
class MyClass:
    def method(self):
        pass

def my_function():
    pass
""")

        result = search("MyClass", mode="symbol", path=str(tmp_path))
        # Should find the class definition
        assert result  # Basic check

    def test_search_no_matches(self, tmp_path):
        """Test search with no matches"""
        (tmp_path / "file.txt").write_text("content")

        result = search("nonexistent", mode="content", path=str(tmp_path))
        assert "No matches" in result or "not found" in result.lower()

    def test_search_invalid_mode(self, tmp_path):
        """Test invalid search mode"""
        result = search("query", mode="invalid", path=str(tmp_path))  # type: ignore
        assert "Error" in result
        assert "Invalid mode" in result


# ========================================
# Integration Tests
# ========================================


class TestIntegration:
    """Integration tests for unified tools"""

    def test_create_edit_read_workflow(self, tmp_path):
        """Test complete workflow: create, edit, read"""
        test_file = tmp_path / "workflow.txt"

        # Create
        result = write(str(test_file), content="Initial content", operation="create")
        assert "Success" in result or "wrote" in result.lower()

        # Edit
        result = write(
            str(test_file),
            operation="edit",
            old_string="Initial",
            new_string="Modified",
        )
        assert "Success" in result or "edited" in result.lower()

        # Read
        result = read(str(test_file), mode="file")
        assert "Modified content" in result

    def test_create_search_delete_workflow(self, tmp_path):
        """Test workflow: create, search, delete"""
        test_file = tmp_path / "searchable.py"

        # Create
        write(str(test_file), content="def test_function():\n    pass", operation="create")

        # Search
        result = search("test_function", mode="content", path=str(tmp_path))
        assert "searchable.py" in result

        # Delete
        result = write(str(test_file), operation="delete")
        assert "Success" in result or "deleted" in result.lower()
        assert not test_file.exists()


# ========================================
# Pytest Fixtures
# ========================================


@pytest.fixture
def tmp_path(tmp_path_factory):
    """Create a temporary directory for tests"""
    return tmp_path_factory.mktemp("test_unified_fs")
