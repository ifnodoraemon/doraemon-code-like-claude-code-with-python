"""
Comprehensive unit tests for the Unified Filesystem Server.

Tests all major functions with mocking: read_file, write_file, list_directory,
create_directory, delete_file, move_file, copy_file, edit_file, and more.

Includes 50+ tests covering success cases, error cases, and edge cases.
"""

import os
import tempfile
from unittest.mock import mock_open, patch

import pytest

from src.servers.filesystem_unified import (
    _human_size,
    copy_file,
    create_directory,
    delete_file,
    edit_file,
    edit_file_multiline,
    find_symbol,
    glob_files,
    grep_search,
    list_directory,
    list_directory_tree,
    move_file,
    read_file,
    read_file_outline,
    rename_file,
    write_file,
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
# Human Size Tests
# ========================================


class TestHumanSize:
    """Tests for _human_size helper function."""

    def test_human_size_bytes(self):
        """Test size in bytes."""
        assert _human_size(512) == "512.0B"

    def test_human_size_kilobytes(self):
        """Test size in kilobytes."""
        assert _human_size(1024) == "1.0KB"

    def test_human_size_megabytes(self):
        """Test size in megabytes."""
        assert _human_size(1024 * 1024) == "1.0MB"

    def test_human_size_gigabytes(self):
        """Test size in gigabytes."""
        assert _human_size(1024 * 1024 * 1024) == "1.0GB"

    def test_human_size_terabytes(self):
        """Test size in terabytes."""
        assert _human_size(1024 * 1024 * 1024 * 1024) == "1.0TB"

    def test_human_size_petabytes(self):
        """Test size in petabytes."""
        assert _human_size(1024 * 1024 * 1024 * 1024 * 1024) == "1.0PB"

    def test_human_size_zero(self):
        """Test zero bytes."""
        assert _human_size(0) == "0.0B"


# ========================================
# Read File Tests
# ========================================


class TestReadFile:
    """Tests for read_file function."""

    def test_read_file_full(self, temp_dir):
        """Test reading entire file."""
        os.chdir(temp_dir)
        content = read_file("test.txt")
        assert "Line 1" in content
        assert "Line 5" in content

    def test_read_file_with_offset(self, temp_dir):
        """Test reading file with offset."""
        os.chdir(temp_dir)
        content = read_file("test.txt", offset=2, limit=2)
        assert "Line 3" in content
        assert "Line 4" in content
        assert "Line 1" not in content

    def test_read_file_with_offset_only(self, temp_dir):
        """Test reading file with offset but no limit."""
        os.chdir(temp_dir)
        content = read_file("test.txt", offset=3)
        assert "Line 4" in content
        assert "Line 5" in content

    def test_read_file_not_found(self, temp_dir):
        """Test reading non-existent file."""
        os.chdir(temp_dir)
        result = read_file("nonexistent.txt")
        assert "Error" in result or "not found" in result.lower()

    def test_read_file_encoding(self, temp_dir):
        """Test reading file with specific encoding."""
        os.chdir(temp_dir)
        content = read_file("test.txt", encoding="utf-8")
        assert "Line 1" in content

    def test_read_file_offset_beyond_file(self, temp_dir):
        """Test reading with offset beyond file length."""
        os.chdir(temp_dir)
        result = read_file("test.txt", offset=100, limit=10)
        assert "No lines found" in result

    def test_read_file_limit_zero(self, temp_dir):
        """Test reading with limit of zero (reads all lines from offset)."""
        os.chdir(temp_dir)
        result = read_file("test.txt", offset=0, limit=0)
        # When limit is 0, it's treated as None, so all lines are read
        assert "Line 1" in result
        assert "Line 5" in result

    @patch("builtins.open", new_callable=mock_open, read_data="test content")
    @patch("os.path.exists")
    def test_read_file_with_mock(self, mock_exists, mock_file):
        """Test read_file with mocked file operations."""
        mock_exists.return_value = True
        result = read_file("test.txt")
        assert "test content" in result

    @patch("builtins.open", side_effect=OSError("Permission denied"))
    @patch("os.path.exists")
    def test_read_file_permission_error(self, mock_exists, mock_file):
        """Test read_file with permission error."""
        mock_exists.return_value = True
        result = read_file("test.txt")
        assert "Error" in result


# ========================================
# Write File Tests
# ========================================


class TestWriteFile:
    """Tests for write_file function."""

    def test_write_file_success(self, temp_dir):
        """Test successful file write."""
        os.chdir(temp_dir)
        result = write_file("new_file.txt", "test content")
        assert "Successfully wrote" in result
        assert os.path.exists(os.path.join(temp_dir, "new_file.txt"))

    def test_write_file_creates_parent_dirs(self, temp_dir):
        """Test that write_file creates parent directories."""
        os.chdir(temp_dir)
        result = write_file("new_dir/subdir/file.txt", "content")
        assert "Successfully wrote" in result
        assert os.path.exists(os.path.join(temp_dir, "new_dir/subdir/file.txt"))

    def test_write_file_overwrites_existing(self, temp_dir):
        """Test that write_file overwrites existing files."""
        os.chdir(temp_dir)
        write_file("test.txt", "new content")
        with open(os.path.join(temp_dir, "test.txt")) as f:
            content = f.read()
        assert content == "new content"

    def test_write_file_empty_content(self, temp_dir):
        """Test writing empty content."""
        os.chdir(temp_dir)
        result = write_file("empty.txt", "")
        assert "Successfully wrote" in result

    @patch("builtins.open", side_effect=OSError("Disk full"))
    @patch("os.makedirs")
    def test_write_file_io_error(self, mock_makedirs, mock_file):
        """Test write_file with IO error."""
        result = write_file("test.txt", "content")
        assert "Error" in result

    def test_write_file_large_content(self, temp_dir):
        """Test writing large content."""
        os.chdir(temp_dir)
        large_content = "x" * 1000000
        result = write_file("large.txt", large_content)
        assert "Successfully wrote" in result


# ========================================
# List Directory Tests
# ========================================


class TestListDirectory:
    """Tests for list_directory function."""

    def test_list_directory_basic(self, temp_dir):
        """Test basic directory listing."""
        os.chdir(temp_dir)
        result = list_directory(".")
        assert "test.txt" in result
        assert "test_module.py" in result
        assert "subdir" in result

    def test_list_directory_detailed(self, temp_dir):
        """Test detailed directory listing."""
        os.chdir(temp_dir)
        result = list_directory(".", detailed=True)
        assert "[file]" in result or "[dir]" in result

    def test_list_directory_not_detailed(self, temp_dir):
        """Test non-detailed directory listing."""
        os.chdir(temp_dir)
        result = list_directory(".", detailed=False)
        assert "test.txt" in result

    def test_list_directory_not_found(self, temp_dir):
        """Test listing non-existent directory."""
        os.chdir(temp_dir)
        result = list_directory("nonexistent_dir")
        assert "Error" in result or "not found" in result.lower()

    def test_list_directory_empty(self, temp_dir):
        """Test listing empty directory."""
        os.chdir(temp_dir)
        empty_dir = os.path.join(temp_dir, "empty")
        os.makedirs(empty_dir)
        result = list_directory("empty")
        assert "(empty directory)" in result

    def test_list_directory_hide_hidden_files(self, temp_dir):
        """Test hiding hidden files."""
        os.chdir(temp_dir)
        hidden_file = os.path.join(temp_dir, ".hidden")
        with open(hidden_file, "w") as f:
            f.write("hidden")
        result = list_directory(".", show_hidden=False)
        assert ".hidden" not in result

    def test_list_directory_show_hidden_files(self, temp_dir):
        """Test showing hidden files."""
        os.chdir(temp_dir)
        hidden_file = os.path.join(temp_dir, ".hidden")
        with open(hidden_file, "w") as f:
            f.write("hidden")
        result = list_directory(".", show_hidden=True)
        assert ".hidden" in result

    @patch("os.listdir", side_effect=OSError("Permission denied"))
    def test_list_directory_permission_error(self, mock_listdir):
        """Test list_directory with permission error."""
        result = list_directory(".")
        assert "Error" in result


# ========================================
# Edit File Tests
# ========================================


class TestEditFile:
    """Tests for edit_file function."""

    def test_edit_file_single_replacement(self, temp_dir):
        """Test single replacement in file."""
        os.chdir(temp_dir)
        result = edit_file("test.txt", "Line 1", "Modified Line 1", count=1)
        assert "Successfully edited" in result
        with open(os.path.join(temp_dir, "test.txt")) as f:
            content = f.read()
        assert "Modified Line 1" in content

    def test_edit_file_all_replacements(self, temp_dir):
        """Test replacing all occurrences."""
        os.chdir(temp_dir)
        result = edit_file("test.txt", "Line", "Modified", count=-1)
        assert "Successfully edited" in result
        with open(os.path.join(temp_dir, "test.txt")) as f:
            content = f.read()
        assert "Modified" in content
        assert content.count("Modified") == 5

    def test_edit_file_not_found(self, temp_dir):
        """Test editing non-existent file."""
        os.chdir(temp_dir)
        result = edit_file("nonexistent.txt", "old", "new")
        assert "Error" in result or "not found" in result.lower()

    def test_edit_file_search_string_not_found(self, temp_dir):
        """Test editing with search string not found."""
        os.chdir(temp_dir)
        result = edit_file("test.txt", "nonexistent", "new")
        assert "Error" in result or "not found" in result.lower()

    def test_edit_file_multiple_replacements(self, temp_dir):
        """Test replacing multiple occurrences."""
        os.chdir(temp_dir)
        result = edit_file("test.txt", "Line", "Modified", count=3)
        assert "Successfully edited" in result

    @patch("builtins.open", side_effect=OSError("Permission denied"))
    def test_edit_file_io_error(self, mock_file):
        """Test edit_file with IO error."""
        result = edit_file("test.txt", "old", "new")
        assert "Error" in result


# ========================================
# Edit File Multiline Tests
# ========================================


class TestEditFileMultiline:
    """Tests for edit_file_multiline function."""

    def test_edit_file_multiline_success(self, temp_dir):
        """Test multiple edits in sequence."""
        os.chdir(temp_dir)
        edits = [
            {"old_string": "Line 1", "new_string": "Modified 1"},
            {"old_string": "Line 2", "new_string": "Modified 2"},
        ]
        result = edit_file_multiline("test.txt", edits)
        assert "Successfully applied" in result
        with open(os.path.join(temp_dir, "test.txt")) as f:
            content = f.read()
        assert "Modified 1" in content
        assert "Modified 2" in content

    def test_edit_file_multiline_not_found(self, temp_dir):
        """Test multiline edit on non-existent file."""
        os.chdir(temp_dir)
        edits = [{"old_string": "old", "new_string": "new"}]
        result = edit_file_multiline("nonexistent.txt", edits)
        assert "Error" in result or "not found" in result.lower()

    def test_edit_file_multiline_missing_old_string(self, temp_dir):
        """Test multiline edit with missing old_string."""
        os.chdir(temp_dir)
        edits = [{"new_string": "new"}]
        result = edit_file_multiline("test.txt", edits)
        assert "Error" in result

    def test_edit_file_multiline_missing_new_string(self, temp_dir):
        """Test multiline edit with missing new_string."""
        os.chdir(temp_dir)
        edits = [{"old_string": "old"}]
        result = edit_file_multiline("test.txt", edits)
        assert "Error" in result

    def test_edit_file_multiline_search_not_found(self, temp_dir):
        """Test multiline edit with search string not found."""
        os.chdir(temp_dir)
        edits = [{"old_string": "nonexistent", "new_string": "new"}]
        result = edit_file_multiline("test.txt", edits)
        assert "Error" in result


# ========================================
# Move File Tests
# ========================================


class TestMoveFile:
    """Tests for move_file function."""

    def test_move_file_success(self, temp_dir):
        """Test successful file move."""
        os.chdir(temp_dir)
        result = move_file("test.txt", "moved.txt")
        assert "Successfully moved" in result
        assert os.path.exists(os.path.join(temp_dir, "moved.txt"))
        assert not os.path.exists(os.path.join(temp_dir, "test.txt"))

    def test_move_file_to_subdirectory(self, temp_dir):
        """Test moving file to subdirectory."""
        os.chdir(temp_dir)
        result = move_file("test.txt", "subdir/test.txt")
        assert "Successfully moved" in result
        assert os.path.exists(os.path.join(temp_dir, "subdir/test.txt"))

    def test_move_file_source_not_found(self, temp_dir):
        """Test moving non-existent file."""
        os.chdir(temp_dir)
        result = move_file("nonexistent.txt", "dest.txt")
        assert "Error" in result or "not found" in result.lower()

    def test_move_file_creates_dest_dir(self, temp_dir):
        """Test that move_file creates destination directory."""
        os.chdir(temp_dir)
        result = move_file("test.txt", "new_dir/test.txt")
        assert "Successfully moved" in result

    @patch("shutil.move", side_effect=OSError("Permission denied"))
    @patch("os.path.exists")
    @patch("os.makedirs")
    def test_move_file_io_error(self, mock_makedirs, mock_exists, mock_move):
        """Test move_file with IO error."""
        mock_exists.return_value = True
        result = move_file("test.txt", "dest.txt")
        assert "Error" in result


# ========================================
# Copy File Tests
# ========================================


class TestCopyFile:
    """Tests for copy_file function."""

    def test_copy_file_success(self, temp_dir):
        """Test successful file copy."""
        os.chdir(temp_dir)
        result = copy_file("test.txt", "copy.txt")
        assert "Successfully copied" in result
        assert os.path.exists(os.path.join(temp_dir, "copy.txt"))
        assert os.path.exists(os.path.join(temp_dir, "test.txt"))

    def test_copy_file_destination_exists_no_overwrite(self, temp_dir):
        """Test copying to existing destination without overwrite."""
        os.chdir(temp_dir)
        result = copy_file("test.txt", "test_module.py")
        assert "Error" in result or "already exists" in result.lower()

    def test_copy_file_destination_exists_with_overwrite(self, temp_dir):
        """Test copying to existing destination with overwrite."""
        os.chdir(temp_dir)
        result = copy_file("test.txt", "test_module.py", overwrite=True)
        assert "Successfully copied" in result

    def test_copy_file_source_not_found(self, temp_dir):
        """Test copying non-existent file."""
        os.chdir(temp_dir)
        result = copy_file("nonexistent.txt", "dest.txt")
        assert "Error" in result or "not found" in result.lower()

    def test_copy_directory(self, temp_dir):
        """Test copying directory."""
        os.chdir(temp_dir)
        result = copy_file("subdir", "subdir_copy")
        assert "Successfully copied" in result
        assert os.path.exists(os.path.join(temp_dir, "subdir_copy"))

    @patch("shutil.copy2", side_effect=OSError("Permission denied"))
    @patch("os.path.exists")
    @patch("os.path.isdir")
    @patch("os.makedirs")
    def test_copy_file_io_error(self, mock_makedirs, mock_isdir, mock_exists, mock_copy):
        """Test copy_file with IO error."""
        mock_exists.side_effect = [True, False]
        mock_isdir.return_value = False
        result = copy_file("test.txt", "dest.txt")
        assert "Error" in result


# ========================================
# Delete File Tests
# ========================================


class TestDeleteFile:
    """Tests for delete_file function."""

    def test_delete_file_success(self, temp_dir):
        """Test successful file deletion."""
        os.chdir(temp_dir)
        result = delete_file("test.txt")
        assert "Successfully deleted" in result
        assert not os.path.exists(os.path.join(temp_dir, "test.txt"))

    def test_delete_directory_without_recursive(self, temp_dir):
        """Test deleting directory without recursive flag."""
        os.chdir(temp_dir)
        result = delete_file("subdir")
        assert "Error" in result or "recursive" in result.lower()

    def test_delete_directory_with_recursive(self, temp_dir):
        """Test deleting directory with recursive flag."""
        os.chdir(temp_dir)
        result = delete_file("subdir", recursive=True)
        assert "Successfully deleted" in result
        assert not os.path.exists(os.path.join(temp_dir, "subdir"))

    def test_delete_file_not_found(self, temp_dir):
        """Test deleting non-existent file."""
        os.chdir(temp_dir)
        result = delete_file("nonexistent.txt")
        assert "Error" in result or "not found" in result.lower()

    @patch("os.remove", side_effect=OSError("Permission denied"))
    @patch("os.path.exists")
    @patch("os.path.isdir")
    def test_delete_file_io_error(self, mock_isdir, mock_exists, mock_remove):
        """Test delete_file with IO error."""
        mock_exists.return_value = True
        mock_isdir.return_value = False
        result = delete_file("test.txt")
        assert "Error" in result


# ========================================
# Rename File Tests
# ========================================


class TestRenameFile:
    """Tests for rename_file function."""

    def test_rename_file_success(self, temp_dir):
        """Test successful file rename."""
        os.chdir(temp_dir)
        result = rename_file("test.txt", "renamed.txt")
        assert "Successfully renamed" in result
        assert os.path.exists(os.path.join(temp_dir, "renamed.txt"))
        assert not os.path.exists(os.path.join(temp_dir, "test.txt"))

    def test_rename_file_not_found(self, temp_dir):
        """Test renaming non-existent file."""
        os.chdir(temp_dir)
        result = rename_file("nonexistent.txt", "new.txt")
        assert "Error" in result or "not found" in result.lower()

    def test_rename_file_destination_exists(self, temp_dir):
        """Test renaming to existing filename."""
        os.chdir(temp_dir)
        result = rename_file("test.txt", "test_module.py")
        assert "Error" in result or "already exists" in result.lower()

    @patch("os.rename", side_effect=OSError("Permission denied"))
    @patch("os.path.exists")
    def test_rename_file_io_error(self, mock_exists, mock_rename):
        """Test rename_file with IO error."""
        mock_exists.side_effect = [True, False]
        result = rename_file("test.txt", "new.txt")
        assert "Error" in result


# ========================================
# Create Directory Tests
# ========================================


class TestCreateDirectory:
    """Tests for create_directory function."""

    def test_create_directory_success(self, temp_dir):
        """Test successful directory creation."""
        os.chdir(temp_dir)
        result = create_directory("new_dir")
        assert "Successfully created" in result
        assert os.path.exists(os.path.join(temp_dir, "new_dir"))

    def test_create_directory_nested(self, temp_dir):
        """Test creating nested directories."""
        os.chdir(temp_dir)
        result = create_directory("a/b/c/d")
        assert "Successfully created" in result
        assert os.path.exists(os.path.join(temp_dir, "a/b/c/d"))

    def test_create_directory_already_exists(self, temp_dir):
        """Test creating directory that already exists."""
        os.chdir(temp_dir)
        result = create_directory("subdir")
        assert "Successfully created" in result

    @patch("os.makedirs", side_effect=OSError("Permission denied"))
    def test_create_directory_io_error(self, mock_makedirs):
        """Test create_directory with IO error."""
        result = create_directory("test_dir")
        assert "Error" in result


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

    def test_glob_with_exclusion(self, temp_dir):
        """Test glob with exclusion pattern."""
        os.chdir(temp_dir)
        result = glob_files("**/*.txt", exclude=["**/sub*"])
        assert "test.txt" in result

    def test_glob_path_traversal_blocked(self, temp_dir):
        """Test that path traversal is blocked."""
        os.chdir(temp_dir)
        result = glob_files("../**/*.txt")
        assert "Error" in result or "cannot contain" in result.lower()

    def test_glob_absolute_path_blocked(self, temp_dir):
        """Test that absolute paths are blocked."""
        result = glob_files("/etc/**/*.txt")
        assert "Error" in result or "cannot be an absolute path" in result.lower()

    def test_glob_max_results(self, temp_dir):
        """Test glob with max results limit."""
        os.chdir(temp_dir)
        result = glob_files("**/*", max_results=1)
        assert "Showing first" in result or "Found" in result


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

    def test_grep_specific_file_type(self, temp_dir):
        """Test grep with specific file type."""
        os.chdir(temp_dir)
        result = grep_search("class", include="*.py")
        assert "test_module.py" in result

    def test_grep_invalid_regex(self, temp_dir):
        """Test grep with invalid regex pattern."""
        os.chdir(temp_dir)
        result = grep_search("[invalid(regex", include="*.txt")
        assert "Error" in result


# ========================================
# List Directory Tree Tests
# ========================================


class TestDirectoryTree:
    """Tests for list_directory_tree function."""

    def test_directory_tree_basic(self, temp_dir):
        """Test basic directory tree."""
        os.chdir(temp_dir)
        result = list_directory_tree(".", depth=2)
        assert "test.txt" in result
        assert "subdir" in result

    def test_directory_tree_depth_limit(self, temp_dir):
        """Test directory tree respects depth limit."""
        os.chdir(temp_dir)
        result = list_directory_tree(".", depth=1)
        assert "Project Tree" in result

    def test_directory_tree_max_depth(self, temp_dir):
        """Test directory tree with max depth."""
        os.chdir(temp_dir)
        result = list_directory_tree(".", depth=100)
        assert "Project Tree" in result

    def test_directory_tree_depth_zero(self, temp_dir):
        """Test directory tree with depth 0."""
        os.chdir(temp_dir)
        result = list_directory_tree(".", depth=0)
        assert "Project Tree" in result


# ========================================
# Read File Outline Tests
# ========================================


class TestReadFileOutline:
    """Tests for read_file_outline function."""

    def test_read_outline_python(self, temp_dir):
        """Test reading Python file outline."""
        os.chdir(temp_dir)
        result = read_file_outline("test_module.py")
        assert "TestClass" in result
        assert "method_one" in result
        assert "standalone_function" in result

    def test_read_outline_not_found(self, temp_dir):
        """Test outline of non-existent file."""
        os.chdir(temp_dir)
        result = read_file_outline("nonexistent.py")
        assert "Error" in result or "not found" in result.lower()


# ========================================
# Find Symbol Tests
# ========================================


class TestFindSymbol:
    """Tests for find_symbol function."""

    @patch("src.services.code_nav.find_definition")
    def test_find_symbol_success(self, mock_find):
        """Test finding symbol definition."""
        mock_find.return_value = "Found at line 10"
        result = find_symbol("TestClass")
        assert "Found at line 10" in result

    @patch("src.services.code_nav.find_definition")
    def test_find_symbol_not_found(self, mock_find):
        """Test finding non-existent symbol."""
        mock_find.return_value = "Symbol not found"
        result = find_symbol("NonExistentClass")
        assert "not found" in result.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
