"""
Comprehensive unit tests for the Unified Filesystem Server.

Tests all major functions with mocking: _read_path_content, _write_path_content, _list_path_entries,
_create_path_directory, _delete_path, _move_path, _copy_path, _replace_path_content, and more.

Includes 50+ tests covering success cases, error cases, and edge cases.
"""

import os
import tempfile
from unittest.mock import mock_open, patch

import pytest

from src.servers.filesystem import (
    _human_size,
    _copy_path,
    _create_path_directory,
    _delete_path,
    _replace_path_content,
    _apply_path_edits,
    find_symbol,
    glob_files,
    grep_search,
    _list_path_entries,
    _list_path_tree,
    _move_path,
    _read_path_content,
    _read_path_outline,
    _rename_path,
    _write_path_content,
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

    def test_read_path_with_offset_only(self, temp_dir):
        """Test reading file with offset but no limit."""
        os.chdir(temp_dir)
        content = _read_path_content("test.txt", offset=3)
        assert "Line 4" in content
        assert "Line 5" in content

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

    def test_read_path_offset_beyond_file(self, temp_dir):
        """Test reading with offset beyond file length."""
        os.chdir(temp_dir)
        result = _read_path_content("test.txt", offset=100, limit=10)
        assert "No lines found" in result

    def test_read_path_limit_zero(self, temp_dir):
        """Test reading with limit of zero (reads all lines from offset)."""
        os.chdir(temp_dir)
        result = _read_path_content("test.txt", offset=0, limit=0)
        # When limit is 0, it's treated as None, so all lines are read
        assert "Line 1" in result
        assert "Line 5" in result

    @patch("os.path.getsize")
    @patch("builtins.open", new_callable=mock_open, read_data="test content")
    @patch("os.path.exists")
    def test_read_path_with_mock(self, mock_exists, mock_file, mock_size):
        """Test _read_path_content with mocked file operations."""
        mock_exists.return_value = True
        mock_size.return_value = 100
        result = _read_path_content("test.txt")
        assert "test content" in result

    @patch("os.path.getsize")
    @patch("builtins.open", side_effect=OSError("Permission denied"))
    @patch("os.path.exists")
    def test_read_path_permission_error(self, mock_exists, mock_file, mock_size):
        """Test _read_path_content with permission error."""
        mock_exists.return_value = True
        mock_size.return_value = 100
        result = _read_path_content("test.txt")
        assert "Error" in result


# ========================================
# Write Path Tests
# ========================================


class TestWritePath:
    """Tests for _write_path_content function."""

    def test_write_path_success(self, temp_dir):
        """Test successful file write."""
        os.chdir(temp_dir)
        result = _write_path_content("new_file.txt", "test content")
        assert "Successfully wrote" in result
        assert os.path.exists(os.path.join(temp_dir, "new_file.txt"))

    def test_write_path_creates_parent_dirs(self, temp_dir):
        """Test that _write_path_content creates parent directories."""
        os.chdir(temp_dir)
        result = _write_path_content("new_dir/subdir/file.txt", "content")
        assert "Successfully wrote" in result
        assert os.path.exists(os.path.join(temp_dir, "new_dir/subdir/file.txt"))

    def test_write_path_overwrites_existing(self, temp_dir):
        """Test that _write_path_content overwrites existing files."""
        os.chdir(temp_dir)
        _write_path_content("test.txt", "new content")
        with open(os.path.join(temp_dir, "test.txt")) as f:
            content = f.read()
        assert content == "new content"

    def test_write_path_empty_content(self, temp_dir):
        """Test writing empty content."""
        os.chdir(temp_dir)
        result = _write_path_content("empty.txt", "")
        assert "Successfully wrote" in result

    @patch("builtins.open", side_effect=OSError("Disk full"))
    @patch("os.makedirs")
    def test_write_path_io_error(self, mock_makedirs, mock_file):
        """Test _write_path_content with IO error."""
        result = _write_path_content("test.txt", "content")
        assert "Error" in result

    def test_write_path_large_content(self, temp_dir):
        """Test writing large content."""
        os.chdir(temp_dir)
        large_content = "x" * 1000000
        result = _write_path_content("large.txt", large_content)
        assert "Successfully wrote" in result


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

    def test_list_path_entries_not_detailed(self, temp_dir):
        """Test non-detailed directory listing."""
        os.chdir(temp_dir)
        result = _list_path_entries(".", detailed=False)
        assert "test.txt" in result

    def test_list_path_entries_not_found(self, temp_dir):
        """Test listing non-existent directory."""
        os.chdir(temp_dir)
        result = _list_path_entries("nonexistent_dir")
        assert "Error" in result or "not found" in result.lower()

    def test_list_path_entries_empty(self, temp_dir):
        """Test listing empty directory."""
        os.chdir(temp_dir)
        empty_dir = os.path.join(temp_dir, "empty")
        os.makedirs(empty_dir)
        result = _list_path_entries("empty")
        assert "(empty directory)" in result

    def test_list_path_entries_hide_hidden_files(self, temp_dir):
        """Test hiding hidden files."""
        os.chdir(temp_dir)
        hidden_file = os.path.join(temp_dir, ".hidden")
        with open(hidden_file, "w") as f:
            f.write("hidden")
        result = _list_path_entries(".", show_hidden=False)
        assert ".hidden" not in result

    def test_list_path_entries_show_hidden_files(self, temp_dir):
        """Test showing hidden files."""
        os.chdir(temp_dir)
        hidden_file = os.path.join(temp_dir, ".hidden")
        with open(hidden_file, "w") as f:
            f.write("hidden")
        result = _list_path_entries(".", show_hidden=True)
        assert ".hidden" in result

    @patch("os.listdir", side_effect=OSError("Permission denied"))
    def test_list_path_entries_permission_error(self, mock_listdir):
        """Test _list_path_entries with permission error."""
        result = _list_path_entries(".")
        assert "Error" in result


# ========================================
# Replace Path Content Tests
# ========================================


class TestReplacePathContent:
    """Tests for _replace_path_content function."""

    def test_replace_path_content_single_replacement(self, temp_dir):
        """Test single replacement in file."""
        os.chdir(temp_dir)
        result = _replace_path_content("test.txt", "Line 1", "Modified Line 1", count=1)
        assert "Successfully edited" in result
        with open(os.path.join(temp_dir, "test.txt")) as f:
            content = f.read()
        assert "Modified Line 1" in content

    def test_replace_path_content_all_replacements(self, temp_dir):
        """Test replacing all occurrences."""
        os.chdir(temp_dir)
        result = _replace_path_content("test.txt", "Line", "Modified", count=-1)
        assert "Successfully edited" in result
        with open(os.path.join(temp_dir, "test.txt")) as f:
            content = f.read()
        assert "Modified" in content
        assert content.count("Modified") == 5

    def test_replace_path_content_not_found(self, temp_dir):
        """Test editing non-existent file."""
        os.chdir(temp_dir)
        result = _replace_path_content("nonexistent.txt", "old", "new")
        assert "Error" in result or "not found" in result.lower()

    def test_replace_path_content_search_string_not_found(self, temp_dir):
        """Test editing with search string not found."""
        os.chdir(temp_dir)
        result = _replace_path_content("test.txt", "nonexistent", "new")
        assert "Error" in result or "not found" in result.lower()

    def test_replace_path_content_multiple_replacements(self, temp_dir):
        """Test replacing multiple occurrences."""
        os.chdir(temp_dir)
        result = _replace_path_content("test.txt", "Line", "Modified", count=3)
        assert "Successfully edited" in result

    @patch("builtins.open", side_effect=OSError("Permission denied"))
    def test_replace_path_content_io_error(self, mock_file):
        """Test _replace_path_content with IO error."""
        result = _replace_path_content("test.txt", "old", "new")
        assert "Error" in result


# ========================================
# Apply Path Edits Tests
# ========================================


class TestApplyPathEdits:
    """Tests for _apply_path_edits function."""

    def test_apply_path_edits_success(self, temp_dir):
        """Test multiple edits in sequence."""
        os.chdir(temp_dir)
        edits = [
            {"old_string": "Line 1", "new_string": "Modified 1"},
            {"old_string": "Line 2", "new_string": "Modified 2"},
        ]
        result = _apply_path_edits("test.txt", edits)
        assert "Successfully applied" in result
        with open(os.path.join(temp_dir, "test.txt")) as f:
            content = f.read()
        assert "Modified 1" in content
        assert "Modified 2" in content

    def test_apply_path_edits_not_found(self, temp_dir):
        """Test multiline edit on non-existent file."""
        os.chdir(temp_dir)
        edits = [{"old_string": "old", "new_string": "new"}]
        result = _apply_path_edits("nonexistent.txt", edits)
        assert "Error" in result or "not found" in result.lower()

    def test_apply_path_edits_missing_old_string(self, temp_dir):
        """Test multiline edit with missing old_string."""
        os.chdir(temp_dir)
        edits = [{"new_string": "new"}]
        result = _apply_path_edits("test.txt", edits)
        assert "Error" in result

    def test_apply_path_edits_missing_new_string(self, temp_dir):
        """Test multiline edit with missing new_string."""
        os.chdir(temp_dir)
        edits = [{"old_string": "old"}]
        result = _apply_path_edits("test.txt", edits)
        assert "Error" in result

    def test_apply_path_edits_search_not_found(self, temp_dir):
        """Test multiline edit with search string not found."""
        os.chdir(temp_dir)
        edits = [{"old_string": "nonexistent", "new_string": "new"}]
        result = _apply_path_edits("test.txt", edits)
        assert "Error" in result


# ========================================
# Move Path Tests
# ========================================


class TestMovePath:
    """Tests for _move_path function."""

    def test_move_path_success(self, temp_dir):
        """Test successful file move."""
        os.chdir(temp_dir)
        result = _move_path("test.txt", "moved.txt")
        assert "Successfully moved" in result
        assert os.path.exists(os.path.join(temp_dir, "moved.txt"))
        assert not os.path.exists(os.path.join(temp_dir, "test.txt"))

    def test_move_path_to_subdirectory(self, temp_dir):
        """Test moving file to subdirectory."""
        os.chdir(temp_dir)
        result = _move_path("test.txt", "subdir/test.txt")
        assert "Successfully moved" in result
        assert os.path.exists(os.path.join(temp_dir, "subdir/test.txt"))

    def test_move_path_source_not_found(self, temp_dir):
        """Test moving non-existent file."""
        os.chdir(temp_dir)
        result = _move_path("nonexistent.txt", "dest.txt")
        assert "Error" in result or "not found" in result.lower()

    def test_move_path_creates_dest_dir(self, temp_dir):
        """Test that _move_path creates destination directory."""
        os.chdir(temp_dir)
        result = _move_path("test.txt", "new_dir/test.txt")
        assert "Successfully moved" in result

    @patch("shutil.move", side_effect=OSError("Permission denied"))
    @patch("os.path.exists")
    @patch("os.makedirs")
    def test_move_path_io_error(self, mock_makedirs, mock_exists, mock_move):
        """Test _move_path with IO error."""
        mock_exists.return_value = True
        result = _move_path("test.txt", "dest.txt")
        assert "Error" in result


# ========================================
# Copy Path Tests
# ========================================


class TestCopyPath:
    """Tests for _copy_path function."""

    def test_copy_path_success(self, temp_dir):
        """Test successful file copy."""
        os.chdir(temp_dir)
        result = _copy_path("test.txt", "copy.txt")
        assert "Successfully copied" in result
        assert os.path.exists(os.path.join(temp_dir, "copy.txt"))
        assert os.path.exists(os.path.join(temp_dir, "test.txt"))

    def test_copy_path_destination_exists_no_overwrite(self, temp_dir):
        """Test copying to existing destination without overwrite."""
        os.chdir(temp_dir)
        result = _copy_path("test.txt", "test_module.py")
        assert "Error" in result or "already exists" in result.lower()

    def test_copy_path_destination_exists_with_overwrite(self, temp_dir):
        """Test copying to existing destination with overwrite."""
        os.chdir(temp_dir)
        result = _copy_path("test.txt", "test_module.py", overwrite=True)
        assert "Successfully copied" in result

    def test_copy_path_source_not_found(self, temp_dir):
        """Test copying non-existent file."""
        os.chdir(temp_dir)
        result = _copy_path("nonexistent.txt", "dest.txt")
        assert "Error" in result or "not found" in result.lower()

    def test_copy_directory(self, temp_dir):
        """Test copying directory."""
        os.chdir(temp_dir)
        result = _copy_path("subdir", "subdir_copy")
        assert "Successfully copied" in result
        assert os.path.exists(os.path.join(temp_dir, "subdir_copy"))

    @patch("shutil.copy2", side_effect=OSError("Permission denied"))
    @patch("os.path.exists")
    @patch("os.path.isdir")
    @patch("os.makedirs")
    def test_copy_path_io_error(self, mock_makedirs, mock_isdir, mock_exists, mock_copy):
        """Test _copy_path with IO error."""
        mock_exists.side_effect = [True, False]
        mock_isdir.return_value = False
        result = _copy_path("test.txt", "dest.txt")
        assert "Error" in result


# ========================================
# Delete Path Tests
# ========================================


class TestDeletePath:
    """Tests for _delete_path function."""

    def test_delete_path_success(self, temp_dir):
        """Test successful file deletion."""
        os.chdir(temp_dir)
        result = _delete_path("test.txt")
        assert "Successfully deleted" in result
        assert not os.path.exists(os.path.join(temp_dir, "test.txt"))

    def test_delete_directory_without_recursive(self, temp_dir):
        """Test deleting directory without recursive flag."""
        os.chdir(temp_dir)
        result = _delete_path("subdir")
        assert "Error" in result or "recursive" in result.lower()

    def test_delete_directory_with_recursive(self, temp_dir):
        """Test deleting directory with recursive flag."""
        os.chdir(temp_dir)
        result = _delete_path("subdir", recursive=True)
        assert "Successfully deleted" in result
        assert not os.path.exists(os.path.join(temp_dir, "subdir"))

    def test_delete_path_not_found(self, temp_dir):
        """Test deleting non-existent file."""
        os.chdir(temp_dir)
        result = _delete_path("nonexistent.txt")
        assert "Error" in result or "not found" in result.lower()

    @patch("os.remove", side_effect=OSError("Permission denied"))
    @patch("os.path.exists")
    @patch("os.path.isdir")
    def test_delete_path_io_error(self, mock_isdir, mock_exists, mock_remove):
        """Test _delete_path with IO error."""
        mock_exists.return_value = True
        mock_isdir.return_value = False
        result = _delete_path("test.txt")
        assert "Error" in result


# ========================================
# Rename Path Tests
# ========================================


class TestRenamePath:
    """Tests for _rename_path function."""

    def test_rename_path_success(self, temp_dir):
        """Test successful file rename."""
        os.chdir(temp_dir)
        result = _rename_path("test.txt", "renamed.txt")
        assert "Successfully renamed" in result
        assert os.path.exists(os.path.join(temp_dir, "renamed.txt"))
        assert not os.path.exists(os.path.join(temp_dir, "test.txt"))

    def test_rename_path_not_found(self, temp_dir):
        """Test renaming non-existent file."""
        os.chdir(temp_dir)
        result = _rename_path("nonexistent.txt", "new.txt")
        assert "Error" in result or "not found" in result.lower()

    def test_rename_path_destination_exists(self, temp_dir):
        """Test renaming to existing filename."""
        os.chdir(temp_dir)
        result = _rename_path("test.txt", "test_module.py")
        assert "Error" in result or "already exists" in result.lower()

    @patch("os.rename", side_effect=OSError("Permission denied"))
    @patch("os.path.exists")
    def test_rename_path_io_error(self, mock_exists, mock_rename):
        """Test _rename_path with IO error."""
        mock_exists.side_effect = [True, False]
        result = _rename_path("test.txt", "new.txt")
        assert "Error" in result


# ========================================
# Create Path Directory Tests
# ========================================


class TestCreatePathDirectory:
    """Tests for _create_path_directory function."""

    def test_create_path_directory_success(self, temp_dir):
        """Test successful directory creation."""
        os.chdir(temp_dir)
        result = _create_path_directory("new_dir")
        assert "Successfully created" in result
        assert os.path.exists(os.path.join(temp_dir, "new_dir"))

    def test_create_path_directory_nested(self, temp_dir):
        """Test creating nested directories."""
        os.chdir(temp_dir)
        result = _create_path_directory("a/b/c/d")
        assert "Successfully created" in result
        assert os.path.exists(os.path.join(temp_dir, "a/b/c/d"))

    def test_create_path_directory_already_exists(self, temp_dir):
        """Test creating directory that already exists."""
        os.chdir(temp_dir)
        result = _create_path_directory("subdir")
        assert "Successfully created" in result

    @patch("os.makedirs", side_effect=OSError("Permission denied"))
    def test_create_path_directory_io_error(self, mock_makedirs):
        """Test _create_path_directory with IO error."""
        result = _create_path_directory("test_dir")
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

    def test_directory_tree_max_depth(self, temp_dir):
        """Test directory tree with max depth."""
        os.chdir(temp_dir)
        result = _list_path_tree(".", depth=100)
        assert "Project Tree" in result

    def test_directory_tree_depth_zero(self, temp_dir):
        """Test directory tree with depth 0."""
        os.chdir(temp_dir)
        result = _list_path_tree(".", depth=0)
        assert "Project Tree" in result


# ========================================
# Read File Outline Tests
# ========================================


class TestReadFileOutline:
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
# Find Symbol Tests
# ========================================


class TestFindSymbol:
    """Tests for find_symbol function."""

    @patch("src.servers._services.code_nav.find_definition")
    def test_find_symbol_success(self, mock_find):
        """Test finding symbol definition."""
        mock_find.return_value = "Found at line 10"
        result = find_symbol("TestClass")
        assert "Found at line 10" in result

    @patch("src.servers._services.code_nav.find_definition")
    def test_find_symbol_not_found(self, mock_find):
        """Test finding non-existent symbol."""
        mock_find.return_value = "Symbol not found"
        result = find_symbol("NonExistentClass")
        assert "not found" in result.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
