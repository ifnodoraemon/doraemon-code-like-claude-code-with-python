"""
Unit tests for the Unified Filesystem Server.

Tests file reading, directory listing, and code navigation functionality.
"""

import os
import tempfile

import pytest

from src.core.security.security import validate_path
from src.servers.filesystem import (
    glob_files,
    grep_search,
    list_directory,
    list_directory_tree,
    read_file,
    read_file_outline,
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

    def test_list_directory_not_found(self, temp_dir):
        """Test listing non-existent directory."""
        os.chdir(temp_dir)
        result = list_directory("nonexistent_dir")
        assert "Error" in result or "not found" in result.lower()


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


# ========================================
# Directory Tree Tests
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
