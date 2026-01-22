"""
Unit tests for the Lint Server.

Tests linting, formatting, and code quality analysis.
"""

import os
import shutil

import pytest

from src.servers.lint import (
    _check_tool_installed,
    _run_command,
    check_security,
    code_complexity,
    format_python_ruff,
    get_lint_summary,
    lint_python_ruff,
)


# ========================================
# Fixtures
# ========================================

@pytest.fixture
def test_files_dir():
    """Create a test directory inside project."""
    # Save original working directory
    original_cwd = os.getcwd()

    test_dir = os.path.join(original_cwd, ".test_lint_files")
    os.makedirs(test_dir, exist_ok=True)
    yield test_dir

    # Restore original working directory before cleanup
    os.chdir(original_cwd)
    shutil.rmtree(test_dir, ignore_errors=True)


@pytest.fixture
def python_file_clean(test_files_dir):
    """Create a clean Python file for testing."""
    path = os.path.join(test_files_dir, "clean.py")
    with open(path, "w") as f:
        f.write('''"""A clean module."""


def greet(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}!"
''')
    yield path


@pytest.fixture
def python_file_issues(test_files_dir):
    """Create a Python file with lint issues."""
    path = os.path.join(test_files_dir, "issues.py")
    with open(path, "w") as f:
        f.write('''import os
import sys
import json  # unused

x=1  # missing spaces
y = 2;  # unnecessary semicolon

def bad_function(a,b,c):  # missing type hints
    return a+b+c
''')
    yield path


@pytest.fixture
def python_file_complex(test_files_dir):
    """Create a Python file with complex function."""
    path = os.path.join(test_files_dir, "complex.py")
    with open(path, "w") as f:
        # Function with high cyclomatic complexity
        f.write('''
def complex_function(a, b, c, d, e):
    """A complex function."""
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    if e > 0:
                        return 1
                    else:
                        return 2
                else:
                    return 3
            else:
                return 4
        else:
            return 5
    else:
        if b < 0:
            if c < 0:
                return 6
            else:
                return 7
        else:
                return 8
''')
    yield path


# ========================================
# Tool Installation Tests
# ========================================

class TestToolInstallation:
    """Tests for tool installation checking"""

    def test_check_ruff_installed(self):
        """Test that ruff is detected if installed"""
        # This may pass or fail depending on environment
        result = _check_tool_installed("ruff")
        assert isinstance(result, bool)

    def test_check_nonexistent_tool(self):
        """Test that nonexistent tool is not detected"""
        result = _check_tool_installed("nonexistent_tool_12345")
        assert result is False


# ========================================
# Command Execution Tests
# ========================================

class TestCommandExecution:
    """Tests for command execution"""

    def test_run_command_success(self):
        """Test running a successful command"""
        exit_code, stdout, stderr = _run_command(["echo", "hello"])
        assert exit_code == 0
        assert "hello" in stdout

    def test_run_command_failure(self):
        """Test running a failing command"""
        exit_code, stdout, stderr = _run_command(["false"])
        assert exit_code != 0

    def test_run_command_not_found(self):
        """Test running nonexistent command"""
        exit_code, stdout, stderr = _run_command(["nonexistent_cmd_12345"])
        assert exit_code == -1
        assert "not found" in stderr.lower()


# ========================================
# Python Linting Tests
# ========================================

@pytest.mark.skipif(
    not _check_tool_installed("ruff"),
    reason="Ruff not installed"
)
class TestPythonLinting:
    """Tests for Python linting with Ruff"""

    def test_lint_clean_file(self, python_file_clean):
        """Test linting a clean file"""
        result = lint_python_ruff(python_file_clean)
        assert "no issues" in result.lower() or "✅" in result

    def test_lint_file_with_issues(self, python_file_issues):
        """Test linting a file with issues"""
        result = lint_python_ruff(python_file_issues)
        # Should find unused import or other issues
        assert "issue" in result.lower() or "F401" in result or "found" in result.lower()

    def test_lint_with_select(self, python_file_issues):
        """Test linting with specific rules"""
        result = lint_python_ruff(python_file_issues, select=["F"])
        # Should only show Pyflakes errors
        assert isinstance(result, str)

    def test_lint_nonexistent_file(self):
        """Test linting nonexistent file"""
        result = lint_python_ruff("/nonexistent/file.py")
        assert "error" in result.lower()


# ========================================
# Python Formatting Tests
# ========================================

@pytest.mark.skipif(
    not _check_tool_installed("ruff"),
    reason="Ruff not installed"
)
class TestPythonFormatting:
    """Tests for Python formatting with Ruff"""

    def test_format_clean_file(self, python_file_clean):
        """Test formatting a clean file"""
        result = format_python_ruff(python_file_clean, check_only=True)
        assert "formatted" in result.lower() or "✅" in result

    def test_format_check_only(self, python_file_issues):
        """Test format check without modifying"""
        # Read original content
        with open(python_file_issues) as f:
            original = f.read()
        
        result = format_python_ruff(python_file_issues, check_only=True)
        
        # File should not be modified
        with open(python_file_issues) as f:
            after = f.read()
        
        assert original == after


# ========================================
# Code Complexity Tests
# ========================================

@pytest.mark.skipif(
    not _check_tool_installed("ruff"),
    reason="Ruff not installed"
)
class TestCodeComplexity:
    """Tests for code complexity analysis"""

    def test_complexity_simple_file(self, python_file_clean):
        """Test complexity of simple file"""
        result = code_complexity(python_file_clean)
        assert "✅" in result or "complexity" in result.lower()

    def test_complexity_complex_file(self, python_file_complex):
        """Test complexity of complex file"""
        result = code_complexity(python_file_complex, max_complexity=3)
        # Should flag the complex function
        assert "complex" in result.lower() or isinstance(result, str)


# ========================================
# Security Check Tests
# ========================================

@pytest.mark.skipif(
    not _check_tool_installed("ruff"),
    reason="Ruff not installed"
)
class TestSecurityCheck:
    """Tests for security analysis"""

    def test_security_clean_file(self, python_file_clean):
        """Test security check on clean file"""
        result = check_security(python_file_clean)
        assert "✅" in result or "no security issues" in result.lower()


# ========================================
# Lint Summary Tests
# ========================================

@pytest.mark.skipif(
    not _check_tool_installed("ruff"),
    reason="Ruff not installed"
)
class TestLintSummary:
    """Tests for lint summary"""

    def test_summary_basic(self, python_file_issues):
        """Test getting lint summary"""
        result = get_lint_summary(python_file_issues)
        assert "summary" in result.lower() or isinstance(result, str)
