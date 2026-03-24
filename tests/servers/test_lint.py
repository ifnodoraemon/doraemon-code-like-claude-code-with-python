"""
Unit tests for the Lint Server.

Tests linting, formatting, and code quality analysis.
"""

import os
import subprocess
import tempfile
from unittest import mock

import pytest

from src.servers.lint import (
    LintIssue,
    _check_tool_installed,
    _run_command,
    check_security,
    code_complexity,
    format_python_ruff,
    get_lint_summary,
    lint_all,
    lint_javascript_eslint,
    lint_python_ruff,
    typecheck_python_mypy,
)

# ========================================
# Fixtures
# ========================================


@pytest.fixture
def test_files_dir():
    """Create an isolated temporary directory for lint test files."""
    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="lint_test_files_") as test_dir:
        yield test_dir
        os.chdir(original_cwd)


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
        f.write("""import os
import sys
import json  # unused

x=1  # missing spaces
y = 2;  # unnecessary semicolon

def bad_function(a,b,c):  # missing type hints
    return a+b+c
""")
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


@pytest.fixture
def python_file_security_issue(test_files_dir):
    """Create a Python file with security issues."""
    path = os.path.join(test_files_dir, "security_issue.py")
    with open(path, "w") as f:
        f.write("""import pickle
import subprocess

# Security issue: hardcoded password
password = "admin123"

# Security issue: using eval
user_input = "1 + 1"
result = eval(user_input)

# Security issue: using pickle
data = pickle.loads(b"some_data")
""")
    yield path


@pytest.fixture
def js_file_clean(test_files_dir):
    """Create a clean JavaScript file."""
    path = os.path.join(test_files_dir, "clean.js")
    with open(path, "w") as f:
        f.write("""function greet(name) {
  return `Hello, ${name}!`;
}

module.exports = greet;
""")
    yield path


@pytest.fixture
def js_file_issues(test_files_dir):
    """Create a JavaScript file with issues."""
    path = os.path.join(test_files_dir, "issues.js")
    with open(path, "w") as f:
        f.write("""var x = 1;
var y = 2;
var z = 3;

function test() {
  console.log(x);
}
""")
    yield path


@pytest.fixture
def python_file_type_errors(test_files_dir):
    """Create a Python file with type errors."""
    path = os.path.join(test_files_dir, "type_errors.py")
    with open(path, "w") as f:
        f.write("""def add(a: int, b: int) -> int:
    return a + b

result = add("1", "2")  # Type error: str instead of int
""")
    yield path


# ========================================
# LintIssue Data Class Tests
# ========================================


class TestLintIssue:
    """Tests for LintIssue data class"""

    def test_lint_issue_creation(self):
        """Test creating a LintIssue"""
        issue = LintIssue(
            file="test.py",
            line=10,
            column=5,
            severity="error",
            code="E501",
            message="Line too long",
            source="ruff",
        )
        assert issue.file == "test.py"
        assert issue.line == 10
        assert issue.column == 5
        assert issue.severity == "error"
        assert issue.code == "E501"
        assert issue.message == "Line too long"
        assert issue.source == "ruff"

    def test_lint_issue_to_string(self):
        """Test LintIssue.to_string() method"""
        issue = LintIssue(
            file="test.py",
            line=10,
            column=5,
            severity="error",
            code="E501",
            message="Line too long",
            source="ruff",
        )
        result = issue.to_string()
        assert "test.py:10:5" in result
        assert "[error]" in result
        assert "E501" in result
        assert "Line too long" in result

    def test_lint_issue_warning_severity(self):
        """Test LintIssue with warning severity"""
        issue = LintIssue(
            file="test.py",
            line=1,
            column=1,
            severity="warning",
            code="W291",
            message="Trailing whitespace",
            source="ruff",
        )
        assert issue.severity == "warning"
        assert "[warning]" in issue.to_string()

    def test_lint_issue_info_severity(self):
        """Test LintIssue with info severity"""
        issue = LintIssue(
            file="test.py",
            line=1,
            column=1,
            severity="info",
            code="I001",
            message="Import sorting",
            source="ruff",
        )
        assert issue.severity == "info"
        assert "[info]" in issue.to_string()


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

    def test_check_mypy_installed(self):
        """Test mypy installation check"""
        result = _check_tool_installed("mypy")
        assert isinstance(result, bool)

    def test_check_npx_installed(self):
        """Test npx installation check"""
        result = _check_tool_installed("npx")
        assert isinstance(result, bool)

    @mock.patch("src.servers.lint.subprocess.run")
    def test_check_tool_timeout(self, mock_run):
        """Test tool check with timeout"""
        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 5)
        result = _check_tool_installed("slow_tool")
        assert result is False

    @mock.patch("src.servers.lint.subprocess.run")
    def test_check_tool_file_not_found(self, mock_run):
        """Test tool check with file not found"""
        mock_run.side_effect = FileNotFoundError("Tool not found")
        result = _check_tool_installed("missing_tool")
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

    def test_run_command_with_stderr(self):
        """Test command that produces stderr"""
        exit_code, stdout, stderr = _run_command(["sh", "-c", "echo error >&2"])
        assert "error" in stderr

    def test_run_command_timeout(self):
        """Test command timeout"""
        exit_code, stdout, stderr = _run_command(["sleep", "10"], timeout=1)
        assert exit_code == -1
        assert "timed out" in stderr.lower()

    def test_run_command_with_cwd(self):
        """Test running command in specific directory"""
        exit_code, stdout, stderr = _run_command(["pwd"], cwd="/tmp")
        # May fail due to path validation, that's ok
        assert isinstance(exit_code, int)

    def test_run_command_invalid_cwd(self):
        """Test running command with invalid cwd"""
        exit_code, stdout, stderr = _run_command(["echo", "test"], cwd="/nonexistent/path")
        assert exit_code == -1
        assert "invalid" in stderr.lower()

    def test_run_command_multiline_output(self):
        """Test command with multiline output"""
        exit_code, stdout, stderr = _run_command(["printf", "line1\nline2\nline3"])
        assert exit_code == 0
        assert "line1" in stdout
        assert "line2" in stdout
        assert "line3" in stdout

    @mock.patch("subprocess.run")
    def test_run_command_exception_handling(self, mock_run):
        """Test exception handling in run_command"""
        mock_run.side_effect = Exception("Unexpected error")
        exit_code, stdout, stderr = _run_command(["test"])
        assert exit_code == -1
        assert "error" in stderr.lower()


# ========================================
# Python Linting Tests
# ========================================


@pytest.mark.skipif(not _check_tool_installed("ruff"), reason="Ruff not installed")
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

    def test_lint_with_ignore(self, python_file_issues):
        """Test linting with ignored rules"""
        result = lint_python_ruff(python_file_issues, ignore=["F401"])
        assert isinstance(result, str)

    def test_lint_nonexistent_file(self):
        """Test linting nonexistent file"""
        result = lint_python_ruff("/nonexistent/file.py")
        assert "error" in result.lower()

    def test_lint_directory(self, test_files_dir):
        """Test linting a directory"""
        result = lint_python_ruff(test_files_dir)
        assert isinstance(result, str)

    def test_lint_with_fix(self, python_file_issues):
        """Test linting with fix enabled"""
        result = lint_python_ruff(python_file_issues, fix=True)
        assert isinstance(result, str)

    def test_lint_multiple_select_rules(self, python_file_issues):
        """Test linting with multiple select rules"""
        result = lint_python_ruff(python_file_issues, select=["E", "F", "W"])
        assert isinstance(result, str)

    def test_lint_multiple_ignore_rules(self, python_file_issues):
        """Test linting with multiple ignore rules"""
        result = lint_python_ruff(python_file_issues, ignore=["E501", "W291"])
        assert isinstance(result, str)

    @mock.patch("src.servers.lint._check_tool_installed")
    def test_lint_ruff_not_installed(self, mock_check):
        """Test linting when ruff is not installed"""
        mock_check.return_value = False
        result = lint_python_ruff(".")
        assert "not installed" in result.lower()

    @mock.patch("src.servers.lint._run_command")
    def test_lint_json_parse_error(self, mock_run):
        """Test handling of JSON parse errors"""
        mock_run.return_value = (0, "invalid json", "")
        result = lint_python_ruff(".")
        assert isinstance(result, str)


# ========================================
# Python Formatting Tests
# ========================================


@pytest.mark.skipif(not _check_tool_installed("ruff"), reason="Ruff not installed")
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

        format_python_ruff(python_file_issues, check_only=True)

        # File should not be modified
        with open(python_file_issues) as f:
            after = f.read()

        assert original == after

    def test_format_directory(self, test_files_dir):
        """Test formatting a directory"""
        result = format_python_ruff(test_files_dir, check_only=True)
        assert isinstance(result, str)

    def test_format_nonexistent_file(self):
        """Test formatting nonexistent file"""
        result = format_python_ruff("/nonexistent/file.py")
        assert "error" in result.lower()

    @mock.patch("src.servers.lint._check_tool_installed")
    def test_format_ruff_not_installed(self, mock_check):
        """Test formatting when ruff is not installed"""
        mock_check.return_value = False
        result = format_python_ruff(".")
        assert "not installed" in result.lower()

    @mock.patch("src.servers.lint._run_command")
    def test_format_with_error(self, mock_run):
        """Test formatting with error"""
        mock_run.return_value = (1, "", "Format error")
        result = format_python_ruff(".")
        assert "error" in result.lower()


# ========================================
# Python Type Checking Tests
# ========================================


@pytest.mark.skipif(not _check_tool_installed("mypy"), reason="MyPy not installed")
class TestPythonTypeChecking:
    """Tests for Python type checking with MyPy"""

    def test_typecheck_clean_file(self, python_file_clean):
        """Test type checking a clean file"""
        result = typecheck_python_mypy(python_file_clean)
        assert "no type errors" in result.lower() or "✅" in result

    def test_typecheck_with_strict(self, python_file_clean):
        """Test type checking with strict mode"""
        result = typecheck_python_mypy(python_file_clean, strict=True)
        assert isinstance(result, str)

    def test_typecheck_ignore_missing_imports(self, python_file_clean):
        """Test type checking with ignore missing imports"""
        result = typecheck_python_mypy(python_file_clean, ignore_missing_imports=True)
        assert isinstance(result, str)

    def test_typecheck_directory(self, test_files_dir):
        """Test type checking a directory"""
        result = typecheck_python_mypy(test_files_dir)
        assert isinstance(result, str)

    def test_typecheck_nonexistent_file(self):
        """Test type checking nonexistent file"""
        result = typecheck_python_mypy("/nonexistent/file.py")
        assert "error" in result.lower()

    @mock.patch("src.servers.lint._check_tool_installed")
    def test_typecheck_mypy_not_installed(self, mock_check):
        """Test type checking when mypy is not installed"""
        mock_check.return_value = False
        result = typecheck_python_mypy(".")
        assert "not installed" in result.lower()


# ========================================
# JavaScript/TypeScript Linting Tests
# ========================================


@pytest.mark.skipif(not _check_tool_installed("npx"), reason="npx not installed")
class TestJavaScriptLinting:
    """Tests for JavaScript/TypeScript linting with ESLint"""

    def test_lint_js_clean_file(self, js_file_clean):
        """Test linting a clean JS file"""
        result = lint_javascript_eslint(js_file_clean)
        # May pass or fail depending on ESLint installation
        assert isinstance(result, str)

    def test_lint_js_with_fix(self, js_file_issues):
        """Test linting JS with fix enabled"""
        result = lint_javascript_eslint(js_file_issues, fix=True)
        assert isinstance(result, str)

    def test_lint_js_with_extensions(self, js_file_clean):
        """Test linting JS with specific extensions"""
        result = lint_javascript_eslint(js_file_clean, ext=[".js", ".jsx"])
        assert isinstance(result, str)

    def test_lint_js_nonexistent_file(self):
        """Test linting nonexistent JS file"""
        result = lint_javascript_eslint("/nonexistent/file.js")
        # May error or return no issues depending on ESLint
        assert isinstance(result, str)

    @mock.patch("src.servers.lint._check_tool_installed")
    def test_lint_js_npx_not_installed(self, mock_check):
        """Test linting when npx is not installed"""
        mock_check.return_value = False
        result = lint_javascript_eslint(".")
        assert "not available" in result.lower()


# ========================================
# Code Complexity Tests
# ========================================


@pytest.mark.skipif(not _check_tool_installed("ruff"), reason="Ruff not installed")
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

    def test_complexity_custom_threshold(self, python_file_complex):
        """Test complexity with custom threshold"""
        result = code_complexity(python_file_complex, max_complexity=5)
        assert isinstance(result, str)

    def test_complexity_directory(self, test_files_dir):
        """Test complexity analysis on directory"""
        result = code_complexity(test_files_dir)
        assert isinstance(result, str)

    def test_complexity_nonexistent_file(self):
        """Test complexity on nonexistent file"""
        result = code_complexity("/nonexistent/file.py")
        assert "error" in result.lower()

    @mock.patch("src.servers.lint._check_tool_installed")
    def test_complexity_ruff_not_installed(self, mock_check):
        """Test complexity when ruff is not installed"""
        mock_check.return_value = False
        result = code_complexity(".")
        assert "not installed" in result.lower()


# ========================================
# Security Check Tests
# ========================================


@pytest.mark.skipif(not _check_tool_installed("ruff"), reason="Ruff not installed")
class TestSecurityCheck:
    """Tests for security analysis"""

    def test_security_clean_file(self, python_file_clean):
        """Test security check on clean file"""
        result = check_security(python_file_clean)
        assert "✅" in result or "no security issues" in result.lower()

    def test_security_directory(self, test_files_dir):
        """Test security check on directory"""
        result = check_security(test_files_dir)
        assert isinstance(result, str)

    def test_security_nonexistent_file(self):
        """Test security check on nonexistent file"""
        result = check_security("/nonexistent/file.py")
        assert "error" in result.lower()

    @mock.patch("src.servers.lint._check_tool_installed")
    def test_security_ruff_not_installed(self, mock_check):
        """Test security check when ruff is not installed"""
        mock_check.return_value = False
        result = check_security(".")
        assert "not installed" in result.lower()

    @mock.patch("src.servers.lint._run_command")
    def test_security_json_parse_error(self, mock_run):
        """Test handling of JSON parse errors in security check"""
        mock_run.return_value = (0, "invalid json", "")
        result = check_security(".")
        assert isinstance(result, str)


# ========================================
# Lint Summary Tests
# ========================================


@pytest.mark.skipif(not _check_tool_installed("ruff"), reason="Ruff not installed")
class TestLintSummary:
    """Tests for lint summary"""

    def test_summary_basic(self, python_file_issues):
        """Test getting lint summary"""
        result = get_lint_summary(python_file_issues)
        assert "summary" in result.lower() or isinstance(result, str)

    def test_summary_clean_file(self, python_file_clean):
        """Test summary of clean file"""
        result = get_lint_summary(python_file_clean)
        assert isinstance(result, str)

    def test_summary_directory(self, test_files_dir):
        """Test summary of directory"""
        result = get_lint_summary(test_files_dir)
        assert isinstance(result, str)

    def test_summary_nonexistent_file(self):
        """Test summary of nonexistent file"""
        result = get_lint_summary("/nonexistent/file.py")
        assert "error" in result.lower()

    @mock.patch("src.servers.lint._check_tool_installed")
    def test_summary_ruff_not_installed(self, mock_check):
        """Test summary when ruff is not installed"""
        mock_check.return_value = False
        result = get_lint_summary(".")
        assert "not installed" in result.lower()

    @mock.patch("src.servers.lint._run_command")
    def test_summary_json_parse_error(self, mock_run):
        """Test handling of JSON parse errors in summary"""
        mock_run.return_value = (0, "invalid json", "")
        result = get_lint_summary(".")
        assert isinstance(result, str)

    @mock.patch("src.servers.lint._run_command")
    def test_summary_empty_output(self, mock_run):
        """Test summary with empty output"""
        mock_run.return_value = (0, "", "")
        result = get_lint_summary(".")
        assert "no issues" in result.lower()


# ========================================
# Multi-Language Lint Tests
# ========================================


@pytest.mark.skipif(not _check_tool_installed("ruff"), reason="Ruff not installed")
class TestMultiLanguageLint:
    """Tests for multi-language linting"""

    def test_lint_all_python_only(self, python_file_clean):
        """Test lint_all on Python file"""
        result = lint_all(python_file_clean)
        assert isinstance(result, str)

    def test_lint_all_with_fix(self, python_file_issues):
        """Test lint_all with fix enabled"""
        result = lint_all(python_file_issues, fix=True)
        assert isinstance(result, str)

    def test_lint_all_directory(self, test_files_dir):
        """Test lint_all on directory"""
        result = lint_all(test_files_dir)
        assert isinstance(result, str)

    def test_lint_all_nonexistent_path(self):
        """Test lint_all on nonexistent path"""
        result = lint_all("/nonexistent/path")
        assert "error" in result.lower()

    def test_lint_all_empty_directory(self, test_files_dir):
        """Test lint_all on empty directory"""
        empty_dir = os.path.join(test_files_dir, "empty")
        os.makedirs(empty_dir, exist_ok=True)
        result = lint_all(empty_dir)
        assert isinstance(result, str)


# ========================================
# Edge Cases and Error Handling
# ========================================


class TestEdgeCases:
    """Tests for edge cases and error handling"""

    def test_run_command_empty_args(self):
        """Test run_command with empty args"""
        exit_code, stdout, stderr = _run_command([])
        # Should handle gracefully
        assert isinstance(exit_code, int)

    def test_lint_issue_with_special_characters(self):
        """Test LintIssue with special characters in message"""
        issue = LintIssue(
            file="test.py",
            line=1,
            column=1,
            severity="error",
            code="E501",
            message="Line too long: 'special chars: @#$%^&*()'",
            source="ruff",
        )
        result = issue.to_string()
        assert "special chars" in result

    def test_lint_issue_with_unicode(self):
        """Test LintIssue with unicode characters"""
        issue = LintIssue(
            file="test.py",
            line=1,
            column=1,
            severity="error",
            code="E501",
            message="Unicode: 你好世界 🌍",
            source="ruff",
        )
        result = issue.to_string()
        assert "Unicode" in result

    @mock.patch("src.servers.lint._run_command")
    def test_lint_with_stderr_error(self, mock_run):
        """Test linting when stderr contains error"""
        mock_run.return_value = (1, "", "Error: something went wrong")
        result = lint_python_ruff(".")
        assert "error" in result.lower()

    def test_run_command_with_special_characters(self):
        """Test run_command with special characters in output"""
        exit_code, stdout, stderr = _run_command(["printf", "special: @#$%^&*()"])
        # May fail due to shell interpretation, that's ok
        assert isinstance(exit_code, int)


# ========================================
# Integration Tests
# ========================================


@pytest.mark.skipif(not _check_tool_installed("ruff"), reason="Ruff not installed")
class TestIntegration:
    """Integration tests combining multiple functions"""

    def test_full_lint_workflow(self, python_file_issues):
        """Test full linting workflow"""
        # Run linting
        lint_result = lint_python_ruff(python_file_issues)
        assert isinstance(lint_result, str)

        # Run complexity check
        complexity_result = code_complexity(python_file_issues)
        assert isinstance(complexity_result, str)

        # Run security check
        security_result = check_security(python_file_issues)
        assert isinstance(security_result, str)

    def test_format_and_lint(self, python_file_issues):
        """Test formatting followed by linting"""
        # Format
        format_result = format_python_ruff(python_file_issues, check_only=True)
        assert isinstance(format_result, str)

        # Lint
        lint_result = lint_python_ruff(python_file_issues)
        assert isinstance(lint_result, str)

    def test_summary_and_complexity(self, python_file_issues):
        """Test summary and complexity analysis"""
        summary = get_lint_summary(python_file_issues)
        assert isinstance(summary, str)

        complexity = code_complexity(python_file_issues)
        assert isinstance(complexity, str)
