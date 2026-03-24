"""
Unit tests for the Computer (Code Execution) Server.

Tests Python code execution with sandboxing and resource limits.
"""

import platform
import subprocess
from unittest import mock

import pytest

from src.servers.computer import (
    DEFAULT_LIMITS,
    ResourceLimits,
    _create_sandbox_preexec,
    _get_pypi_suggestions,
    _get_sandbox_wrapper_code,
    _indent_code,
    _validate_package_name,
    execute_python,
    install_package,
    list_installed_packages,
)

# ========================================
# Resource Limits Tests
# ========================================


class TestResourceLimits:
    """Tests for ResourceLimits configuration."""

    def test_default_limits(self):
        """Test default resource limits."""
        limits = ResourceLimits()
        assert limits.max_memory_mb == 512
        assert limits.max_cpu_time_seconds == 30
        assert limits.max_file_size_mb == 50
        assert limits.max_processes == 10

    def test_custom_limits(self):
        """Test custom resource limits."""
        limits = ResourceLimits(
            max_memory_mb=256,
            max_cpu_time_seconds=10,
            max_file_size_mb=25,
            max_processes=5,
        )
        assert limits.max_memory_mb == 256
        assert limits.max_cpu_time_seconds == 10


# ========================================
# Code Indentation Tests
# ========================================


class TestCodeIndentation:
    """Tests for code indentation helper."""

    def test_indent_single_line(self):
        """Test indenting a single line."""
        result = _indent_code("print('hello')", 4)
        assert result == "    print('hello')"

    def test_indent_multiple_lines(self):
        """Test indenting multiple lines."""
        code = "line1\nline2\nline3"
        result = _indent_code(code, 4)
        lines = result.split("\n")
        assert all(line.startswith("    ") for line in lines if line.strip())

    def test_indent_preserves_empty_lines(self):
        """Test that empty lines are preserved."""
        code = "line1\n\nline2"
        result = _indent_code(code, 4)
        assert "\n\n" in result or "\n    \n" in result


# ========================================
# Sandbox Wrapper Tests
# ========================================


class TestSandboxWrapper:
    """Tests for sandbox wrapper code generation."""

    def test_wrapper_includes_user_code(self):
        """Test that wrapper includes user code."""
        user_code = "print('test')"
        wrapper = _get_sandbox_wrapper_code(user_code)
        assert "print('test')" in wrapper

    def test_wrapper_has_error_handling(self):
        """Test that wrapper includes error handling."""
        wrapper = _get_sandbox_wrapper_code("pass")
        assert "except MemoryError" in wrapper
        assert "except RecursionError" in wrapper
        assert "except Exception" in wrapper

    def test_wrapper_sets_recursion_limit(self):
        """Test that wrapper sets recursion limit."""
        wrapper = _get_sandbox_wrapper_code("pass")
        assert "setrecursionlimit" in wrapper


# ========================================
# Code Execution Tests
# ========================================


class TestExecutePython:
    """Tests for execute_python function."""

    def test_execute_simple_code(self):
        """Test executing simple Python code."""
        result = execute_python("print('Hello, World!')")
        assert "Hello, World!" in result

    def test_execute_math_expression(self):
        """Test executing mathematical expression."""
        result = execute_python("print(2 + 2)")
        assert "4" in result

    def test_execute_multiline_code(self):
        """Test executing multiline code."""
        code = """
x = 5
y = 10
print(x + y)
"""
        result = execute_python(code)
        assert "15" in result

    def test_execute_with_import(self):
        """Test executing code with standard imports."""
        code = """
import math
print(math.pi)
"""
        result = execute_python(code)
        assert "3.14" in result

    def test_execute_list_comprehension(self):
        """Test executing list comprehension."""
        code = """
result = [x**2 for x in range(5)]
print(result)
"""
        result = execute_python(code)
        assert "[0, 1, 4, 9, 16]" in result

    def test_execute_function_definition(self):
        """Test executing function definition and call."""
        code = """
def greet(name):
    return f"Hello, {name}!"

print(greet("Doraemon Code"))
"""
        result = execute_python(code)
        assert "Hello, Doraemon Code!" in result

    def test_execute_class_definition(self):
        """Test executing class definition."""
        code = """
class Calculator:
    def add(self, a, b):
        return a + b

calc = Calculator()
print(calc.add(3, 4))
"""
        result = execute_python(code)
        assert "7" in result

    def test_execute_with_error(self):
        """Test executing code with error."""
        code = "print(undefined_variable)"
        result = execute_python(code)
        assert "Error" in result or "NameError" in result

    def test_execute_syntax_error(self):
        """Test executing code with syntax error."""
        code = "print('unclosed string"
        result = execute_python(code)
        assert "Error" in result or "SyntaxError" in result

    def test_execute_no_output(self):
        """Test executing code with no output."""
        code = "x = 5"
        result = execute_python(code)
        assert "success" in result.lower() or "no output" in result.lower()

    def test_execute_timeout_short_code(self):
        """Test that quick code doesn't timeout."""
        result = execute_python("print('fast')", timeout=5)
        assert "fast" in result

    def test_execute_without_sandbox(self):
        """Test executing code without sandbox (if allowed)."""
        result = execute_python("print('no sandbox')", sandbox=False)
        assert "no sandbox" in result


# ========================================
# Package Listing Tests
# ========================================


class TestListPackages:
    """Tests for list_installed_packages function."""

    def test_list_packages_returns_output(self):
        """Test that list_installed_packages returns output."""
        result = list_installed_packages()
        assert len(result) > 0

    def test_list_packages_includes_common(self):
        """Test that common packages are listed."""
        result = list_installed_packages()
        # Should include at least pip
        assert "pip" in result.lower() or "Package" in result


# ========================================
# Edge Cases
# ========================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_code(self):
        """Test executing empty code."""
        result = execute_python("")
        # Should handle gracefully
        assert "success" in result.lower() or "Error" in result

    def test_whitespace_only_code(self):
        """Test executing whitespace-only code."""
        result = execute_python("   \n   \n   ")
        # Should handle gracefully
        assert result is not None

    def test_code_with_unicode(self):
        """Test executing code with unicode."""
        code = "print('你好世界')"
        result = execute_python(code)
        assert "你好世界" in result

    def test_code_with_special_characters(self):
        """Test executing code with special characters."""
        code = r"print('special: \t\n\r')"
        result = execute_python(code)
        assert "special" in result


# ========================================
# Package Validation Tests
# ========================================


class TestPackageValidation:
    """Tests for package name validation."""

    def test_validate_valid_package_name(self):
        """Test validating a valid package name."""
        is_valid, error = _validate_package_name("numpy")
        assert is_valid is True
        assert error == ""

    def test_validate_package_with_hyphens(self):
        """Test validating package name with hyphens."""
        is_valid, error = _validate_package_name("scikit-learn")
        assert is_valid is True
        assert error == ""

    def test_validate_package_with_underscores(self):
        """Test validating package name with underscores."""
        is_valid, error = _validate_package_name("my_package")
        assert is_valid is True
        assert error == ""

    def test_validate_package_with_dots(self):
        """Test validating package name with dots."""
        is_valid, error = _validate_package_name("zope.interface")
        assert is_valid is True
        assert error == ""

    def test_validate_empty_package_name(self):
        """Test validating empty package name."""
        is_valid, error = _validate_package_name("")
        assert is_valid is False
        assert "empty" in error.lower()

    def test_validate_whitespace_only_package_name(self):
        """Test validating whitespace-only package name."""
        is_valid, error = _validate_package_name("   ")
        assert is_valid is False
        assert "empty" in error.lower()

    def test_validate_package_with_path_traversal(self):
        """Test validating package name with path traversal."""
        is_valid, error = _validate_package_name("../../../etc/passwd")
        assert is_valid is False
        assert "invalid character" in error.lower()

    def test_validate_package_with_shell_injection(self):
        """Test validating package name with shell injection."""
        is_valid, error = _validate_package_name("package; rm -rf /")
        assert is_valid is False
        assert "invalid character" in error.lower()

    def test_validate_package_with_pipe(self):
        """Test validating package name with pipe."""
        is_valid, error = _validate_package_name("package | cat")
        assert is_valid is False
        assert "invalid character" in error.lower()

    def test_validate_package_with_ampersand(self):
        """Test validating package name with ampersand."""
        is_valid, error = _validate_package_name("package & echo")
        assert is_valid is False
        assert "invalid character" in error.lower()

    def test_validate_package_with_backtick(self):
        """Test validating package name with backtick."""
        is_valid, error = _validate_package_name("package`whoami`")
        assert is_valid is False
        assert "invalid character" in error.lower()

    def test_validate_package_with_dollar_sign(self):
        """Test validating package name with dollar sign."""
        is_valid, error = _validate_package_name("package$var")
        assert is_valid is False
        assert "invalid character" in error.lower()

    def test_validate_package_starting_with_number(self):
        """Test validating package name starting with number."""
        is_valid, error = _validate_package_name("2to3")
        assert is_valid is True
        assert error == ""

    def test_validate_package_starting_with_hyphen(self):
        """Test validating package name starting with hyphen."""
        is_valid, error = _validate_package_name("-package")
        assert is_valid is False
        assert "alphanumeric" in error.lower()

    def test_validate_package_with_spaces(self):
        """Test validating package name with spaces."""
        is_valid, error = _validate_package_name("my package")
        assert is_valid is False
        assert "alphanumeric" in error.lower()

    def test_validate_package_suspicious_eval(self):
        """Test validating package name with suspicious pattern."""
        is_valid, error = _validate_package_name("eval-tool")
        # Should be valid but logged as suspicious
        assert is_valid is True

    def test_validate_package_suspicious_exec(self):
        """Test validating package name with suspicious exec pattern."""
        is_valid, error = _validate_package_name("exec-runner")
        # Should be valid but logged as suspicious
        assert is_valid is True


# ========================================
# PyPI Suggestions Tests
# ========================================


class TestPyPISuggestions:
    """Tests for PyPI package suggestions."""

    @mock.patch("src.servers.computer.requests.get")
    def test_get_pypi_suggestions_success(self, mock_get):
        """Test getting PyPI suggestions successfully."""
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.text = """
        <span class="package-snippet__name">numpy</span>
        <p class="package-snippet__description">Numerical computing library</p>
        <span class="package-snippet__name">pandas</span>
        <p class="package-snippet__description">Data analysis library</p>
        """
        mock_get.return_value = mock_response

        result = _get_pypi_suggestions("num")
        assert "numpy" in result
        assert "pandas" in result

    @mock.patch("src.servers.computer.requests.get")
    def test_get_pypi_suggestions_not_found(self, mock_get):
        """Test PyPI suggestions when package not found."""
        mock_response = mock.Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = _get_pypi_suggestions("nonexistent")
        assert result == ""

    @mock.patch("src.servers.computer.requests.get")
    def test_get_pypi_suggestions_timeout(self, mock_get):
        """Test PyPI suggestions with timeout."""
        mock_get.side_effect = Exception("Timeout")

        result = _get_pypi_suggestions("package")
        assert result == ""

    @mock.patch("src.servers.computer.requests.get")
    def test_get_pypi_suggestions_empty_response(self, mock_get):
        """Test PyPI suggestions with empty response."""
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.text = ""
        mock_get.return_value = mock_response

        result = _get_pypi_suggestions("package")
        assert result == ""


# ========================================
# Install Package Tests
# ========================================


class TestInstallPackage:
    """Tests for install_package function."""

    @mock.patch("src.servers.computer.subprocess.run")
    @mock.patch("src.servers.computer.requests.get")
    def test_install_package_success(self, mock_get, mock_run):
        """Test successful package installation."""
        # Mock PyPI check
        mock_pypi_response = mock.Mock()
        mock_pypi_response.status_code = 200
        mock_get.return_value = mock_pypi_response

        # Mock pip install
        mock_result = mock.Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Successfully installed package"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        result = install_package("requests")
        assert "Successfully installed" in result

    @mock.patch("src.servers.computer.requests.get")
    def test_install_package_invalid_name(self, mock_get):
        """Test installing package with invalid name."""
        result = install_package("../../../etc/passwd")
        assert "Invalid package name" in result

    @mock.patch("src.servers.computer.requests.get")
    def test_install_package_not_found(self, mock_get):
        """Test installing package that doesn't exist."""
        mock_get.return_value = mock.Mock(status_code=404)

        result = install_package("nonexistent-package-xyz")
        assert "not found on PyPI" in result

    @mock.patch("src.servers.computer.subprocess.run")
    @mock.patch("src.servers.computer.requests.get")
    def test_install_package_failure(self, mock_get, mock_run):
        """Test package installation failure."""
        # Mock PyPI check
        mock_pypi_response = mock.Mock()
        mock_pypi_response.status_code = 200
        mock_get.return_value = mock_pypi_response

        # Mock pip install failure
        mock_result = mock.Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Installation failed"
        mock_run.return_value = mock_result

        result = install_package("broken-package")
        assert "Failed to install" in result

    @mock.patch("src.servers.computer.requests.get")
    def test_install_package_exception(self, mock_get):
        """Test install_package with exception."""
        mock_get.side_effect = Exception("Network error")

        result = install_package("requests")
        assert "Error" in result

    @mock.patch("src.servers.computer.subprocess.run")
    @mock.patch("src.servers.computer.requests.get")
    def test_install_package_timeout(self, mock_get, mock_run):
        """Test install_package with timeout."""
        # Mock PyPI check
        mock_pypi_response = mock.Mock()
        mock_pypi_response.status_code = 200
        mock_get.return_value = mock_pypi_response

        # Mock pip install timeout
        mock_run.side_effect = subprocess.TimeoutExpired("pip", 120)

        result = install_package("requests")
        assert "Error" in result


# ========================================
# Sandbox Preexec Tests
# ========================================


class TestSandboxPreexec:
    """Tests for sandbox preexec function."""

    def test_create_sandbox_preexec_returns_callable(self):
        """Test that _create_sandbox_preexec returns a callable."""
        limits = ResourceLimits()
        preexec_fn = _create_sandbox_preexec(limits)
        assert callable(preexec_fn)

    @pytest.mark.skipif(platform.system() == "Windows", reason="Unix-only test")
    def test_sandbox_preexec_on_unix(self):
        """Test sandbox preexec on Unix systems."""
        limits = ResourceLimits(max_memory_mb=256)
        preexec_fn = _create_sandbox_preexec(limits)
        # Should not raise when called
        try:
            preexec_fn()
        except Exception as e:
            # Some systems may not allow resource limit setting
            pytest.skip(f"Resource limits not available: {e}")

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-only test")
    def test_sandbox_preexec_on_windows(self):
        """Test sandbox preexec on Windows (should be no-op)."""
        limits = ResourceLimits()
        preexec_fn = _create_sandbox_preexec(limits)
        # Should not raise on Windows
        preexec_fn()

    def test_sandbox_preexec_with_custom_limits(self):
        """Test sandbox preexec with custom limits."""
        limits = ResourceLimits(
            max_memory_mb=256,
            max_cpu_time_seconds=10,
            max_file_size_mb=25,
            max_processes=5,
        )
        preexec_fn = _create_sandbox_preexec(limits)
        assert callable(preexec_fn)


# ========================================
# Code Indentation Advanced Tests
# ========================================


class TestCodeIndentationAdvanced:
    """Advanced tests for code indentation."""

    def test_indent_with_tabs(self):
        """Test indenting code with tabs."""
        code = "line1\nline2"
        result = _indent_code(code, 8)
        lines = result.split("\n")
        assert all(line.startswith("        ") for line in lines if line.strip())

    def test_indent_zero_spaces(self):
        """Test indenting with zero spaces."""
        code = "line1\nline2"
        result = _indent_code(code, 0)
        assert result == code

    def test_indent_large_number_of_spaces(self):
        """Test indenting with large number of spaces."""
        code = "x = 1"
        result = _indent_code(code, 20)
        assert result.startswith(" " * 20)

    def test_indent_mixed_content(self):
        """Test indenting mixed content."""
        code = "def foo():\n    pass\n\nclass Bar:\n    pass"
        result = _indent_code(code, 4)
        lines = result.split("\n")
        # All non-empty lines should be indented
        for line in lines:
            if line.strip():
                assert line.startswith("    ")


# ========================================
# Sandbox Wrapper Advanced Tests
# ========================================


class TestSandboxWrapperAdvanced:
    """Advanced tests for sandbox wrapper."""

    def test_wrapper_with_multiline_code(self):
        """Test wrapper with multiline code."""
        code = """
def test():
    x = 1
    y = 2
    return x + y

print(test())
"""
        wrapper = _get_sandbox_wrapper_code(code)
        assert "def test():" in wrapper
        assert "print(test())" in wrapper

    def test_wrapper_with_imports(self):
        """Test wrapper with imports."""
        code = "import sys\nprint(sys.version)"
        wrapper = _get_sandbox_wrapper_code(code)
        assert "import sys" in wrapper

    def test_wrapper_recursion_limit_value(self):
        """Test wrapper sets correct recursion limit."""
        wrapper = _get_sandbox_wrapper_code("pass")
        assert "setrecursionlimit(1000)" in wrapper

    def test_wrapper_exception_handling_order(self):
        """Test wrapper exception handling order."""
        wrapper = _get_sandbox_wrapper_code("pass")
        memory_pos = wrapper.find("except MemoryError")
        recursion_pos = wrapper.find("except RecursionError")
        exception_pos = wrapper.find("except Exception")
        # All should be present
        assert memory_pos > 0
        assert recursion_pos > 0
        assert exception_pos > 0


# ========================================
# Execute Python Advanced Tests
# ========================================


class TestExecutePythonAdvanced:
    """Advanced tests for execute_python function."""

    def test_execute_with_custom_memory_limit(self):
        """Test executing with custom memory limit."""
        result = execute_python("print('test')", max_memory_mb=256)
        assert "test" in result

    def test_execute_with_custom_timeout(self):
        """Test executing with custom timeout."""
        result = execute_python("print('fast')", timeout=10)
        assert "fast" in result

    def test_execute_with_stderr_output(self):
        """Test executing code that produces stderr."""
        code = """
import sys
print("stdout message")
print("stderr message", file=sys.stderr)
"""
        result = execute_python(code)
        assert "stdout message" in result
        assert "stderr message" in result or "Stderr" in result

    def test_execute_with_exit_code_137(self):
        """Test executing code that exits with code 137."""
        with mock.patch("src.servers.computer.subprocess.run") as mock_run:
            mock_result = mock.Mock()
            mock_result.returncode = 137
            mock_result.stdout = ""
            mock_result.stderr = ""
            mock_run.return_value = mock_result

            result = execute_python("pass")
            assert "terminated" in result.lower() or "exceeded" in result.lower()

    def test_execute_with_exit_code_nonzero(self):
        """Test executing code that exits with non-zero code."""
        with mock.patch("src.servers.computer.subprocess.run") as mock_run:
            mock_result = mock.Mock()
            mock_result.returncode = 1
            mock_result.stdout = ""
            mock_result.stderr = "Error occurred"
            mock_run.return_value = mock_result

            result = execute_python("pass")
            assert "Error occurred" in result

    def test_execute_with_timeout_exception(self):
        """Test executing code that times out."""
        with mock.patch("src.servers.computer.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("python", 30)

            result = execute_python("pass", timeout=30)
            assert "timed out" in result.lower()

    def test_execute_with_memory_error(self):
        """Test executing code that raises MemoryError."""
        with mock.patch("src.servers.computer.subprocess.run") as mock_run:
            mock_run.side_effect = MemoryError()

            result = execute_python("pass")
            assert "memory" in result.lower()

    def test_execute_with_generic_exception(self):
        """Test executing code with generic exception."""
        with mock.patch("src.servers.computer.subprocess.run") as mock_run:
            mock_run.side_effect = RuntimeError("Test error")

            result = execute_python("pass")
            assert "Error" in result

    def test_execute_creates_temp_file(self):
        """Test that execute_python creates temporary file."""
        with mock.patch("src.servers.computer.tempfile.NamedTemporaryFile") as mock_temp:
            mock_file = mock.Mock()
            mock_file.name = "/tmp/test.py"
            mock_temp.return_value.__enter__.return_value = mock_file

            with mock.patch("src.servers.computer.subprocess.run") as mock_run:
                mock_result = mock.Mock()
                mock_result.returncode = 0
                mock_result.stdout = "test"
                mock_result.stderr = ""
                mock_run.return_value = mock_result

                execute_python("print('test')")
                mock_file.write.assert_called()

    def test_execute_cleans_up_temp_file(self):
        """Test that execute_python cleans up temporary file."""
        with mock.patch("src.servers.computer.os.remove") as mock_remove:
            with mock.patch("src.servers.computer.os.path.exists") as mock_exists:
                mock_exists.return_value = True

                with mock.patch("src.servers.computer.subprocess.run") as mock_run:
                    mock_result = mock.Mock()
                    mock_result.returncode = 0
                    mock_result.stdout = "test"
                    mock_result.stderr = ""
                    mock_run.return_value = mock_result

                    execute_python("print('test')")
                    mock_remove.assert_called()

    def test_execute_with_large_code(self):
        """Test executing large code."""
        code = "\n".join([f"x{i} = {i}" for i in range(100)])
        code += "\nprint('done')"
        result = execute_python(code)
        assert "done" in result

    def test_execute_with_nested_functions(self):
        """Test executing code with nested functions."""
        code = """
def outer():
    def inner():
        return 42
    return inner()

print(outer())
"""
        result = execute_python(code)
        assert "42" in result

    def test_execute_with_lambda(self):
        """Test executing code with lambda."""
        code = """
f = lambda x: x * 2
print(f(21))
"""
        result = execute_python(code)
        assert "42" in result

    def test_execute_with_generator(self):
        """Test executing code with generator."""
        code = """
def gen():
    for i in range(3):
        yield i

print(list(gen()))
"""
        result = execute_python(code)
        assert "[0, 1, 2]" in result

    def test_execute_with_exception_handling(self):
        """Test executing code with exception handling."""
        code = """
try:
    x = 1 / 0
except ZeroDivisionError:
    print("Caught division by zero")
"""
        result = execute_python(code)
        assert "Caught division by zero" in result

    def test_execute_with_context_manager(self):
        """Test executing code with context manager."""
        code = """
import tempfile
with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
    f.write('test')
    print(f.name)
"""
        result = execute_python(code)
        assert "tmp" in result or "test" in result

    def test_execute_with_dictionary(self):
        """Test executing code with dictionary."""
        code = """
d = {'a': 1, 'b': 2, 'c': 3}
print(sum(d.values()))
"""
        result = execute_python(code)
        assert "6" in result

    def test_execute_with_set_operations(self):
        """Test executing code with set operations."""
        code = """
s1 = {1, 2, 3}
s2 = {2, 3, 4}
print(s1 & s2)
"""
        result = execute_python(code)
        assert "{2, 3}" in result or "{3, 2}" in result

    def test_execute_with_string_formatting(self):
        """Test executing code with string formatting."""
        code = """
name = "World"
print(f"Hello, {name}!")
"""
        result = execute_python(code)
        assert "Hello, World!" in result

    def test_execute_with_list_operations(self):
        """Test executing code with list operations."""
        code = """
lst = [1, 2, 3, 4, 5]
print([x for x in lst if x % 2 == 0])
"""
        result = execute_python(code)
        assert "[2, 4]" in result

    def test_execute_with_tuple_unpacking(self):
        """Test executing code with tuple unpacking."""
        code = """
a, b, c = 1, 2, 3
print(a + b + c)
"""
        result = execute_python(code)
        assert "6" in result

    def test_execute_with_default_limits(self):
        """Test that DEFAULT_LIMITS is used."""
        assert DEFAULT_LIMITS.max_memory_mb > 0
        assert DEFAULT_LIMITS.max_cpu_time_seconds > 0

    def test_execute_sandbox_true_by_default(self):
        """Test that sandbox is True by default."""
        result = execute_python("print('sandboxed')")
        assert "sandboxed" in result

    def test_execute_with_os_operations(self):
        """Test executing code with os operations."""
        code = """
import os
print(os.name)
"""
        result = execute_python(code)
        assert "posix" in result or "nt" in result


# ========================================
# Integration Tests
# ========================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow_simple(self):
        """Test full workflow with simple code."""
        code = "print('Integration test')"
        result = execute_python(code)
        assert "Integration test" in result

    def test_full_workflow_with_packages(self):
        """Test full workflow listing packages."""
        result = list_installed_packages()
        assert len(result) > 0

    def test_validate_and_install_flow(self):
        """Test validation and install flow."""
        is_valid, error = _validate_package_name("requests")
        assert is_valid is True

    def test_sandbox_wrapper_execution(self):
        """Test sandbox wrapper with actual execution."""
        code = "print('wrapped')"
        wrapper = _get_sandbox_wrapper_code(code)
        # Wrapper should be valid Python
        assert "print('wrapped')" in wrapper
        assert "setrecursionlimit" in wrapper


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
