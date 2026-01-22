"""
Unit tests for the Computer (Code Execution) Server.

Tests Python code execution with sandboxing and resource limits.
"""

import pytest

from src.servers.computer import (
    ResourceLimits,
    _get_sandbox_wrapper_code,
    _indent_code,
    execute_python,
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

print(greet("Polymath"))
"""
        result = execute_python(code)
        assert "Hello, Polymath!" in result

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
