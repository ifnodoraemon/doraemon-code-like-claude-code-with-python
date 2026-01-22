"""
Unit tests for the Shell Command Execution Server.

Tests command execution, security controls, and background processes.
"""

import os
import shutil

import pytest

from src.servers.shell import (
    _is_command_blocked,
    _is_command_sensitive,
    _truncate_output,
    execute_command,
    get_environment_info,
)


# ========================================
# Command Validation Tests
# ========================================

class TestCommandValidation:
    """Tests for command validation functions"""

    def test_blocked_command_rm_rf_root(self):
        """Test that rm -rf / is blocked"""
        assert _is_command_blocked("rm -rf /") is True

    def test_blocked_command_rm_rf_star(self):
        """Test that rm -rf /* is blocked"""
        assert _is_command_blocked("rm -rf /*") is True

    def test_blocked_command_fork_bomb(self):
        """Test that fork bomb is blocked"""
        assert _is_command_blocked(":(){:|:&};:") is True

    def test_safe_command_ls(self):
        """Test that ls is not blocked"""
        assert _is_command_blocked("ls -la") is False

    def test_safe_command_git(self):
        """Test that git commands are not blocked"""
        assert _is_command_blocked("git status") is False

    def test_sensitive_command_rm_rf(self):
        """Test that rm -rf is marked as sensitive"""
        assert _is_command_sensitive("rm -rf some_dir") is True

    def test_sensitive_command_sudo(self):
        """Test that sudo is marked as sensitive"""
        assert _is_command_sensitive("sudo apt update") is True

    def test_safe_command_not_sensitive(self):
        """Test that safe commands are not sensitive"""
        assert _is_command_sensitive("echo hello") is False


# ========================================
# Output Truncation Tests
# ========================================

class TestOutputTruncation:
    """Tests for output truncation"""

    def test_short_output_unchanged(self):
        """Test that short output is not truncated"""
        output = "Hello, World!"
        result = _truncate_output(output, max_size=100)
        assert result == output

    def test_long_output_truncated(self):
        """Test that long output is truncated"""
        output = "x" * 1000
        result = _truncate_output(output, max_size=100)
        assert len(result) < len(output)
        assert "truncated" in result.lower()

    def test_truncation_preserves_start_and_end(self):
        """Test that truncation preserves start and end"""
        output = "START" + "x" * 1000 + "END"
        result = _truncate_output(output, max_size=100)
        assert "START" in result
        assert "END" in result


# ========================================
# Command Execution Tests
# ========================================

class TestExecuteCommand:
    """Tests for command execution"""

    def test_execute_simple_command(self):
        """Test executing a simple command"""
        result = execute_command("echo hello")
        assert "hello" in result

    def test_execute_pwd(self):
        """Test executing pwd command"""
        result = execute_command("pwd")
        assert "/" in result  # Should contain some path

    def test_execute_with_working_directory(self):
        """Test executing command in specific directory"""
        # Use a directory inside the project
        tmpdir = os.path.join(os.getcwd(), ".test_shell_dir")
        os.makedirs(tmpdir, exist_ok=True)
        try:
            result = execute_command("pwd", working_directory=tmpdir)
            assert tmpdir in result or "test_shell_dir" in result
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_execute_blocked_command(self):
        """Test that blocked commands are rejected"""
        result = execute_command("rm -rf /")
        assert "blocked" in result.lower() or "error" in result.lower()

    def test_execute_invalid_directory(self):
        """Test executing in non-existent directory"""
        result = execute_command("echo test", working_directory="/nonexistent/path")
        assert "error" in result.lower()

    def test_execute_with_timeout(self):
        """Test command timeout"""
        # This should complete quickly
        result = execute_command("echo fast", timeout=5)
        assert "fast" in result

    def test_execute_exit_code(self):
        """Test that exit code is captured"""
        result = execute_command("exit 1")
        assert "exit code" in result.lower() or "1" in result

    def test_execute_stderr_capture(self):
        """Test that stderr is captured"""
        result = execute_command("ls /nonexistent_path_12345")
        assert "stderr" in result.lower() or "no such file" in result.lower()


# ========================================
# Environment Info Tests
# ========================================

class TestEnvironmentInfo:
    """Tests for environment info retrieval"""

    def test_get_environment_info(self):
        """Test that environment info is returned"""
        result = get_environment_info()
        assert "OS:" in result
        assert "Python:" in result
        assert "Shell:" in result

    def test_environment_info_includes_path(self):
        """Test that PATH is included"""
        result = get_environment_info()
        assert "PATH" in result


# ========================================
# Edge Cases
# ========================================

class TestEdgeCases:
    """Tests for edge cases"""

    def test_empty_command(self):
        """Test executing empty command"""
        result = execute_command("")
        # Should either succeed with no output or return an error
        assert isinstance(result, str)

    def test_command_with_special_characters(self):
        """Test command with special characters"""
        result = execute_command('echo "hello world"')
        assert "hello world" in result

    def test_command_with_pipes(self):
        """Test command with pipes"""
        result = execute_command("echo hello | cat")
        assert "hello" in result

    def test_command_with_environment_variable(self):
        """Test command using environment variables"""
        result = execute_command("echo $HOME")
        assert result.strip() != "$HOME"  # Should be expanded

    def test_multiline_output(self):
        """Test command with multiline output"""
        result = execute_command("echo -e 'line1\\nline2\\nline3'")
        assert "line1" in result
        assert "line2" in result
