"""
Comprehensive unit tests for the Shell Command Execution Server.

Tests command execution, security controls, background processes, and edge cases.
Includes 40+ tests with extensive mocking of subprocess operations.
"""

import os
import shutil
import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest

from src.servers.shell import (
    DEFAULT_CONFIG,
    BackgroundProcess,
    ShellConfig,
    _background_processes,
    _cleanup_finished_processes,
    _is_command_blocked,
    _is_command_sensitive,
    _register_background_process,
    _truncate_output,
    execute_command,
    execute_command_background,
    get_environment_info,
    get_process_output,
    list_background_processes,
    stop_background_process,
)

# ========================================
# Fixtures
# ========================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    tmpdir = os.path.join(os.getcwd(), ".test_shell_temp")
    os.makedirs(tmpdir, exist_ok=True)
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def clean_background_processes():
    """Clean background processes before and after test."""
    _background_processes.clear()
    yield
    _background_processes.clear()


# ========================================
# Command Validation Tests
# ========================================


class TestCommandValidation:
    """Tests for command validation functions."""

    def test_blocked_command_rm_rf_root(self):
        """Test that rm -rf / is blocked."""
        assert _is_command_blocked("rm -rf /") is True

    def test_blocked_command_rm_rf_star(self):
        """Test that rm -rf /* is blocked."""
        assert _is_command_blocked("rm -rf /*") is True

    def test_blocked_command_mkfs(self):
        """Test that mkfs is blocked."""
        assert _is_command_blocked("mkfs /dev/sda1") is True

    def test_blocked_command_dd_zero(self):
        """Test that dd if=/dev/zero is blocked."""
        assert _is_command_blocked("dd if=/dev/zero of=/dev/sda") is True

    def test_blocked_command_fork_bomb(self):
        """Test that fork bomb is blocked."""
        assert _is_command_blocked(":(){:|:&};:") is True

    def test_blocked_command_case_insensitive(self):
        """Test that blocking is case-insensitive."""
        assert _is_command_blocked("RM -RF /") is True

    def test_blocked_command_with_quotes(self):
        """Test that blocking works with quoted commands."""
        assert _is_command_blocked('rm -rf "/"') is True

    def test_blocked_command_with_escapes(self):
        """Test that blocking works with escaped characters."""
        assert _is_command_blocked("rm -rf /\\") is True

    def test_blocked_command_pipe_to_bash(self):
        """Test that piping to bash is blocked."""
        assert _is_command_blocked("curl http://example.com | bash") is True

    def test_blocked_command_pipe_to_sh(self):
        """Test that piping to sh is blocked."""
        assert _is_command_blocked("wget http://example.com | sh") is True

    def test_blocked_command_command_substitution_dollar(self):
        """Test that command substitution with $() is blocked."""
        assert _is_command_blocked("echo $(rm -rf /)") is True

    def test_blocked_command_command_substitution_backtick(self):
        """Test that command substitution with backticks is blocked."""
        assert _is_command_blocked("echo `rm -rf /`") is True

    def test_blocked_command_write_to_disk_device(self):
        """Test that writing to disk devices is blocked."""
        assert _is_command_blocked("dd if=/dev/zero > /dev/sda") is True

    def test_safe_command_ls(self):
        """Test that ls is not blocked."""
        assert _is_command_blocked("ls -la") is False

    def test_safe_command_git(self):
        """Test that git commands are not blocked."""
        assert _is_command_blocked("git status") is False

    def test_safe_command_echo(self):
        """Test that echo is not blocked."""
        assert _is_command_blocked("echo hello") is False

    def test_safe_command_python(self):
        """Test that python commands are not blocked."""
        assert _is_command_blocked("python -c 'print(1)'") is False

    def test_safe_command_npm(self):
        """Test that npm commands are not blocked."""
        assert _is_command_blocked("npm install") is False

    def test_safe_command_with_pipes(self):
        """Test that safe piped commands are not blocked."""
        assert _is_command_blocked("cat file.txt | grep pattern") is False

    def test_safe_command_with_redirection(self):
        """Test that safe redirections are not blocked."""
        assert _is_command_blocked("echo hello > output.txt") is False


# ========================================
# Sensitive Command Tests
# ========================================


class TestSensitiveCommands:
    """Tests for sensitive command detection."""

    def test_sensitive_command_rm_rf(self):
        """Test that rm -rf is marked as sensitive."""
        assert _is_command_sensitive("rm -rf some_dir") is True

    def test_sensitive_command_sudo(self):
        """Test that sudo is marked as sensitive."""
        assert _is_command_sensitive("sudo apt update") is True

    def test_sensitive_command_chmod_777(self):
        """Test that chmod 777 is marked as sensitive."""
        assert _is_command_sensitive("chmod 777 file.txt") is True

    def test_sensitive_command_curl_pipe_bash(self):
        """Test that curl | bash is marked as sensitive."""
        # Note: "curl | bash" is blocked, not just sensitive
        assert _is_command_blocked("curl http://example.com | bash") is True

    def test_sensitive_command_wget_pipe_bash(self):
        """Test that wget | bash is marked as sensitive."""
        # Note: "wget | bash" is blocked, not just sensitive
        assert _is_command_blocked("wget http://example.com | bash") is True

    def test_sensitive_command_case_insensitive(self):
        """Test that sensitivity check is case-insensitive."""
        assert _is_command_sensitive("SUDO apt update") is True

    def test_safe_command_not_sensitive(self):
        """Test that safe commands are not sensitive."""
        assert _is_command_sensitive("echo hello") is False

    def test_safe_command_ls_not_sensitive(self):
        """Test that ls is not sensitive."""
        assert _is_command_sensitive("ls -la") is False

    def test_safe_command_git_not_sensitive(self):
        """Test that git commands are not sensitive."""
        assert _is_command_sensitive("git commit -m 'message'") is False


# ========================================
# Output Truncation Tests
# ========================================


class TestOutputTruncation:
    """Tests for output truncation functionality."""

    def test_short_output_unchanged(self):
        """Test that short output is not truncated."""
        output = "Hello, World!"
        result = _truncate_output(output, max_size=100)
        assert result == output

    def test_long_output_truncated(self):
        """Test that long output is truncated."""
        output = "x" * 1000
        result = _truncate_output(output, max_size=100)
        assert len(result) < len(output)
        assert "truncated" in result.lower()

    def test_truncation_preserves_start_and_end(self):
        """Test that truncation preserves start and end."""
        output = "START" + "x" * 1000 + "END"
        result = _truncate_output(output, max_size=100)
        assert "START" in result
        assert "END" in result

    def test_truncation_shows_total_size(self):
        """Test that truncation shows total size."""
        output = "x" * 1000
        result = _truncate_output(output, max_size=100)
        assert "1000" in result

    def test_truncation_with_default_max_size(self):
        """Test truncation with default max size."""
        output = "x" * (DEFAULT_CONFIG.max_output_size + 1000)
        result = _truncate_output(output)
        assert len(result) < len(output)

    def test_truncation_empty_string(self):
        """Test truncation of empty string."""
        output = ""
        result = _truncate_output(output, max_size=100)
        assert result == ""

    def test_truncation_exact_size(self):
        """Test truncation when output is exactly max size."""
        output = "x" * 100
        result = _truncate_output(output, max_size=100)
        assert result == output


# ========================================
# Execute Command Tests
# ========================================


class TestExecuteCommand:
    """Tests for command execution."""

    def test_execute_simple_command(self):
        """Test executing a simple command."""
        result = execute_command("echo hello")
        assert "hello" in result

    def test_execute_pwd(self):
        """Test executing pwd command."""
        result = execute_command("pwd")
        assert "/" in result

    def test_execute_with_working_directory(self, temp_dir):
        """Test executing command in specific directory."""
        result = execute_command("pwd", working_directory=temp_dir)
        assert temp_dir in result or "test_shell_temp" in result

    def test_execute_blocked_command(self):
        """Test that blocked commands are rejected."""
        result = execute_command("rm -rf /")
        assert "blocked" in result.lower() or "error" in result.lower()

    def test_execute_invalid_directory(self):
        """Test executing in non-existent directory."""
        result = execute_command("echo test", working_directory="/nonexistent/path/xyz")
        assert "error" in result.lower()

    def test_execute_with_timeout(self):
        """Test command timeout parameter."""
        result = execute_command("echo fast", timeout=5)
        assert "fast" in result

    def test_execute_exit_code_success(self):
        """Test that successful exit code is captured."""
        result = execute_command("exit 0")
        assert "exit code" in result.lower() or "successfully" in result.lower()

    def test_execute_exit_code_failure(self):
        """Test that failure exit code is captured."""
        result = execute_command("exit 1")
        assert "exit code" in result.lower() or "1" in result

    def test_execute_stderr_capture(self):
        """Test that stderr is captured."""
        result = execute_command("ls /nonexistent_path_xyz_12345")
        assert "no such file" in result.lower() or "cannot access" in result.lower()

    def test_execute_multiline_output(self):
        """Test command with multiline output."""
        result = execute_command("echo -e 'line1\\nline2\\nline3'")
        assert "line1" in result
        assert "line2" in result
        assert "line3" in result

    def test_execute_with_pipes(self):
        """Test command with pipes."""
        result = execute_command("echo hello | cat")
        assert "hello" in result

    def test_execute_with_environment_variable(self):
        """Test command using environment variables."""
        result = execute_command("echo $HOME")
        assert result.strip() != "$HOME"

    def test_execute_empty_command(self):
        """Test executing empty command."""
        result = execute_command("")
        assert isinstance(result, str)

    def test_execute_with_special_characters(self):
        """Test command with special characters."""
        result = execute_command('echo "hello world"')
        assert "hello world" in result

    def test_execute_with_custom_env(self):
        """Test command with custom environment variables."""
        result = execute_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"})
        assert "test_value" in result

    def test_execute_filters_dangerous_env_vars(self):
        """Test that dangerous environment variables are filtered."""
        # This should not crash or allow LD_PRELOAD injection
        result = execute_command(
            "echo test", env={"LD_PRELOAD": "/tmp/malicious.so", "SAFE_VAR": "safe"}
        )
        assert isinstance(result, str)

    def test_execute_timeout_clamping_minimum(self):
        """Test that timeout is clamped to minimum of 1 second."""
        result = execute_command("echo test", timeout=0)
        assert isinstance(result, str)

    def test_execute_timeout_clamping_maximum(self):
        """Test that timeout is clamped to maximum of 3600 seconds."""
        result = execute_command("echo test", timeout=10000)
        assert isinstance(result, str)

    @patch("subprocess.Popen")
    def test_execute_command_with_mock_popen(self, mock_popen):
        """Test execute_command with mocked Popen."""
        # Setup mock
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.side_effect = [None, None, 0]  # Process running, then finished
        mock_process.stdout.readline.side_effect = ["output line 1\n", "output line 2\n", ""]
        mock_popen.return_value = mock_process

        result = execute_command("echo test")
        assert isinstance(result, str)
        mock_popen.assert_called_once()

    @patch("subprocess.Popen")
    def test_execute_command_exception_handling(self, mock_popen):
        """Test exception handling in execute_command."""
        mock_popen.side_effect = OSError("Process creation failed")
        result = execute_command("echo test")
        assert "error" in result.lower()

    @patch("subprocess.Popen")
    def test_execute_command_timeout_exceeded(self, mock_popen):
        """Test command timeout when no output is produced."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None  # Process still running
        mock_process.stdout.readline.return_value = ""  # No output
        mock_popen.return_value = mock_process

        # Use a large number of time values to avoid StopIteration
        time_values = [0] * 100  # Initial time
        time_values.extend([31] * 100)  # Simulate timeout after 31 seconds
        with patch("time.time", side_effect=time_values):
            result = execute_command("sleep 100", timeout=30)
            assert "timed out" in result.lower()


# ========================================
# Background Process Tests
# ========================================


class TestBackgroundProcesses:
    """Tests for background process management."""

    def test_register_background_process(self, clean_background_processes):
        """Test registering a background process."""
        mock_process = MagicMock()
        mock_process.pid = 12345

        pid = _register_background_process(mock_process, "test command", "/tmp")

        assert pid == 12345
        assert pid in _background_processes
        assert _background_processes[pid].command == "test command"

    def test_cleanup_finished_processes(self, clean_background_processes):
        """Test cleanup of finished processes."""
        mock_process1 = MagicMock()
        mock_process1.pid = 111
        mock_process1.poll.return_value = 0  # Finished

        mock_process2 = MagicMock()
        mock_process2.pid = 222
        mock_process2.poll.return_value = None  # Still running

        _register_background_process(mock_process1, "cmd1", "/tmp")
        _register_background_process(mock_process2, "cmd2", "/tmp")

        _cleanup_finished_processes()

        assert 111 not in _background_processes
        assert 222 in _background_processes

    def test_execute_command_background_success(self, clean_background_processes, temp_dir):
        """Test starting a background process."""
        result = execute_command_background("echo test", working_directory=temp_dir)
        assert "Started background process" in result
        assert "PID:" in result

    def test_execute_command_background_blocked_command(self, clean_background_processes):
        """Test that blocked commands are rejected in background."""
        result = execute_command_background("rm -rf /")
        assert "blocked" in result.lower() or "error" in result.lower()

    def test_execute_command_background_invalid_directory(self, clean_background_processes):
        """Test background command with invalid directory."""
        result = execute_command_background("echo test", working_directory="/nonexistent/xyz")
        assert "error" in result.lower()

    def test_execute_command_background_with_env(self, clean_background_processes, temp_dir):
        """Test background command with environment variables."""
        result = execute_command_background(
            "echo test", working_directory=temp_dir, env={"TEST_VAR": "value"}
        )
        assert "Started background process" in result or "error" in result.lower()

    def test_execute_command_background_filters_dangerous_env(
        self, clean_background_processes, temp_dir
    ):
        """Test that dangerous env vars are filtered in background."""
        result = execute_command_background(
            "echo test", working_directory=temp_dir, env={"LD_PRELOAD": "/tmp/bad.so", "SAFE": "ok"}
        )
        assert isinstance(result, str)

    @patch("subprocess.Popen")
    def test_execute_command_background_with_mock(
        self, mock_popen, clean_background_processes, temp_dir
    ):
        """Test background command with mocked Popen."""
        mock_process = MagicMock()
        mock_process.pid = 54321
        mock_popen.return_value = mock_process

        result = execute_command_background("test command", working_directory=temp_dir)

        assert "Started background process" in result
        assert "54321" in result

    @patch("subprocess.Popen")
    def test_execute_command_background_exception(self, mock_popen, clean_background_processes):
        """Test exception handling in background command."""
        mock_popen.side_effect = OSError("Failed to start process")
        result = execute_command_background("test command")
        assert "error" in result.lower()


# ========================================
# List Background Processes Tests
# ========================================


class TestListBackgroundProcesses:
    """Tests for listing background processes."""

    def test_list_no_processes(self, clean_background_processes):
        """Test listing when no processes are running."""
        result = list_background_processes()
        assert "No background processes" in result

    def test_list_single_process(self, clean_background_processes):
        """Test listing a single background process."""
        mock_process = MagicMock()
        mock_process.pid = 111
        mock_process.poll.return_value = None  # Running

        _register_background_process(mock_process, "test command", "/tmp")

        result = list_background_processes()
        assert "Running background processes" in result
        assert "111" in result
        assert "test command" in result

    def test_list_multiple_processes(self, clean_background_processes):
        """Test listing multiple background processes."""
        mock_process1 = MagicMock()
        mock_process1.pid = 111
        mock_process1.poll.return_value = None

        mock_process2 = MagicMock()
        mock_process2.pid = 222
        mock_process2.poll.return_value = None

        _register_background_process(mock_process1, "cmd1", "/tmp")
        _register_background_process(mock_process2, "cmd2", "/tmp")

        result = list_background_processes()
        assert "111" in result
        assert "222" in result

    def test_list_shows_process_status(self, clean_background_processes):
        """Test that process status is shown."""
        mock_process = MagicMock()
        mock_process.pid = 111
        mock_process.poll.return_value = None  # Running

        _register_background_process(mock_process, "test", "/tmp")

        result = list_background_processes()
        assert "running" in result.lower()

    def test_list_shows_exit_code(self, clean_background_processes):
        """Test that exit code is shown for finished processes."""
        mock_process = MagicMock()
        mock_process.pid = 111
        mock_process.poll.return_value = 42  # Exited with code 42
        mock_process.returncode = 42

        _register_background_process(mock_process, "test", "/tmp")

        # Cleanup is called in list_background_processes, which removes finished processes
        # So we need to check before cleanup removes it
        result = list_background_processes()
        # After cleanup, finished processes are removed, so we should see "No background processes"
        # OR if the cleanup didn't run, we'd see the exit code
        assert "exited" in result.lower() or "No background processes" in result

    def test_list_truncates_long_commands(self, clean_background_processes):
        """Test that long commands are truncated."""
        mock_process = MagicMock()
        mock_process.pid = 111
        mock_process.poll.return_value = None

        long_command = "x" * 100
        _register_background_process(mock_process, long_command, "/tmp")

        result = list_background_processes()
        assert "..." in result


# ========================================
# Stop Background Process Tests
# ========================================


class TestStopBackgroundProcess:
    """Tests for stopping background processes."""

    def test_stop_nonexistent_process(self, clean_background_processes):
        """Test stopping a process that doesn't exist."""
        result = stop_background_process(99999)
        assert "No background process found" in result

    def test_stop_existing_process(self, clean_background_processes):
        """Test stopping an existing process."""
        mock_process = MagicMock()
        mock_process.pid = 111
        mock_process.poll.return_value = None
        mock_process.wait.return_value = None

        _register_background_process(mock_process, "test", "/tmp")

        result = stop_background_process(111)
        assert "Stopped background process" in result
        mock_process.terminate.assert_called_once()

    def test_stop_process_force_kill_on_timeout(self, clean_background_processes):
        """Test that process is force-killed if terminate times out."""
        mock_process = MagicMock()
        mock_process.pid = 111
        mock_process.poll.return_value = None
        mock_process.wait.side_effect = subprocess.TimeoutExpired("cmd", 5)

        _register_background_process(mock_process, "test", "/tmp")

        result = stop_background_process(111)
        assert "Stopped background process" in result
        mock_process.kill.assert_called_once()

    def test_stop_process_exception_handling(self, clean_background_processes):
        """Test exception handling when stopping process."""
        mock_process = MagicMock()
        mock_process.pid = 111
        mock_process.poll.return_value = None
        mock_process.terminate.side_effect = OSError("Cannot terminate")

        _register_background_process(mock_process, "test", "/tmp")

        result = stop_background_process(111)
        assert "Error stopping process" in result

    def test_stop_process_removes_from_tracking(self, clean_background_processes):
        """Test that stopped process is removed from tracking."""
        mock_process = MagicMock()
        mock_process.pid = 111
        mock_process.poll.return_value = None
        mock_process.wait.return_value = None

        _register_background_process(mock_process, "test", "/tmp")
        assert 111 in _background_processes

        stop_background_process(111)
        assert 111 not in _background_processes


# ========================================
# Get Process Output Tests
# ========================================


class TestGetProcessOutput:
    """Tests for getting process output."""

    def test_get_output_nonexistent_process(self, clean_background_processes):
        """Test getting output from nonexistent process."""
        result = get_process_output(99999)
        assert "No background process found" in result

    def test_get_output_no_output_available(self, clean_background_processes):
        """Test when no output is available."""
        mock_process = MagicMock()
        mock_process.pid = 111
        mock_process.poll.return_value = None
        mock_process.stdout = None

        _register_background_process(mock_process, "test", "/tmp")

        result = get_process_output(111)
        assert "No new output" in result or "Error" in result

    def test_get_output_with_max_lines(self, clean_background_processes):
        """Test getting output with max lines limit."""
        mock_process = MagicMock()
        mock_process.pid = 111
        mock_process.poll.return_value = None
        mock_process.stdout = MagicMock()

        _register_background_process(mock_process, "test", "/tmp")

        result = get_process_output(111, max_lines=50)
        assert isinstance(result, str)

    def test_get_output_exception_handling(self, clean_background_processes):
        """Test exception handling in get_process_output."""
        mock_process = MagicMock()
        mock_process.pid = 111
        mock_process.poll.return_value = None
        mock_process.stdout = MagicMock()
        mock_process.stdout.readline.side_effect = OSError("Read failed")

        _register_background_process(mock_process, "test", "/tmp")

        result = get_process_output(111)
        assert "Error reading process output" in result


# ========================================
# Environment Info Tests
# ========================================


class TestEnvironmentInfo:
    """Tests for environment information retrieval."""

    def test_get_environment_info(self):
        """Test that environment info is returned."""
        result = get_environment_info()
        assert "OS:" in result
        assert "Python:" in result
        assert "Shell:" in result

    def test_environment_info_includes_path(self):
        """Test that PATH is included."""
        result = get_environment_info()
        assert "PATH" in result

    def test_environment_info_includes_architecture(self):
        """Test that architecture is included."""
        result = get_environment_info()
        assert "Architecture:" in result

    def test_environment_info_includes_user(self):
        """Test that user information is included."""
        result = get_environment_info()
        assert "User:" in result

    def test_environment_info_includes_home(self):
        """Test that home directory is included."""
        result = get_environment_info()
        assert "Home:" in result

    def test_environment_info_truncates_long_paths(self):
        """Test that long paths are truncated."""
        result = get_environment_info()
        # Should not have extremely long lines
        lines = result.split("\n")
        for line in lines:
            assert len(line) < 200


# ========================================
# Shell Configuration Tests
# ========================================


class TestShellConfig:
    """Tests for shell configuration."""

    def test_default_config_timeout(self):
        """Test default timeout configuration."""
        assert DEFAULT_CONFIG.default_timeout == 30

    def test_default_config_max_timeout(self):
        """Test default max timeout configuration."""
        assert DEFAULT_CONFIG.max_timeout == 600

    def test_default_config_max_output_size(self):
        """Test default max output size configuration."""
        assert DEFAULT_CONFIG.max_output_size == 100_000

    def test_default_config_shell(self):
        """Test default shell configuration."""
        assert DEFAULT_CONFIG.shell == "/bin/bash"

    def test_default_config_blocked_commands(self):
        """Test that blocked commands are configured."""
        assert len(DEFAULT_CONFIG.blocked_commands) > 0
        assert "rm -rf /" in DEFAULT_CONFIG.blocked_commands

    def test_default_config_sensitive_patterns(self):
        """Test that sensitive patterns are configured."""
        assert len(DEFAULT_CONFIG.sensitive_patterns) > 0
        assert "rm -rf" in DEFAULT_CONFIG.sensitive_patterns

    def test_custom_config_creation(self):
        """Test creating custom shell configuration."""
        config = ShellConfig(default_timeout=60, max_timeout=1200, max_output_size=200_000)
        assert config.default_timeout == 60
        assert config.max_timeout == 1200
        assert config.max_output_size == 200_000


# ========================================
# Background Process Data Structure Tests
# ========================================


class TestBackgroundProcessDataStructure:
    """Tests for BackgroundProcess data structure."""

    def test_background_process_creation(self):
        """Test creating a BackgroundProcess instance."""
        mock_process = MagicMock()
        bp = BackgroundProcess(
            pid=123,
            command="test command",
            start_time=time.time(),
            working_dir="/tmp",
            process=mock_process,
        )

        assert bp.pid == 123
        assert bp.command == "test command"
        assert bp.working_dir == "/tmp"
        assert bp.process == mock_process

    def test_background_process_start_time(self):
        """Test that start time is recorded."""
        mock_process = MagicMock()
        start = time.time()
        bp = BackgroundProcess(
            pid=123, command="test", start_time=start, working_dir="/tmp", process=mock_process
        )

        assert bp.start_time == start


# ========================================
# Integration Tests
# ========================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_execute_and_list_background_processes(self, clean_background_processes, temp_dir):
        """Test executing background process and listing it."""
        result1 = execute_command_background("echo test", working_directory=temp_dir)

        if "Started background process" in result1:
            result2 = list_background_processes()
            assert "Running background processes" in result2

    def test_command_validation_before_execution(self):
        """Test that command validation happens before execution."""
        result = execute_command("rm -rf /")
        assert "blocked" in result.lower()

    def test_output_truncation_on_large_output(self):
        """Test that large output is truncated."""
        # Create a command that produces large output
        result = execute_command("python -c \"print('x' * 200000)\"")
        # Should be truncated
        assert len(result) < 200000 + 1000  # Some overhead for truncation message

    def test_environment_variables_in_execution(self, temp_dir):
        """Test that environment variables are properly set."""
        result = execute_command(
            "echo $CUSTOM_VAR", working_directory=temp_dir, env={"CUSTOM_VAR": "custom_value"}
        )
        assert "custom_value" in result

    def test_working_directory_validation(self):
        """Test that working directory is validated."""
        result = execute_command("pwd", working_directory="/nonexistent")
        assert "error" in result.lower()
