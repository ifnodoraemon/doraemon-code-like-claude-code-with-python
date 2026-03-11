"""
Shell Command Execution MCP Server

Provides general-purpose shell command execution with security controls.
Similar to Claude Code's Shell tool but with additional safety features.

Features:
- Execute arbitrary shell commands
- Working directory support
- Timeout protection
- Output capture (stdout/stderr)
- Background process support
- Command history tracking
"""

import logging
import os
import subprocess
import tempfile
import threading
import time

from mcp.server.fastmcp import FastMCP

from src.core.logger import configure_root_logger
from src.core.security import validate_path
from src.core.shell_security import (
    DEFAULT_CONFIG,
    BackgroundProcess,
    ShellConfig,
    check_git_safety,
    cleanup_finished_processes,
    get_background_processes,
    get_process_lock,
    is_command_blocked,
    is_command_sensitive,
    register_background_process,
    truncate_output,
)
from src.core.subprocess_utils import prepare_safe_env

# Setup logging
configure_root_logger()
logger = logging.getLogger(__name__)

mcp = FastMCP("AgentShell")

# ========================================
# Backward-compatible aliases (used by tests)
# ========================================

_background_processes = get_background_processes()
_process_lock = get_process_lock()
_is_command_blocked = is_command_blocked
_is_command_sensitive = is_command_sensitive
_check_git_safety = check_git_safety
_truncate_output = truncate_output
_register_background_process = register_background_process
_cleanup_finished_processes = cleanup_finished_processes


def _prepare_safe_env(env: dict[str, str] | None) -> dict[str, str]:
    """Build process env, filtering dangerous overrides."""
    return prepare_safe_env(env)


# ========================================
# Shell Tools
# ========================================


@mcp.tool()
def execute_command(
    command: str,
    working_directory: str = ".",
    timeout: int = 30,
    env: dict[str, str] | None = None,
) -> str:
    """
    Execute a shell command and return the output.

    This uses "Active Detection" (Smart Timeout):
    - The process is allowed to run as long as it produces output.
    - It is only killed if it is silent (no output) for the duration of the 'timeout'.
    - This is ideal for long-running builds or downloads that continually log progress.

    Args:
        command: The shell command to execute
        working_directory: Directory to run the command in (default: current directory)
        timeout: Idle timeout in seconds (default: 30). Process is killed if silent for this long.
        env: Additional environment variables to set

    Returns:
        Command output (stdout and stderr combined) or error message
    """
    logger.info(f"Executing command: {command}")

    # Validate command
    if _is_command_blocked(command):
        logger.warning(f"Blocked dangerous command: {command}")
        return "Error: This command is blocked for safety reasons."

    # Git safety check
    git_safety_msg = _check_git_safety(command)
    if git_safety_msg:
        logger.warning(f"Git safety check failed: {command}")
        return git_safety_msg

    # Validate and resolve working directory
    try:
        resolved_dir = validate_path(working_directory)
        if not os.path.isdir(resolved_dir):
            return f"Error: Working directory does not exist: {working_directory}"
    except (PermissionError, ValueError) as e:
        return f"Error: Invalid working directory: {e}"

    # Clamp timeout (minimum 1s, max 1 hour for active processes)
    timeout = min(max(1, timeout), 3600)

    # Prepare environment with safety checks
    process_env = _prepare_safe_env(env)

    try:
        # Start process with Popen
        process = subprocess.Popen(
            command,
            shell=True,
            executable=DEFAULT_CONFIG.shell,
            cwd=resolved_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stdout/stderr
            text=True,
            env=process_env,
            bufsize=1,  # Line buffered
            universal_newlines=True,
        )

        # Use a thread to read output to avoid blocking/buffering issues with select
        import queue

        output_queue: queue.Queue = queue.Queue()

        def reader_thread():
            try:
                for line in iter(process.stdout.readline, ""):
                    output_queue.put(line)
                process.stdout.close()
            except Exception:
                pass

        t = threading.Thread(target=reader_thread, daemon=True)
        t.start()

        output_lines = []
        last_activity_time = time.time()

        idle_poll_interval = 0.1

        while True:
            # Poll frequently enough that idle timeouts stay responsive even if
            # the process is silent and wall-clock time is being simulated.
            try:
                line = output_queue.get(timeout=idle_poll_interval)
                output_lines.append(line)
                last_activity_time = time.time()
                # Drain any additional lines already queued
                while True:
                    try:
                        line = output_queue.get_nowait()
                        output_lines.append(line)
                        last_activity_time = time.time()
                    except queue.Empty:
                        break
            except queue.Empty:
                pass  # No output in the last second

            return_code = process.poll()
            if return_code is not None and not t.is_alive() and output_queue.empty():
                break

            # Check Idle Timeout
            if time.time() - last_activity_time > timeout:
                process.kill()
                logger.warning(f"Command timed out due to inactivity ({timeout}s): {command}")
                return (
                    "".join(output_lines)
                    + f"\n\nError: Command timed out. No output for {timeout} seconds."
                )

        # Process finished
        output = "".join(output_lines)
        if return_code != 0:
            output += f"\n\n[Exit code: {return_code}]"

        output = _truncate_output(output)

        if not output.strip():
            if return_code == 0:
                return "Command completed successfully (exit code: 0)"
            return f"Command completed (exit code: {return_code})"

        return output

    except Exception as e:
        logger.error(f"Command execution failed: {e}")
        return f"Error executing command: {str(e)}"


@mcp.tool()
def execute_command_background(
    command: str,
    working_directory: str = ".",
    env: dict[str, str] | None = None,
) -> str:
    """
    Start a long-running command in the background.

    Use this for:
    - Development servers (npm run dev, python -m http.server)
    - Watch processes (npm run watch)
    - Long-running tasks

    Args:
        command: The shell command to execute
        working_directory: Directory to run the command in
        env: Additional environment variables

    Returns:
        Process ID (PID) of the background process

    Example:
        execute_command_background("npm run dev", working_directory="frontend")
    """
    logger.info(f"Starting background command: {command}")

    if _is_command_blocked(command):
        return "Error: This command is blocked for safety reasons."

    # Git safety check
    git_safety_msg = _check_git_safety(command)
    if git_safety_msg:
        return git_safety_msg

    try:
        resolved_dir = validate_path(working_directory)
        if not os.path.isdir(resolved_dir):
            return f"Error: Working directory does not exist: {working_directory}"
    except (PermissionError, ValueError) as e:
        return f"Error: Invalid working directory: {e}"

    process_env = _prepare_safe_env(env)

    try:
        # Capture stdout/stderr to temp files so get_process_output can read them
        stdout_file = tempfile.NamedTemporaryFile(
            mode='w', prefix='doraemon_bg_', suffix='.log', delete=False
        )
        log_file_path = stdout_file.name
        proc = subprocess.Popen(
            command,
            shell=True,
            executable=DEFAULT_CONFIG.shell,
            cwd=resolved_dir,
            stdout=stdout_file,
            stderr=subprocess.STDOUT,
            env=process_env,
            start_new_session=True,  # Detach from parent
        )

        pid = _register_background_process(proc, command, resolved_dir, log_file=log_file_path)

        return f"Started background process with PID: {pid}\nCommand: {command}"

    except Exception as e:
        logger.error(f"Failed to start background command: {e}")
        return f"Error starting background process: {str(e)}"


@mcp.tool()
def list_background_processes() -> str:
    """
    List all running background processes started by Doraemon Code.

    Returns:
        List of background processes with their PIDs, commands, and runtime
    """
    _cleanup_finished_processes()

    with _process_lock:
        if not _background_processes:
            return "No background processes running."

        lines = ["Running background processes:\n"]
        for pid, bp in _background_processes.items():
            runtime = time.time() - bp.start_time
            status = "running" if bp.process.poll() is None else f"exited ({bp.process.returncode})"
            lines.append(
                f"  PID {pid}: {bp.command[:50]}{'...' if len(bp.command) > 50 else ''}\n"
                f"    Status: {status}, Runtime: {runtime:.1f}s\n"
                f"    Directory: {bp.working_dir}"
            )

        return "\n".join(lines)


@mcp.tool()
def stop_background_process(pid: int) -> str:
    """
    Stop a background process by PID.

    Args:
        pid: Process ID to stop

    Returns:
        Confirmation message
    """
    with _process_lock:
        if pid not in _background_processes:
            return f"Error: No background process found with PID {pid}"

        bp = _background_processes[pid]

        try:
            bp.process.terminate()
            # Give it a moment to terminate gracefully
            try:
                bp.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                bp.process.kill()

            # Clean up temp log file
            if bp.log_file and os.path.exists(bp.log_file):
                try:
                    os.unlink(bp.log_file)
                except OSError:
                    pass

            del _background_processes[pid]
            return f"Stopped background process {pid}: {bp.command[:50]}"

        except Exception as e:
            return f"Error stopping process {pid}: {str(e)}"


@mcp.tool()
def get_process_output(pid: int, max_lines: int = 100) -> str:
    """
    Get the recent output from a background process.

    Args:
        pid: Process ID
        max_lines: Maximum number of lines to return

    Returns:
        Recent output from the process
    """
    with _process_lock:
        if pid not in _background_processes:
            return f"Error: No background process found with PID {pid}"

        bp = _background_processes[pid]

        try:
            # Read output from the temp log file
            if bp.log_file and os.path.exists(bp.log_file):
                with open(bp.log_file, encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()

                if lines:
                    # Return the last max_lines lines
                    tail = lines[-max_lines:]
                    output = "".join(tail)
                    total = len(lines)
                    if total > max_lines:
                        output = f"[Showing last {max_lines} of {total} lines]\n" + output
                    return output

            # Backward-compatible fallback for tests or processes without temp logs.
            if bp.process.stdout is None:
                status = (
                    "running" if bp.process.poll() is None else f"exited ({bp.process.returncode})"
                )
                return f"No new output available. Process status: {status}"

            lines: list[str] = []
            for _ in range(max_lines):
                line = bp.process.stdout.readline()
                if not line:
                    break
                lines.append(line)

            if lines:
                return "".join(lines)

            status = (
                "running" if bp.process.poll() is None else f"exited ({bp.process.returncode})"
            )
            return f"No new output available. Process status: {status}"

        except Exception as e:
            return f"Error reading process output: {str(e)}"


@mcp.tool()
def get_environment_info() -> str:
    """
    Get information about the current shell environment.

    Returns:
        System information including OS, shell, Python version, etc.
    """
    import platform
    import sys

    info = [
        "Shell Environment Information:",
        f"  OS: {platform.system()} {platform.release()}",
        f"  Architecture: {platform.machine()}",
        f"  Python: {sys.version}",
        f"  Shell: {DEFAULT_CONFIG.shell}",
        f"  Working Directory: {os.getcwd()}",
        f"  User: {os.getenv('USER', 'unknown')}",
        f"  Home: {os.getenv('HOME', 'unknown')}",
        "",
        "Key Environment Variables:",
    ]

    key_vars = ["PATH", "PYTHONPATH", "NODE_PATH", "GOPATH", "JAVA_HOME"]
    for var in key_vars:
        value = os.getenv(var)
        if value:
            # Truncate long paths
            if len(value) > 100:
                value = value[:100] + "..."
            info.append(f"  {var}: {value}")

    return "\n".join(info)


if __name__ == "__main__":
    mcp.run()
