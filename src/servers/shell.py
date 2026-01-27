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
import threading
import time
from dataclasses import dataclass, field

from mcp.server.fastmcp import FastMCP

from src.core.security import validate_path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("DoraemonShell")


# ========================================
# Configuration
# ========================================


@dataclass
class ShellConfig:
    """Configuration for shell execution."""

    default_timeout: int = 30  # seconds
    max_timeout: int = 600  # 10 minutes
    max_output_size: int = 100_000  # characters
    shell: str = "/bin/bash"

    # Commands that are blocked for safety
    blocked_commands: list[str] = field(
        default_factory=lambda: [
            "rm -rf /",
            "rm -rf /*",
            "mkfs",
            "dd if=/dev/zero",
            ":(){:|:&};:",  # fork bomb
        ]
    )

    # Commands that require confirmation (handled by HITL in main CLI)
    sensitive_patterns: list[str] = field(
        default_factory=lambda: [
            "rm -rf",
            "sudo",
            "chmod 777",
            "curl | bash",
            "wget | bash",
        ]
    )


DEFAULT_CONFIG = ShellConfig()


# ========================================
# Background Process Management
# ========================================


@dataclass
class BackgroundProcess:
    """Tracks a background process."""

    pid: int
    command: str
    start_time: float
    working_dir: str
    process: subprocess.Popen


# Store for background processes
_background_processes: dict[int, BackgroundProcess] = {}
_process_lock = threading.Lock()


def _register_background_process(proc: subprocess.Popen, command: str, working_dir: str) -> int:
    """Register a background process for tracking."""
    with _process_lock:
        bp = BackgroundProcess(
            pid=proc.pid,
            command=command,
            start_time=time.time(),
            working_dir=working_dir,
            process=proc,
        )
        _background_processes[proc.pid] = bp
        return proc.pid


def _cleanup_finished_processes():
    """Remove finished processes from tracking."""
    with _process_lock:
        finished = []
        for pid, bp in _background_processes.items():
            if bp.process.poll() is not None:
                finished.append(pid)
        for pid in finished:
            del _background_processes[pid]


# ========================================
# Command Validation
# ========================================


def _is_command_blocked(command: str, config: ShellConfig = DEFAULT_CONFIG) -> bool:
    """Check if a command is blocked for safety."""
    command_lower = command.lower().strip()

    # Remove quotes and escape sequences for better detection
    # This helps prevent simple bypass attempts
    normalized = command_lower.replace('"', '').replace("'", '').replace('\\', '')

    for blocked in config.blocked_commands:
        blocked_lower = blocked.lower()
        if blocked_lower in command_lower or blocked_lower in normalized:
            return True

    # Additional checks for dangerous patterns
    dangerous_patterns = [
        r'>\s*/dev/sd',  # Writing to disk devices
        r'>\s*/dev/null.*2>&1.*&',  # Hiding output and backgrounding
        r'\|\s*bash',  # Piping to bash
        r'\|\s*sh\b',  # Piping to sh
        r'\$\(.*\)',  # Command substitution (potential injection)
        r'`.*`',  # Backtick command substitution
    ]

    import re
    for pattern in dangerous_patterns:
        if re.search(pattern, command_lower):
            return True

    return False


def _is_command_sensitive(command: str, config: ShellConfig = DEFAULT_CONFIG) -> bool:
    """Check if a command requires extra confirmation."""
    command_lower = command.lower()

    for pattern in config.sensitive_patterns:
        if pattern.lower() in command_lower:
            return True

    return False


def _truncate_output(output: str, max_size: int = DEFAULT_CONFIG.max_output_size) -> str:
    """Truncate output if it exceeds max size."""
    if len(output) <= max_size:
        return output

    half = max_size // 2
    return (
        output[:half]
        + f"\n\n... [Output truncated: {len(output)} chars total, showing first and last {half} chars] ...\n\n"
        + output[-half:]
    )


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
    process_env = os.environ.copy()
    if env:
        dangerous_env_vars = {'PATH', 'LD_PRELOAD', 'LD_LIBRARY_PATH', 'DYLD_INSERT_LIBRARIES'}
        safe_env = {k: v for k, v in env.items() if k.upper() not in dangerous_env_vars}
        process_env.update(safe_env)

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
        output_queue = queue.Queue()
        
        def reader_thread():
            try:
                for line in iter(process.stdout.readline, ''):
                    output_queue.put(line)
                process.stdout.close()
            except Exception:
                pass

        t = threading.Thread(target=reader_thread, daemon=True)
        t.start()

        output_lines = []
        last_activity_time = time.time()
        
        while True:
            # Check if process has finished
            return_code = process.poll()
            
            # Read all available output from queue
            has_new_output = False
            while True:
                try:
                    line = output_queue.get_nowait()
                    output_lines.append(line)
                    last_activity_time = time.time()
                    has_new_output = True
                except queue.Empty:
                    break
            
            if return_code is not None and not t.is_alive() and output_queue.empty():
                # Process finished and thread finished
                break

            # Check Idle Timeout
            if time.time() - last_activity_time > timeout:
                process.kill()
                logger.warning(f"Command timed out due to inactivity ({timeout}s): {command}")
                return "".join(output_lines) + f"\n\nError: Command timed out. No output for {timeout} seconds."

            # Sleep briefly to avoid busy loop
            time.sleep(0.1)

        # Process finished
        output = "".join(output_lines)
        if return_code != 0:
            output += f"\n\n[Exit code: {return_code}]"

        output = _truncate_output(output)
        
        if not output.strip():
            return f"Command completed successfully (exit code: {return_code})"

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

    try:
        resolved_dir = validate_path(working_directory)
        if not os.path.isdir(resolved_dir):
            return f"Error: Working directory does not exist: {working_directory}"
    except (PermissionError, ValueError) as e:
        return f"Error: Invalid working directory: {e}"

    process_env = os.environ.copy()
    if env:
        # Filter out dangerous environment variable overrides
        dangerous_env_vars = {'PATH', 'LD_PRELOAD', 'LD_LIBRARY_PATH', 'DYLD_INSERT_LIBRARIES'}
        safe_env = {k: v for k, v in env.items() if k.upper() not in dangerous_env_vars}
        process_env.update(safe_env)

    try:
        # Start the process
        proc = subprocess.Popen(
            command,
            shell=True,
            executable=DEFAULT_CONFIG.shell,
            cwd=resolved_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=process_env,
            start_new_session=True,  # Detach from parent
        )

        pid = _register_background_process(proc, command, resolved_dir)

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

        # Note: This is a simple implementation. For production, you'd want
        # to capture output to a file or use a more sophisticated approach.
        try:
            # Non-blocking read attempt
            import select

            output_lines = []

            # Check if there's output available
            if bp.process.stdout and select.select([bp.process.stdout], [], [], 0.1)[0]:
                for _ in range(max_lines):
                    line = bp.process.stdout.readline()
                    if not line:
                        break
                    output_lines.append(line.decode() if isinstance(line, bytes) else line)

            if output_lines:
                return "".join(output_lines)
            else:
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
