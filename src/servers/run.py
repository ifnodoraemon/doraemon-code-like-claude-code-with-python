"""
Unified Run Tool - Combines shell, python, background, and install operations.

Follows Occam's Razor principle: one tool with mode parameter instead of 4 separate tools.

Modes:
  - shell: Execute shell commands (from shell.py)
  - python: Execute Python code (from computer.py)
  - background: Run commands in background (from shell.py)
  - install: Install Python packages (from computer.py)
"""

import logging
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Literal

from mcp.server.fastmcp import FastMCP

from src.core.logger import configure_root_logger
from src.core.security.security import validate_path
from src.core.security.shell_security import (
    DEFAULT_CONFIG as DEFAULT_SHELL_CONFIG,
)
from src.core.security.shell_security import (
    check_git_safety,
    is_command_blocked,
    register_background_process,
    truncate_output,
)

# Setup logging
configure_root_logger()
logger = logging.getLogger(__name__)

mcp = FastMCP("AgentRunUnified")


# ========================================
# Configuration (Python execution specific)
# ========================================


@dataclass
class ResourceLimits:
    """Configuration for code execution resource limits."""

    max_memory_mb: int = 512
    max_cpu_time_seconds: int = 30
    max_file_size_mb: int = 50
    max_processes: int = 10


DEFAULT_LIMITS = ResourceLimits(
    max_memory_mb=int(os.getenv("AGENT_MAX_MEMORY_MB", "512")),
    max_cpu_time_seconds=int(os.getenv("AGENT_MAX_CPU_TIME", "30")),
    max_file_size_mb=int(os.getenv("AGENT_MAX_FILE_SIZE_MB", "50")),
    max_processes=int(os.getenv("AGENT_MAX_PROCESSES", "10")),
)


# ========================================
# Python Execution Helpers
# ========================================


def _create_sandbox_preexec(limits: ResourceLimits):
    """Create a preexec_fn for subprocess that sets resource limits."""

    def set_limits():
        if platform.system() == "Windows":
            return

        try:
            import resource

            memory_bytes = limits.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (limits.max_cpu_time_seconds, limits.max_cpu_time_seconds),
            )
            file_size_bytes = limits.max_file_size_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_FSIZE, (file_size_bytes, file_size_bytes))
            resource.setrlimit(resource.RLIMIT_NPROC, (limits.max_processes, limits.max_processes))
        except (ImportError, OSError) as e:
            logger.warning(f"Could not set resource limits: {e}")

    return set_limits


def _indent_code(code: str, spaces: int) -> str:
    """Indent code by the specified number of spaces."""
    indent = " " * spaces
    lines = code.split("\n")
    return "\n".join(indent + line if line.strip() else line for line in lines)


def _get_sandbox_wrapper_code(user_code: str) -> str:
    """Wrap user code with safety measures.

    Enhanced (C2): blocks access to dangerous modules within the sandbox.
    """
    return f"""
import sys
import os

# Block dangerous modules to prevent sandbox escape
_BLOCKED_MODULES = {{
    'subprocess', 'shutil', 'socket', 'http', 'urllib',
    'ftplib', 'smtplib', 'telnetlib', 'ctypes', 'multiprocessing',
}}

class _SandboxImportBlocker:
    def find_module(self, name, path=None):
        top_level = name.split('.')[0]
        if top_level in _BLOCKED_MODULES:
            return self
        return None
    def load_module(self, name):
        raise ImportError(f"Module '{{name}}' is blocked in sandbox mode")

sys.meta_path.insert(0, _SandboxImportBlocker())
sys.setrecursionlimit(1000)

try:
{_indent_code(user_code, 4)}
except MemoryError:
    print("Error: Code exceeded memory limit", file=sys.stderr)
    sys.exit(137)
except RecursionError:
    print("Error: Maximum recursion depth exceeded", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error: {{type(e).__name__}}: {{e}}", file=sys.stderr)
    sys.exit(1)
"""


def _validate_package_name(name: str) -> tuple[bool, str]:
    """Validate package name for safety."""
    import re

    name = name.strip()

    if not name:
        return False, "Package name cannot be empty"

    dangerous_patterns = ["..", "/", "\\", ";", "|", "&", "$", "`", ">", "<", "(", ")"]
    for pattern in dangerous_patterns:
        if pattern in name:
            return False, f"Package name contains invalid character: {pattern}"

    if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$", name):
        return False, "Package name must start with alphanumeric"

    return True, ""


# ========================================
# Output truncation helper
# ========================================

_truncate_output = truncate_output
_is_command_blocked = is_command_blocked
_check_git_safety = check_git_safety
_register_background_process = register_background_process


# ========================================
# Unified Run Tool
# ========================================


@mcp.tool()
def run(
    command: str,
    mode: Literal["shell", "python", "background", "install"] = "shell",
    timeout: int = 120,
    working_dir: str | None = None,
) -> str:
    """
    Unified execution tool - run shell commands, Python code, background processes, or install packages.

    Args:
        command: The command/code/package to execute
        mode: Execution mode
            - "shell": Execute shell command (default)
            - "python": Execute Python code
            - "background": Run command in background
            - "install": Install Python package via pip
        timeout: Timeout in seconds (default: 120, max: 600)
        working_dir: Working directory (optional, defaults to current directory)

    Examples:
        run("ls -la")                           # Shell command
        run("print('hello')", mode="python")    # Python code
        run("npm run dev", mode="background")   # Background process
        run("requests", mode="install")         # pip install requests

    Returns:
        Command output, execution result, or error message
    """
    logger.info(f"run(mode={mode}, command={command[:50]}...)")

    if mode == "shell":
        return _run_shell(command, timeout, working_dir)
    elif mode == "python":
        return _run_python(command, timeout)
    elif mode == "background":
        return _run_background(command, working_dir)
    elif mode == "install":
        return _run_install(command)
    else:
        return f"Error: Unknown mode '{mode}'. Use: shell, python, background, install"


def _run_shell(command: str, timeout: int, working_dir: str | None) -> str:
    """Execute a shell command."""
    import queue

    if _is_command_blocked(command):
        logger.warning(f"Blocked dangerous command: {command}")
        return "Error: This command is blocked for safety reasons."

    git_safety_msg = _check_git_safety(command)
    if git_safety_msg:
        logger.warning(f"Git safety check failed: {command}")
        return git_safety_msg

    resolved_dir = working_dir or os.getcwd()
    try:
        resolved_dir = validate_path(resolved_dir)
        if not os.path.isdir(resolved_dir):
            return f"Error: Working directory does not exist: {resolved_dir}"
    except (PermissionError, ValueError) as e:
        return f"Error: Invalid working directory: {e}"

    timeout = min(max(1, timeout), 600)

    try:
        process = subprocess.Popen(
            command,
            shell=True,
            executable=DEFAULT_SHELL_CONFIG.shell,
            cwd=resolved_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

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

        while True:
            # Block waiting for output (up to 1s), avoids CPU-burning busy-loop
            try:
                line = output_queue.get(timeout=1.0)
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

            if time.time() - last_activity_time > timeout:
                process.kill()
                logger.warning(f"Command timed out: {command}")
                return (
                    "".join(output_lines)
                    + f"\n\nError: Command timed out. No output for {timeout} seconds."
                )

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


def _run_python(code: str, timeout: int) -> str:
    """Execute Python code in a sandboxed environment."""
    logger.info(f"Executing Python code ({len(code)} chars)")

    limits = ResourceLimits(
        max_memory_mb=DEFAULT_LIMITS.max_memory_mb,
        max_cpu_time_seconds=min(timeout, DEFAULT_LIMITS.max_cpu_time_seconds),
        max_file_size_mb=DEFAULT_LIMITS.max_file_size_mb,
        max_processes=DEFAULT_LIMITS.max_processes,
    )

    wrapped_code = _get_sandbox_wrapper_code(code)

    script_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(wrapped_code)
            script_path = f.name

        subprocess_kwargs: dict = {
            "capture_output": True,
            "text": True,
            "timeout": timeout,
        }

        if platform.system() != "Windows":
            subprocess_kwargs["preexec_fn"] = _create_sandbox_preexec(limits)

        result = subprocess.run([sys.executable, script_path], **subprocess_kwargs)

        output = result.stdout
        if result.stderr:
            output += f"\n[Stderr]:\n{result.stderr}"

        if result.returncode == 137:
            return (
                f"Error: Code was terminated (exit code 137). "
                f"Exceeded memory ({limits.max_memory_mb}MB) or CPU time limits."
            )

        return output if output.strip() else "Code executed successfully (no output)."

    except subprocess.TimeoutExpired:
        return f"Error: Code execution timed out ({timeout}s)."
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        if script_path and os.path.exists(script_path):
            try:
                os.remove(script_path)
            except OSError:
                pass


def _run_background(command: str, working_dir: str | None) -> str:
    """Start a command in the background."""
    logger.info(f"Starting background command: {command}")

    if _is_command_blocked(command):
        return "Error: This command is blocked for safety reasons."

    git_safety_msg = _check_git_safety(command)
    if git_safety_msg:
        return git_safety_msg

    resolved_dir = working_dir or os.getcwd()
    try:
        resolved_dir = validate_path(resolved_dir)
        if not os.path.isdir(resolved_dir):
            return f"Error: Working directory does not exist: {resolved_dir}"
    except (PermissionError, ValueError) as e:
        return f"Error: Invalid working directory: {e}"

    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            executable=DEFAULT_SHELL_CONFIG.shell,
            cwd=resolved_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )

        pid = _register_background_process(proc, command, resolved_dir)

        return f"Started background process with PID: {pid}\nCommand: {command}"

    except Exception as e:
        logger.error(f"Failed to start background command: {e}")
        return f"Error starting background process: {str(e)}"


def _run_install(package_name: str) -> str:
    """Install a Python package via pip."""
    from urllib.error import URLError
    from urllib.request import urlopen

    logger.info(f"Installing package: {package_name}")

    is_valid, error_msg = _validate_package_name(package_name)
    if not is_valid:
        return f"Error: Invalid package name: {error_msg}"

    try:
        # Check if package exists on PyPI (best-effort, skip on network error)
        try:
            check_url = f"https://pypi.org/pypi/{package_name}/json"
            resp = urlopen(check_url, timeout=5)  # noqa: S310
            if resp.status != 200:
                return f"Error: Package '{package_name}' not found on PyPI."
        except URLError:
            logger.debug(f"PyPI check skipped for {package_name} (network unavailable)")

        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            return f"Successfully installed {package_name}."
        else:
            return f"Failed to install {package_name}.\nError: {result.stderr}"

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    mcp.run()
