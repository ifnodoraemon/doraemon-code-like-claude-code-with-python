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
from dataclasses import dataclass, field
from typing import Literal

from mcp.server.fastmcp import FastMCP

from src.core.security import validate_path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("DoraemonRunUnified")


# ========================================
# Configuration (from shell.py and computer.py)
# ========================================


@dataclass
class ShellConfig:
    """Configuration for shell execution."""

    default_timeout: int = 30
    max_timeout: int = 600
    max_output_size: int = 100_000
    shell: str = "/bin/bash"

    blocked_commands: list[str] = field(
        default_factory=lambda: [
            "rm -rf /",
            "rm -rf /*",
            "mkfs",
            "dd if=/dev/zero",
            ":(){:|:&};:",
            "chmod -R 777 /",
            "chown -R",
            "> /dev/sda",
            "mv / ",
            "wget -O- | sh",
            "curl -s | sh",
        ]
    )

    sensitive_patterns: list[str] = field(
        default_factory=lambda: [
            "rm -rf",
            "rm -r",
            "sudo",
            "chmod 777",
            "curl | bash",
            "wget | bash",
            "pip install",
            "npm install -g",
            "apt install",
            "apt remove",
            "systemctl",
            "service ",
            "kill -9",
            "pkill",
            "docker rm",
            "docker rmi",
        ]
    )

    blocked_base_commands: list[str] = field(
        default_factory=lambda: [
            "mkfs",
            "fdisk",
            "parted",
            "wipefs",
            "shred",
            "halt",
            "poweroff",
            "reboot",
            "shutdown",
            "init",
            "telinit",
        ]
    )


@dataclass
class ResourceLimits:
    """Configuration for code execution resource limits."""

    max_memory_mb: int = 512
    max_cpu_time_seconds: int = 30
    max_file_size_mb: int = 50
    max_processes: int = 10


DEFAULT_SHELL_CONFIG = ShellConfig()
DEFAULT_LIMITS = ResourceLimits(
    max_memory_mb=int(os.getenv("DORAEMON_MAX_MEMORY_MB", "512")),
    max_cpu_time_seconds=int(os.getenv("DORAEMON_MAX_CPU_TIME", "30")),
    max_file_size_mb=int(os.getenv("DORAEMON_MAX_FILE_SIZE_MB", "50")),
    max_processes=int(os.getenv("DORAEMON_MAX_PROCESSES", "10")),
)


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


# ========================================
# Security Functions (from shell.py)
# ========================================


def _is_command_blocked(command: str, config: ShellConfig = DEFAULT_SHELL_CONFIG) -> bool:
    """Check if a command is blocked for safety."""
    import re
    import shlex

    command_lower = command.lower().strip()
    normalized = command_lower.replace('"', "").replace("'", "").replace("\\", "")

    for blocked in config.blocked_commands:
        blocked_lower = blocked.lower()
        if blocked_lower in command_lower or blocked_lower in normalized:
            return True

    try:
        tokens = shlex.split(command_lower)
    except ValueError:
        tokens = command_lower.split()

    if tokens:
        base_cmd = os.path.basename(tokens[0])
        if base_cmd in config.blocked_base_commands:
            return True

        if base_cmd == "rm" and any(t in tokens for t in ["-rf", "-fr"]):
            for t in tokens:
                if t in ("/", "/*", "/.", "/.."):
                    return True

    for separator in [";", "&&", "||"]:
        if separator in command:
            subcmds = command.split(separator)
            for subcmd in subcmds:
                subcmd = subcmd.strip()
                if subcmd and _is_single_command_blocked(subcmd, config):
                    return True

    dangerous_patterns = [
        r">\s*/dev/sd",
        r"\|\s*bash",
        r"\|\s*sh\b",
        r"\|\s*zsh\b",
        r"eval\s+",
        r"exec\s+\d*[<>]",
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, command_lower):
            return True

    return False


def _is_single_command_blocked(command: str, config: ShellConfig) -> bool:
    """Check a single command against blocked list."""
    import shlex

    command_lower = command.lower().strip()
    normalized = command_lower.replace('"', "").replace("'", "").replace("\\", "")

    for blocked in config.blocked_commands:
        blocked_lower = blocked.lower()
        if blocked_lower in command_lower or blocked_lower in normalized:
            return True

    try:
        tokens = shlex.split(command_lower)
    except ValueError:
        tokens = command_lower.split()

    if tokens:
        base_cmd = os.path.basename(tokens[0])
        if base_cmd in config.blocked_base_commands:
            return True

    return False


def _check_git_safety(command: str) -> str | None:
    """Check git commands for dangerous operations."""
    import shlex

    command_stripped = command.strip()

    if not command_stripped.startswith("git "):
        return None

    try:
        tokens = shlex.split(command_stripped)
    except ValueError:
        tokens = command_stripped.split()

    if len(tokens) < 2:
        return None

    subcommand = tokens[1]

    if subcommand == "push":
        for token in tokens[2:]:
            if token in ("--force", "-f", "--force-with-lease"):
                for t in tokens[2:]:
                    if t in ("main", "master", "origin/main", "origin/master"):
                        return (
                            "Error: Force push to main/master is blocked for safety. "
                            "This could overwrite shared history."
                        )
                return (
                    "Error: Force push (--force) is blocked for safety. "
                    "Use --force-with-lease if you must, or ask the user to confirm."
                )

    if subcommand == "reset":
        if "--hard" in tokens:
            return (
                "Warning: 'git reset --hard' will discard all uncommitted changes. "
                "This is blocked for safety. Use 'git stash' first to preserve changes."
            )

    if subcommand == "checkout":
        if "." in tokens:
            return (
                "Error: 'git checkout .' will discard all uncommitted changes. "
                "Use 'git stash' to preserve them first."
            )

    if subcommand == "clean":
        if "-f" in tokens or "--force" in tokens:
            return (
                "Error: 'git clean -f' will permanently delete untracked files. "
                "This is blocked for safety."
            )

    if subcommand == "branch":
        if "-D" in tokens:
            for t in tokens[2:]:
                if t in ("main", "master"):
                    return "Error: Deleting main/master branch is blocked for safety."

    if "--no-verify" in tokens:
        return (
            "Error: --no-verify bypasses pre-commit hooks. "
            "This is blocked unless explicitly authorized."
        )

    return None


def _truncate_output(output: str, max_size: int = DEFAULT_SHELL_CONFIG.max_output_size) -> str:
    """Truncate output if it exceeds max size."""
    if len(output) <= max_size:
        return output

    half = max_size // 2
    return (
        output[:half]
        + f"\n\n... [Output truncated: {len(output)} chars total] ...\n\n"
        + output[-half:]
    )


# ========================================
# Python Execution Helpers (from computer.py)
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
    """Wrap user code with safety measures."""
    return f"""
import sys
import os

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
            return_code = process.poll()

            while True:
                try:
                    line = output_queue.get_nowait()
                    output_lines.append(line)
                    last_activity_time = time.time()
                except queue.Empty:
                    break

            if return_code is not None and not t.is_alive() and output_queue.empty():
                break

            if time.time() - last_activity_time > timeout:
                process.kill()
                logger.warning(f"Command timed out: {command}")
                return (
                    "".join(output_lines)
                    + f"\n\nError: Command timed out. No output for {timeout} seconds."
                )

            time.sleep(0.1)

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
    import requests

    logger.info(f"Installing package: {package_name}")

    is_valid, error_msg = _validate_package_name(package_name)
    if not is_valid:
        return f"Error: Invalid package name: {error_msg}"

    try:
        # Check if package exists on PyPI
        check_url = f"https://pypi.org/pypi/{package_name}/json"
        check_resp = requests.get(check_url, timeout=5)

        if check_resp.status_code != 200:
            return f"Error: Package '{package_name}' not found on PyPI."

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


# ========================================
# Backward Compatibility Aliases
# ========================================


@mcp.tool()
def shell_execute(command: str, timeout: int = 30, working_directory: str = ".") -> str:
    """[Deprecated] Use run(command, mode='shell') instead."""
    return run(command, mode="shell", timeout=timeout, working_dir=working_directory)


@mcp.tool()
def shell_background(command: str, working_directory: str = ".") -> str:
    """[Deprecated] Use run(command, mode='background') instead."""
    return run(command, mode="background", working_dir=working_directory)


@mcp.tool()
def execute_python(code: str, timeout: int = 30) -> str:
    """[Deprecated] Use run(code, mode='python') instead."""
    return run(code, mode="python", timeout=timeout)


@mcp.tool()
def install_package(package_name: str) -> str:
    """[Deprecated] Use run(package_name, mode='install') instead."""
    return run(package_name, mode="install")


if __name__ == "__main__":
    mcp.run()
