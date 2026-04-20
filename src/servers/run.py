"""
Unified Run Tool - Combines shell, python, background, and install operations.

Follows Occam's Razor principle: one tool with mode parameter instead of 4 separate tools.

Modes:
  - shell: Execute shell commands
  - python: Execute Python code
  - background: Run commands in background
  - install: Install Python packages
"""

import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Literal

from src.core.logger import configure_root_logger
from src.core.security.security import validate_path
from src.core.security.shell_security import (
    DEFAULT_CONFIG as DEFAULT_SHELL_CONFIG,
)
from src.core.security.shell_security import (
    check_git_safety,
    is_command_blocked,
    register_background_process,
)

configure_root_logger()
logger = logging.getLogger(__name__)


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
            logger.warning("Could not set resource limits: %s", e)

    return set_limits


def _indent_code(code: str, spaces: int) -> str:
    """Indent code by the specified number of spaces."""
    indent = " " * spaces
    lines = code.split("\n")
    return "\n".join(indent + line if line.strip() else line for line in lines)


def _get_sandbox_wrapper_code(user_code: str) -> str:
    """Wrap user code with safety measures.

    NOTE: This wrapper provides defense-in-depth but is NOT a security
    boundary. See _SANDBOX_WARNING for details.
    """
    return f"""
import sys

_BLOCKED_MODULES = frozenset({{
    'subprocess', 'shutil', 'socket', 'http', 'urllib',
    'ftplib', 'smtplib', 'telnetlib', 'ctypes', 'multiprocessing',
    'importlib', 'codecs', 'code', 'codeop', 'compileall',
    'zipimport', 'pkgutil', 'site', 'runpy', 'pdb',
    'os', 'signal', 'posixpath', 'ntpath',
}})

_UNSAFE_BUILTINS = frozenset({{
    'exec', 'eval', 'compile', '__import__', 'globals', 'locals',
    'breakpoint', 'exit', 'quit',
}})

_UNSAFE_DUNDERS = frozenset({{
    '__import__',
}})

for _mod in list(sys.modules.keys()):
    if _mod.split('.')[0] in _BLOCKED_MODULES:
        del sys.modules[_mod]

class _SandboxImportBlocker:
    def find_spec(self, name, path=None, target=None):
        top_level = name.split('.')[0]
        if top_level in _BLOCKED_MODULES:
            raise ImportError(f"Module '{{name}}' is blocked in sandbox mode")
        return None

sys.meta_path.insert(0, _SandboxImportBlocker())

import builtins as _builtins
_SAFE_BUILTINS = frozenset({{
    'abs', 'all', 'any', 'bin', 'bool', 'divmod', 'enumerate', 'filter',
    'float', 'hash', 'hex', 'id', 'int', 'isinstance',
    'issubclass', 'iter', 'len', 'list', 'map', 'max', 'min', 'next', 'oct',
    'ord', 'pow', 'print', 'range', 'repr', 'reversed', 'round', 'set',
    'slice', 'sorted', 'str', 'sum', 'tuple', 'zip',
    'True', 'False', 'None', 'bool', 'bytes', 'dict', 'float', 'frozenset',
    'int', 'list', 'set', 'str', 'tuple', 'complex', 'bytearray',
    'classmethod', 'staticmethod', 'property', 'super',
    'Exception', 'TypeError', 'ValueError', 'KeyError', 'IndexError',
    'AttributeError', 'RuntimeError', 'StopIteration', 'NotImplementedError',
    'IsADirectoryError', 'FileExistsError', 'FileNotFoundError',
    'PermissionError', 'IsADirectoryError', 'OSError', 'IOError',
}})
_original_getattr = _builtins.__dict__.get
_ALLOWED_DUNDERS = frozenset({{
    '__name__', '__doc__', '__module__',
    '__init__', '__repr__', '__str__', '__len__',
    '__iter__', '__next__', '__getitem__',
    '__contains__', '__bool__', '__hash__', '__eq__',
    '__enter__', '__exit__',
}})

class _SandboxBuiltins:
    def __getattr__(self, name):
        if name in _UNSAFE_BUILTINS or name in _UNSAFE_DUNDERS:
            raise NameError(f"name '{{name}}' is not available in sandbox mode")
        if name in _ALLOWED_DUNDERS or name in _SAFE_BUILTINS:
            return _original_getattr(name)
        if name.startswith('__'):
            raise NameError(f"name '{{name}}' is not available in sandbox mode")
        raise NameError(f"name '{{name}}' is not available in sandbox mode")
    def __dir__(self):
        return list(_SAFE_BUILTINS)

sys.modules['builtins'] = _SandboxBuiltins()

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


_is_command_blocked = is_command_blocked
_check_git_safety = check_git_safety
_register_background_process = register_background_process


def run(
    command: str,
    mode: Literal["shell", "python", "background", "install"] = "shell",
    timeout: int = 120,
    working_dir: str | None = None,
) -> str:
    """Unified execution tool."""
    logger.info("run(mode=%s, command=%s...)", mode, command[:50])

    if mode == "shell":
        return _run_shell(command, timeout, working_dir)
    if mode == "python":
        return _run_python(command, timeout)
    if mode == "background":
        return _run_background(command, working_dir)
    if mode == "install":
        return _run_install(command)
    return f"Error: Unknown mode '{mode}'. Use: shell, python, background, install"


def _run_shell(command: str, timeout: int, working_dir: str | None) -> str:
    """Execute a shell command."""
    import queue

    if _is_command_blocked(command):
        logger.warning("Blocked dangerous command: %s", command)
        return "Error: This command is blocked for safety reasons."

    git_safety_msg = _check_git_safety(command)
    if git_safety_msg:
        logger.warning("Git safety check failed: %s", command)
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
            [DEFAULT_SHELL_CONFIG.shell, "-c", command],
            shell=False,
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
            try:
                line = output_queue.get(timeout=1.0)
                output_lines.append(line)
                last_activity_time = time.time()
                while True:
                    try:
                        line = output_queue.get_nowait()
                        output_lines.append(line)
                        last_activity_time = time.time()
                    except queue.Empty:
                        break
            except queue.Empty:
                pass

            return_code = process.poll()
            if return_code is not None:
                t.join(timeout=2.0)
                while True:
                    try:
                        line = output_queue.get_nowait()
                        output_lines.append(line)
                    except queue.Empty:
                        break
                break

            if time.time() - last_activity_time > timeout:
                process.kill()
                logger.warning("Command timed out: %s", command)
                return (
                    "".join(output_lines)
                    + f"\n\nError: Command timed out. No output for {timeout} seconds."
                )

        output = "".join(output_lines)
        if return_code != 0:
            output += f"\n\n[Exit code: {return_code}]"

        if not output.strip():
            return f"Command completed successfully (exit code: {return_code})"

        return output

    except Exception as e:
        logger.error("Command execution failed: %s", e)
        return f"Error executing command: {str(e)}"


_SANDBOX_WARNING = (
    "Python sandbox mode runs code in a subprocess with resource limits "
    "and import/builtin restrictions. It is NOT a true security boundary — "
    "a determined attacker with knowledge of Python internals can escape "
    "via object introspection chains. For untrusted code, use container "
    "or VM isolation instead."
)


def _run_python(code: str, timeout: int) -> str:
    """Execute Python code in a sandboxed environment.

    WARNING: The sandbox provides defense-in-depth (import blocking, builtin
    restrictions, resource limits) but is NOT a security boundary. It runs
    in the same OS user context and can be escaped via MRO chain traversal.
    Only run code from sources you trust, or add container/VM isolation.
    """
    logger.info("Executing Python code (%s chars)", len(code))

    limits = ResourceLimits(
        max_memory_mb=DEFAULT_LIMITS.max_memory_mb,
        max_cpu_time_seconds=min(timeout, DEFAULT_LIMITS.max_cpu_time_seconds),
        max_file_size_mb=DEFAULT_LIMITS.max_file_size_mb,
        max_processes=DEFAULT_LIMITS.max_processes,
    )

    wrapped_code = _get_sandbox_wrapper_code(code)

    script_path = None
    try:
        tmp_dir = tempfile.mkdtemp(prefix="doraemon_sandbox_")
        os.chmod(tmp_dir, 0o700)
        script_path = os.path.join(tmp_dir, "_sandbox_code.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(wrapped_code)
        os.chmod(script_path, 0o600)

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
        if script_path:
            tmp_dir = os.path.dirname(script_path)
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass


def _run_background(command: str, working_dir: str | None) -> str:
    """Start a command in the background."""
    logger.info("Starting background command: %s", command)

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
            [DEFAULT_SHELL_CONFIG.shell, "-c", command],
            shell=False,
            cwd=resolved_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )

        pid = _register_background_process(proc, command, resolved_dir)

        return f"Started background process with PID: {pid}\nCommand: {command}"

    except Exception as e:
        logger.error("Failed to start background command: %s", e)
        return f"Error starting background process: {str(e)}"


def _run_install(package_name: str) -> str:
    """Install a Python package via pip."""
    from urllib.error import URLError
    from urllib.request import urlopen

    logger.info("Installing package: %s", package_name)

    is_valid, error_msg = _validate_package_name(package_name)
    if not is_valid:
        return f"Error: Invalid package name: {error_msg}"

    try:
        try:
            check_url = f"https://pypi.org/pypi/{package_name}/json"
            resp = urlopen(check_url, timeout=5)  # noqa: S310
            if resp.status != 200:
                return f"Error: Package '{package_name}' not found on PyPI."
        except URLError:
            logger.warning("PyPI unreachable for %s; refusing install for safety", package_name)
            return f"Error: Cannot verify package '{package_name}' on PyPI (network unavailable). Installation refused for safety."

        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            return f"Successfully installed {package_name}."
        return f"Failed to install {package_name}.\nError: {result.stderr}"

    except Exception as e:
        return f"Error: {str(e)}"
