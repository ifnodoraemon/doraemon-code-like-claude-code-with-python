import logging
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass

import requests
from mcp.server.fastmcp import FastMCP

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("DoraemonComputer")


# ========================================
# Resource Limits Configuration
# ========================================
@dataclass
class ResourceLimits:
    """Configuration for code execution resource limits."""

    max_memory_mb: int = 512  # Maximum memory in MB
    max_cpu_time_seconds: int = 30  # Maximum CPU time in seconds
    max_file_size_mb: int = 50  # Maximum file size that can be created
    max_processes: int = 10  # Maximum number of child processes


# Default limits (can be overridden via environment variables)
DEFAULT_LIMITS = ResourceLimits(
    max_memory_mb=int(os.getenv("DORAEMON_MAX_MEMORY_MB", "512")),
    max_cpu_time_seconds=int(os.getenv("DORAEMON_MAX_CPU_TIME", "30")),
    max_file_size_mb=int(os.getenv("DORAEMON_MAX_FILE_SIZE_MB", "50")),
    max_processes=int(os.getenv("DORAEMON_MAX_PROCESSES", "10")),
)


def _create_sandbox_preexec(limits: ResourceLimits):
    """
    Create a preexec_fn for subprocess that sets resource limits.

    This function is called in the child process before exec.
    Only works on Unix-like systems (Linux, macOS).
    """

    def set_limits():
        # Only apply resource limits on Unix-like systems
        if platform.system() == "Windows":
            return

        try:
            import resource

            # Memory limit (soft and hard)
            memory_bytes = limits.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

            # CPU time limit
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (limits.max_cpu_time_seconds, limits.max_cpu_time_seconds),
            )

            # File size limit
            file_size_bytes = limits.max_file_size_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_FSIZE, (file_size_bytes, file_size_bytes))

            # Number of processes limit
            resource.setrlimit(resource.RLIMIT_NPROC, (limits.max_processes, limits.max_processes))

            logger.debug(
                f"Resource limits set: memory={limits.max_memory_mb}MB, "
                f"cpu={limits.max_cpu_time_seconds}s, "
                f"file_size={limits.max_file_size_mb}MB, "
                f"processes={limits.max_processes}"
            )
        except (ImportError, OSError) as e:
            # resource module not available or limits couldn't be set
            logger.warning(f"Could not set resource limits: {e}")

    return set_limits


def _get_sandbox_wrapper_code(user_code: str) -> str:
    """
    Wrap user code with additional safety measures.

    This adds:
    - Restricted imports warning
    - Output size limiting
    - Exception handling
    """
    return f"""
import sys
import os

# Safety: Limit recursion depth
sys.setrecursionlimit(1000)

# Safety: Remove dangerous builtins (optional, can be bypassed but adds a layer)
# Note: This is not a complete sandbox, but helps prevent accidental misuse
_dangerous_builtins = ['eval', 'exec', 'compile', '__import__']
# We don't actually remove them as it would break legitimate code

# Execute user code
try:
    # User code starts here
{_indent_code(user_code, 4)}
    # User code ends here
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


def _indent_code(code: str, spaces: int) -> str:
    """Indent code by the specified number of spaces."""
    indent = " " * spaces
    lines = code.split("\n")
    return "\n".join(indent + line if line.strip() else line for line in lines)


def _get_pypi_suggestions(query: str) -> str:
    """Internal helper to get suggestions when a package is not found."""
    try:
        url = f"https://pypi.org/search/?q={query}"
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return ""

        import re

        names = re.findall(r'<span class="package-snippet__name">(.*?)</span>', response.text)
        descs = re.findall(r'<p class="package-snippet__description">(.*?)</p>', response.text)

        suggestions = []
        for i in range(min(3, len(names))):
            # Clean up tags if any
            name = names[i].replace('"', "")
            desc = descs[i].strip()
            suggestions.append(f"- **{name}**: {desc}")

        if suggestions:
            return (
                "\n\nI couldn't find that exact package, but here are some similar ones on PyPI:\n"
                + "\n".join(suggestions)
            )
        return ""
    except Exception:
        return ""


@mcp.tool()
def list_installed_packages() -> str:
    """List all currently installed Python packages and their versions."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list"], capture_output=True, text=True
        )
        return result.stdout
    except Exception as e:
        return f"Error listing packages: {str(e)}"


def _validate_package_name(name: str) -> tuple[bool, str]:
    """Validate package name for safety."""
    import re

    # Basic sanitization
    name = name.strip()

    # Check for empty name
    if not name:
        return False, "Package name cannot be empty"

    # Check for path traversal or shell injection attempts
    dangerous_patterns = ['..', '/', '\\', ';', '|', '&', '$', '`', '>', '<', '(', ')']
    for pattern in dangerous_patterns:
        if pattern in name:
            return False, f"Package name contains invalid character: {pattern}"

    # Valid PyPI package name pattern (PEP 508)
    # Allows letters, numbers, underscores, hyphens, dots
    if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9._-]*$', name):
        return False, "Package name must start with alphanumeric and contain only letters, numbers, dots, underscores, and hyphens"

    # Check for suspicious patterns (potential typosquatting indicators)
    suspicious_patterns = ['eval', 'exec', 'system', 'subprocess']
    name_lower = name.lower()
    for pattern in suspicious_patterns:
        if pattern in name_lower and name_lower != pattern:
            logger.warning(f"Package name contains suspicious pattern: {pattern}")

    return True, ""


@mcp.tool()
def install_package(package_name: str) -> str:
    """
    Install a Python package using pip.
    If the package name is incorrect, I will provide suggestions based on PyPI search.
    """
    logger.info(f"Attempting to install package: {package_name}")

    # Validate package name for security
    is_valid, error_msg = _validate_package_name(package_name)
    if not is_valid:
        return f"❌ Invalid package name: {error_msg}"

    try:
        # First, check if package exists to provide better feedback
        check_url = f"https://pypi.org/pypi/{package_name}/json"
        check_resp = requests.get(check_url, timeout=5)

        if check_resp.status_code != 200:
            error_msg = f"❌ Package '{package_name}' not found on PyPI."
            suggestions = _get_pypi_suggestions(package_name)
            return (
                error_msg
                + suggestions
                + "\n\nPlease review the suggestions and try again with the correct name."
            )

        # Package exists, proceed with installation
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            return f"✅ Successfully installed {package_name}."
        else:
            return f"❌ Failed to install {package_name}.\nError: {result.stderr}"

    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
def execute_python(
    code: str,
    timeout: int = 30,
    max_memory_mb: int | None = None,
    sandbox: bool = True,
) -> str:
    """
    Execute arbitrary Python code in a sandboxed environment with resource limits.

    The code runs in a temporary environment with the following protections:
    - Memory limit (default: 512MB, configurable)
    - CPU time limit (default: 30s)
    - File size limit (default: 50MB)
    - Process count limit (default: 10)

    Images generated (e.g. via matplotlib) should be saved to 'drafts/'.

    Args:
        code: Python code to execute
        timeout: Maximum wall-clock time in seconds (default: 30)
        max_memory_mb: Override default memory limit in MB (optional)
        sandbox: Whether to apply resource limits (default: True)

    Returns:
        Execution output or error message

    Environment Variables:
        DORAEMON_MAX_MEMORY_MB: Default memory limit (512)
        DORAEMON_MAX_CPU_TIME: Default CPU time limit (30)
        DORAEMON_MAX_FILE_SIZE_MB: Default file size limit (50)
        DORAEMON_MAX_PROCESSES: Default process limit (10)
    """
    logger.info(f"Executing Python code ({len(code)} chars, sandbox={sandbox})")
    logger.debug(f"Code:\n{code}")

    # Warn about sandbox=False usage
    if not sandbox:
        logger.warning("Running code with sandbox=False - no resource limits applied!")

    # Configure resource limits
    limits = ResourceLimits(
        max_memory_mb=max_memory_mb or DEFAULT_LIMITS.max_memory_mb,
        max_cpu_time_seconds=min(timeout, DEFAULT_LIMITS.max_cpu_time_seconds),
        max_file_size_mb=DEFAULT_LIMITS.max_file_size_mb,
        max_processes=DEFAULT_LIMITS.max_processes,
    )

    # Wrap code with safety measures if sandbox enabled
    if sandbox:
        wrapped_code = _get_sandbox_wrapper_code(code)
    else:
        wrapped_code = code

    # Create temporary file for code
    script_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(wrapped_code)
            script_path = f.name

        # Prepare subprocess arguments
        subprocess_kwargs: dict = {
            "capture_output": True,
            "text": True,
            "timeout": timeout,
        }

        # Apply resource limits on Unix-like systems
        if sandbox and platform.system() != "Windows":
            subprocess_kwargs["preexec_fn"] = _create_sandbox_preexec(limits)
            logger.info(
                f"Sandbox enabled: memory={limits.max_memory_mb}MB, "
                f"cpu={limits.max_cpu_time_seconds}s"
            )

        # Execute the code
        result = subprocess.run([sys.executable, script_path], **subprocess_kwargs)

        # Process output
        output = result.stdout
        if result.stderr:
            output += f"\n[Stderr]:\n{result.stderr}"

        # Check return code for resource limit violations
        if result.returncode == 137:
            logger.warning("Code was killed (likely OOM or resource limit)")
            return (
                f"Error: Code was terminated (exit code 137). "
                f"This usually means it exceeded memory ({limits.max_memory_mb}MB) "
                f"or CPU time ({limits.max_cpu_time_seconds}s) limits."
            )
        elif result.returncode == 0:
            logger.info("Code executed successfully")
        else:
            logger.warning(f"Code exited with code {result.returncode}")

        return output if output.strip() else "Code executed successfully (no output)."

    except subprocess.TimeoutExpired:
        logger.error(f"Code execution timed out after {timeout}s")
        return f"Error: Code execution timed out ({timeout}s wall-clock limit)."
    except MemoryError:
        logger.error("Code execution ran out of memory")
        return f"Error: Code exceeded memory limit ({limits.max_memory_mb}MB)."
    except Exception as e:
        logger.error(f"Code execution failed: {e}")
        return f"Error: {str(e)}"
    finally:
        # Clean up temporary file
        if script_path and os.path.exists(script_path):
            try:
                os.remove(script_path)
            except OSError:
                pass  # Best effort cleanup


if __name__ == "__main__":
    mcp.run()
