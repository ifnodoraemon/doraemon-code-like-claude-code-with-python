"""
Git Common Utilities

Shared helper functions for all git modules.
"""

import logging
import re
import subprocess

from src.core.security import validate_path

logger = logging.getLogger(__name__)


def run_git_command(
    args: list[str],
    cwd: str = ".",
    timeout: int = 30,
) -> tuple[bool, str]:
    """
    Run a git command and return (success, output).

    Args:
        args: Git command arguments (without 'git' prefix)
        cwd: Working directory
        timeout: Timeout in seconds

    Returns:
        Tuple of (success: bool, output: str)
    """
    try:
        resolved_cwd = validate_path(cwd)
    except (PermissionError, ValueError) as e:
        return False, f"Invalid path: {e}"

    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=resolved_cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        output = result.stdout
        if result.stderr:
            if output:
                output += "\n"
            output += result.stderr

        return result.returncode == 0, output.strip()

    except subprocess.TimeoutExpired:
        return False, f"Git command timed out after {timeout}s"
    except FileNotFoundError:
        return False, "Git is not installed or not in PATH"
    except Exception as e:
        return False, f"Error: {str(e)}"


def is_git_repo(path: str = ".") -> bool:
    """Check if the path is inside a git repository."""
    success, _ = run_git_command(["rev-parse", "--git-dir"], cwd=path)
    return success


def validate_git_ref(ref: str) -> tuple[bool, str]:
    """
    Validate git reference (branch/tag name) is safe.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not ref:
        return False, "Reference cannot be empty"

    if len(ref) > 255:
        return False, "Reference name too long"

    # Check for invalid characters
    invalid_chars = [" ", "~", "^", ":", "?", "*", "[", "\\", ".."]
    for char in invalid_chars:
        if char in ref:
            return False, f"Reference contains invalid character: {char}"

    # Cannot start with dash or dot
    if ref.startswith("-") or ref.startswith("."):
        return False, "Reference cannot start with '-' or '.'"

    # Cannot end with dot or slash
    if ref.endswith(".") or ref.endswith("/"):
        return False, "Reference cannot end with '.' or '/'"

    # Cannot contain consecutive slashes
    if "//" in ref:
        return False, "Reference cannot contain consecutive slashes"

    return True, ""


def sanitize_git_arg(value: str) -> str:
    """Sanitize git argument to prevent injection."""
    # Remove any shell metacharacters
    dangerous = [";", "&", "|", "`", "$", "(", ")", "<", ">", "\n", "\r"]
    for char in dangerous:
        value = value.replace(char, "")
    return value


def run_gh_command(args: list[str], cwd: str = ".", timeout: int = 30) -> tuple[bool, str]:
    """
    Run a GitHub CLI command.

    Args:
        args: gh command arguments (without 'gh' prefix)
        cwd: Working directory
        timeout: Timeout in seconds

    Returns:
        Tuple of (success: bool, output: str)
    """
    try:
        resolved_cwd = validate_path(cwd)
    except (PermissionError, ValueError) as e:
        return False, f"Invalid path: {e}"

    try:
        result = subprocess.run(
            ["gh"] + args,
            cwd=resolved_cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        output = result.stdout
        if result.stderr:
            if output:
                output += "\n"
            output += result.stderr

        return result.returncode == 0, output.strip()

    except subprocess.TimeoutExpired:
        return False, f"Command timed out after {timeout}s"
    except FileNotFoundError:
        return False, "GitHub CLI (gh) is not installed"
    except Exception as e:
        return False, f"Error: {str(e)}"
