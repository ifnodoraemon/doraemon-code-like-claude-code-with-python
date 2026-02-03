"""
Git Remote Operations

Provides remote repository operations:
- Fetch changes from remote
- Pull changes from remote
- Push changes to remote
"""

import logging

from mcp.server.fastmcp import FastMCP

from src.core.security import validate_path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("DoraemonGitRemote")


# ========================================
# Helper Functions (imported from git.py)
# ========================================


def _run_git_command(
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
    import subprocess

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


def _is_git_repo(path: str = ".") -> bool:
    """Check if the path is inside a git repository."""
    success, _ = _run_git_command(["rev-parse", "--git-dir"], cwd=path)
    return success


def _validate_git_ref(ref: str) -> tuple[bool, str]:
    """
    Validate git reference (branch/tag name) is safe.

    Returns:
        Tuple of (is_valid, error_message)
    """
    import re

    if not ref:
        return False, "Reference cannot be empty"

    if len(ref) > 255:
        return False, "Reference name too long"

    # Git refs cannot contain: spaces, ~, ^, :, ?, *, [, \, control chars
    if re.search(r'[~\^:?*\[\s\\]', ref):
        return False, "Reference contains invalid characters"

    # Cannot start with - (looks like option)
    if ref.startswith('-'):
        return False, "Reference cannot start with '-'"

    # Cannot end with . or /
    if ref.endswith('.') or ref.endswith('/'):
        return False, "Reference cannot end with '.' or '/'"

    # Cannot contain ..
    if '..' in ref:
        return False, "Reference cannot contain '..'"

    # Cannot contain //
    if '//' in ref:
        return False, "Reference cannot contain '//'"

    return True, ""


# ========================================
# Remote Operations Tools
# ========================================


@mcp.tool()
def git_pull(
    path: str = ".",
    remote: str = "origin",
    branch: str | None = None,
) -> str:
    """
    Pull changes from remote repository.

    Args:
        path: Repository path
        remote: Remote name (default: origin)
        branch: Branch to pull (default: current branch)

    Returns:
        Pull result or error message
    """
    if not _is_git_repo(path):
        return f"Error: {path} is not a git repository"

    # Validate remote name
    is_valid, error_msg = _validate_git_ref(remote)
    if not is_valid:
        return f"Error: Invalid remote '{remote}': {error_msg}"

    # Validate branch if provided
    if branch:
        is_valid, error_msg = _validate_git_ref(branch)
        if not is_valid:
            return f"Error: Invalid branch '{branch}': {error_msg}"

    args = ["pull", remote]
    if branch:
        args.append(branch)

    success, output = _run_git_command(args, cwd=path, timeout=60)
    return output if success else f"Error: {output}"


@mcp.tool()
def git_push(
    path: str = ".",
    remote: str = "origin",
    branch: str | None = None,
    set_upstream: bool = False,
) -> str:
    """
    Push commits to remote repository.

    Args:
        path: Repository path
        remote: Remote name (default: origin)
        branch: Branch to push (default: current branch)
        set_upstream: Set upstream tracking reference

    Returns:
        Push result or error message
    """
    if not _is_git_repo(path):
        return f"Error: {path} is not a git repository"

    # Validate remote name
    is_valid, error_msg = _validate_git_ref(remote)
    if not is_valid:
        return f"Error: Invalid remote '{remote}': {error_msg}"

    # Validate branch if provided
    if branch:
        is_valid, error_msg = _validate_git_ref(branch)
        if not is_valid:
            return f"Error: Invalid branch '{branch}': {error_msg}"

    args = ["push"]
    if set_upstream:
        args.append("-u")
    args.append(remote)
    if branch:
        args.append(branch)

    success, output = _run_git_command(args, cwd=path, timeout=60)
    return output if success else f"Error: {output}"


@mcp.tool()
def git_fetch(
    path: str = ".",
    remote: str = "origin",
    prune: bool = False,
) -> str:
    """
    Fetch changes from remote without merging.

    Args:
        path: Repository path
        remote: Remote name (default: origin)
        prune: Remove remote-tracking branches that no longer exist

    Returns:
        Fetch result or error message
    """
    if not _is_git_repo(path):
        return f"Error: {path} is not a git repository"

    args = ["fetch", remote]
    if prune:
        args.append("--prune")

    success, output = _run_git_command(args, cwd=path, timeout=60)
    return output if output else "Fetch completed (up to date)"


if __name__ == "__main__":
    mcp.run()
