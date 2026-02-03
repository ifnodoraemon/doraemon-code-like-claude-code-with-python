"""
Git Branch Management Operations

Provides branch-related git operations:
- List branches
- Create and delete branches
- Switch branches (checkout)
- Merge branches
"""

import logging

from mcp.server.fastmcp import FastMCP

from src.core.security import validate_path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("DoraemonGitBranch")


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
# Branch Management Tools
# ========================================


@mcp.tool()
def git_branch(
    path: str = ".",
    all_branches: bool = False,
) -> str:
    """
    List branches in the repository.

    Args:
        path: Repository path
        all_branches: Include remote branches

    Returns:
        List of branches (current branch marked with *)
    """
    if not _is_git_repo(path):
        return f"Error: {path} is not a git repository"

    args = ["branch"]
    if all_branches:
        args.append("-a")

    success, output = _run_git_command(args, cwd=path)
    return output if success else f"Error: {output}"


@mcp.tool()
def git_checkout(
    target: str,
    path: str = ".",
    create: bool = False,
) -> str:
    """
    Switch branches or restore files.

    Args:
        target: Branch name, commit hash, or file path
        path: Repository path
        create: If True, create a new branch with this name

    Returns:
        Confirmation or error message

    Examples:
        git_checkout("main")  # Switch to main branch
        git_checkout("feature/new", create=True)  # Create and switch to new branch
    """
    if not _is_git_repo(path):
        return f"Error: {path} is not a git repository"

    # Validate target reference
    is_valid, error_msg = _validate_git_ref(target)
    if not is_valid:
        return f"Error: Invalid target '{target}': {error_msg}"

    args = ["checkout"]
    if create:
        args.append("-b")
    args.extend(["--", target])  # Use -- to prevent option injection

    success, output = _run_git_command(args, cwd=path)
    return output if success else f"Error: {output}"


@mcp.tool()
def git_merge(
    branch: str,
    path: str = ".",
    no_ff: bool = False,
) -> str:
    """
    Merge a branch into the current branch.

    Args:
        branch: Branch to merge
        path: Repository path
        no_ff: Create a merge commit even for fast-forward merges

    Returns:
        Merge result or error message
    """
    if not _is_git_repo(path):
        return f"Error: {path} is not a git repository"

    # Validate branch reference
    is_valid, error_msg = _validate_git_ref(branch)
    if not is_valid:
        return f"Error: Invalid branch '{branch}': {error_msg}"

    args = ["merge"]
    if no_ff:
        args.append("--no-ff")
    args.append(branch)

    success, output = _run_git_command(args, cwd=path)
    return output if success else f"Error: {output}"


if __name__ == "__main__":
    mcp.run()
