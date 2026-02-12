"""
Git Advanced Operations

Provides advanced git operations:
- Stash management
- Worktree management
- GitHub integration (via gh CLI)
"""

import logging
import re

from mcp.server.fastmcp import FastMCP

from src.core.security import validate_path
from src.servers.git_common import is_git_repo as _is_git_repo
from src.servers.git_common import run_gh_command as _run_gh_command
from src.servers.git_common import run_git_command as _run_git_command
from src.servers.git_common import sanitize_git_arg as _sanitize_git_arg
from src.servers.git_common import validate_git_ref as _validate_git_ref

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("DoraemonGitAdvanced")


# ========================================
# Stash Management Tools
# ========================================


@mcp.tool()
def git_stash(
    action: str = "push",
    path: str = ".",
    message: str | None = None,
) -> str:
    """
    Stash changes temporarily.

    Args:
        action: "push" (save), "pop" (restore and remove), "list", "drop"
        path: Repository path
        message: Optional message for stash (only for push)

    Returns:
        Stash result or list
    """
    if not _is_git_repo(path):
        return f"Error: {path} is not a git repository"

    if action not in ["push", "pop", "list", "drop", "apply"]:
        return f"Error: Invalid action '{action}'. Use: push, pop, list, drop, apply"

    args = ["stash", action]
    if action == "push" and message:
        args.extend(["-m", message])

    success, output = _run_git_command(args, cwd=path)
    return output if output else f"Stash {action} completed"


# ========================================
# Git Worktrees (Parallel Branch Development)
# ========================================


@mcp.tool()
def git_worktree_list(path: str = ".") -> str:
    """
    List all git worktrees.

    Worktrees allow you to have multiple working directories for the same
    repository, enabling parallel work on different branches.

    Args:
        path: Repository path

    Returns:
        List of worktrees with their paths and branches
    """
    if not _is_git_repo(path):
        return f"Error: {path} is not a git repository"

    success, output = _run_git_command(["worktree", "list"], cwd=path)
    return output if success else f"Error: {output}"


@mcp.tool()
def git_worktree_add(
    worktree_path: str,
    branch: str,
    path: str = ".",
    create_branch: bool = False,
) -> str:
    """
    Create a new worktree for parallel development.

    This allows you to work on multiple branches simultaneously without
    switching branches in your main directory.

    Args:
        worktree_path: Path where the new worktree will be created
        branch: Branch to checkout in the new worktree
        path: Repository path
        create_branch: If True, create a new branch with this name

    Returns:
        Success message or error

    Example:
        git_worktree_add("../myproject-feature", "feature/new-ui", create_branch=True)
    """
    if not _is_git_repo(path):
        return f"Error: {path} is not a git repository"

    # Validate branch name
    is_valid, error_msg = _validate_git_ref(branch)
    if not is_valid:
        return f"Error: Invalid branch name '{branch}': {error_msg}"

    # Validate worktree path
    try:
        validate_path(worktree_path)
    except (PermissionError, ValueError) as e:
        return f"Error: Invalid worktree path: {e}"

    args = ["worktree", "add"]
    if create_branch:
        args.append("-b")
        args.append(branch)
    args.append(worktree_path)
    if not create_branch:
        args.append(branch)

    success, output = _run_git_command(args, cwd=path)

    if success:
        return f"Created worktree at {worktree_path} on branch {branch}"
    return f"Error: {output}"


@mcp.tool()
def git_worktree_remove(
    worktree_path: str,
    path: str = ".",
    force: bool = False,
) -> str:
    """
    Remove a worktree.

    Args:
        worktree_path: Path of the worktree to remove
        path: Repository path
        force: Force removal even if there are uncommitted changes

    Returns:
        Success message or error
    """
    if not _is_git_repo(path):
        return f"Error: {path} is not a git repository"

    # Validate worktree path
    try:
        validate_path(worktree_path)
    except (PermissionError, ValueError) as e:
        return f"Error: Invalid worktree path: {e}"

    args = ["worktree", "remove"]
    if force:
        args.append("--force")
    args.append(worktree_path)

    success, output = _run_git_command(args, cwd=path)

    if success:
        return f"Removed worktree at {worktree_path}"
    return f"Error: {output}"


@mcp.tool()
def git_worktree_prune(path: str = ".") -> str:
    """
    Prune stale worktree information.

    Removes worktree entries that no longer exist on disk.

    Args:
        path: Repository path

    Returns:
        Prune result
    """
    if not _is_git_repo(path):
        return f"Error: {path} is not a git repository"

    success, output = _run_git_command(["worktree", "prune"], cwd=path)
    return output if output else "Worktree prune completed"


# ========================================
# GitHub Integration (via gh CLI)
# ========================================


@mcp.tool()
def gh_pr_create(
    title: str,
    body: str,
    path: str = ".",
    base: str | None = None,
    draft: bool = False,
) -> str:
    """
    Create a GitHub Pull Request.

    Args:
        title: PR title
        body: PR description (supports markdown)
        path: Repository path
        base: Base branch (default: repository default branch)
        draft: Create as draft PR

    Returns:
        PR URL or error message

    Requires:
        GitHub CLI (gh) must be installed and authenticated.
    """
    # Sanitize title and body to prevent argument injection
    safe_title = _sanitize_git_arg(title)
    safe_body = _sanitize_git_arg(body)

    args = ["pr", "create", "--title", safe_title, "--body", safe_body]

    if base:
        # Validate base branch name (alphanumeric, dash, underscore, slash)
        import re
        if not re.match(r'^[\w\-/]+$', base):
            return "Error: Invalid base branch name"
        args.extend(["--base", base])
    if draft:
        args.append("--draft")

    success, output = _run_gh_command(args, cwd=path, timeout=60)
    return output if success else f"Error: {output}"


@mcp.tool()
def gh_pr_list(
    path: str = ".",
    state: str = "open",
    limit: int = 10,
) -> str:
    """
    List Pull Requests in the repository.

    Args:
        path: Repository path
        state: PR state - "open", "closed", "merged", or "all"
        limit: Maximum number of PRs to list

    Returns:
        List of PRs with their numbers, titles, and status
    """
    args = ["pr", "list", "--state", state, "--limit", str(limit)]

    success, output = _run_gh_command(args, cwd=path)
    return output if success else f"Error: {output}"


@mcp.tool()
def gh_pr_view(
    pr_number: int | None = None,
    path: str = ".",
) -> str:
    """
    View details of a Pull Request.

    Args:
        pr_number: PR number (default: PR for current branch)
        path: Repository path

    Returns:
        PR details including title, body, status, and reviews
    """
    args = ["pr", "view"]
    if pr_number:
        args.append(str(pr_number))

    success, output = _run_gh_command(args, cwd=path)
    return output if success else f"Error: {output}"


@mcp.tool()
def gh_issue_list(
    path: str = ".",
    state: str = "open",
    limit: int = 10,
    labels: list[str] | None = None,
) -> str:
    """
    List GitHub Issues in the repository.

    Args:
        path: Repository path
        state: Issue state - "open", "closed", or "all"
        limit: Maximum number of issues to list
        labels: Filter by labels

    Returns:
        List of issues
    """
    args = ["issue", "list", "--state", state, "--limit", str(limit)]

    if labels:
        for label in labels:
            args.extend(["--label", label])

    success, output = _run_gh_command(args, cwd=path)
    return output if success else f"Error: {output}"


if __name__ == "__main__":
    mcp.run()
