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
from src.servers.git_common import check_ref, require_repo
from src.servers.git_common import run_gh_command as _run_gh_command
from src.servers.git_common import run_git_command as _run_git_command
from src.servers.git_common import sanitize_git_arg as _sanitize_git_arg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("DoraemonGitAdvanced")


# ========================================
# Stash
# ========================================


@mcp.tool()
def git_stash(action: str = "push", path: str = ".", message: str | None = None) -> str:
    """Stash changes. Actions: push, pop, list, drop, apply."""
    if err := require_repo(path):
        return err
    valid = {"push", "pop", "list", "drop", "apply"}
    if action not in valid:
        return f"Error: Invalid action '{action}'. Use: {', '.join(sorted(valid))}"
    args = ["stash", action]
    if action == "push" and message:
        args.extend(["-m", message])
    success, output = _run_git_command(args, cwd=path)
    if not success:
        return f"Error: {output}"
    return output if output else f"Stash {action} completed"


def _validate_worktree_path(worktree_path: str) -> str | None:
    """Validate worktree path. Return error string or None."""
    try:
        validate_path(worktree_path)
        return None
    except (PermissionError, ValueError) as e:
        return f"Error: Invalid worktree path: {e}"


# ========================================
# Worktrees
# ========================================


@mcp.tool()
def git_worktree_list(path: str = ".") -> str:
    """List all git worktrees."""
    if err := require_repo(path):
        return err
    success, output = _run_git_command(["worktree", "list"], cwd=path)
    return output if success else f"Error: {output}"


@mcp.tool()
def git_worktree_add(
    worktree_path: str, branch: str, path: str = ".", create_branch: bool = False,
) -> str:
    """Create a new worktree for parallel development."""
    if err := require_repo(path):
        return err
    if err := check_ref(branch, "branch"):
        return err
    if err := _validate_worktree_path(worktree_path):
        return err
    args = ["worktree", "add"]
    if create_branch:
        args += ["-b", branch, worktree_path]
    else:
        args += [worktree_path, branch]
    success, output = _run_git_command(args, cwd=path)
    return f"Created worktree at {worktree_path} on branch {branch}" if success else f"Error: {output}"


@mcp.tool()
def git_worktree_remove(worktree_path: str, path: str = ".", force: bool = False) -> str:
    """Remove a worktree."""
    if err := require_repo(path):
        return err
    if err := _validate_worktree_path(worktree_path):
        return err
    args = ["worktree", "remove"] + (["--force"] if force else []) + [worktree_path]
    success, output = _run_git_command(args, cwd=path)
    return f"Removed worktree at {worktree_path}" if success else f"Error: {output}"


@mcp.tool()
def git_worktree_prune(path: str = ".") -> str:
    """Prune stale worktree information."""
    if err := require_repo(path):
        return err
    success, output = _run_git_command(["worktree", "prune"], cwd=path)
    return output if output else "Worktree prune completed"


# ========================================
# GitHub Integration (via gh CLI)
# ========================================


@mcp.tool()
def gh_pr_create(
    title: str, body: str, path: str = ".", base: str | None = None, draft: bool = False,
) -> str:
    """Create a GitHub Pull Request."""
    args = ["pr", "create", "--title", title, "--body", body]
    if base:
        if not re.match(r'^[\w\-/]+$', base):
            return "Error: Invalid base branch name"
        args.extend(["--base", base])
    if draft:
        args.append("--draft")
    success, output = _run_gh_command(args, cwd=path, timeout=60)
    return output if success else f"Error: {output}"


@mcp.tool()
def gh_pr_list(path: str = ".", state: str = "open", limit: int = 10) -> str:
    """List Pull Requests."""
    args = ["pr", "list", "--state", state, "--limit", str(limit)]
    success, output = _run_gh_command(args, cwd=path)
    return output if success else f"Error: {output}"


@mcp.tool()
def gh_pr_view(pr_number: int | None = None, path: str = ".") -> str:
    """View details of a Pull Request."""
    args = ["pr", "view"] + ([str(pr_number)] if pr_number else [])
    success, output = _run_gh_command(args, cwd=path)
    return output if success else f"Error: {output}"


@mcp.tool()
def gh_issue_list(
    path: str = ".", state: str = "open", limit: int = 10, labels: list[str] | None = None,
) -> str:
    """List GitHub Issues."""
    args = ["issue", "list", "--state", state, "--limit", str(limit)]
    for label in (labels or []):
        args.extend(["--label", label])
    success, output = _run_gh_command(args, cwd=path)
    return output if success else f"Error: {output}"


if __name__ == "__main__":
    mcp.run()
