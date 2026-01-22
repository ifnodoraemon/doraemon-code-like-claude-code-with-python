"""
Git Operations MCP Server

Provides comprehensive Git version control operations.
Similar to Claude Code's git workflow capabilities.

Features:
- Repository status and diff
- Staging and committing changes
- Branch management
- Log history
- GitHub PR integration (via gh CLI)
"""

import logging
import subprocess

from mcp.server.fastmcp import FastMCP

from src.core.security import validate_path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("PolymathGit")


# ========================================
# Helper Functions
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


# ========================================
# Repository Status Tools
# ========================================


@mcp.tool()
def git_status(path: str = ".") -> str:
    """
    Get the current git repository status.

    Shows:
    - Current branch
    - Staged changes
    - Unstaged changes
    - Untracked files

    Args:
        path: Repository path (default: current directory)

    Returns:
        Git status output
    """
    if not _is_git_repo(path):
        return f"Error: {path} is not a git repository"

    success, output = _run_git_command(["status"], cwd=path)
    return output if success else f"Error: {output}"


@mcp.tool()
def git_diff(
    path: str = ".",
    staged: bool = False,
    file_path: str | None = None,
) -> str:
    """
    Show changes in the repository.

    Args:
        path: Repository path
        staged: If True, show staged changes; otherwise show unstaged changes
        file_path: Specific file to diff (optional)

    Returns:
        Diff output showing changes
    """
    if not _is_git_repo(path):
        return f"Error: {path} is not a git repository"

    args = ["diff"]
    if staged:
        args.append("--staged")
    if file_path:
        args.extend(["--", file_path])

    success, output = _run_git_command(args, cwd=path)

    if not output:
        return "No changes to show."

    return output if success else f"Error: {output}"


@mcp.tool()
def git_log(
    path: str = ".",
    count: int = 10,
    oneline: bool = False,
    author: str | None = None,
    since: str | None = None,
) -> str:
    """
    Show commit history.

    Args:
        path: Repository path
        count: Number of commits to show (default: 10)
        oneline: Use compact one-line format
        author: Filter by author name/email
        since: Show commits since date (e.g., "2024-01-01", "1 week ago")

    Returns:
        Commit history
    """
    if not _is_git_repo(path):
        return f"Error: {path} is not a git repository"

    args = ["log", f"-{count}"]

    if oneline:
        args.append("--oneline")
    else:
        args.append("--format=%h | %an | %ar | %s")

    if author:
        args.append(f"--author={author}")
    if since:
        args.append(f"--since={since}")

    success, output = _run_git_command(args, cwd=path)
    return output if success else f"Error: {output}"


@mcp.tool()
def git_show(
    commit: str = "HEAD",
    path: str = ".",
    stat_only: bool = False,
) -> str:
    """
    Show details of a specific commit.

    Args:
        commit: Commit hash or reference (default: HEAD)
        path: Repository path
        stat_only: Only show file statistics, not full diff

    Returns:
        Commit details
    """
    if not _is_git_repo(path):
        return f"Error: {path} is not a git repository"

    args = ["show", commit]
    if stat_only:
        args.append("--stat")

    success, output = _run_git_command(args, cwd=path)
    return output if success else f"Error: {output}"


# ========================================
# Staging and Committing
# ========================================


@mcp.tool()
def git_add(
    files: list[str] | str,
    path: str = ".",
) -> str:
    """
    Stage files for commit.

    Args:
        files: File(s) to stage. Use "." for all files, or a list of specific files
        path: Repository path

    Returns:
        Confirmation or error message

    Examples:
        git_add(".")  # Stage all changes
        git_add(["src/main.py", "README.md"])  # Stage specific files
    """
    if not _is_git_repo(path):
        return f"Error: {path} is not a git repository"

    if isinstance(files, str):
        files = [files]

    args = ["add"] + files
    success, output = _run_git_command(args, cwd=path)

    if success:
        return f"Staged: {', '.join(files)}"
    return f"Error: {output}"


@mcp.tool()
def git_commit(
    message: str,
    path: str = ".",
    add_all: bool = False,
) -> str:
    """
    Create a commit with staged changes.

    Args:
        message: Commit message
        path: Repository path
        add_all: If True, automatically stage all tracked files before committing

    Returns:
        Commit confirmation or error message

    Note:
        The commit message should follow conventional commit format:
        - feat: new feature
        - fix: bug fix
        - docs: documentation
        - refactor: code refactoring
        - test: adding tests
        - chore: maintenance
    """
    if not _is_git_repo(path):
        return f"Error: {path} is not a git repository"

    if not message.strip():
        return "Error: Commit message cannot be empty"

    args = ["commit"]
    if add_all:
        args.append("-a")
    args.extend(["-m", message])

    success, output = _run_git_command(args, cwd=path)
    return output if success else f"Error: {output}"


@mcp.tool()
def git_reset(
    files: list[str] | str | None = None,
    path: str = ".",
    mode: str = "mixed",
) -> str:
    """
    Unstage files or reset commits.

    Args:
        files: Specific files to unstage (None = reset staging area)
        path: Repository path
        mode: Reset mode - "soft" (keep changes staged), "mixed" (unstage), "hard" (discard)

    Returns:
        Confirmation or error message

    Warning:
        Using mode="hard" will DISCARD all uncommitted changes!
    """
    if not _is_git_repo(path):
        return f"Error: {path} is not a git repository"

    if files:
        # Unstage specific files
        if isinstance(files, str):
            files = [files]
        args = ["reset", "HEAD", "--"] + files
    else:
        # Reset staging area or commits
        if mode not in ["soft", "mixed", "hard"]:
            return f"Error: Invalid mode '{mode}'. Use: soft, mixed, or hard"
        args = ["reset", f"--{mode}"]

    success, output = _run_git_command(args, cwd=path)

    if success:
        if files:
            return f"Unstaged: {', '.join(files)}"
        return f"Reset completed (mode: {mode})"
    return f"Error: {output}"


# ========================================
# Branch Management
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

    args = ["checkout"]
    if create:
        args.append("-b")
    args.append(target)

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

    args = ["merge"]
    if no_ff:
        args.append("--no-ff")
    args.append(branch)

    success, output = _run_git_command(args, cwd=path)
    return output if success else f"Error: {output}"


# ========================================
# Remote Operations
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


# ========================================
# GitHub Integration (via gh CLI)
# ========================================


def _run_gh_command(args: list[str], cwd: str = ".", timeout: int = 30) -> tuple[bool, str]:
    """Run a GitHub CLI command."""
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

    except FileNotFoundError:
        return False, "GitHub CLI (gh) is not installed. Install it from: https://cli.github.com/"
    except subprocess.TimeoutExpired:
        return False, f"Command timed out after {timeout}s"
    except Exception as e:
        return False, f"Error: {str(e)}"


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
    args = ["pr", "create", "--title", title, "--body", body]

    if base:
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


if __name__ == "__main__":
    mcp.run()
