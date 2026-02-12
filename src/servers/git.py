"""
Git Operations MCP Server

Provides core Git version control operations.

Features:
- Repository status and diff
- Staging and committing changes
- Log history and show commits
- Reset operations

Branch, remote, stash, worktree, and GitHub CLI operations
are in git_branch.py, git_remote.py, and git_advanced.py.
"""

import logging

from mcp.server.fastmcp import FastMCP

from src.core.security import validate_path
from src.servers.git_common import is_git_repo as _is_git_repo
from src.servers.git_common import run_git_command as _run_git_command

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("DoraemonGit")


# ========================================
# Unified Git Tool (Recommended)
# ========================================


@mcp.tool()
def git(
    operation: str = "status",
    path: str = ".",
    # diff parameters
    staged: bool = False,
    file: str | None = None,
    # log parameters
    count: int = 10,
    oneline: bool = True,
    author: str | None = None,
    since: str | None = None,
    # add parameters
    files: list[str] | None = None,
    # commit parameters
    message: str | None = None,
    add_all: bool = False,
) -> str:
    """
    Unified Git operations tool.

    Combines git_status, git_diff, git_log, git_add, git_commit into a single
    parameterized tool following Doraemon's Occam's Razor principle.

    Args:
        operation: Git operation to perform
            - "status": Show repository status (default)
            - "diff": Show changes (use staged=True for staged changes)
            - "log": Show commit history
            - "add": Stage files for commit
            - "commit": Create a commit
        path: Repository path (default: current directory)

        # diff parameters:
        staged: If True, show staged changes (for diff operation)
        file: Specific file to diff (for diff operation)

        # log parameters:
        count: Number of commits to show (default: 10)
        oneline: Use compact one-line format (default: True)
        author: Filter by author name/email
        since: Show commits since date (e.g., "2024-01-01", "1 week ago")

        # add parameters:
        files: List of files to stage (for add operation)

        # commit parameters:
        message: Commit message (required for commit operation)
        add_all: If True, automatically stage all tracked files before committing

    Returns:
        Git command output

    Examples:
        git()                                    # git status
        git("status")                            # git status
        git("diff")                              # git diff (unstaged changes)
        git("diff", staged=True)                 # git diff --staged
        git("diff", file="src/main.py")          # git diff -- src/main.py
        git("log")                               # git log -10 --oneline
        git("log", count=5)                      # git log -5 --oneline
        git("log", oneline=False)                # git log -10 (detailed format)
        git("log", author="john")                # git log --author=john
        git("add", files=["src/main.py"])        # git add src/main.py
        git("add", files=["."])                  # git add .
        git("commit", message="fix: bug")        # git commit -m "fix: bug"
        git("commit", message="feat: new", add_all=True)  # git commit -a -m "..."
    """
    if not _is_git_repo(path):
        return f"Error: {path} is not a git repository"

    operation = operation.lower().strip()

    # --- STATUS ---
    if operation == "status":
        success, output = _run_git_command(["status"], cwd=path)
        return output if success else f"Error: {output}"

    # --- DIFF ---
    elif operation == "diff":
        args = ["diff"]
        if staged:
            args.append("--staged")
        if file:
            # Validate file path
            try:
                validate_path(file)
            except (PermissionError, ValueError) as e:
                return f"Error: Invalid file path: {e}"
            args.extend(["--", file])

        success, output = _run_git_command(args, cwd=path)

        if not output:
            return "No changes to show."

        return output if success else f"Error: {output}"

    # --- LOG ---
    elif operation == "log":
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

    # --- ADD ---
    elif operation == "add":
        if not files:
            return "Error: 'files' parameter is required for add operation. Use files=['.'] to stage all."

        # Validate each file path (except for "." which means all)
        validated_files = []
        for f in files:
            if f == ".":
                validated_files.append(f)
            else:
                try:
                    # Validate the file is within the workspace
                    validate_path(f)
                    validated_files.append(f)
                except (PermissionError, ValueError) as e:
                    return f"Error: Invalid file path '{f}': {e}"

        # Use "--" separator to prevent argument injection
        args = ["add", "--"] + validated_files
        success, output = _run_git_command(args, cwd=path)

        if success:
            return f"Staged: {', '.join(validated_files)}"
        return f"Error: {output}"

    # --- COMMIT ---
    elif operation == "commit":
        if not message or not message.strip():
            return "Error: 'message' parameter is required for commit operation"

        args = ["commit"]
        if add_all:
            args.append("-a")
        args.extend(["-m", message])

        success, output = _run_git_command(args, cwd=path)
        return output if success else f"Error: {output}"

    else:
        return (
            f"Error: Unknown operation '{operation}'. "
            f"Valid operations: status, diff, log, add, commit"
        )


# ========================================
# Individual Git Tools
# ========================================


@mcp.tool()
def git_status(path: str = ".") -> str:
    """Get the current git repository status."""
    return git(operation="status", path=path)


@mcp.tool()
def git_diff(path: str = ".", staged: bool = False, file_path: str | None = None) -> str:
    """Show changes in the repository."""
    return git(operation="diff", path=path, staged=staged, file=file_path)


@mcp.tool()
def git_log(
    path: str = ".", count: int = 10, oneline: bool = False,
    author: str | None = None, since: str | None = None,
) -> str:
    """Show commit history."""
    return git(operation="log", path=path, count=count, oneline=oneline, author=author, since=since)


@mcp.tool()
def git_show(commit: str = "HEAD", path: str = ".", stat_only: bool = False) -> str:
    """Show details of a specific commit."""
    if not _is_git_repo(path):
        return f"Error: {path} is not a git repository"
    args = ["show", commit] + (["--stat"] if stat_only else [])
    success, output = _run_git_command(args, cwd=path)
    return output if success else f"Error: {output}"


@mcp.tool()
def git_add(files: list[str] | str, path: str = ".") -> str:
    """Stage files for commit. Use '.' for all files."""
    if isinstance(files, str):
        files = [files]
    return git(operation="add", path=path, files=files)


@mcp.tool()
def git_commit(message: str, path: str = ".", add_all: bool = False) -> str:
    """Create a commit with staged changes."""
    return git(operation="commit", path=path, message=message, add_all=add_all)


@mcp.tool()
def git_reset(files: list[str] | str | None = None, path: str = ".", mode: str = "mixed") -> str:
    """Unstage files or reset commits. Modes: soft, mixed, hard."""
    if not _is_git_repo(path):
        return f"Error: {path} is not a git repository"
    if files:
        if isinstance(files, str):
            files = [files]
        args = ["reset", "HEAD", "--"] + files
    else:
        if mode not in ("soft", "mixed", "hard"):
            return f"Error: Invalid mode '{mode}'. Use: soft, mixed, hard"
        args = ["reset", f"--{mode}"]
    success, output = _run_git_command(args, cwd=path)
    if success:
        return f"Unstaged: {', '.join(files)}" if files else f"Reset completed (mode: {mode})"
    return f"Error: {output}"


if __name__ == "__main__":
    mcp.run()
