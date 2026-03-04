"""
Git Remote Operations

Provides remote repository operations:
- Fetch changes from remote
- Pull changes from remote
- Push changes to remote
"""

import logging

from mcp.server.fastmcp import FastMCP

from src.servers.git_common import check_ref, require_repo
from src.servers.git_common import run_git_command as _run_git_command

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("DoraemonGitRemote")


@mcp.tool()
def git_pull(path: str = ".", remote: str = "origin", branch: str | None = None) -> str:
    """Pull changes from remote repository."""
    if err := require_repo(path):
        return err
    if err := check_ref(remote, "remote"):
        return err
    if branch and (err := check_ref(branch, "branch")):
        return err
    args = ["pull", remote] + ([branch] if branch else [])
    success, output = _run_git_command(args, cwd=path, timeout=60)
    return output if success else f"Error: {output}"


@mcp.tool()
def git_push(
    path: str = ".", remote: str = "origin",
    branch: str | None = None, set_upstream: bool = False,
) -> str:
    """Push commits to remote repository."""
    if err := require_repo(path):
        return err
    if err := check_ref(remote, "remote"):
        return err
    if branch and (err := check_ref(branch, "branch")):
        return err
    args = ["push"] + (["-u"] if set_upstream else []) + [remote] + ([branch] if branch else [])
    success, output = _run_git_command(args, cwd=path, timeout=60)
    return output if success else f"Error: {output}"


@mcp.tool()
def git_fetch(path: str = ".", remote: str = "origin", prune: bool = False) -> str:
    """Fetch changes from remote without merging."""
    if err := require_repo(path):
        return err
    if err := check_ref(remote, "remote"):
        return err
    args = ["fetch", remote] + (["--prune"] if prune else [])
    success, output = _run_git_command(args, cwd=path, timeout=60)
    if not success:
        return f"Error: {output}"
    return output if output else "Fetch completed (up to date)"


if __name__ == "__main__":
    mcp.run()
