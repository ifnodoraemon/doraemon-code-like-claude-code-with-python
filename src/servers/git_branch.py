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

from src.servers.git_common import check_ref, require_repo
from src.servers.git_common import run_git_command as _run_git_command

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("DoraemonGitBranch")


@mcp.tool()
def git_branch(path: str = ".", all_branches: bool = False) -> str:
    """List branches in the repository."""
    if err := require_repo(path):
        return err
    args = ["branch"] + (["-a"] if all_branches else [])
    success, output = _run_git_command(args, cwd=path)
    return output if success else f"Error: {output}"


@mcp.tool()
def git_checkout(target: str, path: str = ".", create: bool = False) -> str:
    """Switch branches or restore files."""
    if err := require_repo(path):
        return err
    if err := check_ref(target, "target"):
        return err
    args = ["checkout"] + (["-b"] if create else []) + [target]
    success, output = _run_git_command(args, cwd=path)
    return output if success else f"Error: {output}"


@mcp.tool()
def git_merge(branch: str, path: str = ".", no_ff: bool = False) -> str:
    """Merge a branch into the current branch."""
    if err := require_repo(path):
        return err
    if err := check_ref(branch, "branch"):
        return err
    args = ["merge"] + (["--no-ff"] if no_ff else []) + [branch]
    success, output = _run_git_command(args, cwd=path)
    return output if success else f"Error: {output}"


if __name__ == "__main__":
    mcp.run()
