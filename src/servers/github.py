
import logging
import os

from mcp.server.fastmcp import FastMCP

try:
    from github import Auth, Github
    _PYGITHUB_AVAILABLE = True
except ImportError:
    _PYGITHUB_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("DoraemonGitHub")

def get_github_client() -> "Github | None":
    if not _PYGITHUB_AVAILABLE:
        return None
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        return None
    auth = Auth.Token(token)
    return Github(auth=auth)

@mcp.tool()
def github_list_issues(owner: str, repo: str, state: str = "open") -> str:
    """List issues for a repository."""
    g = get_github_client()
    if not g:
        return "Error: GITHUB_TOKEN not configured."

    try:
        r = g.get_repo(f"{owner}/{repo}")
        issues = r.get_issues(state=state)

        output = []
        for issue in issues[:10]: # Limit to 10
            output.append(f"#{issue.number} {issue.title} (by {issue.user.login})")

        return "\n".join(output) if output else "No issues found."
    except Exception as e:
        logger.error(f"GitHub list error: {e}")
        return f"Error: {e}"

@mcp.tool()
def github_create_issue(owner: str, repo: str, title: str, body: str) -> str:
    """Create a new issue."""
    g = get_github_client()
    if not g:
        return "Error: GITHUB_TOKEN not configured."

    try:
        r = g.get_repo(f"{owner}/{repo}")
        issue = r.create_issue(title=title, body=body)
        return f"Issue created: #{issue.number} {issue.html_url}"
    except Exception as e:
        logger.error(f"GitHub create issue error: {e}")
        return f"Error: {e}"

if __name__ == "__main__":
    mcp.run()
