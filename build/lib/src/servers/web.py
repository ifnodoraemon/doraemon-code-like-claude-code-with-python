from mcp.server.fastmcp import FastMCP
from duckduckgo_search import DDGS

mcp = FastMCP("PolymathWeb")

@mcp.tool()
def search_internet(query: str, max_results: int = 5) -> str:
    """Search the internet for up-to-date information."""
    try:
        results = DDGS().text(query, max_results=max_results)
        if not results:
            return "No results found."
        
        output = []
        for r in results:
            output.append(f"Title: {r['title']}\nLink: {r['href']}\nSnippet: {r['body']}\n---")
        return "\n".join(output)
    except Exception as e:
        return f"Search error: {str(e)}"

if __name__ == "__main__":
    mcp.run()

