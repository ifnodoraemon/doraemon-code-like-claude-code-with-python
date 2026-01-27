import logging

import trafilatura
from duckduckgo_search import DDGS
from mcp.server.fastmcp import FastMCP

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("DoraemonWeb")


@mcp.tool()
def search_internet(query: str, max_results: int = 5) -> str:
    """Search the internet for up-to-date information."""
    logger.info(f"Searching for: {query}")
    try:
        results = DDGS().text(query, max_results=max_results)
        if not results:
            logger.info("No search results found")
            return "No results found."

        output = []
        for r in results:
            output.append(f"Title: {r['title']}\nLink: {r['href']}\nSnippet: {r['body']}\n---")

        logger.info(f"Found {len(results)} search results")
        return "\n".join(output)
    except Exception as e:
        logger.error(f"Search error: {e}")
        return f"Search error: {str(e)}"


@mcp.tool()
def fetch_page(url: str) -> str:
    """Fetch and extract main content from a URL."""
    logger.info(f"Fetching URL: {url}")
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            logger.warning(f"Failed to fetch URL: {url}")
            return f"Error: Failed to fetch {url}"

        result = trafilatura.extract(downloaded)
        if not result:
            logger.warning(f"Could not extract content from: {url}")
            return "Error: Could not extract content from page."

        logger.info(f"Successfully extracted {len(result)} characters from {url}")
        return result
    except Exception as e:
        logger.error(f"Fetch error for {url}: {e}")
        return f"Fetch error: {str(e)}"


if __name__ == "__main__":
    mcp.run()
