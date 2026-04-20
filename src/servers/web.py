import logging

import httpx

try:
    import trafilatura
except ImportError:
    trafilatura = None

try:
    from duckduckgo_search import DDGS
    from duckduckgo_search.exceptions import DuckDuckGoSearchException
except ImportError:
    DDGS = None
    DuckDuckGoSearchException = Exception

from src.core.logger import configure_root_logger

# Setup logging
configure_root_logger()
logger = logging.getLogger(__name__)


def web_search(query: str, max_results: int = 5) -> str:
    """Search the internet for up-to-date information."""
    logger.info("Searching for: %s", query)
    if DDGS is None:
        return "Search error: duckduckgo-search not installed"
    try:
        results = DDGS().text(query, max_results=max_results)
        if not results:
            logger.info("No search results found")
            return "No results found."

        output = []
        for r in results:
            output.append(f"Title: {r['title']}\nLink: {r['href']}\nSnippet: {r['body']}\n---")

        logger.info("Found %s search results", len(results))
        return "\n".join(output)
    except (DuckDuckGoSearchException, httpx.HTTPError, ConnectionError, TimeoutError) as e:
        logger.error("Search error: %s", e)
        return f"Search error: {str(e)}"
    except Exception as e:
        logger.error("Unexpected search error: %s: %s", type(e).__name__, e)
        return f"Search error: {str(e)}"


def web_fetch(url: str) -> str:
    """Fetch and extract main content from a web page."""
    logger.info("Fetching URL: %s", url)
    if not url.startswith(("http://", "https://")):
        return "Error: Only http/https URLs are allowed"
    if trafilatura is None:
        return "Fetch error: trafilatura not installed"
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            logger.warning("Failed to fetch URL: %s", url)
            return f"Error: Failed to fetch {url}"

        result = trafilatura.extract(downloaded)
        if not result:
            logger.warning("Could not extract content from: %s", url)
            return "Error: Could not extract content from page."

        logger.info("Successfully extracted %s characters from %s", len(result), url)
        return result
    except Exception as e:
        logger.error("Fetch error for %s: %s", url, e)
        return f"Fetch error: {str(e)}"
