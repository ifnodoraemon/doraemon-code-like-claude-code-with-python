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
    logger.info(f"Searching for: {query}")
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

        logger.info(f"Found {len(results)} search results")
        return "\n".join(output)
    except (DuckDuckGoSearchException, httpx.HTTPError, ConnectionError, TimeoutError) as e:
        logger.error(f"Search error: {e}")
        return f"Search error: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected search error: {type(e).__name__}: {e}")
        return f"Search error: {str(e)}"


def web_fetch(url: str) -> str:
    """Fetch and extract main content from a web page."""
    logger.info(f"Fetching URL: {url}")
    if not url.startswith(("http://", "https://")):
        return "Error: Only http/https URLs are allowed"
    if trafilatura is None:
        return "Fetch error: trafilatura not installed"
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
