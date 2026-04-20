import ipaddress
import logging
import urllib.parse

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

_PRIVATE_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("::ffff:0.0.0.0/96"),
]


def _is_private_url(url: str) -> bool:
    """Check if a URL resolves to a private/internal network address.

    Note: This checks at URL-parse time, not at connection time.
    DNS rebinding attacks are possible but require a higher level of
    mitigation (custom DNS resolver / connection-level validation).
    HTTP redirect chains are not followed by this check.

    IMPORTANT: For defense-in-depth against DNS rebinding, log a warning
    when fetching from a hostname that resolved differently than expected.
    """
    try:
        parsed = urllib.parse.urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return True

        try:
            addr = ipaddress.ip_address(hostname)
        except ValueError:
            if hostname in ("localhost", "localhost.localdomain"):
                return True
            return False

        if isinstance(addr, ipaddress.IPv6Address):
            try:
                mapped_ipv4 = addr.ipv4_mapped
                if mapped_ipv4 is not None:
                    addr = mapped_ipv4
            except ValueError:
                pass

        return any(addr in net for net in _PRIVATE_NETWORKS)
    except Exception:
        return True


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
        return "Search error: Unable to complete search. Please try again."
    except Exception as e:
        logger.error("Unexpected search error: %s: %s", type(e).__name__, e)
        return "Search error: An unexpected error occurred. Please try again."


def web_fetch(url: str) -> str:
    """Fetch and extract main content from a web page."""
    logger.info("Fetching URL: %s", url)
    if not url.startswith(("http://", "https://")):
        return "Error: Only http/https URLs are allowed"
    if _is_private_url(url):
        return "Error: URLs pointing to private/internal network addresses are not allowed"
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
        return "Fetch error: Unable to extract content from the page."
