
import asyncio
import logging

from mcp.server.fastmcp import FastMCP
from playwright.async_api import Browser, async_playwright

from src.core.security import validate_path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("DoraemonBrowser")

# Global browser instance
_browser: Browser | None = None
_playwright = None
_browser_lock = asyncio.Lock()

async def get_browser():
    global _browser, _playwright
    async with _browser_lock:
        if _browser is None:
            _playwright = await async_playwright().start()
            _browser = await _playwright.chromium.launch(headless=True)
        return _browser


async def close_browser():
    """Close the browser and playwright instances to release resources."""
    global _browser, _playwright
    if _browser is not None:
        await _browser.close()
        _browser = None
    if _playwright is not None:
        await _playwright.stop()
        _playwright = None


@mcp.tool()
async def browse_page(url: str) -> str:
    """
    Navigate to a URL and extract its text content significantly better than a simple fetch.
    Handles JavaScript rendering.
    """
    logger.info(f"Browsing: {url}")
    try:
        browser = await get_browser()
        page = await browser.new_page()
        try:
            await page.goto(url, timeout=30000)
            # Wait for some content to load
            await page.wait_for_load_state("domcontentloaded")

            # Simple text extraction
            # For better results, one might use readability.js or similar,
            # but for now, innerText of body is a good start.
            text = await page.evaluate("document.body.innerText")
            title = await page.title()

            return f"Title: {title}\nURL: {url}\n\n{text[:10000]}..." # Limit output size

        finally:
            await page.close()

    except Exception as e:
        logger.error(f"Browser error for {url}: {e}")
        return f"Error: {str(e)}"

@mcp.tool()
async def take_screenshot(url: str, path: str) -> str:
    """
    Take a screenshot of a webpage.
    """
    logger.info(f"Screenshotting: {url} -> {path}")
    try:
        validate_path(path)
    except (ValueError, PermissionError) as e:
        return f"Error: {e}"
    try:
        browser = await get_browser()
        page = await browser.new_page()
        try:
            await page.goto(url, timeout=30000)
            await page.wait_for_load_state("networkidle")
            await page.screenshot(path=path)
            return f"Screenshot saved to {path}"
        finally:
            await page.close()
    except Exception as e:
        logger.error(f"Screenshot error: {e}")
        return f"Error: {str(e)}"

if __name__ == "__main__":
    mcp.run()
