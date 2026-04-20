"""
Playwright Browser Tools

Provides browser automation using Playwright. This is an enhanced version
with more capabilities for web interaction.

If you need an external browser automation endpoint, prefer the official
Puppeteer MCP server in its own container or process:
`npx -y @modelcontextprotocol/server-puppeteer`

Tools provided:
- browse_page: Navigate and extract page content
- take_screenshot: Capture page screenshots
- browser_click: Click elements on a page
- browser_fill: Fill form inputs
- browser_evaluate: Execute JavaScript
- browser_wait: Wait for elements
- browser_pdf: Save page as PDF
"""

import asyncio
import logging
from pathlib import Path

from playwright.async_api import Browser, Page, async_playwright

from src.core.logger import configure_root_logger
from src.core.security.security import validate_path

configure_root_logger()
logger = logging.getLogger(__name__)

_browser: Browser | None = None
_playwright = None
_browser_lock = asyncio.Lock()
_pages: dict[str, Page] = {}


async def get_browser():
    global _browser, _playwright
    async with _browser_lock:
        if _browser is None:
            _playwright = await async_playwright().start()
            _browser = await _playwright.chromium.launch(headless=True)
        return _browser


async def close_browser():
    global _browser, _playwright, _pages
    if _browser is not None:
        await _browser.close()
        _browser = None
    if _playwright is not None:
        await _playwright.stop()
        _playwright = None
    _pages.clear()


async def get_or_create_page(page_id: str | None = None) -> tuple[Page, str]:
    import uuid

    browser = await get_browser()
    if page_id and page_id in _pages:
        return _pages[page_id], page_id

    page = await browser.new_page()
    new_id = str(uuid.uuid4())[:8]
    _pages[new_id] = page
    return page, new_id


async def browse_page(
    url: str,
    page_id: str | None = None,
    wait_until: str = "domcontentloaded",
) -> str:
    """
    Navigate to a URL and extract its text content.

    Args:
        url: URL to navigate to
        page_id: Optional page ID to reuse an existing page
        wait_until: When to consider navigation complete
            (load, domcontentloaded, networkidle)

    Returns:
        Page title, URL, and text content
    """
    logger.info("Browsing: %s", url)
    try:
        page, pid = await get_or_create_page(page_id)
        await page.goto(url, timeout=30000, wait_until=wait_until)

        text = await page.evaluate("document.body.innerText")
        title = await page.title()

        return f"Page ID: {pid}\nTitle: {title}\nURL: {url}\n\n{text[:10000]}..."

    except Exception as e:
        logger.error("Browser error for %s: %s", url, e)
        return f"Error: {str(e)}"


async def take_screenshot(
    url: str,
    path: str,
    full_page: bool = False,
    selector: str | None = None,
) -> str:
    """
    Take a screenshot of a webpage.

    Args:
        url: URL to navigate to
        path: Path to save the screenshot
        full_page: Whether to capture the full page
        selector: Optional CSS selector to screenshot specific element

    Returns:
        Success message with file path
    """
    logger.info("Screenshotting: %s -> %s", url, path)
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

            Path(path).parent.mkdir(parents=True, exist_ok=True)

            if selector:
                element = await page.query_selector(selector)
                if element:
                    await element.screenshot(path=path)
                else:
                    return f"Error: Element '{selector}' not found"
            else:
                await page.screenshot(path=path, full_page=full_page)

            return f"Screenshot saved to {path}"
        finally:
            await page.close()
    except Exception as e:
        logger.error("Screenshot error: %s", e)
        return f"Error: {str(e)}"


async def browser_click(
    page_id: str,
    selector: str,
    timeout: int = 10000,
) -> str:
    """
    Click an element on a page.

    Args:
        page_id: ID of the page
        selector: CSS selector for the element
        timeout: Timeout in milliseconds

    Returns:
        Success or error message
    """
    if page_id not in _pages:
        return f"Error: Page '{page_id}' not found"

    try:
        page = _pages[page_id]
        await page.click(selector, timeout=timeout)
        return f"Clicked: {selector}"
    except Exception as e:
        return f"Error: {str(e)}"


async def browser_fill(
    page_id: str,
    selector: str,
    value: str,
    clear_first: bool = True,
) -> str:
    """
    Fill an input field on a page.

    Args:
        page_id: ID of the page
        selector: CSS selector for the input
        value: Value to fill
        clear_first: Whether to clear existing value first

    Returns:
        Success or error message
    """
    if page_id not in _pages:
        return f"Error: Page '{page_id}' not found"

    try:
        page = _pages[page_id]
        if clear_first:
            await page.fill(selector, value)
        else:
            await page.type(selector, value)
        return f"Filled: {selector}"
    except Exception as e:
        return f"Error: {str(e)}"


async def browser_evaluate(
    page_id: str,
    script: str,
) -> str:
    """
    Execute JavaScript on a page.

    Args:
        page_id: ID of the page
        script: JavaScript code to execute

    Returns:
        Script result
    """
    if page_id not in _pages:
        return f"Error: Page '{page_id}' not found"

    try:
        page = _pages[page_id]
        result = await page.evaluate(script)
        import json

        return json.dumps(result, indent=2, default=str) if result else "Success"
    except Exception as e:
        return f"Error: {str(e)}"


async def browser_wait(
    page_id: str,
    selector: str,
    timeout: int = 30000,
) -> str:
    """
    Wait for an element to appear on a page.

    Args:
        page_id: ID of the page
        selector: CSS selector for the element
        timeout: Timeout in milliseconds

    Returns:
        Success or timeout message
    """
    if page_id not in _pages:
        return f"Error: Page '{page_id}' not found"

    try:
        page = _pages[page_id]
        await page.wait_for_selector(selector, timeout=timeout)
        return f"Element found: {selector}"
    except Exception as e:
        return f"Error: {str(e)}"


async def browser_pdf(
    page_id: str,
    path: str,
    format: str = "A4",
) -> str:
    """
    Save a page as PDF.

    Args:
        page_id: ID of the page
        path: Path to save the PDF
        format: Paper format (A4, Letter, etc.)

    Returns:
        Success or error message
    """
    if page_id not in _pages:
        return f"Error: Page '{page_id}' not found"

    try:
        validate_path(path)
        page = _pages[page_id]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        await page.pdf(path=path, format=format)
        return f"PDF saved to {path}"
    except Exception as e:
        return f"Error: {str(e)}"


async def browser_get_html(page_id: str) -> str:
    """
    Get the HTML content of a page.

    Args:
        page_id: ID of the page

    Returns:
        HTML content
    """
    if page_id not in _pages:
        return f"Error: Page '{page_id}' not found"

    try:
        page = _pages[page_id]
        html = await page.content()
        return html[:50000] if len(html) > 50000 else html
    except Exception as e:
        return f"Error: {str(e)}"


async def browser_close_page(page_id: str) -> str:
    """
    Close a specific page.

    Args:
        page_id: ID of the page to close

    Returns:
        Success message
    """
    if page_id not in _pages:
        return f"Error: Page '{page_id}' not found"

    try:
        page = _pages.pop(page_id)
        await page.close()
        return f"Page {page_id} closed"
    except Exception as e:
        return f"Error: {str(e)}"


async def browser_list_pages() -> str:
    """
    List all open pages.

    Returns:
        JSON list of page IDs and URLs
    """
    import json

    pages_info = []
    for pid, page in _pages.items():
        url = page.url
        pages_info.append({"id": pid, "url": url})

    return json.dumps(pages_info, indent=2)

