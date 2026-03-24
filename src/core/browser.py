"""
Browser Automation

Playwright-based browser automation for web testing and scraping.

Features:
- Browser launch and control
- Page navigation and interaction
- Screenshot and PDF capture
- Element selection and action
- Cookie and storage management
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BrowserConfig:
    """Browser configuration."""

    headless: bool = True
    browser_type: str = "chromium"  # chromium, firefox, webkit
    viewport_width: int = 1280
    viewport_height: int = 720
    timeout: int = 30000  # ms
    slow_mo: int = 0  # Slow down operations (ms)
    user_agent: str | None = None


@dataclass
class PageInfo:
    """Information about the current page."""

    url: str
    title: str
    viewport: dict[str, int]
    cookies_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "viewport": self.viewport,
            "cookies_count": self.cookies_count,
        }


class BrowserManager:
    """
    Manages browser automation with Playwright.

    Usage:
        browser = BrowserManager()

        # Start browser
        await browser.launch()

        # Navigate
        await browser.navigate("https://example.com")

        # Interact
        await browser.click("button#submit")
        await browser.fill("input[name=email]", "test@example.com")

        # Screenshot
        screenshot = await browser.screenshot()

        # Close
        await browser.close()
    """

    def __init__(self, config: BrowserConfig | None = None):
        """
        Initialize browser manager.

        Args:
            config: Browser configuration
        """
        self.config = config or BrowserConfig()
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._available = False

        # Check if Playwright is installed
        try:
            import playwright  # noqa: F401

            self._available = True
        except ImportError:
            logger.warning(
                "Playwright not installed. Run: pip install playwright && playwright install"
            )

    def is_available(self) -> bool:
        """Check if browser automation is available."""
        return self._available

    async def launch(self) -> bool:
        """
        Launch the browser.

        Returns:
            True if launched successfully
        """
        if not self._available:
            logger.error("Playwright not available")
            return False

        try:
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()

            # Select browser type
            if self.config.browser_type == "firefox":
                browser_type = self._playwright.firefox
            elif self.config.browser_type == "webkit":
                browser_type = self._playwright.webkit
            else:
                browser_type = self._playwright.chromium

            # Launch browser
            self._browser = await browser_type.launch(
                headless=self.config.headless,
                slow_mo=self.config.slow_mo,
            )

            # Create context
            context_options = {
                "viewport": {
                    "width": self.config.viewport_width,
                    "height": self.config.viewport_height,
                },
            }

            if self.config.user_agent:
                context_options["user_agent"] = self.config.user_agent

            self._context = await self._browser.new_context(**context_options)

            # Create page
            self._page = await self._context.new_page()
            self._page.set_default_timeout(self.config.timeout)

            logger.info(f"Browser launched: {self.config.browser_type}")
            return True

        except Exception as e:
            logger.error(f"Failed to launch browser: {e}")
            return False

    async def close(self):
        """Close the browser."""
        try:
            if self._page:
                await self._page.close()
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()

            self._page = None
            self._context = None
            self._browser = None
            self._playwright = None

            logger.info("Browser closed")

        except Exception as e:
            logger.error(f"Error closing browser: {e}")

    async def navigate(self, url: str, wait_until: str = "load") -> bool:
        """
        Navigate to a URL.

        Args:
            url: URL to navigate to
            wait_until: Wait condition (load, domcontentloaded, networkidle)

        Returns:
            True if navigation successful
        """
        if not self._page:
            logger.error("Browser not launched")
            return False

        try:
            await self._page.goto(url, wait_until=wait_until)
            logger.info(f"Navigated to: {url}")
            return True
        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            return False

    async def click(self, selector: str) -> bool:
        """
        Click an element.

        Args:
            selector: CSS selector

        Returns:
            True if click successful
        """
        if not self._page:
            return False

        try:
            await self._page.click(selector)
            return True
        except Exception as e:
            logger.error(f"Click failed on {selector}: {e}")
            return False

    async def fill(self, selector: str, value: str) -> bool:
        """
        Fill a form field.

        Args:
            selector: CSS selector
            value: Value to fill

        Returns:
            True if fill successful
        """
        if not self._page:
            return False

        try:
            await self._page.fill(selector, value)
            return True
        except Exception as e:
            logger.error(f"Fill failed on {selector}: {e}")
            return False

    async def type(self, selector: str, text: str, delay: int = 50) -> bool:
        """
        Type text into an element.

        Args:
            selector: CSS selector
            text: Text to type
            delay: Delay between keystrokes (ms)

        Returns:
            True if type successful
        """
        if not self._page:
            return False

        try:
            await self._page.type(selector, text, delay=delay)
            return True
        except Exception as e:
            logger.error(f"Type failed on {selector}: {e}")
            return False

    async def press(self, key: str) -> bool:
        """
        Press a key.

        Args:
            key: Key to press (e.g., "Enter", "Tab")

        Returns:
            True if press successful
        """
        if not self._page:
            return False

        try:
            await self._page.keyboard.press(key)
            return True
        except Exception as e:
            logger.error(f"Press failed for {key}: {e}")
            return False

    async def select(self, selector: str, value: str) -> bool:
        """
        Select an option from a dropdown.

        Args:
            selector: CSS selector
            value: Option value

        Returns:
            True if select successful
        """
        if not self._page:
            return False

        try:
            await self._page.select_option(selector, value)
            return True
        except Exception as e:
            logger.error(f"Select failed on {selector}: {e}")
            return False

    async def wait_for_selector(self, selector: str, timeout: int | None = None) -> bool:
        """
        Wait for an element to appear.

        Args:
            selector: CSS selector
            timeout: Timeout in ms

        Returns:
            True if element found
        """
        if not self._page:
            return False

        try:
            await self._page.wait_for_selector(selector, timeout=timeout)
            return True
        except Exception as e:
            logger.error(f"Wait failed for {selector}: {e}")
            return False

    async def get_text(self, selector: str) -> str | None:
        """
        Get text content of an element.

        Args:
            selector: CSS selector

        Returns:
            Text content or None
        """
        if not self._page:
            return None

        try:
            return await self._page.text_content(selector)
        except Exception as e:
            logger.error(f"Get text failed for {selector}: {e}")
            return None

    async def get_attribute(self, selector: str, attribute: str) -> str | None:
        """
        Get attribute value of an element.

        Args:
            selector: CSS selector
            attribute: Attribute name

        Returns:
            Attribute value or None
        """
        if not self._page:
            return None

        try:
            return await self._page.get_attribute(selector, attribute)
        except Exception as e:
            logger.error(f"Get attribute failed: {e}")
            return None

    async def screenshot(
        self,
        path: str | Path | None = None,
        full_page: bool = False,
        selector: str | None = None,
    ) -> bytes | None:
        """
        Take a screenshot.

        Args:
            path: Optional path to save screenshot
            full_page: Capture full page
            selector: Optional selector for element screenshot

        Returns:
            Screenshot bytes or None
        """
        if not self._page:
            return None

        try:
            options = {"full_page": full_page}

            if path:
                options["path"] = str(path)

            if selector:
                element = await self._page.query_selector(selector)
                if element:
                    return await element.screenshot(**options)
                return None

            return await self._page.screenshot(**options)

        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return None

    async def pdf(self, path: str | Path) -> bool:
        """
        Save page as PDF (Chromium only).

        Args:
            path: Path to save PDF

        Returns:
            True if successful
        """
        if not self._page:
            return False

        try:
            await self._page.pdf(path=str(path))
            return True
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            return False

    async def evaluate(self, script: str) -> Any:
        """
        Execute JavaScript on the page.

        Args:
            script: JavaScript code

        Returns:
            Evaluation result
        """
        if not self._page:
            return None

        try:
            return await self._page.evaluate(script)
        except Exception as e:
            logger.error(f"Evaluate failed: {e}")
            return None

    async def get_page_info(self) -> PageInfo | None:
        """Get information about the current page."""
        if not self._page:
            return None

        try:
            cookies = await self._context.cookies() if self._context else []
            viewport = self._page.viewport_size or {}

            return PageInfo(
                url=self._page.url,
                title=await self._page.title(),
                viewport=viewport,
                cookies_count=len(cookies),
            )
        except Exception as e:
            logger.error(f"Get page info failed: {e}")
            return None

    async def get_content(self) -> str | None:
        """Get page HTML content."""
        if not self._page:
            return None

        try:
            return await self._page.content()
        except Exception as e:
            logger.error(f"Get content failed: {e}")
            return None

    async def set_cookies(self, cookies: list[dict]) -> bool:
        """
        Set cookies.

        Args:
            cookies: List of cookie dicts

        Returns:
            True if successful
        """
        if not self._context:
            return False

        try:
            await self._context.add_cookies(cookies)
            return True
        except Exception as e:
            logger.error(f"Set cookies failed: {e}")
            return False

    async def clear_cookies(self) -> bool:
        """Clear all cookies."""
        if not self._context:
            return False

        try:
            await self._context.clear_cookies()
            return True
        except Exception as e:
            logger.error(f"Clear cookies failed: {e}")
            return False

    def is_launched(self) -> bool:
        """Check if browser is launched."""
        return self._page is not None


# Global browser instance
_browser_manager: BrowserManager | None = None


def get_browser_manager() -> BrowserManager:
    """Get the global browser manager."""
    global _browser_manager
    if _browser_manager is None:
        _browser_manager = BrowserManager()
    return _browser_manager
