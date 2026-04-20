"""Tests for src/core/browser.py"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.browser import BrowserConfig, BrowserManager, PageInfo, get_browser_manager


class TestBrowserConfig:
    def test_defaults(self):
        cfg = BrowserConfig()
        assert cfg.headless is True
        assert cfg.browser_type == "chromium"
        assert cfg.viewport_width == 1280
        assert cfg.viewport_height == 720
        assert cfg.timeout == 30000
        assert cfg.slow_mo == 0
        assert cfg.user_agent is None

    def test_custom(self):
        cfg = BrowserConfig(headless=False, browser_type="firefox", timeout=5000, user_agent="bot")
        assert cfg.headless is False
        assert cfg.browser_type == "firefox"
        assert cfg.timeout == 5000
        assert cfg.user_agent == "bot"


class TestPageInfo:
    def test_to_dict(self):
        info = PageInfo(url="http://example.com", title="Example", viewport={"width": 800, "height": 600}, cookies_count=3)
        d = info.to_dict()
        assert d["url"] == "http://example.com"
        assert d["title"] == "Example"
        assert d["viewport"]["width"] == 800
        assert d["cookies_count"] == 3


class TestBrowserManagerInit:
    def test_init_default(self):
        bm = BrowserManager()
        assert isinstance(bm.config, BrowserConfig)
        assert bm.is_launched() is False

    def test_init_custom_config(self):
        cfg = BrowserConfig(browser_type="webkit")
        bm = BrowserManager(config=cfg)
        assert bm.config.browser_type == "webkit"

    def test_is_available(self):
        bm = BrowserManager()
        result = bm.is_available()
        assert isinstance(result, bool)

    def test_is_launched_false(self):
        bm = BrowserManager()
        assert bm.is_launched() is False


class TestBrowserManagerNoPage:
    @pytest.mark.asyncio
    async def test_launch_without_playwright(self):
        bm = BrowserManager()
        bm._available = False
        result = await bm.launch()
        assert result is False

    @pytest.mark.asyncio
    async def test_navigate_without_page(self):
        bm = BrowserManager()
        assert await bm.navigate("http://example.com") is False

    @pytest.mark.asyncio
    async def test_click_without_page(self):
        bm = BrowserManager()
        assert await bm.click("button") is False

    @pytest.mark.asyncio
    async def test_fill_without_page(self):
        bm = BrowserManager()
        assert await bm.fill("input", "val") is False

    @pytest.mark.asyncio
    async def test_type_without_page(self):
        bm = BrowserManager()
        assert await bm.type("input", "text") is False

    @pytest.mark.asyncio
    async def test_press_without_page(self):
        bm = BrowserManager()
        assert await bm.press("Enter") is False

    @pytest.mark.asyncio
    async def test_select_without_page(self):
        bm = BrowserManager()
        assert await bm.select("select", "opt") is False

    @pytest.mark.asyncio
    async def test_wait_for_selector_without_page(self):
        bm = BrowserManager()
        assert await bm.wait_for_selector("div") is False

    @pytest.mark.asyncio
    async def test_get_text_without_page(self):
        bm = BrowserManager()
        assert await bm.get_text("div") is None

    @pytest.mark.asyncio
    async def test_get_attribute_without_page(self):
        bm = BrowserManager()
        assert await bm.get_attribute("div", "class") is None

    @pytest.mark.asyncio
    async def test_screenshot_without_page(self):
        bm = BrowserManager()
        assert await bm.screenshot() is None

    @pytest.mark.asyncio
    async def test_pdf_without_page(self):
        bm = BrowserManager()
        assert await bm.pdf("/tmp/test.pdf") is False

    @pytest.mark.asyncio
    async def test_evaluate_without_page(self):
        bm = BrowserManager()
        assert await bm.evaluate("1+1") is None

    @pytest.mark.asyncio
    async def test_get_page_info_without_page(self):
        bm = BrowserManager()
        assert await bm.get_page_info() is None

    @pytest.mark.asyncio
    async def test_get_content_without_page(self):
        bm = BrowserManager()
        assert await bm.get_content() is None

    @pytest.mark.asyncio
    async def test_set_cookies_without_context(self):
        bm = BrowserManager()
        assert await bm.set_cookies([]) is False

    @pytest.mark.asyncio
    async def test_clear_cookies_without_context(self):
        bm = BrowserManager()
        assert await bm.clear_cookies() is False


class TestBrowserManagerLaunch:
    @pytest.mark.asyncio
    async def test_launch_chromium(self):
        bm = BrowserManager()
        bm._available = True
        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()
        mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)
        mock_pw.firefox = MagicMock()
        mock_pw.webkit = MagicMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_page.set_default_timeout = MagicMock()

        class FakePlaywrightCtx:
            async def start(self):
                return mock_pw
            async def __aenter__(self):
                return mock_pw
            async def __aexit__(self, *args):
                pass

        with patch.dict("sys.modules", {"playwright": MagicMock(), "playwright.async_api": MagicMock(async_playwright=lambda: FakePlaywrightCtx())}):
            result = await bm.launch()
            assert result is True

    @pytest.mark.asyncio
    async def test_launch_firefox(self):
        cfg = BrowserConfig(browser_type="firefox")
        bm = BrowserManager(config=cfg)
        bm._available = True
        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()
        mock_pw.firefox.launch = AsyncMock(return_value=mock_browser)
        mock_pw.chromium = MagicMock()
        mock_pw.webkit = MagicMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_page.set_default_timeout = MagicMock()

        class FakePlaywrightCtx:
            async def start(self):
                return mock_pw

        with patch.dict("sys.modules", {"playwright": MagicMock(), "playwright.async_api": MagicMock(async_playwright=lambda: FakePlaywrightCtx())}):
            result = await bm.launch()
            assert result is True

    @pytest.mark.asyncio
    async def test_launch_webkit(self):
        cfg = BrowserConfig(browser_type="webkit")
        bm = BrowserManager(config=cfg)
        bm._available = True
        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()
        mock_pw.webkit.launch = AsyncMock(return_value=mock_browser)
        mock_pw.chromium = MagicMock()
        mock_pw.firefox = MagicMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_page.set_default_timeout = MagicMock()

        class FakePlaywrightCtx:
            async def start(self):
                return mock_pw

        with patch.dict("sys.modules", {"playwright": MagicMock(), "playwright.async_api": MagicMock(async_playwright=lambda: FakePlaywrightCtx())}):
            result = await bm.launch()
            assert result is True

    @pytest.mark.asyncio
    async def test_launch_with_user_agent(self):
        cfg = BrowserConfig(user_agent="TestBot/1.0")
        bm = BrowserManager(config=cfg)
        bm._available = True
        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()
        mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)
        mock_pw.firefox = MagicMock()
        mock_pw.webkit = MagicMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_page.set_default_timeout = MagicMock()

        class FakePlaywrightCtx:
            async def start(self):
                return mock_pw

        with patch.dict("sys.modules", {"playwright": MagicMock(), "playwright.async_api": MagicMock(async_playwright=lambda: FakePlaywrightCtx())}):
            result = await bm.launch()
            assert result is True
            mock_browser.new_context.assert_called_once()
            call_kwargs = mock_browser.new_context.call_args[1]
            assert call_kwargs["user_agent"] == "TestBot/1.0"

    @pytest.mark.asyncio
    async def test_launch_exception(self):
        bm = BrowserManager()
        bm._available = True
        result = await bm.launch()
        assert result is False


class TestBrowserManagerWithPage:
    def _make_manager_with_page(self):
        bm = BrowserManager()
        bm._page = MagicMock()
        bm._context = MagicMock()
        bm._browser = MagicMock()
        return bm

    @pytest.mark.asyncio
    async def test_navigate_success(self):
        bm = self._make_manager_with_page()
        bm._page.goto = AsyncMock()
        result = await bm.navigate("https://example.com")
        assert result is True
        bm._page.goto.assert_called_once_with("https://example.com", wait_until="load")

    @pytest.mark.asyncio
    async def test_navigate_failure(self):
        bm = self._make_manager_with_page()
        bm._page.goto = AsyncMock(side_effect=Exception("timeout"))
        result = await bm.navigate("https://example.com")
        assert result is False

    @pytest.mark.asyncio
    async def test_click_success(self):
        bm = self._make_manager_with_page()
        bm._page.click = AsyncMock()
        result = await bm.click("button#submit")
        assert result is True

    @pytest.mark.asyncio
    async def test_click_failure(self):
        bm = self._make_manager_with_page()
        bm._page.click = AsyncMock(side_effect=Exception("not found"))
        result = await bm.click("button#missing")
        assert result is False

    @pytest.mark.asyncio
    async def test_fill_success(self):
        bm = self._make_manager_with_page()
        bm._page.fill = AsyncMock()
        result = await bm.fill("input[name=email]", "test@test.com")
        assert result is True

    @pytest.mark.asyncio
    async def test_fill_failure(self):
        bm = self._make_manager_with_page()
        bm._page.fill = AsyncMock(side_effect=Exception("not found"))
        result = await bm.fill("input", "val")
        assert result is False

    @pytest.mark.asyncio
    async def test_type_success(self):
        bm = self._make_manager_with_page()
        bm._page.type = AsyncMock()
        result = await bm.type("input", "hello", delay=30)
        assert result is True
        bm._page.type.assert_called_once_with("input", "hello", delay=30)

    @pytest.mark.asyncio
    async def test_type_failure(self):
        bm = self._make_manager_with_page()
        bm._page.type = AsyncMock(side_effect=Exception("err"))
        result = await bm.type("input", "text")
        assert result is False

    @pytest.mark.asyncio
    async def test_press_success(self):
        bm = self._make_manager_with_page()
        bm._page.keyboard.press = AsyncMock()
        result = await bm.press("Enter")
        assert result is True

    @pytest.mark.asyncio
    async def test_press_failure(self):
        bm = self._make_manager_with_page()
        bm._page.keyboard.press = AsyncMock(side_effect=Exception("err"))
        result = await bm.press("Enter")
        assert result is False

    @pytest.mark.asyncio
    async def test_select_success(self):
        bm = self._make_manager_with_page()
        bm._page.select_option = AsyncMock()
        result = await bm.select("select", "opt1")
        assert result is True

    @pytest.mark.asyncio
    async def test_select_failure(self):
        bm = self._make_manager_with_page()
        bm._page.select_option = AsyncMock(side_effect=Exception("err"))
        result = await bm.select("select", "opt1")
        assert result is False

    @pytest.mark.asyncio
    async def test_wait_for_selector_success(self):
        bm = self._make_manager_with_page()
        bm._page.wait_for_selector = AsyncMock()
        result = await bm.wait_for_selector(".loaded", timeout=5000)
        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_selector_failure(self):
        bm = self._make_manager_with_page()
        bm._page.wait_for_selector = AsyncMock(side_effect=Exception("timeout"))
        result = await bm.wait_for_selector(".missing")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_text_success(self):
        bm = self._make_manager_with_page()
        bm._page.text_content = AsyncMock(return_value="Hello World")
        result = await bm.get_text("h1")
        assert result == "Hello World"

    @pytest.mark.asyncio
    async def test_get_text_failure(self):
        bm = self._make_manager_with_page()
        bm._page.text_content = AsyncMock(side_effect=Exception("err"))
        result = await bm.get_text("h1")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_attribute_success(self):
        bm = self._make_manager_with_page()
        bm._page.get_attribute = AsyncMock(return_value="active")
        result = await bm.get_attribute("div", "class")
        assert result == "active"

    @pytest.mark.asyncio
    async def test_get_attribute_failure(self):
        bm = self._make_manager_with_page()
        bm._page.get_attribute = AsyncMock(side_effect=Exception("err"))
        result = await bm.get_attribute("div", "class")
        assert result is None

    @pytest.mark.asyncio
    async def test_screenshot_full_page(self):
        bm = self._make_manager_with_page()
        bm._page.screenshot = AsyncMock(return_value=b"imagedata")
        result = await bm.screenshot(full_page=True)
        assert result == b"imagedata"

    @pytest.mark.asyncio
    async def test_screenshot_with_path(self):
        bm = self._make_manager_with_page()
        bm._page.screenshot = AsyncMock(return_value=b"imagedata")
        result = await bm.screenshot(path="/tmp/shot.png")
        assert result == b"imagedata"

    @pytest.mark.asyncio
    async def test_screenshot_selector_found(self):
        bm = self._make_manager_with_page()
        mock_element = MagicMock()
        mock_element.screenshot = AsyncMock(return_value=b"elemdata")
        bm._page.query_selector = AsyncMock(return_value=mock_element)
        result = await bm.screenshot(selector="#chart")
        assert result == b"elemdata"

    @pytest.mark.asyncio
    async def test_screenshot_selector_not_found(self):
        bm = self._make_manager_with_page()
        bm._page.query_selector = AsyncMock(return_value=None)
        result = await bm.screenshot(selector="#missing")
        assert result is None

    @pytest.mark.asyncio
    async def test_screenshot_failure(self):
        bm = self._make_manager_with_page()
        bm._page.screenshot = AsyncMock(side_effect=Exception("err"))
        result = await bm.screenshot()
        assert result is None

    @pytest.mark.asyncio
    async def test_pdf_success(self):
        bm = self._make_manager_with_page()
        bm._page.pdf = AsyncMock()
        result = await bm.pdf("/tmp/page.pdf")
        assert result is True

    @pytest.mark.asyncio
    async def test_pdf_failure(self):
        bm = self._make_manager_with_page()
        bm._page.pdf = AsyncMock(side_effect=Exception("not chromium"))
        result = await bm.pdf("/tmp/page.pdf")
        assert result is False

    @pytest.mark.asyncio
    async def test_evaluate_success(self):
        bm = self._make_manager_with_page()
        bm._page.evaluate = AsyncMock(return_value=42)
        result = await bm.evaluate("1+1")
        assert result == 42

    @pytest.mark.asyncio
    async def test_evaluate_failure(self):
        bm = self._make_manager_with_page()
        bm._page.evaluate = AsyncMock(side_effect=Exception("err"))
        result = await bm.evaluate("bad code")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_page_info_success(self):
        bm = self._make_manager_with_page()
        bm._context.cookies = AsyncMock(return_value=[{"name": "session"}])
        bm._page.viewport_size = {"width": 1280, "height": 720}
        bm._page.url = "https://example.com"
        bm._page.title = AsyncMock(return_value="Example")
        result = await bm.get_page_info()
        assert result.url == "https://example.com"
        assert result.title == "Example"
        assert result.cookies_count == 1
        assert result.viewport == {"width": 1280, "height": 720}

    @pytest.mark.asyncio
    async def test_get_page_info_failure(self):
        bm = self._make_manager_with_page()
        bm._context.cookies = AsyncMock(side_effect=Exception("err"))
        result = await bm.get_page_info()
        assert result is None

    @pytest.mark.asyncio
    async def test_get_content_success(self):
        bm = self._make_manager_with_page()
        bm._page.content = AsyncMock(return_value="<html></html>")
        result = await bm.get_content()
        assert result == "<html></html>"

    @pytest.mark.asyncio
    async def test_get_content_failure(self):
        bm = self._make_manager_with_page()
        bm._page.content = AsyncMock(side_effect=Exception("err"))
        result = await bm.get_content()
        assert result is None

    @pytest.mark.asyncio
    async def test_set_cookies_success(self):
        bm = self._make_manager_with_page()
        bm._context.add_cookies = AsyncMock()
        result = await bm.set_cookies([{"name": "test", "value": "1", "domain": "example.com"}])
        assert result is True

    @pytest.mark.asyncio
    async def test_set_cookies_failure(self):
        bm = self._make_manager_with_page()
        bm._context.add_cookies = AsyncMock(side_effect=Exception("err"))
        result = await bm.set_cookies([])
        assert result is False

    @pytest.mark.asyncio
    async def test_clear_cookies_success(self):
        bm = self._make_manager_with_page()
        bm._context.clear_cookies = AsyncMock()
        result = await bm.clear_cookies()
        assert result is True

    @pytest.mark.asyncio
    async def test_clear_cookies_failure(self):
        bm = self._make_manager_with_page()
        bm._context.clear_cookies = AsyncMock(side_effect=Exception("err"))
        result = await bm.clear_cookies()
        assert result is False

    @pytest.mark.asyncio
    async def test_close_success(self):
        bm = BrowserManager()
        mock_page = MagicMock()
        mock_page.close = AsyncMock()
        mock_context = MagicMock()
        mock_context.close = AsyncMock()
        mock_browser = MagicMock()
        mock_browser.close = AsyncMock()
        mock_pw = MagicMock()
        mock_pw.stop = AsyncMock()
        bm._page = mock_page
        bm._context = mock_context
        bm._browser = mock_browser
        bm._playwright = mock_pw
        await bm.close()
        assert bm._page is None
        assert bm._context is None
        assert bm._browser is None
        assert bm._playwright is None

    @pytest.mark.asyncio
    async def test_close_with_exception(self):
        bm = BrowserManager()
        mock_page = MagicMock()
        mock_page.close = AsyncMock(side_effect=Exception("err"))
        bm._page = mock_page
        await bm.close()

    def test_is_launched_true(self):
        bm = BrowserManager()
        bm._page = MagicMock()
        assert bm.is_launched() is True


class TestGetBrowserManager:
    def test_returns_instance(self):
        import src.core.browser as mod
        mod._browser_manager = None
        bm = get_browser_manager()
        assert isinstance(bm, BrowserManager)
        mod._browser_manager = None

    def test_singleton(self):
        import src.core.browser as mod
        mod._browser_manager = None
        bm1 = get_browser_manager()
        bm2 = get_browser_manager()
        assert bm1 is bm2
        mod._browser_manager = None
