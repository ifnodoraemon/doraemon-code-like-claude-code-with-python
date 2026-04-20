"""Tests for src/servers/browser.py

Playwright may not be installed, so we mock it at import time.
"""

import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock playwright before importing the module
_mock_playwright = MagicMock()
sys.modules.setdefault("playwright", _mock_playwright)
sys.modules.setdefault("playwright.async_api", MagicMock())

from src.servers import browser as browser_mod


@pytest.fixture(autouse=True)
def _reset_browser_state():
    browser_mod._browser = None
    browser_mod._playwright = None
    browser_mod._pages.clear()
    yield
    browser_mod._browser = None
    browser_mod._playwright = None
    browser_mod._pages.clear()


class TestBrowsePage:
    @pytest.mark.asyncio
    async def test_browse_page_creates_page_and_navigates(self):
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value="page text content")
        mock_page.title = AsyncMock(return_value="Test Title")

        mock_browser = AsyncMock()
        mock_browser.new_page = AsyncMock(return_value=mock_page)

        with patch.object(browser_mod, "get_browser", return_value=mock_browser), \
             patch("uuid.uuid4", return_value=MagicMock(__str__=lambda s: "abc12345")):
            result = await browser_mod.browse_page("https://example.com")
            assert "Test Title" in result
            assert "abc12345" in result

    @pytest.mark.asyncio
    async def test_browse_page_with_existing_page(self):
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value="text")
        mock_page.title = AsyncMock(return_value="Title")

        browser_mod._pages["existing"] = mock_page

        with patch.object(browser_mod, "get_browser"):
            result = await browser_mod.browse_page("https://example.com", page_id="existing")
            assert "Title" in result


class TestBrowserClick:
    @pytest.mark.asyncio
    async def test_click_missing_page(self):
        result = await browser_mod.browser_click("missing", "button")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_click_success(self):
        mock_page = AsyncMock()
        mock_page.click = AsyncMock()
        browser_mod._pages["p1"] = mock_page

        result = await browser_mod.browser_click("p1", "button")
        assert "Clicked" in result


class TestBrowserFill:
    @pytest.mark.asyncio
    async def test_fill_missing_page(self):
        result = await browser_mod.browser_fill("missing", "input", "val")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_fill_success(self):
        mock_page = AsyncMock()
        mock_page.fill = AsyncMock()
        browser_mod._pages["p1"] = mock_page

        result = await browser_mod.browser_fill("p1", "input", "hello")
        assert "Filled" in result


class TestBrowserEvaluate:
    @pytest.mark.asyncio
    async def test_evaluate_missing_page(self):
        result = await browser_mod.browser_evaluate("missing", "1+1")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_evaluate_success(self):
        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value={"key": "val"})
        browser_mod._pages["p1"] = mock_page

        result = await browser_mod.browser_evaluate("p1", "JSON.stringify({key:'val'})")
        assert "key" in result


class TestBrowserWait:
    @pytest.mark.asyncio
    async def test_wait_missing_page(self):
        result = await browser_mod.browser_wait("missing", "div")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_wait_success(self):
        mock_page = AsyncMock()
        mock_page.wait_for_selector = AsyncMock()
        browser_mod._pages["p1"] = mock_page

        result = await browser_mod.browser_wait("p1", "div")
        assert "found" in result.lower() or "Element" in result


class TestBrowserPdf:
    @pytest.mark.asyncio
    async def test_pdf_missing_page(self):
        result = await browser_mod.browser_pdf("missing", "/tmp/out.pdf")
        assert "Error" in result


class TestBrowserGetHtml:
    @pytest.mark.asyncio
    async def test_get_html_missing_page(self):
        result = await browser_mod.browser_get_html("missing")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_get_html_success(self):
        mock_page = AsyncMock()
        mock_page.content = AsyncMock(return_value="<html>hello</html>")
        browser_mod._pages["p1"] = mock_page

        result = await browser_mod.browser_get_html("p1")
        assert "<html>" in result


class TestBrowserClosePage:
    @pytest.mark.asyncio
    async def test_close_missing_page(self):
        result = await browser_mod.browser_close_page("missing")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_close_success(self):
        mock_page = AsyncMock()
        mock_page.close = AsyncMock()
        browser_mod._pages["p1"] = mock_page

        result = await browser_mod.browser_close_page("p1")
        assert "closed" in result
        assert "p1" not in browser_mod._pages


class TestBrowserListPages:
    @pytest.mark.asyncio
    async def test_list_empty(self):
        result = await browser_mod.browser_list_pages()
        data = json.loads(result)
        assert data == []

    @pytest.mark.asyncio
    async def test_list_with_pages(self):
        mock_page = MagicMock()
        mock_page.url = "https://example.com"
        browser_mod._pages["p1"] = mock_page

        result = await browser_mod.browser_list_pages()
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["id"] == "p1"
        assert data[0]["url"] == "https://example.com"


class TestGetOrCreatePage:
    @pytest.mark.asyncio
    async def test_existing_page_reused(self):
        mock_page = AsyncMock()
        browser_mod._pages["abc"] = mock_page

        mock_browser = AsyncMock()
        with patch.object(browser_mod, "get_browser", return_value=mock_browser):
            page, pid = await browser_mod.get_or_create_page("abc")
            assert pid == "abc"
            assert page is mock_page

    @pytest.mark.asyncio
    async def test_new_page_created(self):
        mock_page = AsyncMock()
        mock_browser = AsyncMock()
        mock_browser.new_page = AsyncMock(return_value=mock_page)

        with patch.object(browser_mod, "get_browser", return_value=mock_browser), \
             patch("uuid.uuid4", return_value=MagicMock(__str__=lambda s: "newpage1")):
            page, pid = await browser_mod.get_or_create_page()
            assert page is mock_page


class TestBrowserFillAdvanced:
    @pytest.mark.asyncio
    async def test_fill_without_clear(self):
        mock_page = AsyncMock()
        mock_page.type = AsyncMock()
        browser_mod._pages["p1"] = mock_page

        result = await browser_mod.browser_fill("p1", "input", "hello", clear_first=False)
        assert "Filled" in result
        mock_page.type.assert_called_once()

    @pytest.mark.asyncio
    async def test_fill_with_clear(self):
        mock_page = AsyncMock()
        mock_page.fill = AsyncMock()
        browser_mod._pages["p1"] = mock_page

        result = await browser_mod.browser_fill("p1", "input", "hello", clear_first=True)
        assert "Filled" in result
        mock_page.fill.assert_called_once()

    @pytest.mark.asyncio
    async def test_fill_exception(self):
        mock_page = AsyncMock()
        mock_page.fill = AsyncMock(side_effect=Exception("fill error"))
        browser_mod._pages["p1"] = mock_page

        result = await browser_mod.browser_fill("p1", "input", "hello")
        assert "Error" in result


class TestBrowserClickAdvanced:
    @pytest.mark.asyncio
    async def test_click_exception(self):
        mock_page = AsyncMock()
        mock_page.click = AsyncMock(side_effect=Exception("click error"))
        browser_mod._pages["p1"] = mock_page

        result = await browser_mod.browser_click("p1", "button")
        assert "Error" in result


class TestBrowserWaitAdvanced:
    @pytest.mark.asyncio
    async def test_wait_exception(self):
        mock_page = AsyncMock()
        mock_page.wait_for_selector = AsyncMock(side_effect=Exception("timeout"))
        browser_mod._pages["p1"] = mock_page

        result = await browser_mod.browser_wait("p1", "div")
        assert "Error" in result


class TestBrowserEvaluateAdvanced:
    @pytest.mark.asyncio
    async def test_evaluate_none_result(self):
        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value=None)
        browser_mod._pages["p1"] = mock_page

        result = await browser_mod.browser_evaluate("p1", "void(0)")
        assert result == "Success"

    @pytest.mark.asyncio
    async def test_evaluate_exception(self):
        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(side_effect=Exception("eval error"))
        browser_mod._pages["p1"] = mock_page

        result = await browser_mod.browser_evaluate("p1", "bad_code")
        assert "Error" in result


class TestBrowserPdfAdvanced:
    @pytest.mark.asyncio
    async def test_pdf_success(self):
        mock_page = AsyncMock()
        mock_page.pdf = AsyncMock()
        browser_mod._pages["p1"] = mock_page

        with patch("src.servers.browser.validate_path"):
            result = await browser_mod.browser_pdf("p1", "/tmp/test.pdf")
            assert "PDF saved" in result

    @pytest.mark.asyncio
    async def test_pdf_invalid_path(self):
        with patch("src.servers.browser.validate_path", side_effect=ValueError("bad path")):
            result = await browser_mod.browser_pdf("p1", "/tmp/test.pdf")
            assert "Error" in result

    @pytest.mark.asyncio
    async def test_pdf_exception(self):
        mock_page = AsyncMock()
        mock_page.pdf = AsyncMock(side_effect=Exception("pdf error"))
        browser_mod._pages["p1"] = mock_page

        with patch("src.servers.browser.validate_path"):
            result = await browser_mod.browser_pdf("p1", "/tmp/test.pdf")
            assert "Error" in result


class TestBrowserGetHtmlAdvanced:
    @pytest.mark.asyncio
    async def test_get_html_long_content_truncated(self):
        mock_page = AsyncMock()
        mock_page.content = AsyncMock(return_value="<html>" + "x" * 60000 + "</html>")
        browser_mod._pages["p1"] = mock_page

        result = await browser_mod.browser_get_html("p1")
        assert len(result) == 50000

    @pytest.mark.asyncio
    async def test_get_html_exception(self):
        mock_page = AsyncMock()
        mock_page.content = AsyncMock(side_effect=Exception("content error"))
        browser_mod._pages["p1"] = mock_page

        result = await browser_mod.browser_get_html("p1")
        assert "Error" in result


class TestBrowserClosePageAdvanced:
    @pytest.mark.asyncio
    async def test_close_page_exception(self):
        mock_page = AsyncMock()
        mock_page.close = AsyncMock(side_effect=Exception("close error"))
        browser_mod._pages["p1"] = mock_page

        result = await browser_mod.browser_close_page("p1")
        assert "Error" in result


class TestBrowsePageAdvanced:
    @pytest.mark.asyncio
    async def test_browse_page_exception(self):
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(side_effect=Exception("nav error"))

        browser_mod._pages["p1"] = mock_page

        with patch.object(browser_mod, "get_browser"):
            result = await browser_mod.browse_page("https://example.com", page_id="p1")
            assert "Error" in result


class TestTakeScreenshot:
    @pytest.mark.asyncio
    async def test_screenshot_invalid_path(self):
        with patch("src.servers.browser.validate_path", side_effect=ValueError("bad")):
            result = await browser_mod.take_screenshot("https://example.com", "/bad/path.png")
            assert "Error" in result

    @pytest.mark.asyncio
    async def test_screenshot_selector_found(self):
        mock_element = AsyncMock()
        mock_element.screenshot = AsyncMock()

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.wait_for_load_state = AsyncMock()
        mock_page.query_selector = AsyncMock(return_value=mock_element)
        mock_page.close = AsyncMock()

        mock_browser = AsyncMock()
        mock_browser.new_page = AsyncMock(return_value=mock_page)

        with patch("src.servers.browser.validate_path"), \
             patch.object(browser_mod, "get_browser", return_value=mock_browser):
            result = await browser_mod.take_screenshot(
                "https://example.com", "/tmp/shot.png", selector=".main"
            )
            assert "Screenshot saved" in result

    @pytest.mark.asyncio
    async def test_screenshot_selector_not_found(self):
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.wait_for_load_state = AsyncMock()
        mock_page.query_selector = AsyncMock(return_value=None)
        mock_page.close = AsyncMock()

        mock_browser = AsyncMock()
        mock_browser.new_page = AsyncMock(return_value=mock_page)

        with patch("src.servers.browser.validate_path"), \
             patch.object(browser_mod, "get_browser", return_value=mock_browser):
            result = await browser_mod.take_screenshot(
                "https://example.com", "/tmp/shot.png", selector=".missing"
            )
            assert "Error" in result
            assert "not found" in result

    @pytest.mark.asyncio
    async def test_screenshot_full_page(self):
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.wait_for_load_state = AsyncMock()
        mock_page.screenshot = AsyncMock()
        mock_page.close = AsyncMock()

        mock_browser = AsyncMock()
        mock_browser.new_page = AsyncMock(return_value=mock_page)

        with patch("src.servers.browser.validate_path"), \
             patch.object(browser_mod, "get_browser", return_value=mock_browser):
            result = await browser_mod.take_screenshot(
                "https://example.com", "/tmp/shot.png", full_page=True
            )
            assert "Screenshot saved" in result

    @pytest.mark.asyncio
    async def test_screenshot_exception(self):
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(side_effect=Exception("network error"))
        mock_page.close = AsyncMock()

        mock_browser = AsyncMock()
        mock_browser.new_page = AsyncMock(return_value=mock_page)

        with patch("src.servers.browser.validate_path"), \
             patch.object(browser_mod, "get_browser", return_value=mock_browser):
            result = await browser_mod.take_screenshot(
                "https://example.com", "/tmp/shot.png"
            )
            assert "Error" in result
