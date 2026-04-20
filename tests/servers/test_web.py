"""Tests for servers.web — web_search and web_fetch."""

from unittest.mock import MagicMock, patch

import pytest

from src.servers.web import web_fetch, web_search


class TestWebSearch:
    def test_ddgs_not_installed(self, monkeypatch):
        import src.servers.web as mod
        monkeypatch.setattr(mod, "DDGS", None)
        result = web_search("test")
        assert "not installed" in result

    @patch("src.servers.web.DDGS")
    def test_returns_results(self, mock_ddgs_cls):
        mock_ddgs = MagicMock()
        mock_ddgs.text.return_value = [
            {"title": "Result 1", "href": "http://a.com", "body": "snippet 1"},
            {"title": "Result 2", "href": "http://b.com", "body": "snippet 2"},
        ]
        mock_ddgs_cls.return_value = mock_ddgs
        result = web_search("test query")
        assert "Result 1" in result
        assert "http://a.com" in result
        assert "snippet 2" in result

    @patch("src.servers.web.DDGS")
    def test_no_results(self, mock_ddgs_cls):
        mock_ddgs = MagicMock()
        mock_ddgs.text.return_value = []
        mock_ddgs_cls.return_value = mock_ddgs
        result = web_search("obscure query")
        assert "No results found" in result

    @patch("src.servers.web.DDGS")
    def test_search_exception(self, mock_ddgs_cls):
        mock_ddgs = MagicMock()
        mock_ddgs.text.side_effect = Exception("network error")
        mock_ddgs_cls.return_value = mock_ddgs
        result = web_search("test")
        assert "error" in result.lower()

    @patch("src.servers.web.DDGS")
    def test_max_results_passed(self, mock_ddgs_cls):
        mock_ddgs = MagicMock()
        mock_ddgs.text.return_value = []
        mock_ddgs_cls.return_value = mock_ddgs
        web_search("test", max_results=10)
        mock_ddgs.text.assert_called_once_with("test", max_results=10)


class TestWebFetch:
    def test_invalid_url_scheme(self):
        result = web_fetch("ftp://example.com")
        assert "Only http/https" in result

    def test_relative_url(self):
        result = web_fetch("/some/path")
        assert "Only http/https" in result

    def test_trafilatura_not_installed(self, monkeypatch):
        import src.servers.web as mod
        monkeypatch.setattr(mod, "trafilatura", None)
        result = web_fetch("https://example.com")
        assert "not installed" in result

    @patch("src.servers.web.trafilatura")
    def test_successful_fetch(self, mock_traf):
        mock_traf.fetch_url.return_value = "<html>content</html>"
        mock_traf.extract.return_value = "Extracted article text"
        result = web_fetch("https://example.com")
        assert result == "Extracted article text"
        mock_traf.fetch_url.assert_called_once_with("https://example.com")

    @patch("src.servers.web.trafilatura")
    def test_fetch_url_returns_none(self, mock_traf):
        mock_traf.fetch_url.return_value = None
        result = web_fetch("https://example.com")
        assert "Failed to fetch" in result

    @patch("src.servers.web.trafilatura")
    def test_extract_returns_none(self, mock_traf):
        mock_traf.fetch_url.return_value = "<html></html>"
        mock_traf.extract.return_value = None
        result = web_fetch("https://example.com")
        assert "Could not extract" in result

    @patch("src.servers.web.trafilatura")
    def test_fetch_exception(self, mock_traf):
        mock_traf.fetch_url.side_effect = Exception("connection failed")
        result = web_fetch("https://example.com")
        assert "error" in result.lower()

    def test_https_url_accepted(self, monkeypatch):
        import src.servers.web as mod
        monkeypatch.setattr(mod, "trafilatura", None)
        result = web_fetch("https://example.com")
        assert "not installed" in result

    def test_http_url_accepted(self, monkeypatch):
        import src.servers.web as mod
        monkeypatch.setattr(mod, "trafilatura", None)
        result = web_fetch("http://example.com")
        assert "not installed" in result
