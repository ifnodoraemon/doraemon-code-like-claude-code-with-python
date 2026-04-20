"""Tests for src/webui/__init__.py — lazy import coverage."""

import pytest


class TestWebuiInit:
    def test_getattr_app(self):
        from src.webui import __getattr__
        result = __getattr__("app")
        assert result is not None

    def test_getattr_start_server(self):
        from src.webui import __getattr__
        result = __getattr__("start_server")
        assert callable(result)

    def test_getattr_unknown_raises(self):
        from src.webui import __getattr__
        with pytest.raises(AttributeError, match="nonexistent"):
            __getattr__("nonexistent")
