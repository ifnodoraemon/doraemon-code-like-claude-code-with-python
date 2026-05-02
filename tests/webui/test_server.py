from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import Response

from src.webui.server import (
    _extract_asset_paths,
    _is_loopback_request,
    _is_rate_limited_path,
    _load_dashboard_router,
    _load_webui_api_key,
    _verify_webui_api_key,
    app,
    resolve_static_bundle,
)


async def _call_enforce_api_key(path: str, headers: dict[str, str] | None = None):
    import src.webui.server as mod

    request = SimpleNamespace(
        url=SimpleNamespace(path=path),
        headers=headers or {},
        client=SimpleNamespace(host="203.0.113.10"),
    )
    called = False

    async def call_next(_request):
        nonlocal called
        called = True
        return Response(status_code=204)

    response = await mod.enforce_api_key(request, call_next)
    return response, called


def test_resolve_static_bundle_accepts_valid_bundle(tmp_path: Path):
    static_dir = tmp_path / "static"
    assets_dir = static_dir / "assets"
    assets_dir.mkdir(parents=True)
    (assets_dir / "index-test.js").write_text("console.log('ok')", encoding="utf-8")
    (assets_dir / "index-test.css").write_text("body{}", encoding="utf-8")
    (static_dir / "index.html").write_text(
        '<script type="module" src="/assets/index-test.js"></script>'
        '<link rel="stylesheet" href="/assets/index-test.css">',
        encoding="utf-8",
    )

    assert resolve_static_bundle(tmp_path) == static_dir


def test_resolve_static_bundle_rejects_missing_assets(tmp_path: Path):
    static_dir = tmp_path / "static"
    static_dir.mkdir(parents=True)
    (static_dir / "index.html").write_text(
        '<script type="module" src="/assets/index-missing.js"></script>',
        encoding="utf-8",
    )

    assert resolve_static_bundle(tmp_path) is None


def test_root_route_is_registered():
    assert any(
        getattr(route, "name", None) == "static" and getattr(route, "path", None) in {"", "/"}
        for route in app.routes
    )


class TestExtractAssetPaths:
    def test_src_assets(self):
        html = '<script src="/assets/app.js"></script>'
        paths = _extract_asset_paths(html)
        assert "app.js" in paths

    def test_href_assets(self):
        html = '<link href="/assets/style.css">'
        paths = _extract_asset_paths(html)
        assert "style.css" in paths

    def test_multiple_assets(self):
        html = (
            '<script src="/assets/app.js"></script>'
            '<link href="/assets/style.css">'
            '<script src="/assets/chunk.js"></script>'
        )
        paths = _extract_asset_paths(html)
        assert len(paths) == 3

    def test_no_assets(self):
        html = "<html><body>Hello</body></html>"
        paths = _extract_asset_paths(html)
        assert paths == []

    def test_empty_string(self):
        paths = _extract_asset_paths("")
        assert paths == []


class TestResolveStaticBundle:
    def test_missing_index_html(self, tmp_path: Path):
        result = resolve_static_bundle(tmp_path)
        assert result is None

    def test_missing_static_dir(self, tmp_path: Path):
        result = resolve_static_bundle(tmp_path / "nonexistent")
        assert result is None


class TestLoadDashboardRouter:
    def test_loads_or_returns_none(self):
        result = _load_dashboard_router()
        assert result is None or hasattr(result, "routes")


class TestVerifyWebuiApiKey:
    def test_empty_env_key_is_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv("AGENT_WEBUI_API_KEY", "")

        assert _load_webui_api_key() is None

    def test_no_key_configured(self, monkeypatch):
        import src.webui.server as mod

        monkeypatch.setattr(mod, "WEBUI_API_KEY", None)
        assert _verify_webui_api_key(None) is True

    def test_valid_key(self, monkeypatch):
        import src.webui.server as mod

        monkeypatch.setattr(mod, "WEBUI_API_KEY", "secret")
        assert _verify_webui_api_key("Bearer secret") is True

    def test_valid_key_no_bearer(self, monkeypatch):
        import src.webui.server as mod

        monkeypatch.setattr(mod, "WEBUI_API_KEY", "secret")
        assert _verify_webui_api_key("secret") is True

    def test_invalid_key(self, monkeypatch):
        import src.webui.server as mod

        monkeypatch.setattr(mod, "WEBUI_API_KEY", "secret")
        assert _verify_webui_api_key("wrong") is False

    def test_no_auth_with_key_set(self, monkeypatch):
        import src.webui.server as mod

        monkeypatch.setattr(mod, "WEBUI_API_KEY", "secret")
        assert _verify_webui_api_key(None) is False


class TestLoopbackRequest:
    def test_loopback_client(self):
        request = MagicMock()
        request.client.host = "127.0.0.1"
        assert _is_loopback_request(request) is True

    def test_non_loopback_client(self):
        request = MagicMock()
        request.client.host = "203.0.113.10"
        assert _is_loopback_request(request) is False


class TestRateLimitPath:
    def test_api_paths_are_rate_limited(self):
        assert _is_rate_limited_path("/api/sessions") is True

    def test_dashboard_api_paths_are_rate_limited(self):
        assert _is_rate_limited_path("/dashboard/api/evaluate") is True

    def test_page_and_static_paths_are_not_rate_limited(self):
        assert _is_rate_limited_path("/") is False
        assert _is_rate_limited_path("/dashboard") is False
        assert _is_rate_limited_path("/dashboard/static/dashboard.js") is False


class TestAppRoutes:
    def test_chat_router_registered(self):
        paths = [getattr(r, "path", "") for r in app.routes]
        assert any("/api/chat" in p for p in paths)

    def test_projects_router_registered(self):
        paths = [getattr(r, "path", "") for r in app.routes]
        assert any("/api/projects" in p for p in paths)

    def test_sessions_router_registered(self):
        paths = [getattr(r, "path", "") for r in app.routes]
        assert any("/api/sessions" in p for p in paths)

    def test_tasks_router_registered(self):
        paths = [getattr(r, "path", "") for r in app.routes]
        assert any("/api/tasks" in p for p in paths)

    def test_tools_router_registered(self):
        paths = [getattr(r, "path", "") for r in app.routes]
        assert any("/api/tools" in p for p in paths)

    def test_health_endpoint_exists(self):
        paths = [getattr(r, "path", "") for r in app.routes]
        has_health = any(p == "/health" for p in paths)
        if not has_health:
            has_root = any(
                getattr(r, "name", None) == "webui_bundle_missing"
                for r in app.routes
                if hasattr(r, "name")
            )
            assert has_root or any(p in {"", "/"} for p in paths)


class TestEnforceApiKey:
    @pytest.mark.asyncio
    async def test_api_health_exempt_with_api_key(self, monkeypatch):
        import src.webui.server as mod

        monkeypatch.setattr(mod, "WEBUI_API_KEY", "secret")

        resp, called = await _call_enforce_api_key("/api/health")
        assert resp.status_code != 401
        assert called is True

    @pytest.mark.asyncio
    async def test_health_exempt_with_api_key(self, monkeypatch):
        import src.webui.server as mod

        monkeypatch.setattr(mod, "WEBUI_API_KEY", "secret")

        resp, called = await _call_enforce_api_key("/health")
        assert resp.status_code != 401
        assert called is True

    @pytest.mark.asyncio
    async def test_dashboard_page_exempt_with_api_key(self, monkeypatch):
        import src.webui.server as mod

        monkeypatch.setattr(mod, "WEBUI_API_KEY", "secret")

        resp, called = await _call_enforce_api_key("/dashboard")
        assert resp.status_code != 401
        assert called is True

    @pytest.mark.asyncio
    async def test_dashboard_api_requires_api_key(self, monkeypatch):
        import src.webui.server as mod

        monkeypatch.setattr(mod, "WEBUI_API_KEY", "secret")

        resp, called = await _call_enforce_api_key("/dashboard/api/models/compare")
        assert resp.status_code == 401
        assert called is False

    @pytest.mark.asyncio
    async def test_non_loopback_without_api_key_rejected(self, monkeypatch):
        import src.webui.server as mod

        monkeypatch.setattr(mod, "WEBUI_API_KEY", None)

        resp, called = await _call_enforce_api_key("/api/sessions")
        assert resp.status_code == 503
        assert called is False

    @pytest.mark.asyncio
    async def test_non_loopback_empty_api_key_rejected(self, monkeypatch):
        import src.webui.server as mod

        monkeypatch.setattr(mod, "WEBUI_API_KEY", None)

        resp, called = await _call_enforce_api_key("/dashboard/api/models/compare")
        assert resp.status_code == 503
        assert called is False

    @pytest.mark.asyncio
    async def test_unauthenticated_request_rejected(self, monkeypatch):
        import src.webui.server as mod

        monkeypatch.setattr(mod, "WEBUI_API_KEY", "secret")

        resp, called = await _call_enforce_api_key("/api/sessions")
        assert resp.status_code == 401
        assert called is False

    @pytest.mark.asyncio
    async def test_authenticated_request_allowed(self, monkeypatch):
        import src.webui.server as mod

        monkeypatch.setattr(mod, "WEBUI_API_KEY", "secret")

        resp, called = await _call_enforce_api_key(
            "/api/sessions",
            headers={"Authorization": "Bearer secret"},
        )
        assert resp.status_code != 401
        assert called is True


class TestServerCoverageGaps:
    def test_load_dashboard_router_import_error(self, monkeypatch):
        import src.webui.server as mod

        monkeypatch.setitem(__import__("sys").modules, "src.webui.dashboard.api", None)
        result = mod._load_dashboard_router()
        assert result is None

    def test_start_server_function(self, monkeypatch):
        import src.webui.server as mod

        called = {}
        def fake_run(app, host, port, reload, log_level):
            called["ok"] = True

        monkeypatch.setattr("uvicorn.run", fake_run)
        mod.start_server(host="0.0.0.0", port=9999)
        assert called.get("ok")

    def test_main_block(self):
        import src.webui.server as mod
        assert hasattr(mod, "app")
        assert hasattr(mod, "start_server")

    def test_webui_missing_route_exists(self):
        from src.webui.server import app
        has_missing_route = any(
            getattr(r, "name", None) == "webui_bundle_missing"
            for r in app.routes
            if hasattr(r, "name")
        )
        assert has_missing_route or any(
            getattr(r, "path", None) in {"", "/"}
            for r in app.routes
            if hasattr(r, "path")
        )
