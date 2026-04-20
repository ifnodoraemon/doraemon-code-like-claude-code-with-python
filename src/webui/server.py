"""
Doraemon Code Web UI Server

Main FastAPI application for the Web UI.
"""

import logging
import os
import secrets
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from src.core.logger import configure_root_logger
from src.webui.routes import chat, projects, sessions, tasks, tools

configure_root_logger()
logger = logging.getLogger(__name__)


def _extract_asset_paths(index_html: str) -> list[str]:
    asset_paths: list[str] = []
    for marker in ('src="/assets/', 'href="/assets/'):
        start = 0
        while True:
            offset = index_html.find(marker, start)
            if offset == -1:
                break
            asset_start = offset + len(marker)
            asset_end = index_html.find('"', asset_start)
            asset_paths.append(index_html[asset_start:asset_end])
            start = asset_end + 1
    return asset_paths


def resolve_static_bundle(webui_dir: Path) -> Path | None:
    static_dir = webui_dir / "static"
    index_path = static_dir / "index.html"
    if not index_path.exists():
        logger.warning(
            "Web UI bundle missing: %s not found. Run `cd webui && npm ci && npm run build`.",
            index_path,
        )
        return None

    index_html = index_path.read_text(encoding="utf-8")
    missing_assets = [
        asset_path
        for asset_path in _extract_asset_paths(index_html)
        if not (static_dir / "assets" / asset_path).exists()
    ]
    if missing_assets:
        logger.warning(
            "Web UI bundle incomplete: missing assets %s. Run `cd webui && npm run build`.",
            ", ".join(missing_assets),
        )
        return None

    return static_dir


def _load_dashboard_router():
    try:
        from src.webui.dashboard.api import router as dashboard_router
    except (ImportError, AssertionError) as exc:
        logger.warning(
            "Dashboard router disabled: %s. Install optional template dependencies to enable /dashboard.",
            exc,
        )
        return None
    return dashboard_router


# ---------------------------------------------------------------------------
# API key authentication
# ---------------------------------------------------------------------------
WEBUI_API_KEY = os.getenv("AGENT_WEBUI_API_KEY")
_AUTH_EXEMPT_PATHS = {"/", "/health", "/api/health"}
_AUTH_EXEMPT_PREFIXES = ("/assets/", "/dashboard/static/")


def _verify_webui_api_key(authorization: str | None) -> bool:
    if not WEBUI_API_KEY:
        return True
    if not authorization:
        return False
    key = authorization[7:] if authorization.startswith("Bearer ") else authorization
    return secrets.compare_digest(key, WEBUI_API_KEY)


app = FastAPI(
    title="Doraemon Code API",
    description="Backend API for Doraemon Code Web UI",
    version="0.8.0",
)

# CORS configuration — configurable via environment
_cors_origins = os.getenv(
    "AGENT_WEBUI_CORS_ORIGINS",
    "http://localhost:5173,http://localhost:8000,http://127.0.0.1:5173,http://127.0.0.1:8000",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response


@app.middleware("http")
async def enforce_api_key(request: Request, call_next):
    """Reject requests without a valid API key when AGENT_WEBUI_API_KEY is set."""
    if WEBUI_API_KEY is None:
        return await call_next(request)

    path = request.url.path
    if path in _AUTH_EXEMPT_PATHS or any(path.startswith(p) for p in _AUTH_EXEMPT_PREFIXES):
        return await call_next(request)

    authorization = request.headers.get("Authorization")
    if _verify_webui_api_key(authorization):
        return await call_next(request)

    return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})


# Include routers
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(projects.router, prefix="/api/projects", tags=["projects"])
app.include_router(sessions.router, prefix="/api/sessions", tags=["sessions"])
app.include_router(tasks.router, prefix="/api/tasks", tags=["tasks"])
app.include_router(tools.router, prefix="/api/tools", tags=["tools"])
dashboard_router = _load_dashboard_router()
if dashboard_router is not None:
    app.include_router(dashboard_router, prefix="/dashboard", tags=["dashboard"])

# Mount dashboard static files
dashboard_static_dir = Path(__file__).parent / "dashboard" / "static"
if dashboard_static_dir.exists():
    app.mount(
        "/dashboard/static",
        StaticFiles(directory=str(dashboard_static_dir)),
        name="dashboard_static",
    )

# Mount static files (frontend)
static_dir = resolve_static_bundle(Path(__file__).parent)
if static_dir is not None:
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
else:

    @app.get("/", include_in_schema=False)
    async def webui_bundle_missing():
        return PlainTextResponse(
            "Web UI bundle is missing or incomplete. Run `cd webui && npm ci && npm run build`.",
            status_code=503,
        )


def start_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Start the Web UI server."""
    import uvicorn

    logger.info("Starting Doraemon Code Web UI at http://%s:%s", host, port)
    if WEBUI_API_KEY:
        logger.info("WebUI API key authentication enabled")
    uvicorn.run(
        "src.webui.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    start_server(reload=True)
