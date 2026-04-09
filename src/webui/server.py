"""
Doraemon Code Web UI Server

Main FastAPI application for the Web UI.
"""

import logging
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from src.core.logger import configure_root_logger
from src.webui.routes import chat, projects, sessions, tasks, tools

# Setup logging
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

app = FastAPI(
    title="Doraemon Code API",
    description="Backend API for Doraemon Code Web UI",
    version="0.8.0",
)

# CORS configuration
origins = [
    "http://localhost:5173",  # Vite dev server
    "http://localhost:8000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response


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

    logger.info(f"Starting Doraemon Code Web UI at http://{host}:{port}")
    uvicorn.run(
        "src.webui.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    start_server(reload=True)
