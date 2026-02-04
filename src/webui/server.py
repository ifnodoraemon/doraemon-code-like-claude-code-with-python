"""
Doraemon Code Web UI Server

Main FastAPI application for the Web UI.
"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.webui.dashboard.api import router as dashboard_router
from src.webui.routes import chat, sessions, tools

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(sessions.router, prefix="/api/sessions", tags=["sessions"])
app.include_router(tools.router, prefix="/api/tools", tags=["tools"])
app.include_router(dashboard_router, prefix="/dashboard", tags=["dashboard"])

# Mount dashboard static files
dashboard_static_dir = Path(__file__).parent / "dashboard" / "static"
if dashboard_static_dir.exists():
    app.mount(
        "/dashboard/static",
        StaticFiles(directory=str(dashboard_static_dir)),
        name="dashboard_static"
    )

# Mount static files (frontend)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


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
