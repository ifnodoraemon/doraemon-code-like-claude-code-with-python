"""
Doraemon Code Web UI Module

Provides a modern web interface for the AI coding assistant.
"""

__all__ = ["app", "start_server"]


def __getattr__(name: str):
    if name in {"app", "start_server"}:
        from src.webui.server import app, start_server

        return {"app": app, "start_server": start_server}[name]
    raise AttributeError(name)
