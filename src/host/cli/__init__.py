"""
Doraemon CLI - Main Module Entry Point

This package provides the CLI interface for Doraemon.
"""

from typing import Any


def entry_point() -> None:
    from src.host.cli.main import entry_point as _entry_point

    _entry_point()


def __getattr__(name: str) -> Any:
    if name == "app":
        from src.host.cli.main import app

        return app
    raise AttributeError(name)

__all__ = ["app", "entry_point"]
