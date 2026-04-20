import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rich.logging import RichHandler

from .paths import logs_dir

# ================================
# Standard Logging Setup
# ================================


def _resolve_log_level(level: str | None = None) -> str:
    """Resolve the effective log level from args/env/defaults."""
    return (level or os.getenv("AGENT_LOG_LEVEL") or "INFO").upper()


def _build_file_handler(log_path: Path) -> logging.FileHandler:
    """Create a UTF-8 file handler with a consistent formatter."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    return file_handler


def configure_root_logger(level: str | None = None, log_file: str | None = None) -> logging.Logger:
    """Configure root logging so unconfigured modules still emit project-local logs."""
    effective_level = _resolve_log_level(level)
    root = logging.getLogger()

    root.setLevel(getattr(logging, effective_level, logging.INFO))
    root.handlers.clear()

    console_handler = RichHandler(
        rich_tracebacks=True, markup=True, show_time=True, show_path=False
    )
    console_handler.setLevel(getattr(logging, effective_level, logging.INFO))
    root.addHandler(console_handler)

    file_target = Path(log_file or os.getenv("AGENT_LOG_FILE") or (logs_dir() / "agent.log"))
    root.addHandler(_build_file_handler(file_target))
    return root


def setup_logger(name: str, level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    """
    Setup a standard logger with console and optional file output.

    Args:
        name: Logger name (usually __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output

    Returns:
        Configured logger instance
    """
    effective_level = _resolve_log_level(level)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, effective_level, logging.INFO))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    logger.propagate = False

    # Console handler with Rich formatting
    console_handler = RichHandler(
        rich_tracebacks=True, markup=True, show_time=True, show_path=False
    )
    console_handler.setLevel(getattr(logging, effective_level, logging.INFO))
    logger.addHandler(console_handler)

    # File handler (if specified)
    file_target = log_file or os.getenv("AGENT_LOG_FILE")
    if file_target:
        logger.addHandler(_build_file_handler(Path(file_target)))

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with project-local default configuration."""
    # Check if already configured
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    # Auto-configure
    log_dir = logs_dir()
    log_file = os.getenv("AGENT_LOG_FILE") or str(log_dir / f"{name.split('.')[-1]}.log")

    return setup_logger(name, log_file=log_file)


# ================================
# Trace Logger (for debugging)
# ================================


@dataclass
class TraceEvent:
    type: str  # "tool_call", "tool_result", "model_think", "user_input"
    name: str  # tool name or event name
    data: Any
    timestamp: float
    duration_ms: float | None = None


class TraceLogger:
    """Lightweight trace logger for tool execution tracking."""

    def __init__(self):
        self.events: list[TraceEvent] = []
        self.logger = get_logger(__name__)

    def log(self, type: str, name: str, data: Any, duration_ms: float = 0):
        event = TraceEvent(
            type=type, name=name, data=data, timestamp=time.time(), duration_ms=duration_ms
        )
        self.events.append(event)

        # Also log to standard logger
        payload = json.dumps(data, ensure_ascii=False, default=str)
        self.logger.debug("Trace: %s - %s (%sms) | data=%s", type, name, duration_ms, payload)

    def export(self) -> list[dict]:
        return [asdict(e) for e in self.events]
