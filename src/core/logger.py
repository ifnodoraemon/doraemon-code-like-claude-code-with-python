import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rich.logging import RichHandler

# ================================
# Standard Logging Setup
# ================================


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
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler with Rich formatting
    console_handler = RichHandler(
        rich_tracebacks=True, markup=True, show_time=True, show_path=False
    )
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # File gets all levels
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with default Polymath configuration."""
    # Check if already configured
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    # Auto-configure
    log_dir = Path.home() / ".doraemon" / "logs"
    log_file = log_dir / f"{name.split('.')[-1]}.log"

    return setup_logger(name, level="INFO", log_file=str(log_file))


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
        self.logger.debug(f"Trace: {type} - {name} ({duration_ms}ms)", extra={"data": data})

    def export(self) -> list[dict]:
        return [asdict(e) for e in self.events]
