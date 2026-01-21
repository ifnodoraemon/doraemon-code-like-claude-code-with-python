"""
Structured Logging and Tracing System

Provides structured logging with context, correlation IDs, and distributed tracing support.
Inspired by OpenTelemetry and Serilog.

Status: AVAILABLE FOR EXTENSION
    This module provides advanced logging and tracing capabilities beyond the basic
    logger in logger.py. It's designed for:
    - Production deployments requiring structured JSON logs
    - Distributed tracing across multiple MCP servers
    - Correlation of logs across request boundaries
    - Performance monitoring and debugging

Note:
    The basic logger (src.core.logger) is used by default. This module can be
    integrated when more advanced telemetry features are needed.

Example Usage:
    from src.core.telemetry import StructuredLogger, Tracer

    logger = StructuredLogger("MyService")
    tracer = Tracer()

    with tracer.start_span("process_request") as span:
        logger.info("Processing started", request_id="123")
        span.tags["status"] = "success"
"""

import json
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class LogLevel(Enum):
    """Log levels"""

    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


@dataclass
class LogContext:
    """Contextual information for log entries"""

    correlation_id: str
    user_id: str | None = None
    session_id: str | None = None
    operation: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class LogEvent:
    """Structured log event"""

    timestamp: datetime
    level: LogLevel
    message: str
    logger_name: str
    context: LogContext
    exception: Exception | None = None
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        data = {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.name,
            "message": self.message,
            "logger": self.logger_name,
            "correlation_id": self.context.correlation_id,
        }

        if self.context.user_id:
            data["user_id"] = self.context.user_id
        if self.context.session_id:
            data["session_id"] = self.context.session_id
        if self.context.operation:
            data["operation"] = self.context.operation

        # Merge context properties
        if self.context.properties:
            data["context"] = self.context.properties

        # Merge event properties
        if self.properties:
            data["properties"] = self.properties

        # Add exception if present
        if self.exception:
            data["exception"] = {
                "type": type(self.exception).__name__,
                "message": str(self.exception),
            }

        return data


class LogSink:
    """Base class for log sinks"""

    def emit(self, event: LogEvent):
        """Emit log event"""
        raise NotImplementedError


class ConsoleLogSink(LogSink):
    """Sink that writes to console"""

    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors
        self.colors = {
            LogLevel.TRACE: "\033[37m",  # White
            LogLevel.DEBUG: "\033[36m",  # Cyan
            LogLevel.INFO: "\033[32m",  # Green
            LogLevel.WARNING: "\033[33m",  # Yellow
            LogLevel.ERROR: "\033[31m",  # Red
            LogLevel.CRITICAL: "\033[35m",  # Magenta
        }
        self.reset = "\033[0m"

    def emit(self, event: LogEvent):
        """Emit to console"""
        color = self.colors.get(event.level, "") if self.use_colors else ""
        reset = self.reset if self.use_colors else ""

        # Format: [TIMESTAMP] LEVEL: Message (correlation_id)
        output = (
            f"{color}[{event.timestamp.strftime('%H:%M:%S')}] {event.level.name}: {event.message}"
        )

        if event.properties:
            output += f" | {json.dumps(event.properties)}"

        output += f" (cid:{event.context.correlation_id[:8]}){reset}"

        print(output)


class FileLogSink(LogSink):
    """Sink that writes JSON logs to file"""

    def __init__(self, file_path: Path, max_size_mb: int = 10, backup_count: int = 5):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size_mb * 1024 * 1024
        self.backup_count = backup_count
        self._lock = threading.Lock()

    def emit(self, event: LogEvent):
        """Emit to file"""
        with self._lock:
            # Check file size and rotate if needed
            if self.file_path.exists() and self.file_path.stat().st_size > self.max_size:
                self._rotate()

            # Write log entry as JSON
            with open(self.file_path, "a", encoding="utf-8") as f:
                json.dump(event.to_dict(), f, ensure_ascii=False)
                f.write("\n")

    def _rotate(self):
        """Rotate log files"""
        # Delete oldest backup
        oldest = self.file_path.with_suffix(f".{self.backup_count}.log")
        if oldest.exists():
            oldest.unlink()

        # Rotate existing backups
        for i in range(self.backup_count - 1, 0, -1):
            old_file = self.file_path.with_suffix(f".{i}.log")
            new_file = self.file_path.with_suffix(f".{i + 1}.log")
            if old_file.exists():
                old_file.rename(new_file)

        # Rename current file to .1.log
        if self.file_path.exists():
            self.file_path.rename(self.file_path.with_suffix(".1.log"))


class StructuredLogger:
    """
    Structured logger with rich contextual information.

    Example:
        logger = StructuredLogger("MyApp")
        logger.info("User logged in", user_id="123", action="login")

        with logger.operation("process_payment"):
            logger.debug("Processing payment", amount=100)
    """

    def __init__(
        self, name: str, sinks: list[LogSink] | None = None, min_level: LogLevel = LogLevel.DEBUG
    ):
        self.name = name
        self.sinks = sinks or [ConsoleLogSink()]
        self.min_level = min_level
        self._context_stack: list[LogContext] = []
        self._local = threading.local()

    def _get_current_context(self) -> LogContext:
        """Get current log context"""
        if not hasattr(self._local, "context"):
            self._local.context = LogContext(correlation_id=str(uuid.uuid4()))
        return self._local.context

    def _set_context(self, context: LogContext):
        """Set current log context"""
        self._local.context = context

    def log(self, level: LogLevel, message: str, exception: Exception | None = None, **properties):
        """Log a message"""
        if level.value < self.min_level.value:
            return

        event = LogEvent(
            timestamp=datetime.now(),
            level=level,
            message=message,
            logger_name=self.name,
            context=self._get_current_context(),
            exception=exception,
            properties=properties,
        )

        for sink in self.sinks:
            try:
                sink.emit(event)
            except Exception as e:
                # Don't let sink errors break the application
                print(f"Error in log sink: {e}")

    def trace(self, message: str, **properties):
        """Log trace message"""
        self.log(LogLevel.TRACE, message, **properties)

    def debug(self, message: str, **properties):
        """Log debug message"""
        self.log(LogLevel.DEBUG, message, **properties)

    def info(self, message: str, **properties):
        """Log info message"""
        self.log(LogLevel.INFO, message, **properties)

    def warning(self, message: str, **properties):
        """Log warning message"""
        self.log(LogLevel.WARNING, message, **properties)

    def error(self, message: str, exception: Exception | None = None, **properties):
        """Log error message"""
        self.log(LogLevel.ERROR, message, exception=exception, **properties)

    def critical(self, message: str, exception: Exception | None = None, **properties):
        """Log critical message"""
        self.log(LogLevel.CRITICAL, message, exception=exception, **properties)

    @contextmanager
    def operation(self, name: str, **properties):
        """
        Create an operation scope for grouping related logs.

        Example:
            with logger.operation("database_query", table="users"):
                logger.debug("Executing query")
                # ... do work ...
        """
        context = self._get_current_context()
        old_operation = context.operation
        old_properties = context.properties.copy()

        # Set operation context
        context.operation = name
        context.properties.update(properties)

        start_time = time.time()
        self.debug(f"Starting operation: {name}", **properties)

        try:
            yield
            duration = time.time() - start_time
            self.debug(
                f"Completed operation: {name}", duration_ms=int(duration * 1000), **properties
            )
        except Exception as e:
            duration = time.time() - start_time
            self.error(
                f"Operation failed: {name}",
                exception=e,
                duration_ms=int(duration * 1000),
                **properties,
            )
            raise
        finally:
            # Restore context
            context.operation = old_operation
            context.properties = old_properties

    @contextmanager
    def correlation(self, correlation_id: str):
        """
        Set correlation ID for this scope.

        Example:
            with logger.correlation("req-12345"):
                logger.info("Processing request")
        """
        old_context = self._get_current_context()
        new_context = LogContext(
            correlation_id=correlation_id,
            user_id=old_context.user_id,
            session_id=old_context.session_id,
            operation=old_context.operation,
            properties=old_context.properties.copy(),
        )

        self._set_context(new_context)

        try:
            yield
        finally:
            self._set_context(old_context)


@dataclass
class Span:
    """Distributed tracing span"""

    span_id: str
    trace_id: str
    parent_id: str | None
    operation: str
    start_time: float
    end_time: float | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    logs: list[dict[str, Any]] = field(default_factory=list)

    def finish(self):
        """Mark span as finished"""
        self.end_time = time.time()

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds"""
        if self.end_time is None:
            return 0
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "operation": self.operation,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "tags": self.tags,
            "logs": self.logs,
        }


class Tracer:
    """Simple distributed tracer"""

    def __init__(self):
        self._active_spans: dict[str, Span] = {}
        self._local = threading.local()

    @contextmanager
    def start_span(self, operation: str, **tags):
        """
        Start a new tracing span.

        Example:
            with tracer.start_span("database_query", table="users") as span:
                # ... do work ...
                span.tags["rows"] = 10
        """
        # Generate IDs
        span_id = str(uuid.uuid4())[:8]
        trace_id = getattr(self._local, "trace_id", str(uuid.uuid4()))
        parent_id = getattr(self._local, "current_span_id", None)

        # Create span
        span = Span(
            span_id=span_id,
            trace_id=trace_id,
            parent_id=parent_id,
            operation=operation,
            start_time=time.time(),
            tags=tags,
        )

        # Set as current span
        old_span_id = getattr(self._local, "current_span_id", None)
        self._local.trace_id = trace_id
        self._local.current_span_id = span_id

        try:
            yield span
        finally:
            span.finish()
            self._local.current_span_id = old_span_id


# Global instances
_loggers: dict[str, StructuredLogger] = {}
_tracer = Tracer()


def get_logger(name: str, **kwargs) -> StructuredLogger:
    """Get or create a logger"""
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name, **kwargs)
    return _loggers[name]


def get_tracer() -> Tracer:
    """Get the global tracer"""
    return _tracer
