"""
Service Configuration and Dependency Injection Setup

This module configures all services for the Polymath application using
the DI container. It provides a centralized place to register and
manage application services.
"""

import os
from pathlib import Path
from typing import Any

from rich.console import Console

from src.core.configuration import Configuration, ConfigurationBuilder, configure
from src.core.container import ServiceCollection, configure_services
from src.core.events import EventBus, get_event_bus
from src.core.telemetry import (
    ConsoleLogSink,
    FileLogSink,
    LogLevel,
    StructuredLogger,
    Tracer,
    get_logger,
    get_tracer,
)

# ========================================
# Service Interfaces (Protocols)
# ========================================


class IConsole:
    """Console output interface"""

    def print(self, *args, **kwargs) -> None:
        raise NotImplementedError


class IConfiguration:
    """Configuration interface"""

    def get(self, path: str, default: Any = None) -> Any:
        raise NotImplementedError


# ========================================
# Service Implementations
# ========================================


class RichConsole(IConsole):
    """Rich console implementation"""

    def __init__(self):
        self._console = Console()

    def print(self, *args, **kwargs) -> None:
        self._console.print(*args, **kwargs)

    @property
    def raw(self) -> Console:
        """Access the underlying Rich Console"""
        return self._console


# ========================================
# Service Configuration
# ========================================


def _create_configuration() -> Configuration:
    """Create application configuration from multiple sources."""

    def setup_config(builder: ConfigurationBuilder):
        # 1. Default values
        builder.add_defaults(
            {
                "app": {
                    "name": "Polymath",
                    "version": "0.4.0",
                },
                "logging": {
                    "level": "INFO",
                    "structured": False,
                },
                "security": {
                    "sensitive_tools": [
                        "execute_python",
                        "write_file",
                        "save_note",
                        "move_file",
                        "delete_file",
                    ],
                },
            }
        )

        # 2. JSON config file (optional)
        builder.add_json_file(".polymath/config.json", optional=True)

        # 3. Environment variables with POLYMATH_ prefix
        builder.add_environment_variables(prefix="POLYMATH_")

    return configure(setup_config)


def _create_logger(config: Configuration) -> StructuredLogger:
    """Create structured logger based on configuration."""
    log_level_str = config.get_str("logging.level", "INFO").upper()
    log_level = getattr(LogLevel, log_level_str, LogLevel.INFO)

    return get_logger("polymath", min_level=log_level)


def _create_structured_logger() -> StructuredLogger:
    """Create a structured logger with console and file sinks."""
    log_level_str = os.getenv("POLYMATH_LOG_LEVEL", "INFO").upper()
    log_level = getattr(LogLevel, log_level_str, LogLevel.INFO)

    # Determine if structured logging (JSON file) is enabled
    enable_file_logging = os.getenv("POLYMATH_LOG_FILE", "").lower() in ("true", "1", "yes")

    sinks = [ConsoleLogSink(use_colors=True)]

    if enable_file_logging:
        # Create log directory
        log_dir = Path.home() / ".polymath" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "polymath.log"

        sinks.append(FileLogSink(log_file, max_size_mb=10, backup_count=5))

    return StructuredLogger("polymath", sinks=sinks, min_level=log_level)


def setup_services(services: ServiceCollection) -> None:
    """
    Configure all application services.

    This is the main service configuration function that registers
    all services with their appropriate lifetimes.

    Args:
        services: The service collection to configure
    """
    # Configuration (singleton - one instance for entire app)
    services.add_singleton(Configuration, _create_configuration)

    # Console (singleton - shared console instance)
    services.add_singleton(IConsole, RichConsole)
    services.add_singleton(Console, lambda: Console())

    # Event Bus (singleton - central event hub)
    services.add_singleton(EventBus, get_event_bus)

    # Tracer (singleton - distributed tracing)
    services.add_singleton(Tracer, get_tracer)

    # Logger (singleton - shared structured logger)
    services.add_singleton(StructuredLogger, _create_structured_logger)


# ========================================
# Global Service Access
# ========================================

_service_provider = None


def initialize_services():
    """
    Initialize the global service provider.

    This should be called once at application startup.
    """
    global _service_provider
    _service_provider = configure_services(setup_services)
    return _service_provider


def get_service_provider():
    """Get the global service provider."""
    global _service_provider
    if _service_provider is None:
        _service_provider = initialize_services()
    return _service_provider


def resolve(service_type: type):
    """
    Resolve a service from the global container.

    Args:
        service_type: The service type to resolve

    Returns:
        The resolved service instance

    Example:
        console = resolve(Console)
        console.print("Hello!")
    """
    return get_service_provider().get_service(service_type)


def require(service_type: type):
    """
    Resolve a required service from the global container.

    Args:
        service_type: The service type to resolve

    Returns:
        The resolved service instance

    Raises:
        ValueError: If the service is not registered

    Example:
        config = require(Configuration)
        debug = config.get_bool("debug")
    """
    return get_service_provider().get_required_service(service_type)
