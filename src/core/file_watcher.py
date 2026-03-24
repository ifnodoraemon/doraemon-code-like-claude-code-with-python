"""
File Watcher

File system monitoring with watchdog.

Features:
- Watch files and directories
- Event callbacks
- Debouncing
- Pattern filtering
- Cache invalidation integration
"""

import asyncio
import fnmatch
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Try to import watchdog
try:
    from watchdog.events import (
        DirCreatedEvent,
        DirDeletedEvent,
        DirMovedEvent,
        FileSystemEventHandler,
    )
    from watchdog.observers import Observer

    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    logger.warning("watchdog not installed. File watching disabled.")


class FileEventType(Enum):
    """File event types."""

    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"


@dataclass
class FileEvent:
    """A file system event."""

    type: FileEventType
    path: Path
    is_directory: bool
    src_path: Path | None = None  # For moves
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "path": str(self.path),
            "is_directory": self.is_directory,
            "src_path": str(self.src_path) if self.src_path else None,
            "timestamp": self.timestamp,
        }


@dataclass
class WatchConfig:
    """Watch configuration."""

    path: Path
    recursive: bool = True
    patterns: list[str] = field(default_factory=lambda: ["*"])  # Glob patterns
    ignore_patterns: list[str] = field(
        default_factory=lambda: [
            "*.pyc",
            "__pycache__",
            ".git",
            ".venv",
            "node_modules",
            "*.swp",
            "*.swo",
            "*~",
        ]
    )
    debounce_ms: int = 100  # Debounce rapid events


class DebouncedHandler:
    """Debounces rapid file events."""

    def __init__(self, callback: Callable[[FileEvent], None], delay_ms: int = 100):
        self._callback = callback
        self._delay = delay_ms / 1000
        self._pending: dict[str, FileEvent] = {}
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def handle(self, event: FileEvent):
        """Handle an event with debouncing."""
        with self._lock:
            key = str(event.path)
            self._pending[key] = event

            if self._timer:
                self._timer.cancel()

            self._timer = threading.Timer(self._delay, self._flush)
            self._timer.start()

    def _flush(self):
        """Flush pending events."""
        with self._lock:
            for event in self._pending.values():
                try:
                    self._callback(event)
                except Exception as e:
                    logger.error(f"Error in file event callback: {e}")
            self._pending.clear()


if WATCHDOG_AVAILABLE:

    class WatchdogHandler(FileSystemEventHandler):
        """Watchdog event handler."""

        def __init__(
            self,
            callback: Callable[[FileEvent], None],
            patterns: list[str],
            ignore_patterns: list[str],
        ):
            super().__init__()
            self._callback = callback
            self._patterns = patterns
            self._ignore_patterns = ignore_patterns

        def _should_process(self, path: str) -> bool:
            """Check if path should be processed."""
            name = Path(path).name

            # Check ignore patterns
            for pattern in self._ignore_patterns:
                if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(path, pattern):
                    return False

            # Check include patterns
            for pattern in self._patterns:
                if fnmatch.fnmatch(name, pattern) or pattern == "*":
                    return True

            return False

        def _create_event(
            self, event_type: FileEventType, path: str, is_dir: bool
        ) -> FileEvent | None:
            """Create a FileEvent if path should be processed."""
            if not self._should_process(path):
                return None

            return FileEvent(
                type=event_type,
                path=Path(path),
                is_directory=is_dir,
            )

        def on_created(self, event):
            is_dir = isinstance(event, DirCreatedEvent)
            file_event = self._create_event(FileEventType.CREATED, event.src_path, is_dir)
            if file_event:
                self._callback(file_event)

        def on_modified(self, event):
            if isinstance(event, DirCreatedEvent | DirDeletedEvent | DirMovedEvent):
                return
            file_event = self._create_event(FileEventType.MODIFIED, event.src_path, False)
            if file_event:
                self._callback(file_event)

        def on_deleted(self, event):
            is_dir = isinstance(event, DirDeletedEvent)
            file_event = self._create_event(FileEventType.DELETED, event.src_path, is_dir)
            if file_event:
                self._callback(file_event)

        def on_moved(self, event):
            is_dir = isinstance(event, DirMovedEvent)
            if self._should_process(event.dest_path):
                file_event = FileEvent(
                    type=FileEventType.MOVED,
                    path=Path(event.dest_path),
                    is_directory=is_dir,
                    src_path=Path(event.src_path),
                )
                self._callback(file_event)


class FileWatcher:
    """
    Watches file system for changes.

    Usage:
        watcher = FileWatcher()

        # Add callback
        watcher.on_change(lambda e: print(f"Changed: {e.path}"))

        # Start watching
        watcher.watch("/path/to/project")

        # Stop watching
        watcher.stop()
    """

    def __init__(self, debounce_ms: int = 100):
        """
        Initialize file watcher.

        Args:
            debounce_ms: Debounce delay in milliseconds
        """
        self._observers: dict[str, Any] = {}
        self._callbacks: list[Callable[[FileEvent], None]] = []
        self._debouncer = DebouncedHandler(self._dispatch_event, debounce_ms)
        self._running = False

    def is_available(self) -> bool:
        """Check if file watching is available."""
        return WATCHDOG_AVAILABLE

    def on_change(self, callback: Callable[[FileEvent], None]):
        """
        Register a callback for file changes.

        Args:
            callback: Function to call on change
        """
        self._callbacks.append(callback)

    def off_change(self, callback: Callable[[FileEvent], None]):
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def watch(self, config: WatchConfig | Path | str) -> bool:
        """
        Start watching a path.

        Args:
            config: Watch configuration or path

        Returns:
            True if watching started
        """
        if not WATCHDOG_AVAILABLE:
            logger.warning("watchdog not available")
            return False

        # Convert to config
        if isinstance(config, str | Path):
            config = WatchConfig(path=Path(config))

        path = config.path
        if not path.exists():
            logger.error(f"Path does not exist: {path}")
            return False

        key = str(path)
        if key in self._observers:
            logger.warning(f"Already watching: {path}")
            return True

        try:
            handler = WatchdogHandler(
                callback=self._debouncer.handle,
                patterns=config.patterns,
                ignore_patterns=config.ignore_patterns,
            )

            observer = Observer()
            observer.schedule(handler, str(path), recursive=config.recursive)
            observer.start()

            self._observers[key] = observer
            self._running = True
            logger.info(f"Watching: {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to watch {path}: {e}")
            return False

    def unwatch(self, path: Path | str):
        """Stop watching a path."""
        key = str(path)
        if key in self._observers:
            self._observers[key].stop()
            self._observers[key].join()
            del self._observers[key]
            logger.info(f"Stopped watching: {path}")

    def stop(self):
        """Stop all watchers."""
        for key in list(self._observers.keys()):
            self.unwatch(key)
        self._running = False

    def _dispatch_event(self, event: FileEvent):
        """Dispatch event to callbacks."""
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in file change callback: {e}")

    def get_watched_paths(self) -> list[str]:
        """Get list of watched paths."""
        return list(self._observers.keys())

    def is_running(self) -> bool:
        """Check if watcher is running."""
        return self._running and len(self._observers) > 0


class AsyncFileWatcher:
    """
    Async wrapper for FileWatcher.

    Usage:
        async with AsyncFileWatcher() as watcher:
            async for event in watcher.watch("/path"):
                print(f"Changed: {event.path}")
    """

    def __init__(self):
        self._watcher = FileWatcher()
        self._queue: asyncio.Queue[FileEvent] = asyncio.Queue()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._watcher.stop()

    def _on_event(self, event: FileEvent):
        """Put event in async queue."""
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            pass

    async def watch(self, path: Path | str):
        """
        Watch a path and yield events.

        Args:
            path: Path to watch

        Yields:
            FileEvent objects
        """
        self._watcher.on_change(self._on_event)
        self._watcher.watch(path)

        try:
            while self._watcher.is_running():
                try:
                    event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                    yield event
                except asyncio.TimeoutError:
                    continue
        finally:
            self._watcher.stop()


# Global watcher instance
_file_watcher: FileWatcher | None = None


def get_file_watcher() -> FileWatcher:
    """Get the global file watcher."""
    global _file_watcher
    if _file_watcher is None:
        _file_watcher = FileWatcher()
    return _file_watcher
