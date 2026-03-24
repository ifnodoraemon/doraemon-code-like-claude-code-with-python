"""
Configuration Hot Reload

Automatic configuration reloading on file changes.

Features:
- Watch config files
- Automatic reload
- Change callbacks
- Validation before apply
"""

import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .file_watcher import FileEvent, FileEventType, FileWatcher

logger = logging.getLogger(__name__)


@dataclass
class ConfigFile:
    """A watched configuration file."""

    path: Path
    parser: Callable[[str], dict]  # Function to parse file content
    validator: Callable[[dict], bool] | None = None  # Optional validator
    on_change: Callable[[dict], None] | None = None  # Change callback
    last_modified: float = 0
    last_value: dict = field(default_factory=dict)


class HotReloadManager:
    """
    Manages hot reloading of configuration files.

    Usage:
        manager = HotReloadManager()

        # Watch a JSON config
        manager.watch_json(
            Path(".agent/config.json"),
            on_change=lambda cfg: print(f"Config changed: {cfg}")
        )

        # Watch custom format
        manager.watch(
            path=Path(".env"),
            parser=parse_env,
            on_change=update_env
        )

        # Start watching
        manager.start()

        # Get current config
        config = manager.get_config(".agent/config.json")

        # Stop watching
        manager.stop()
    """

    def __init__(self):
        """Initialize hot reload manager."""
        self._configs: dict[str, ConfigFile] = {}
        self._watcher = FileWatcher(debounce_ms=200)
        self._global_callbacks: list[Callable[[str, dict], None]] = []
        self._running = False
        self._lock = threading.Lock()

    def watch(
        self,
        path: Path,
        parser: Callable[[str], dict],
        validator: Callable[[dict], bool] | None = None,
        on_change: Callable[[dict], None] | None = None,
    ) -> bool:
        """
        Watch a configuration file.

        Args:
            path: Path to config file
            parser: Function to parse file content
            validator: Optional function to validate config
            on_change: Optional callback on change

        Returns:
            True if watching started
        """
        path = path.resolve()
        key = str(path)

        if not path.exists():
            logger.warning(f"Config file does not exist: {path}")
            return False

        # Load initial value
        try:
            content = path.read_text(encoding="utf-8")
            value = parser(content)

            if validator and not validator(value):
                logger.error(f"Invalid config: {path}")
                return False

            config_file = ConfigFile(
                path=path,
                parser=parser,
                validator=validator,
                on_change=on_change,
                last_modified=path.stat().st_mtime,
                last_value=value,
            )

            with self._lock:
                self._configs[key] = config_file

            logger.info(f"Watching config: {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load config {path}: {e}")
            return False

    def watch_json(
        self,
        path: Path,
        validator: Callable[[dict], bool] | None = None,
        on_change: Callable[[dict], None] | None = None,
    ) -> bool:
        """Watch a JSON configuration file."""
        return self.watch(
            path=path,
            parser=json.loads,
            validator=validator,
            on_change=on_change,
        )

    def watch_env(
        self,
        path: Path,
        on_change: Callable[[dict], None] | None = None,
    ) -> bool:
        """Watch an .env file."""

        def parse_env(content: str) -> dict:
            result = {}
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    result[key.strip()] = value.strip().strip('"').strip("'")
            return result

        return self.watch(
            path=path,
            parser=parse_env,
            on_change=on_change,
        )

    def unwatch(self, path: Path | str):
        """Stop watching a config file."""
        key = str(Path(path).resolve())
        with self._lock:
            if key in self._configs:
                del self._configs[key]
                logger.info(f"Stopped watching: {path}")

    def on_change(self, callback: Callable[[str, dict], None]):
        """
        Register a global callback for any config change.

        Args:
            callback: Function(path, new_config)
        """
        self._global_callbacks.append(callback)

    def start(self):
        """Start watching all config files."""
        if self._running:
            return

        # Get unique directories to watch
        dirs_to_watch = set()
        for config in self._configs.values():
            dirs_to_watch.add(config.path.parent)

        # Setup watcher
        self._watcher.on_change(self._handle_change)

        for dir_path in dirs_to_watch:
            self._watcher.watch(dir_path)

        self._running = True
        logger.info(f"Hot reload started for {len(self._configs)} configs")

    def stop(self):
        """Stop watching all config files."""
        self._watcher.stop()
        self._running = False
        logger.info("Hot reload stopped")

    def _handle_change(self, event: FileEvent):
        """Handle a file change event."""
        if event.type not in (FileEventType.MODIFIED, FileEventType.CREATED):
            return

        key = str(event.path.resolve())

        with self._lock:
            if key not in self._configs:
                return

            config = self._configs[key]

        # Check if actually modified (not just accessed)
        try:
            current_mtime = event.path.stat().st_mtime
            if current_mtime <= config.last_modified:
                return
        except FileNotFoundError:
            return

        # Reload config
        self._reload_config(key)

    def _reload_config(self, key: str):
        """Reload a specific config file."""
        with self._lock:
            if key not in self._configs:
                return
            config = self._configs[key]

        try:
            content = config.path.read_text(encoding="utf-8")
            new_value = config.parser(content)

            # Validate
            if config.validator and not config.validator(new_value):
                logger.error(f"Invalid config after reload: {config.path}")
                return

            # Check if actually changed
            if new_value == config.last_value:
                return

            # Update
            with self._lock:
                config.last_value = new_value
                config.last_modified = config.path.stat().st_mtime

            logger.info(f"Config reloaded: {config.path}")

            # Call callbacks
            if config.on_change:
                try:
                    config.on_change(new_value)
                except Exception as e:
                    logger.error(f"Error in config callback: {e}")

            for callback in self._global_callbacks:
                try:
                    callback(str(config.path), new_value)
                except Exception as e:
                    logger.error(f"Error in global callback: {e}")

        except Exception as e:
            logger.error(f"Failed to reload config {config.path}: {e}")

    def get_config(self, path: Path | str) -> dict | None:
        """
        Get current config value.

        Args:
            path: Config file path

        Returns:
            Current config dict or None
        """
        key = str(Path(path).resolve())
        with self._lock:
            if key in self._configs:
                return self._configs[key].last_value.copy()
        return None

    def reload(self, path: Path | str | None = None):
        """
        Force reload of config(s).

        Args:
            path: Specific path or None for all
        """
        if path:
            key = str(Path(path).resolve())
            self._reload_config(key)
        else:
            for key in list(self._configs.keys()):
                self._reload_config(key)

    def get_watched_files(self) -> list[str]:
        """Get list of watched config files."""
        with self._lock:
            return list(self._configs.keys())

    def is_running(self) -> bool:
        """Check if hot reload is running."""
        return self._running


# Global instance
_hot_reload_manager: HotReloadManager | None = None


def get_hot_reload_manager() -> HotReloadManager:
    """Get the global hot reload manager."""
    global _hot_reload_manager
    if _hot_reload_manager is None:
        _hot_reload_manager = HotReloadManager()
    return _hot_reload_manager
