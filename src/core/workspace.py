"""
Multi-Directory Workspace Support

Enables working with multiple directories simultaneously.

Features:
- Add additional working directories
- Directory-aware file operations
- Cross-directory search
- Workspace configuration
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class WorkspaceDirectory:
    """A directory in the workspace."""

    path: Path
    alias: str | None = None  # Optional short name
    readonly: bool = False
    primary: bool = False  # Primary working directory

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "alias": self.alias,
            "readonly": self.readonly,
            "primary": self.primary,
        }


class WorkspaceManager:
    """
    Manages multiple working directories.

    Usage:
        ws = WorkspaceManager()

        # Add directories
        ws.add_directory("/path/to/project")
        ws.add_directory("/path/to/lib", alias="lib", readonly=True)

        # List directories
        dirs = ws.list_directories()

        # Resolve path
        full_path = ws.resolve_path("lib/src/utils.py")

        # Check if path is in workspace
        if ws.is_in_workspace("/some/path"):
            ...
    """

    def __init__(self, primary_dir: Path | None = None):
        """
        Initialize workspace manager.

        Args:
            primary_dir: Primary working directory (defaults to cwd)
        """
        self._directories: list[WorkspaceDirectory] = []

        # Add primary directory
        primary = primary_dir or Path.cwd()
        self.add_directory(primary, primary=True)

    def add_directory(
        self,
        path: str | Path,
        alias: str | None = None,
        readonly: bool = False,
        primary: bool = False,
    ) -> bool:
        """
        Add a directory to the workspace.

        Args:
            path: Directory path
            alias: Optional alias for the directory
            readonly: Whether directory is read-only
            primary: Whether this is the primary directory

        Returns:
            True if added successfully
        """
        dir_path = Path(path).resolve()

        if not dir_path.exists():
            logger.error("Directory does not exist: %s", dir_path)
            return False

        if not dir_path.is_dir():
            logger.error("Path is not a directory: %s", dir_path)
            return False

        # Check if already added
        for d in self._directories:
            if d.path == dir_path:
                logger.warning("Directory already in workspace: %s", dir_path)
                return False

        # If setting as primary, unset existing primary
        if primary:
            for d in self._directories:
                d.primary = False

        workspace_dir = WorkspaceDirectory(
            path=dir_path,
            alias=alias,
            readonly=readonly,
            primary=primary,
        )

        self._directories.append(workspace_dir)
        logger.info("Added directory to workspace: %s%s", dir_path, f" ({alias})" if alias else "")
        return True

    def remove_directory(self, path_or_alias: str) -> bool:
        """
        Remove a directory from the workspace.

        Args:
            path_or_alias: Directory path or alias

        Returns:
            True if removed
        """
        for i, d in enumerate(self._directories):
            if str(d.path) == path_or_alias or d.alias == path_or_alias:
                if d.primary:
                    logger.error("Cannot remove primary directory")
                    return False
                del self._directories[i]
                logger.info("Removed directory from workspace: %s", d.path)
                return True

        logger.warning("Directory not found in workspace: %s", path_or_alias)
        return False

    def list_directories(self) -> list[WorkspaceDirectory]:
        """List all directories in workspace."""
        return self._directories.copy()

    def get_primary(self) -> Path:
        """Get the primary working directory."""
        for d in self._directories:
            if d.primary:
                return d.path
        # Fallback to first directory
        if self._directories:
            return self._directories[0].path
        return Path.cwd()

    def resolve_path(self, path: str) -> Path | None:
        """
        Resolve a path that may use an alias.

        Args:
            path: Path that may start with an alias

        Returns:
            Resolved absolute path or None if not found

        Examples:
            ws.resolve_path("lib/src/utils.py")  # Uses alias
            ws.resolve_path("./main.py")  # Relative to primary
            ws.resolve_path("/abs/path")  # Absolute path
        """
        path_obj = Path(path)

        # Absolute path
        if path_obj.is_absolute():
            return path_obj if path_obj.exists() else None

        # Check if starts with alias
        parts = path.split("/", 1)
        if parts:
            for d in self._directories:
                if d.alias == parts[0]:
                    if len(parts) > 1:
                        return d.path / parts[1]
                    return d.path

        # Relative to primary
        primary = self.get_primary()
        resolved = primary / path
        if resolved.exists():
            return resolved

        # Search in all directories
        for d in self._directories:
            candidate = d.path / path
            if candidate.exists():
                return candidate

        return None

    def is_in_workspace(self, path: str | Path) -> bool:
        """
        Check if a path is within the workspace.

        Args:
            path: Path to check

        Returns:
            True if path is in any workspace directory
        """
        check_path = Path(path).resolve()

        for d in self._directories:
            try:
                check_path.relative_to(d.path)
                return True
            except ValueError:
                continue

        return False

    def get_directory_for_path(self, path: str | Path) -> WorkspaceDirectory | None:
        """
        Get the workspace directory containing a path.

        Args:
            path: Path to check

        Returns:
            WorkspaceDirectory or None
        """
        check_path = Path(path).resolve()

        for d in self._directories:
            try:
                check_path.relative_to(d.path)
                return d
            except ValueError:
                continue

        return None

    def is_readonly(self, path: str | Path) -> bool:
        """
        Check if a path is in a readonly directory.

        Args:
            path: Path to check

        Returns:
            True if path is in a readonly directory
        """
        ws_dir = self.get_directory_for_path(path)
        return ws_dir.readonly if ws_dir else False

    def get_all_paths(self) -> list[Path]:
        """Get all directory paths."""
        return [d.path for d in self._directories]

    def format_path(self, path: str | Path) -> str:
        """
        Format a path using aliases if available.

        Args:
            path: Path to format

        Returns:
            Formatted path string
        """
        path_obj = Path(path).resolve()

        for d in self._directories:
            try:
                relative = path_obj.relative_to(d.path)
                if d.alias:
                    return f"{d.alias}/{relative}"
                return str(relative)
            except ValueError:
                continue

        return str(path_obj)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "directories": [d.to_dict() for d in self._directories],
            "primary": str(self.get_primary()),
        }

    def get_summary(self) -> str:
        """Get workspace summary string."""
        lines = []
        for d in self._directories:
            prefix = "📁" if d.primary else "  "
            suffix = ""
            if d.readonly:
                suffix += " [readonly]"
            if d.alias:
                suffix += f" ({d.alias})"
            lines.append(f"{prefix} {d.path}{suffix}")
        return "\n".join(lines)
