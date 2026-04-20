"""
Layered Memory System

Hierarchical memory with organization, project, and user levels.

Features:
- Multi-layer memory hierarchy
- Layer inheritance
- Priority resolution
- Cross-layer search
- Memory persistence
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .paths import layered_memory_dir

logger = logging.getLogger(__name__)


class MemoryLayer(Enum):
    """Memory hierarchy layers (higher = more specific)."""

    GLOBAL = "global"  # System-wide defaults
    ORGANIZATION = "organization"  # Organization/team level
    PROJECT = "project"  # Project level
    USER = "user"  # User preferences
    SESSION = "session"  # Current session only


@dataclass
class MemoryEntry:
    """A memory entry."""

    key: str
    value: Any
    layer: MemoryLayer
    created_at: float
    updated_at: float
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    expires_at: float | None = None

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "layer": self.layer.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "tags": self.tags,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        return cls(
            key=data["key"],
            value=data["value"],
            layer=MemoryLayer(data["layer"]),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            expires_at=data.get("expires_at"),
        )


class LayeredMemory:
    """
    Hierarchical memory system with multiple layers.

    Usage:
        memory = LayeredMemory()

        # Set values at different layers
        memory.set("api_key", "xxx", layer=MemoryLayer.USER)
        memory.set("project_name", "myapp", layer=MemoryLayer.PROJECT)

        # Get with automatic layer resolution
        value = memory.get("api_key")  # Resolves from highest priority layer

        # Get from specific layer
        value = memory.get("setting", layer=MemoryLayer.PROJECT)

        # Search across layers
        results = memory.search("config")

        # List all from a layer
        entries = memory.list_layer(MemoryLayer.USER)
    """

    # Layer priority (higher index = higher priority)
    LAYER_PRIORITY = [
        MemoryLayer.GLOBAL,
        MemoryLayer.ORGANIZATION,
        MemoryLayer.PROJECT,
        MemoryLayer.USER,
        MemoryLayer.SESSION,
    ]

    def __init__(
        self,
        storage_dir: Path | None = None,
        organization_id: str | None = None,
        project_id: str | None = None,
        user_id: str | None = None,
        layer_paths: dict[MemoryLayer, Path] | None = None,
        simple_layers: set[MemoryLayer] | None = None,
    ):
        """
        Initialize layered memory.

        Args:
            storage_dir: Directory for memory persistence
            organization_id: Organization identifier
            project_id: Project identifier
            user_id: User identifier
        """
        self._storage_dir = storage_dir or layered_memory_dir()
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        self._organization_id = organization_id
        self._project_id = project_id
        self._user_id = user_id
        self._layer_paths = layer_paths or {}
        self._simple_layers = simple_layers or set()

        # Memory storage per layer
        self._memory: dict[MemoryLayer, dict[str, MemoryEntry]] = {
            layer: {} for layer in MemoryLayer
        }

        # Load persisted memory
        self._load_all()

    def set(
        self,
        key: str,
        value: Any,
        layer: MemoryLayer = MemoryLayer.SESSION,
        tags: list[str] | None = None,
        metadata: dict | None = None,
        ttl: int | None = None,
    ):
        """
        Set a value in memory.

        Args:
            key: Memory key
            value: Value to store
            layer: Memory layer
            tags: Optional tags
            metadata: Optional metadata
            ttl: Time-to-live in seconds
        """
        now = time.time()
        expires_at = now + ttl if ttl else None

        entry = MemoryEntry(
            key=key,
            value=value,
            layer=layer,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
            tags=tags or [],
            expires_at=expires_at,
        )

        self._memory[layer][key] = entry

        # Persist if not session layer
        if layer != MemoryLayer.SESSION:
            self._save_layer(layer)

    def get(
        self,
        key: str,
        default: Any = None,
        layer: MemoryLayer | None = None,
    ) -> Any:
        """
        Get a value from memory.

        Args:
            key: Memory key
            default: Default value if not found
            layer: Specific layer (None = resolve from all)

        Returns:
            Value or default
        """
        if layer is not None:
            # Get from specific layer
            entry = self._memory[layer].get(key)
            if entry and not entry.is_expired():
                return entry.value
            return default

        # Resolve from highest priority layer down
        for layer in reversed(self.LAYER_PRIORITY):
            entry = self._memory[layer].get(key)
            if entry and not entry.is_expired():
                return entry.value

        return default

    def get_entry(
        self,
        key: str,
        layer: MemoryLayer | None = None,
    ) -> MemoryEntry | None:
        """Get full entry including metadata."""
        if layer is not None:
            entry = self._memory[layer].get(key)
            if entry and not entry.is_expired():
                return entry
            return None

        for layer in reversed(self.LAYER_PRIORITY):
            entry = self._memory[layer].get(key)
            if entry and not entry.is_expired():
                return entry

        return None

    def delete(self, key: str, layer: MemoryLayer | None = None):
        """
        Delete a key from memory.

        Args:
            key: Key to delete
            layer: Specific layer (None = all layers)
        """
        if layer is not None:
            if key in self._memory[layer]:
                del self._memory[layer][key]
                if layer != MemoryLayer.SESSION:
                    self._save_layer(layer)
        else:
            for layer in MemoryLayer:
                if key in self._memory[layer]:
                    del self._memory[layer][key]
                    if layer != MemoryLayer.SESSION:
                        self._save_layer(layer)

    def has(self, key: str, layer: MemoryLayer | None = None) -> bool:
        """Check if key exists."""
        return self.get(key, layer=layer) is not None

    def search(
        self,
        query: str,
        layers: list[MemoryLayer] | None = None,
        tags: list[str] | None = None,
    ) -> list[MemoryEntry]:
        """
        Search memory entries.

        Args:
            query: Search query (matches key or value)
            layers: Layers to search (None = all)
            tags: Filter by tags

        Returns:
            List of matching entries
        """
        results = []
        search_layers = layers or list(MemoryLayer)

        for layer in search_layers:
            for entry in self._memory[layer].values():
                if entry.is_expired():
                    continue

                # Match query
                match = False
                if query.lower() in entry.key.lower():
                    match = True
                elif isinstance(entry.value, str) and query.lower() in entry.value.lower():
                    match = True

                # Match tags
                if match and tags:
                    if not any(t in entry.tags for t in tags):
                        match = False

                if match:
                    results.append(entry)

        return results

    def list_layer(self, layer: MemoryLayer) -> list[MemoryEntry]:
        """List all entries in a layer."""
        return [e for e in self._memory[layer].values() if not e.is_expired()]

    def list_keys(self, layer: MemoryLayer | None = None) -> list[str]:
        """List all keys."""
        if layer is not None:
            return [k for k, v in self._memory[layer].items() if not v.is_expired()]

        keys = set()
        for layer in MemoryLayer:
            for key, entry in self._memory[layer].items():
                if not entry.is_expired():
                    keys.add(key)
        return list(keys)

    def clear_layer(self, layer: MemoryLayer):
        """Clear all entries in a layer."""
        self._memory[layer].clear()
        if layer != MemoryLayer.SESSION:
            self._save_layer(layer)

    def clear_session(self):
        """Clear session memory."""
        self.clear_layer(MemoryLayer.SESSION)

    def _get_layer_path(self, layer: MemoryLayer) -> Path:
        """Get storage path for a layer."""
        if layer in self._layer_paths:
            return self._layer_paths[layer]
        if layer == MemoryLayer.GLOBAL:
            return self._storage_dir / "global.json"
        elif layer == MemoryLayer.ORGANIZATION:
            org_id = self._organization_id or "default"
            return self._storage_dir / f"org_{org_id}.json"
        elif layer == MemoryLayer.PROJECT:
            proj_id = self._project_id or "default"
            return self._storage_dir / f"project_{proj_id}.json"
        elif layer == MemoryLayer.USER:
            user_id = self._user_id or "default"
            return self._storage_dir / f"user_{user_id}.json"
        else:
            return self._storage_dir / "session.json"

    def _save_layer(self, layer: MemoryLayer):
        """Save a layer to disk."""
        if layer == MemoryLayer.SESSION:
            return  # Don't persist session

        path = self._get_layer_path(layer)
        if layer in self._simple_layers:
            data = {k: v.value for k, v in self._memory[layer].items() if not v.is_expired()}
        else:
            data = {k: v.to_dict() for k, v in self._memory[layer].items() if not v.is_expired()}

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error("Failed to save memory layer %s: %s", layer.value, e)

    def _load_layer(self, layer: MemoryLayer):
        """Load a layer from disk."""
        if layer == MemoryLayer.SESSION:
            return

        path = self._get_layer_path(layer)
        if not path.exists():
            return

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for key, entry_data in data.items():
                if isinstance(entry_data, dict) and "layer" in entry_data:
                    entry = MemoryEntry.from_dict(entry_data)
                else:
                    now = time.time()
                    entry = MemoryEntry(
                        key=key,
                        value=entry_data,
                        layer=layer,
                        created_at=now,
                        updated_at=now,
                    )
                if not entry.is_expired():
                    self._memory[layer][key] = entry
        except Exception as e:
            logger.error("Failed to load memory layer %s: %s", layer.value, e)

    def _load_all(self):
        """Load all layers from disk."""
        for layer in MemoryLayer:
            self._load_layer(layer)

    def merge_from_file(self, path: Path, layer: MemoryLayer):
        """
        Merge memory entries from a file.

        Args:
            path: Path to JSON file
            layer: Target layer
        """
        if not path.exists():
            return

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for key, value in data.items():
                if isinstance(value, dict) and "value" in value:
                    # Full entry format
                    self.set(
                        key=key,
                        value=value["value"],
                        layer=layer,
                        tags=value.get("tags", []),
                        metadata=value.get("metadata", {}),
                    )
                else:
                    # Simple key-value format
                    self.set(key, value, layer=layer)

            logger.info("Merged %s entries to %s", len(data), layer.value)

        except Exception as e:
            logger.error("Failed to merge memory file: %s", e)

    def export_layer(self, layer: MemoryLayer) -> dict:
        """Export a layer as dict."""
        return {k: v.to_dict() for k, v in self._memory[layer].items() if not v.is_expired()}

    def get_summary(self) -> dict[str, Any]:
        """Get memory summary."""
        summary = {
            "layers": {},
            "total_entries": 0,
        }

        for layer in MemoryLayer:
            count = len([e for e in self._memory[layer].values() if not e.is_expired()])
            summary["layers"][layer.value] = count
            summary["total_entries"] += count

        return summary


# Global instance
_layered_memory: LayeredMemory | None = None


def get_layered_memory() -> LayeredMemory:
    """Get the global layered memory instance."""
    global _layered_memory
    if _layered_memory is None:
        _layered_memory = LayeredMemory()
    return _layered_memory
