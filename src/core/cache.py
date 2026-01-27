"""
Tool Result Cache

TTL-based caching for tool results.

Features:
- Time-to-live (TTL) based expiration
- Size-limited cache
- Selective caching (read operations)
- Cache invalidation
- Persistence support
"""

import hashlib
import json
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cached entry."""

    key: str
    value: Any
    created_at: float
    expires_at: float
    hits: int = 0
    size_bytes: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        return time.time() > self.expires_at

    @property
    def ttl_remaining(self) -> float:
        """Get remaining TTL in seconds."""
        remaining = self.expires_at - time.time()
        return max(0, remaining)


@dataclass
class CacheConfig:
    """Cache configuration."""

    max_size: int = 1000  # Maximum number of entries
    max_memory_mb: int = 100  # Maximum memory usage
    default_ttl: int = 300  # Default TTL in seconds (5 minutes)
    persist: bool = False  # Persist to disk
    persist_path: Path | None = None


class ToolCache:
    """
    Cache for tool results.

    Usage:
        cache = ToolCache()

        # Cache a result
        cache.set("file_read", {"path": "/file"}, "content", ttl=60)

        # Get cached result
        result = cache.get("file_read", {"path": "/file"})

        # Check if cached
        if cache.has("file_read", {"path": "/file"}):
            ...

        # Invalidate
        cache.invalidate("file_read", {"path": "/file"})
    """

    # Tools that are safe to cache (read-only operations)
    CACHEABLE_TOOLS = {
        "file_read",
        "file_list",
        "file_search",
        "git_status",
        "git_log",
        "git_diff",
        "git_show",
        "semantic_search",
        "grep",
        "glob",
        "outline",
        "web_search",
    }

    # TTL overrides for specific tools
    TOOL_TTL = {
        "git_status": 10,  # Changes frequently
        "git_diff": 30,
        "file_read": 60,
        "file_list": 30,
        "semantic_search": 300,
        "web_search": 600,  # Web results change slowly
    }

    def __init__(self, config: CacheConfig | None = None):
        """
        Initialize cache.

        Args:
            config: Cache configuration
        """
        self.config = config or CacheConfig()
        self._cache: dict[str, CacheEntry] = {}
        self._memory_usage = 0
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "invalidations": 0,
        }

        # Load persisted cache
        if self.config.persist and self.config.persist_path:
            self._load_cache()

    def _make_key(self, tool: str, arguments: dict) -> str:
        """Create a cache key from tool and arguments."""
        # Sort arguments for consistent key
        sorted_args = json.dumps(arguments, sort_keys=True)
        key_str = f"{tool}:{sorted_args}"
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]

    def _estimate_size(self, value: Any) -> int:
        """Estimate size of a value in bytes."""
        try:
            return len(pickle.dumps(value))
        except Exception:
            return len(str(value))

    def is_cacheable(self, tool: str) -> bool:
        """Check if a tool's results should be cached."""
        return tool in self.CACHEABLE_TOOLS

    def get_ttl(self, tool: str) -> int:
        """Get TTL for a tool."""
        return self.TOOL_TTL.get(tool, self.config.default_ttl)

    def get(self, tool: str, arguments: dict) -> Any | None:
        """
        Get cached result.

        Args:
            tool: Tool name
            arguments: Tool arguments

        Returns:
            Cached value or None
        """
        if not self.is_cacheable(tool):
            return None

        key = self._make_key(tool, arguments)
        entry = self._cache.get(key)

        if entry is None:
            self._stats["misses"] += 1
            return None

        if entry.is_expired:
            self._remove(key)
            self._stats["misses"] += 1
            return None

        entry.hits += 1
        self._stats["hits"] += 1
        return entry.value

    def set(
        self,
        tool: str,
        arguments: dict,
        value: Any,
        ttl: int | None = None,
    ):
        """
        Cache a result.

        Args:
            tool: Tool name
            arguments: Tool arguments
            value: Result to cache
            ttl: Time-to-live in seconds
        """
        if not self.is_cacheable(tool):
            return

        # Evict if necessary
        self._evict_if_needed()

        key = self._make_key(tool, arguments)
        ttl = ttl or self.get_ttl(tool)
        size = self._estimate_size(value)

        entry = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            expires_at=time.time() + ttl,
            size_bytes=size,
        )

        # Update memory tracking
        if key in self._cache:
            self._memory_usage -= self._cache[key].size_bytes
        self._memory_usage += size

        self._cache[key] = entry

        # Persist if enabled
        if self.config.persist:
            self._save_cache()

    def has(self, tool: str, arguments: dict) -> bool:
        """Check if result is cached and not expired."""
        if not self.is_cacheable(tool):
            return False

        key = self._make_key(tool, arguments)
        entry = self._cache.get(key)

        if entry is None:
            return False

        if entry.is_expired:
            self._remove(key)
            return False

        return True

    def invalidate(self, tool: str, arguments: dict):
        """Invalidate a specific cache entry."""
        key = self._make_key(tool, arguments)
        if key in self._cache:
            self._remove(key)
            self._stats["invalidations"] += 1

    def invalidate_tool(self, tool: str):
        """Invalidate all entries for a tool."""
        prefix = tool + ":"
        keys_to_remove = [
            k for k, v in self._cache.items()
            if self._make_key(tool, {}).startswith(prefix[:10])
        ]
        for key in keys_to_remove:
            self._remove(key)
            self._stats["invalidations"] += 1

    def invalidate_path(self, path: str):
        """Invalidate all entries containing a path."""
        keys_to_remove = []
        for key, entry in self._cache.items():
            # Check if entry involves this path (rough check)
            if path in str(entry.value):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            self._remove(key)
            self._stats["invalidations"] += 1

    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()
        self._memory_usage = 0

        if self.config.persist and self.config.persist_path:
            self.config.persist_path.unlink(missing_ok=True)

    def _remove(self, key: str):
        """Remove an entry from cache."""
        if key in self._cache:
            self._memory_usage -= self._cache[key].size_bytes
            del self._cache[key]

    def _evict_if_needed(self):
        """Evict entries if cache is full."""
        # Check entry count
        while len(self._cache) >= self.config.max_size:
            self._evict_lru()

        # Check memory usage
        max_bytes = self.config.max_memory_mb * 1024 * 1024
        while self._memory_usage > max_bytes and self._cache:
            self._evict_lru()

    def _evict_lru(self):
        """Evict the least recently used entry."""
        if not self._cache:
            return

        # First try to evict expired entries
        expired = [k for k, v in self._cache.items() if v.is_expired]
        if expired:
            self._remove(expired[0])
            self._stats["evictions"] += 1
            return

        # Evict entry with oldest access (fewest hits as approximation)
        lru_key = min(self._cache.keys(), key=lambda k: self._cache[k].hits)
        self._remove(lru_key)
        self._stats["evictions"] += 1

    def _save_cache(self):
        """Save cache to disk."""
        if not self.config.persist_path:
            return

        try:
            # Only save non-expired entries
            data = {
                k: {
                    "value": v.value,
                    "created_at": v.created_at,
                    "expires_at": v.expires_at,
                    "hits": v.hits,
                }
                for k, v in self._cache.items()
                if not v.is_expired
            }

            self.config.persist_path.parent.mkdir(parents=True, exist_ok=True)
            self.config.persist_path.write_bytes(pickle.dumps(data))

        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _load_cache(self):
        """Load cache from disk."""
        if not self.config.persist_path or not self.config.persist_path.exists():
            return

        try:
            data = pickle.loads(self.config.persist_path.read_bytes())

            for key, entry_data in data.items():
                # Skip expired entries
                if time.time() > entry_data["expires_at"]:
                    continue

                entry = CacheEntry(
                    key=key,
                    value=entry_data["value"],
                    created_at=entry_data["created_at"],
                    expires_at=entry_data["expires_at"],
                    hits=entry_data["hits"],
                    size_bytes=self._estimate_size(entry_data["value"]),
                )

                self._cache[key] = entry
                self._memory_usage += entry.size_bytes

            logger.info(f"Loaded {len(self._cache)} cache entries")

        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = (
            self._stats["hits"] / total_requests if total_requests > 0 else 0
        )

        return {
            "entries": len(self._cache),
            "memory_mb": self._memory_usage / (1024 * 1024),
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": f"{hit_rate:.1%}",
            "evictions": self._stats["evictions"],
            "invalidations": self._stats["invalidations"],
        }

    def get_entries(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent cache entries."""
        entries = sorted(
            self._cache.values(),
            key=lambda e: e.created_at,
            reverse=True,
        )[:limit]

        return [
            {
                "key": e.key[:16] + "...",
                "ttl_remaining": f"{e.ttl_remaining:.0f}s",
                "hits": e.hits,
                "size_kb": e.size_bytes / 1024,
            }
            for e in entries
        ]


# Global cache instance
_tool_cache: ToolCache | None = None


def get_tool_cache() -> ToolCache:
    """Get the global tool cache."""
    global _tool_cache
    if _tool_cache is None:
        _tool_cache = ToolCache()
    return _tool_cache
