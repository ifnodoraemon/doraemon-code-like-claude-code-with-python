"""
Tool Cache - LRU Cache for Tool Results

Caches read-only tool results to avoid redundant calls.
"""

import hashlib
import time
from collections import OrderedDict
from threading import Lock


class ToolCache:
    """
    Thread-safe LRU cache for tool results.

    Usage:
        cache = ToolCache(max_size=100, ttl=300)

        # Cache result
        cache.set("read", {"path": "file.py"}, "content...")

        # Get cached result
        result = cache.get("read", {"path": "file.py"})
    """

    _instance = None
    _lock = Lock()

    def __new__(cls, max_size: int = 100, ttl: int = 300):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._cache = OrderedDict()
                    cls._instance._timestamps = {}
                    cls._instance.max_size = max_size
                    cls._instance.ttl = ttl
                    cls._instance._cache_lock = Lock()
                    cls._instance._stats = {"hits": 0, "misses": 0}
        return cls._instance

    def _make_key(self, tool_name: str, args: dict) -> str:
        """Create cache key from tool name and args."""
        args_hash = hashlib.md5(str(sorted(args.items())).encode()).hexdigest()[:8]
        return f"{tool_name}:{args_hash}"

    def get(self, tool_name: str, args: dict) -> str | None:
        """Get cached result if available and not expired."""
        key = self._make_key(tool_name, args)

        with self._cache_lock:
            if key not in self._cache:
                self._stats["misses"] += 1
                return None

            timestamp = self._timestamps.get(key, 0)
            if time.time() - timestamp > self.ttl:
                del self._cache[key]
                del self._timestamps[key]
                self._stats["misses"] += 1
                return None

            self._cache.move_to_end(key)
            self._stats["hits"] += 1
            return self._cache[key]

    def set(self, tool_name: str, args: dict, result: str) -> None:
        """Cache a result."""
        if not result or len(result) > 100000:
            return

        key = self._make_key(tool_name, args)

        with self._cache_lock:
            if key in self._cache:
                del self._cache[key]

            self._cache[key] = result
            self._timestamps[key] = time.time()

            while len(self._cache) > self.max_size:
                oldest = next(iter(self._cache))
                del self._cache[oldest]
                del self._timestamps[oldest]

    def invalidate(self, tool_name: str, args: dict | None = None) -> None:
        """Invalidate cache entries."""
        with self._cache_lock:
            if args is None:
                keys_to_delete = [k for k in self._cache if k.startswith(f"{tool_name}:")]
            else:
                key = self._make_key(tool_name, args)
                keys_to_delete = [key]

            for key in keys_to_delete:
                self._cache.pop(key, None)
                self._timestamps.pop(key, None)

    def clear(self) -> None:
        """Clear all cache."""
        with self._cache_lock:
            self._cache.clear()
            self._timestamps.clear()

    def get_stats(self) -> dict:
        """Get cache statistics."""
        with self._cache_lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total if total > 0 else 0
            return {
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate": f"{hit_rate:.1%}",
                "size": len(self._cache),
                "max_size": self.max_size,
            }

    @classmethod
    def get_instance(cls) -> "ToolCache":
        """Get singleton instance."""
        return cls()


READ_ONLY_TOOLS = {
    "read",
    "search",
    "glob",
    "list_directory",
    "get_note",
    "list_notes",
    "search_notes",
    "browse_page",
    "take_screenshot",
    "semantic_search",
    "lsp_diagnostics",
}


def should_cache_tool(tool_name: str) -> bool:
    """Check if a tool result should be cached."""
    return tool_name in READ_ONLY_TOOLS
