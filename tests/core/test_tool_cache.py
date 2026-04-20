"""Tests for src.core.tool_cache."""

import time

from src.core.tool_cache import ToolCache, should_cache_tool, READ_ONLY_TOOLS


class TestToolCache:
    def setup_method(self):
        ToolCache._instance = None

    def test_set_and_get(self):
        cache = ToolCache(max_size=10, ttl=300)
        cache.set("read", {"path": "f.py"}, "content")
        assert cache.get("read", {"path": "f.py"}) == "content"

    def test_get_miss(self):
        cache = ToolCache(max_size=10, ttl=300)
        assert cache.get("read", {"path": "missing.py"}) is None

    def test_ttl_expired(self):
        cache = ToolCache(max_size=10, ttl=0)
        cache.set("read", {"path": "f.py"}, "content")
        time.sleep(0.01)
        assert cache.get("read", {"path": "f.py"}) is None

    def test_lru_eviction(self):
        cache = ToolCache(max_size=3, ttl=300)
        for i in range(4):
            cache.set("read", {"path": f"f{i}.py"}, f"content{i}")
        assert cache.get("read", {"path": "f0.py"}) is None
        assert cache.get("read", {"path": "f3.py"}) == "content3"

    def test_invalidate_specific(self):
        cache = ToolCache(max_size=10, ttl=300)
        cache.set("read", {"path": "a.py"}, "a")
        cache.set("read", {"path": "b.py"}, "b")
        cache.invalidate("read", {"path": "a.py"})
        assert cache.get("read", {"path": "a.py"}) is None
        assert cache.get("read", {"path": "b.py"}) == "b"

    def test_invalidate_all_for_tool(self):
        cache = ToolCache(max_size=10, ttl=300)
        cache.set("read", {"path": "a.py"}, "a")
        cache.set("search", {"pattern": "x"}, "found")
        cache.invalidate("read")
        assert cache.get("read", {"path": "a.py"}) is None
        assert cache.get("search", {"pattern": "x"}) == "found"

    def test_clear(self):
        cache = ToolCache(max_size=10, ttl=300)
        cache.set("read", {"path": "a.py"}, "a")
        cache.clear()
        assert cache.get("read", {"path": "a.py"}) is None

    def test_get_stats(self):
        cache = ToolCache(max_size=10, ttl=300)
        cache.get("read", {"path": "a.py"})
        cache.set("read", {"path": "a.py"}, "content")
        cache.get("read", {"path": "a.py"})
        stats = cache.get_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 1

    def test_set_empty_result_skipped(self):
        cache = ToolCache(max_size=10, ttl=300)
        cache.set("read", {"path": "a.py"}, "")
        assert cache.get("read", {"path": "a.py"}) is None

    def test_set_oversized_result_skipped(self):
        cache = ToolCache(max_size=10, ttl=300)
        cache.set("read", {"path": "a.py"}, "x" * 100001)
        assert cache.get("read", {"path": "a.py"}) is None

    def test_singleton(self):
        c1 = ToolCache(max_size=10, ttl=300)
        c2 = ToolCache(max_size=20, ttl=600)
        assert c1 is c2

    def test_get_instance(self):
        c = ToolCache.get_instance()
        assert isinstance(c, ToolCache)

    def test_different_args_different_key(self):
        cache = ToolCache(max_size=10, ttl=300)
        cache.set("read", {"path": "a.py"}, "a")
        cache.set("read", {"path": "b.py"}, "b")
        assert cache.get("read", {"path": "a.py"}) == "a"
        assert cache.get("read", {"path": "b.py"}) == "b"


class TestShouldCacheTool:
    def test_read_only_tools(self):
        for tool in READ_ONLY_TOOLS:
            assert should_cache_tool(tool) is True

    def test_write_tools_not_cached(self):
        assert not should_cache_tool("write")
        assert not should_cache_tool("run")

    def test_unknown_tool(self):
        assert not should_cache_tool("unknown_tool")
