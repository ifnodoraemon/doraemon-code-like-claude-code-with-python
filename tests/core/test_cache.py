import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from src.core.cache import CacheConfig, CacheEntry, ToolCache, get_tool_cache


class TestCacheEntry:
    def test_is_expired_false(self):
        entry = CacheEntry(
            key="k",
            value="v",
            created_at=time.time(),
            expires_at=time.time() + 100,
        )
        assert entry.is_expired is False

    def test_is_expired_true(self):
        entry = CacheEntry(
            key="k",
            value="v",
            created_at=time.time() - 200,
            expires_at=time.time() - 1,
        )
        assert entry.is_expired is True

    def test_ttl_remaining_positive(self):
        future = time.time() + 60
        entry = CacheEntry(
            key="k",
            value="v",
            created_at=time.time(),
            expires_at=future,
        )
        remaining = entry.ttl_remaining
        assert 50 < remaining <= 60

    def test_ttl_remaining_zero_when_expired(self):
        entry = CacheEntry(
            key="k",
            value="v",
            created_at=time.time() - 200,
            expires_at=time.time() - 100,
        )
        assert entry.ttl_remaining == 0

    def test_ttl_remaining_clamped_to_zero(self):
        entry = CacheEntry(
            key="k",
            value="v",
            created_at=time.time() - 10,
            expires_at=time.time() - 5,
        )
        assert entry.ttl_remaining == 0

    def test_default_fields(self):
        entry = CacheEntry(
            key="k",
            value="v",
            created_at=1.0,
            expires_at=2.0,
        )
        assert entry.hits == 0
        assert entry.size_bytes == 0
        assert entry.tool_name == ""
        assert entry.last_accessed == 0.0


class TestCacheConfig:
    def test_defaults(self):
        cfg = CacheConfig()
        assert cfg.max_size == 1000
        assert cfg.max_memory_mb == 100
        assert cfg.default_ttl == 300
        assert cfg.persist is False
        assert cfg.persist_path is None

    def test_custom_values(self, tmp_path):
        p = tmp_path / "cache.json"
        cfg = CacheConfig(
            max_size=50,
            max_memory_mb=10,
            default_ttl=60,
            persist=True,
            persist_path=p,
        )
        assert cfg.max_size == 50
        assert cfg.max_memory_mb == 10
        assert cfg.default_ttl == 60
        assert cfg.persist is True
        assert cfg.persist_path == p


class TestToolCache:
    def test_init_defaults(self):
        cache = ToolCache()
        assert cache.config.max_size == 1000
        assert len(cache._cache) == 0
        assert cache._memory_usage == 0

    def test_init_with_config(self):
        cfg = CacheConfig(max_size=10)
        cache = ToolCache(config=cfg)
        assert cache.config.max_size == 10

    def test_make_key_deterministic(self):
        cache = ToolCache()
        k1 = cache._make_key("file_read", {"path": "/a"})
        k2 = cache._make_key("file_read", {"path": "/a"})
        assert k1 == k2

    def test_make_key_arg_order_independent(self):
        cache = ToolCache()
        k1 = cache._make_key("tool", {"a": 1, "b": 2})
        k2 = cache._make_key("tool", {"b": 2, "a": 1})
        assert k1 == k2

    def test_make_key_different_tools(self):
        cache = ToolCache()
        k1 = cache._make_key("file_read", {"path": "/a"})
        k2 = cache._make_key("file_list", {"path": "/a"})
        assert k1 != k2

    def test_estimate_size_string(self):
        cache = ToolCache()
        size = cache._estimate_size("hello")
        assert size > 0

    def test_estimate_size_dict(self):
        cache = ToolCache()
        size = cache._estimate_size({"key": "value"})
        assert size > 0

    def test_estimate_size_non_serializable(self):
        cache = ToolCache()

        class NonSerializable:
            pass

        size = cache._estimate_size(NonSerializable())
        assert size > 0

    def test_is_cacheable_known_tools(self):
        cache = ToolCache()
        assert cache.is_cacheable("file_read") is True
        assert cache.is_cacheable("web_search") is True
        assert cache.is_cacheable("memory_get") is True

    def test_is_cacheable_unknown_tool(self):
        cache = ToolCache()
        assert cache.is_cacheable("file_write") is False
        assert cache.is_cacheable("shell_exec") is False

    def test_get_ttl_known_tool(self):
        cache = ToolCache()
        assert cache.get_ttl("file_read") == 60
        assert cache.get_ttl("web_search") == 600

    def test_get_ttl_unknown_tool_uses_default(self):
        cache = ToolCache(config=CacheConfig(default_ttl=500))
        assert cache.get_ttl("unknown_tool") == 500

    def test_set_and_get(self):
        cache = ToolCache()
        cache.set("file_read", {"path": "/a"}, "content")
        result = cache.get("file_read", {"path": "/a"})
        assert result == "content"

    def test_get_miss(self):
        cache = ToolCache()
        result = cache.get("file_read", {"path": "/missing"})
        assert result is None

    def test_get_non_cacheable(self):
        cache = ToolCache()
        cache.set("file_write", {"path": "/a"}, "data")
        result = cache.get("file_write", {"path": "/a"})
        assert result is None

    def test_set_non_cacheable(self):
        cache = ToolCache()
        cache.set("file_write", {"path": "/a"}, "data")
        assert len(cache._cache) == 0

    def test_has_true(self):
        cache = ToolCache()
        cache.set("file_read", {"path": "/a"}, "content")
        assert cache.has("file_read", {"path": "/a"}) is True

    def test_has_false_missing(self):
        cache = ToolCache()
        assert cache.has("file_read", {"path": "/missing"}) is False

    def test_has_false_non_cacheable(self):
        cache = ToolCache()
        assert cache.has("file_write", {"path": "/a"}) is False

    def test_has_false_expired(self):
        cache = ToolCache()
        cache.set("file_read", {"path": "/a"}, "content", ttl=1)
        with patch("time.time", return_value=time.time() + 10):
            assert cache.has("file_read", {"path": "/a"}) is False

    def test_get_expired_returns_none(self):
        cache = ToolCache()
        cache.set("file_read", {"path": "/a"}, "content", ttl=1)
        with patch("time.time", return_value=time.time() + 10):
            result = cache.get("file_read", {"path": "/a"})
            assert result is None

    def test_get_updates_hits(self):
        cache = ToolCache()
        cache.set("file_read", {"path": "/a"}, "content")
        cache.get("file_read", {"path": "/a"})
        cache.get("file_read", {"path": "/a"})
        key = cache._make_key("file_read", {"path": "/a"})
        assert cache._cache[key].hits == 2

    def test_get_updates_last_accessed(self):
        cache = ToolCache()
        cache.set("file_read", {"path": "/a"}, "content")
        key = cache._make_key("file_read", {"path": "/a"})
        first_access = cache._cache[key].last_accessed
        time.sleep(0.01)
        cache.get("file_read", {"path": "/a"})
        assert cache._cache[key].last_accessed >= first_access

    def test_set_custom_ttl(self):
        cache = ToolCache()
        cache.set("file_read", {"path": "/a"}, "content", ttl=999)
        key = cache._make_key("file_read", {"path": "/a"})
        now = time.time()
        entry = cache._cache[key]
        assert abs(entry.expires_at - (entry.created_at + 999)) < 1

    def test_set_overwrites_existing(self):
        cache = ToolCache()
        cache.set("file_read", {"path": "/a"}, "old")
        cache.set("file_read", {"path": "/a"}, "new")
        result = cache.get("file_read", {"path": "/a"})
        assert result == "new"

    def test_invalidate_specific(self):
        cache = ToolCache()
        cache.set("file_read", {"path": "/a"}, "content")
        cache.invalidate("file_read", {"path": "/a"})
        assert cache.get("file_read", {"path": "/a"}) is None

    def test_invalidate_updates_stats(self):
        cache = ToolCache()
        cache.set("file_read", {"path": "/a"}, "content")
        cache.invalidate("file_read", {"path": "/a"})
        assert cache._stats["invalidations"] == 1

    def test_invalidate_nonexistent(self):
        cache = ToolCache()
        cache.invalidate("file_read", {"path": "/missing"})
        assert cache._stats["invalidations"] == 0

    def test_invalidate_tool(self):
        cache = ToolCache()
        cache.set("file_read", {"path": "/a"}, "c1")
        cache.set("file_read", {"path": "/b"}, "c2")
        cache.set("file_list", {"dir": "/x"}, "c3")
        cache.invalidate_tool("file_read")
        assert cache.get("file_read", {"path": "/a"}) is None
        assert cache.get("file_read", {"path": "/b"}) is None
        assert cache.get("file_list", {"dir": "/x"}) is not None

    def test_invalidate_tool_stats(self):
        cache = ToolCache()
        cache.set("file_read", {"path": "/a"}, "c1")
        cache.set("file_read", {"path": "/b"}, "c2")
        cache.invalidate_tool("file_read")
        assert cache._stats["invalidations"] == 2

    def test_invalidate_path(self):
        cache = ToolCache()
        cache.set("file_read", {"path": "/src/main.py"}, "content of /src/main.py")
        cache.set("file_read", {"path": "/other.py"}, "other")
        cache.invalidate_path("/src/main.py")
        assert cache.get("file_read", {"path": "/src/main.py"}) is None
        assert cache.get("file_read", {"path": "/other.py"}) is not None

    def test_invalidate_path_stats(self):
        cache = ToolCache()
        cache.set("file_read", {"path": "/a.py"}, "/a.py content")
        cache.invalidate_path("/a.py")
        assert cache._stats["invalidations"] == 1

    def test_clear(self):
        cache = ToolCache()
        cache.set("file_read", {"path": "/a"}, "c1")
        cache.set("file_list", {"dir": "/x"}, "c2")
        cache.clear()
        assert len(cache._cache) == 0
        assert cache._memory_usage == 0

    def test_clear_removes_persist_file(self, tmp_path):
        p = tmp_path / "cache.json"
        cfg = CacheConfig(persist=True, persist_path=p)
        cache = ToolCache(config=cfg)
        cache.set("file_read", {"path": "/a"}, "c1")
        assert p.exists()
        cache.clear()
        assert not p.exists()

    def test_remove(self):
        cache = ToolCache()
        cache.set("file_read", {"path": "/a"}, "content")
        key = cache._make_key("file_read", {"path": "/a"})
        assert cache._memory_usage > 0
        cache._remove(key)
        assert len(cache._cache) == 0
        assert cache._memory_usage == 0

    def test_evict_if_needed_entry_limit(self):
        cache = ToolCache(config=CacheConfig(max_size=3))
        for i in range(5):
            cache.set("file_read", {"path": f"/{i}"}, f"content_{i}")
        assert len(cache._cache) <= 3

    def test_evict_if_needed_memory_limit(self):
        cache = ToolCache(config=CacheConfig(max_memory_mb=0))
        assert cache._cache == {}

    def test_evict_lru_expired_first(self):
        cache = ToolCache(config=CacheConfig(max_size=2))
        cache.set("file_read", {"path": "/old"}, "old", ttl=1)
        now = time.time()
        with patch("time.time", return_value=now):
            cache.set("file_read", {"path": "/new1"}, "new1")
        with patch("time.time", return_value=now + 10):
            cache.set("file_read", {"path": "/new2"}, "new2")
            assert cache._stats["evictions"] > 0

    def test_evict_lru_least_recently_accessed(self):
        cache = ToolCache(config=CacheConfig(max_size=3))
        cache.set("file_read", {"path": "/a"}, "a")
        cache.set("file_read", {"path": "/b"}, "b")
        cache.set("file_read", {"path": "/c"}, "c")
        cache.get("file_read", {"path": "/a"})
        cache.get("file_read", {"path": "/c"})
        cache.set("file_read", {"path": "/d"}, "d")
        assert not cache.has("file_read", {"path": "/b"})

    def test_save_and_load_cache(self, tmp_path):
        p = tmp_path / "cache.json"
        cfg = CacheConfig(persist=True, persist_path=p)
        cache = ToolCache(config=cfg)
        cache.set("file_read", {"path": "/a"}, "content_a")
        cache.set("memory_get", {"key": "k"}, "val_k")
        assert p.exists()

        data = json.loads(p.read_text())
        assert len(data) >= 2

    def test_load_cache_from_disk(self, tmp_path):
        p = tmp_path / "cache.json"
        cfg = CacheConfig(persist=True, persist_path=p)
        cache1 = ToolCache(config=cfg)
        cache1.set("file_read", {"path": "/a"}, "content_a")

        cache2 = ToolCache(config=cfg)
        result = cache2.get("file_read", {"path": "/a"})
        assert result == "content_a"

    def test_load_cache_skips_expired(self, tmp_path):
        p = tmp_path / "cache.json"
        cfg = CacheConfig(persist=True, persist_path=p)
        now = time.time()
        with patch("time.time", return_value=now):
            cache1 = ToolCache(config=cfg)
            cache1.set("file_read", {"path": "/a"}, "content_a", ttl=1)

        import time as _time

        _time.sleep(1.5)

        cache2 = ToolCache(config=CacheConfig(persist=True, persist_path=p))
        assert cache2.get("file_read", {"path": "/a"}) is None

    def test_load_cache_missing_file(self, tmp_path):
        p = tmp_path / "nonexistent.json"
        cfg = CacheConfig(persist=True, persist_path=p)
        cache = ToolCache(config=cfg)
        assert len(cache._cache) == 0

    def test_load_cache_corrupt_file(self, tmp_path):
        p = tmp_path / "cache.json"
        p.write_text("not valid json{{{")
        cfg = CacheConfig(persist=True, persist_path=p)
        cache = ToolCache(config=cfg)
        assert len(cache._cache) == 0

    def test_save_cache_handles_error(self, tmp_path):
        p = tmp_path / "readonly" / "cache.json"
        cfg = CacheConfig(persist=True, persist_path=p)
        cache = ToolCache(config=cfg)
        cache._save_cache()

    def test_get_stats_initial(self):
        cache = ToolCache()
        stats = cache.get_stats()
        assert stats["entries"] == 0
        assert stats["memory_mb"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == "0.0%"
        assert stats["evictions"] == 0
        assert stats["invalidations"] == 0

    def test_get_stats_after_operations(self):
        cache = ToolCache()
        cache.set("file_read", {"path": "/a"}, "content")
        cache.get("file_read", {"path": "/a"})
        cache.get("file_read", {"path": "/missing"})
        stats = cache.get_stats()
        assert stats["entries"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == "50.0%"

    def test_get_entries(self):
        cache = ToolCache()
        cache.set("file_read", {"path": "/a"}, "content_a")
        cache.set("file_list", {"dir": "/x"}, "content_x")
        entries = cache.get_entries(limit=10)
        assert len(entries) == 2
        assert "key" in entries[0]
        assert "ttl_remaining" in entries[0]
        assert "hits" in entries[0]
        assert "size_kb" in entries[0]

    def test_get_entries_limit(self):
        cache = ToolCache()
        for i in range(20):
            cache.set("file_read", {"path": f"/{i}"}, f"content_{i}")
        entries = cache.get_entries(limit=5)
        assert len(entries) == 5

    def test_set_memory_tracking(self):
        cache = ToolCache()
        cache.set("file_read", {"path": "/a"}, "content")
        assert cache._memory_usage > 0

    def test_set_overwrite_updates_memory(self):
        cache = ToolCache()
        cache.set("file_read", {"path": "/a"}, "short")
        mem1 = cache._memory_usage
        cache.set("file_read", {"path": "/a"}, "much longer content here")
        mem2 = cache._memory_usage
        assert mem2 != mem1


class TestGetToolCache:
    def test_returns_tool_cache(self):
        import src.core.cache as mod

        mod._tool_cache = None
        cache = get_tool_cache()
        assert isinstance(cache, ToolCache)

    def test_returns_same_instance(self):
        import src.core.cache as mod

        mod._tool_cache = None
        c1 = get_tool_cache()
        c2 = get_tool_cache()
        assert c1 is c2

    def test_reset_global(self):
        import src.core.cache as mod

        mod._tool_cache = None
        c1 = get_tool_cache()
        assert c1 is not None
        mod._tool_cache = None
        c2 = get_tool_cache()
        assert c2 is not c1
