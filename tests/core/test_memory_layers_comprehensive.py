"""Additional tests for src/core/memory_layers.py"""

import json
import time

import pytest

from src.core.memory_layers import (
    LayeredMemory,
    MemoryEntry,
    MemoryLayer,
    get_layered_memory,
)


class TestMemoryEntry:
    def test_is_expired_false(self):
        entry = MemoryEntry(
            key="k", value="v", layer=MemoryLayer.GLOBAL,
            created_at=time.time(), updated_at=time.time(),
        )
        assert entry.is_expired() is False

    def test_is_expired_true(self):
        entry = MemoryEntry(
            key="k", value="v", layer=MemoryLayer.GLOBAL,
            created_at=0, updated_at=0, expires_at=1,
        )
        assert entry.is_expired() is True

    def test_is_expired_none(self):
        entry = MemoryEntry(
            key="k", value="v", layer=MemoryLayer.GLOBAL,
            created_at=time.time(), updated_at=time.time(), expires_at=None,
        )
        assert entry.is_expired() is False

    def test_to_dict(self):
        entry = MemoryEntry(
            key="k", value="v", layer=MemoryLayer.PROJECT,
            created_at=100.0, updated_at=200.0,
            metadata={"source": "test"}, tags=["a", "b"],
            expires_at=300.0,
        )
        d = entry.to_dict()
        assert d["key"] == "k"
        assert d["value"] == "v"
        assert d["layer"] == "project"
        assert d["tags"] == ["a", "b"]
        assert d["expires_at"] == 300.0

    def test_from_dict(self):
        d = {
            "key": "k", "value": 42, "layer": "user",
            "created_at": 1.0, "updated_at": 2.0,
            "metadata": {}, "tags": [], "expires_at": None,
        }
        entry = MemoryEntry.from_dict(d)
        assert entry.key == "k"
        assert entry.value == 42
        assert entry.layer == MemoryLayer.USER

    def test_roundtrip(self):
        entry = MemoryEntry(
            key="k", value="v", layer=MemoryLayer.ORGANIZATION,
            created_at=1.0, updated_at=2.0, tags=["t"],
        )
        d = entry.to_dict()
        restored = MemoryEntry.from_dict(d)
        assert restored.key == entry.key
        assert restored.value == entry.value
        assert restored.layer == entry.layer


class TestLayeredMemorySetGet:
    def test_set_and_get_same_layer(self, tmp_path):
        mem = LayeredMemory(storage_dir=tmp_path)
        mem.set("k", "v", layer=MemoryLayer.PROJECT)
        assert mem.get("k", layer=MemoryLayer.PROJECT) == "v"

    def test_get_default(self, tmp_path):
        mem = LayeredMemory(storage_dir=tmp_path)
        assert mem.get("missing", default="def") == "def"

    def test_get_resolves_priority(self, tmp_path):
        mem = LayeredMemory(storage_dir=tmp_path)
        mem.set("k", "global_val", layer=MemoryLayer.GLOBAL)
        mem.set("k", "user_val", layer=MemoryLayer.USER)
        assert mem.get("k") == "user_val"

    def test_get_skips_expired(self, tmp_path):
        mem = LayeredMemory(storage_dir=tmp_path)
        mem.set("k", "expired", layer=MemoryLayer.USER, ttl=-1)
        assert mem.get("k") is None

    def test_get_entry(self, tmp_path):
        mem = LayeredMemory(storage_dir=tmp_path)
        mem.set("k", "v", layer=MemoryLayer.PROJECT, tags=["t"])
        entry = mem.get_entry("k", layer=MemoryLayer.PROJECT)
        assert entry is not None
        assert entry.tags == ["t"]

    def test_get_entry_missing(self, tmp_path):
        mem = LayeredMemory(storage_dir=tmp_path)
        assert mem.get_entry("missing") is None

    def test_get_entry_resolves_priority(self, tmp_path):
        mem = LayeredMemory(storage_dir=tmp_path)
        mem.set("k", "low", layer=MemoryLayer.GLOBAL)
        mem.set("k", "high", layer=MemoryLayer.SESSION)
        entry = mem.get_entry("k")
        assert entry.value == "high"


class TestLayeredMemoryDelete:
    def test_delete_from_specific_layer(self, tmp_path):
        mem = LayeredMemory(storage_dir=tmp_path)
        mem.set("k", "v", layer=MemoryLayer.PROJECT)
        mem.delete("k", layer=MemoryLayer.PROJECT)
        assert mem.get("k", layer=MemoryLayer.PROJECT) is None

    def test_delete_from_all_layers(self, tmp_path):
        mem = LayeredMemory(storage_dir=tmp_path)
        mem.set("k", "a", layer=MemoryLayer.GLOBAL)
        mem.set("k", "b", layer=MemoryLayer.USER)
        mem.delete("k")
        assert mem.get("k") is None


class TestLayeredMemoryHas:
    def test_has_true(self, tmp_path):
        mem = LayeredMemory(storage_dir=tmp_path)
        mem.set("k", "v", layer=MemoryLayer.GLOBAL)
        assert mem.has("k") is True

    def test_has_false(self, tmp_path):
        mem = LayeredMemory(storage_dir=tmp_path)
        assert mem.has("missing") is False


class TestLayeredMemorySearch:
    def test_search_by_key(self, tmp_path):
        mem = LayeredMemory(storage_dir=tmp_path)
        mem.set("api_key", "xxx", layer=MemoryLayer.PROJECT)
        mem.set("db_url", "localhost", layer=MemoryLayer.PROJECT)
        results = mem.search("api")
        assert len(results) >= 1
        assert results[0].key == "api_key"

    def test_search_by_value(self, tmp_path):
        mem = LayeredMemory(storage_dir=tmp_path)
        mem.set("k", "hello world", layer=MemoryLayer.GLOBAL)
        results = mem.search("world")
        assert len(results) >= 1

    def test_search_with_tags(self, tmp_path):
        mem = LayeredMemory(storage_dir=tmp_path)
        mem.set("k1", "v1", layer=MemoryLayer.PROJECT, tags=["config"])
        mem.set("k2", "v2", layer=MemoryLayer.PROJECT, tags=["runtime"])
        results = mem.search("k", tags=["config"])
        assert len(results) == 1
        assert results[0].key == "k1"

    def test_search_in_specific_layers(self, tmp_path):
        mem = LayeredMemory(storage_dir=tmp_path)
        mem.set("k", "v", layer=MemoryLayer.PROJECT)
        results = mem.search("k", layers=[MemoryLayer.USER])
        assert len(results) == 0

    def test_search_expires_entry(self, tmp_path):
        mem = LayeredMemory(storage_dir=tmp_path)
        mem.set("k", "v", layer=MemoryLayer.GLOBAL, ttl=-1)
        results = mem.search("k")
        assert len(results) == 0


class TestLayeredMemoryList:
    def test_list_layer(self, tmp_path):
        mem = LayeredMemory(storage_dir=tmp_path)
        mem.set("a", 1, layer=MemoryLayer.PROJECT)
        mem.set("b", 2, layer=MemoryLayer.PROJECT)
        entries = mem.list_layer(MemoryLayer.PROJECT)
        assert len(entries) == 2

    def test_list_keys(self, tmp_path):
        mem = LayeredMemory(storage_dir=tmp_path)
        mem.set("a", 1, layer=MemoryLayer.GLOBAL)
        mem.set("b", 2, layer=MemoryLayer.USER)
        keys = mem.list_keys()
        assert "a" in keys
        assert "b" in keys

    def test_list_keys_specific_layer(self, tmp_path):
        mem = LayeredMemory(storage_dir=tmp_path)
        mem.set("a", 1, layer=MemoryLayer.GLOBAL)
        mem.set("b", 2, layer=MemoryLayer.USER)
        keys = mem.list_keys(layer=MemoryLayer.GLOBAL)
        assert keys == ["a"]


class TestLayeredMemoryClear:
    def test_clear_layer(self, tmp_path):
        mem = LayeredMemory(storage_dir=tmp_path)
        mem.set("k", "v", layer=MemoryLayer.PROJECT)
        mem.clear_layer(MemoryLayer.PROJECT)
        assert mem.list_layer(MemoryLayer.PROJECT) == []

    def test_clear_session(self, tmp_path):
        mem = LayeredMemory(storage_dir=tmp_path)
        mem.set("k", "v", layer=MemoryLayer.SESSION)
        mem.clear_session()
        assert mem.list_layer(MemoryLayer.SESSION) == []


class TestLayeredMemoryPersistence:
    def test_save_and_reload(self, tmp_path):
        storage = tmp_path / "mem"
        mem1 = LayeredMemory(storage_dir=storage)
        mem1.set("k", "v", layer=MemoryLayer.PROJECT)

        mem2 = LayeredMemory(storage_dir=storage)
        assert mem2.get("k", layer=MemoryLayer.PROJECT) == "v"

    def test_session_not_persisted(self, tmp_path):
        storage = tmp_path / "mem"
        mem1 = LayeredMemory(storage_dir=storage)
        mem1.set("k", "v", layer=MemoryLayer.SESSION)

        mem2 = LayeredMemory(storage_dir=storage)
        assert mem2.get("k", layer=MemoryLayer.SESSION) is None


class TestLayeredMemoryMergeExport:
    def test_merge_from_file(self, tmp_path):
        storage = tmp_path / "mem"
        mem = LayeredMemory(storage_dir=storage)

        merge_file = tmp_path / "import.json"
        merge_file.write_text(json.dumps({"imported_key": "imported_val"}))

        mem.merge_from_file(merge_file, MemoryLayer.PROJECT)
        assert mem.get("imported_key", layer=MemoryLayer.PROJECT) == "imported_val"

    def test_merge_from_file_with_full_entries(self, tmp_path):
        storage = tmp_path / "mem"
        mem = LayeredMemory(storage_dir=storage)

        merge_file = tmp_path / "import.json"
        merge_file.write_text(json.dumps({
            "k": {"value": "v", "tags": ["t"], "metadata": {"src": "imp"}}
        }))

        mem.merge_from_file(merge_file, MemoryLayer.PROJECT)
        entry = mem.get_entry("k", layer=MemoryLayer.PROJECT)
        assert entry.value == "v"
        assert "t" in entry.tags

    def test_export_layer(self, tmp_path):
        mem = LayeredMemory(storage_dir=tmp_path)
        mem.set("k", "v", layer=MemoryLayer.PROJECT)
        exported = mem.export_layer(MemoryLayer.PROJECT)
        assert "k" in exported


class TestLayeredMemorySummary:
    def test_get_summary(self, tmp_path):
        mem = LayeredMemory(storage_dir=tmp_path)
        mem.set("a", 1, layer=MemoryLayer.GLOBAL)
        mem.set("b", 2, layer=MemoryLayer.PROJECT)
        summary = mem.get_summary()
        assert summary["total_entries"] == 2
        assert summary["layers"]["global"] == 1
        assert summary["layers"]["project"] == 1


class TestLayeredMemoryCustomPaths:
    def test_custom_layer_path(self, tmp_path):
        user_path = tmp_path / "custom_user.json"
        mem = LayeredMemory(
            storage_dir=tmp_path / "layers",
            layer_paths={MemoryLayer.USER: user_path},
        )
        mem.set("k", "v", layer=MemoryLayer.USER)
        assert user_path.exists()

    def test_org_project_user_ids(self, tmp_path):
        mem = LayeredMemory(
            storage_dir=tmp_path,
            organization_id="acme",
            project_id="widget",
            user_id="alice",
        )
        mem.set("k", "v", layer=MemoryLayer.ORGANIZATION)
        assert (tmp_path / "org_acme.json").exists()


class TestGetLayeredMemory:
    def test_returns_instance(self):
        import src.core.memory_layers as mod
        mod._layered_memory = None
        mem = get_layered_memory()
        assert isinstance(mem, LayeredMemory)
        mod._layered_memory = None

    def test_singleton(self):
        import src.core.memory_layers as mod
        mod._layered_memory = None
        m1 = get_layered_memory()
        m2 = get_layered_memory()
        assert m1 is m2
        mod._layered_memory = None
