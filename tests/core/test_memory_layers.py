"""Tests for LayeredMemory path overrides and simple layer compatibility."""

import json

from src.core.memory_layers import LayeredMemory, MemoryEntry, MemoryLayer, get_layered_memory


class TestLayeredMemoryCompatibility:
    def test_simple_layer_round_trip_uses_flat_json(self, tmp_path):
        user_path = tmp_path / "memory.json"
        memory = LayeredMemory(
            storage_dir=tmp_path / "layers",
            layer_paths={MemoryLayer.USER: user_path},
            simple_layers={MemoryLayer.USER},
        )

        memory.set("role", "engineer", layer=MemoryLayer.USER)

        saved = json.loads(user_path.read_text(encoding="utf-8"))
        assert saved == {"role": "engineer"}

    def test_simple_layer_loads_legacy_flat_json(self, tmp_path):
        user_path = tmp_path / "memory.json"
        user_path.write_text(json.dumps({"name": "Ada"}), encoding="utf-8")

        memory = LayeredMemory(
            storage_dir=tmp_path / "layers",
            layer_paths={MemoryLayer.USER: user_path},
            simple_layers={MemoryLayer.USER},
        )

        assert memory.get("name", layer=MemoryLayer.USER) == "Ada"


class TestMemoryEntryExpired:
    def test_is_expired_with_future_expiry(self):
        entry = MemoryEntry(
            key="k", value="v", layer=MemoryLayer.SESSION,
            created_at=0, updated_at=0, expires_at=9999999999,
        )
        assert entry.is_expired() is False

    def test_is_expired_with_past_expiry(self):
        entry = MemoryEntry(
            key="k", value="v", layer=MemoryLayer.SESSION,
            created_at=0, updated_at=0, expires_at=1,
        )
        assert entry.is_expired() is True


class TestLayeredMemorySessionLayer:
    def test_session_layer_not_persisted(self, tmp_path):
        memory = LayeredMemory(storage_dir=tmp_path)
        memory.set("temp", "val", layer=MemoryLayer.SESSION)
        assert memory.get("temp", layer=MemoryLayer.SESSION) == "val"
        session_path = tmp_path / "session.json"
        assert not session_path.exists()

    def test_get_entry_returns_entry(self, tmp_path):
        memory = LayeredMemory(storage_dir=tmp_path)
        memory.set("k", "v", layer=MemoryLayer.PROJECT)
        entry = memory.get_entry("k", layer=MemoryLayer.PROJECT)
        assert entry is not None
        assert entry.value == "v"

    def test_get_entry_returns_none_for_expired(self, tmp_path):
        memory = LayeredMemory(storage_dir=tmp_path)
        memory.set("k", "v", layer=MemoryLayer.PROJECT, ttl=-1)
        entry = memory.get_entry("k", layer=MemoryLayer.PROJECT)
        assert entry is None

    def test_delete_from_all_layers(self, tmp_path):
        memory = LayeredMemory(storage_dir=tmp_path)
        memory.set("k", "global_v", layer=MemoryLayer.GLOBAL)
        memory.set("k", "session_v", layer=MemoryLayer.SESSION)
        memory.delete("k")
        assert memory.get("k") is None

    def test_clear_session(self, tmp_path):
        memory = LayeredMemory(storage_dir=tmp_path)
        memory.set("k", "v", layer=MemoryLayer.SESSION)
        memory.clear_session()
        assert memory.get("k", layer=MemoryLayer.SESSION) is None

    def test_export_layer(self, tmp_path):
        memory = LayeredMemory(storage_dir=tmp_path)
        memory.set("k", "v", layer=MemoryLayer.PROJECT)
        exported = memory.export_layer(MemoryLayer.PROJECT)
        assert "k" in exported

    def test_get_summary(self, tmp_path):
        memory = LayeredMemory(storage_dir=tmp_path)
        memory.set("k", "v", layer=MemoryLayer.SESSION)
        summary = memory.get_summary()
        assert summary["total_entries"] == 1

    def test_has_key(self, tmp_path):
        memory = LayeredMemory(storage_dir=tmp_path)
        memory.set("k", "v", layer=MemoryLayer.SESSION)
        assert memory.has("k") is True
        assert memory.has("nonexistent") is False

    def test_list_keys(self, tmp_path):
        memory = LayeredMemory(storage_dir=tmp_path)
        memory.set("a", "1", layer=MemoryLayer.SESSION)
        memory.set("b", "2", layer=MemoryLayer.SESSION)
        keys = memory.list_keys()
        assert "a" in keys
        assert "b" in keys

    def test_merge_from_file(self, tmp_path):
        memory = LayeredMemory(storage_dir=tmp_path)
        merge_file = tmp_path / "merge.json"
        merge_file.write_text(json.dumps({"x": "merged_val"}))
        memory.merge_from_file(merge_file, MemoryLayer.SESSION)
        assert memory.get("x", layer=MemoryLayer.SESSION) == "merged_val"

    def test_merge_from_file_full_entry_format(self, tmp_path):
        memory = LayeredMemory(storage_dir=tmp_path)
        merge_file = tmp_path / "merge.json"
        merge_file.write_text(json.dumps({
            "y": {"value": "full_val", "tags": ["t1"], "metadata": {"m": 1}}
        }))
        memory.merge_from_file(merge_file, MemoryLayer.SESSION)
        assert memory.get("y", layer=MemoryLayer.SESSION) == "full_val"

    def test_merge_from_nonexistent_file(self, tmp_path):
        memory = LayeredMemory(storage_dir=tmp_path)
        memory.merge_from_file(tmp_path / "nope.json", MemoryLayer.SESSION)

    def test_merge_from_invalid_file(self, tmp_path):
        memory = LayeredMemory(storage_dir=tmp_path)
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json")
        memory.merge_from_file(bad_file, MemoryLayer.SESSION)

    def test_get_layer_path_session(self, tmp_path):
        memory = LayeredMemory(storage_dir=tmp_path)
        path = memory._get_layer_path(MemoryLayer.SESSION)
        assert path.name == "session.json"

    def test_save_layer_session_noop(self, tmp_path):
        memory = LayeredMemory(storage_dir=tmp_path)
        memory.set("k", "v", layer=MemoryLayer.SESSION)
        memory._save_layer(MemoryLayer.SESSION)
        assert not (tmp_path / "session.json").exists()

    def test_load_layer_invalid_data(self, tmp_path):
        memory = LayeredMemory(storage_dir=tmp_path)
        bad_file = tmp_path / "project_default.json"
        bad_file.write_text("not json")
        memory._load_layer(MemoryLayer.PROJECT)

    def test_get_layered_memory(self):
        m1 = get_layered_memory()
        m2 = get_layered_memory()
        assert m1 is m2
