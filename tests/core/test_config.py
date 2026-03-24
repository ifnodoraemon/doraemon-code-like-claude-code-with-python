import json

from src.core.config import config as config_module


def test_load_config_uses_cache_and_invalidates_on_change(tmp_path):
    config_module._CONFIG_CACHE.clear()

    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"model": "first"}), encoding="utf-8")

    first = config_module.load_config(str(config_file), validate=False)
    second = config_module.load_config(str(config_file), validate=False)

    assert first["model"] == "first"
    assert second["model"] == "first"

    config_file.write_text(json.dumps({"model": "second"}), encoding="utf-8")

    third = config_module.load_config(str(config_file), validate=False)

    assert third["model"] == "second"
