import json

import pytest

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


def test_load_config_default_when_no_file(tmp_path, monkeypatch):
    config_module._CONFIG_CACHE.clear()
    monkeypatch.setattr(
        "src.core.config.config.default_config_path", lambda: tmp_path / "nonexistent.json"
    )
    result = config_module.load_config(validate=False)
    assert isinstance(result, dict)


def test_load_config_no_validate(tmp_path):
    config_module._CONFIG_CACHE.clear()
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"model": "test"}), encoding="utf-8")
    result = config_module.load_config(str(config_file), validate=False)
    assert result["model"] == "test"


def test_load_config_invalid_json(tmp_path):
    config_module._CONFIG_CACHE.clear()
    config_file = tmp_path / "bad.json"
    config_file.write_text("{invalid json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        config_module.load_config(str(config_file), validate=False)


def test_load_config_project_fallback(tmp_path, monkeypatch):
    config_module._CONFIG_CACHE.clear()
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"model": "proj"}), encoding="utf-8")
    monkeypatch.setattr("src.core.config.config.default_config_path", lambda: config_file)
    result = config_module.load_config(validate=False)
    assert result["model"] == "proj"


def test_get_config_signature_none_path():
    result = config_module._get_config_signature(None)
    assert result is None


def test_get_config_signature_missing_file():
    result = config_module._get_config_signature(
        type("P", (), {"exists": lambda s: False, "resolve": lambda s: "/x"})()
    )
    assert result is None


def test_get_required_config_value_missing(tmp_path):
    config_module._CONFIG_CACHE.clear()
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"model": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing required"):
        config_module.get_required_config_value("nonexistent_key", override_path=str(config_file))


def test_get_required_config_value_empty_string(tmp_path):
    config_module._CONFIG_CACHE.clear()
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"model": "test", "custom": ""}), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing required"):
        config_module.get_required_config_value("custom", override_path=str(config_file))


def test_get_optional_config_value_with_default(tmp_path):
    config_module._CONFIG_CACHE.clear()
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"model": "x"}), encoding="utf-8")
    result = config_module.get_optional_config_value(
        "missing", default="def", override_path=str(config_file)
    )
    assert result == "def"


def test_get_optional_config_value_present(tmp_path):
    config_module._CONFIG_CACHE.clear()
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"model": "x"}), encoding="utf-8")
    result = config_module.get_optional_config_value("model", override_path=str(config_file))
    assert result == "x"
