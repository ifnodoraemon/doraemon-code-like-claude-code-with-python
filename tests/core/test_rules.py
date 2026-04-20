from pathlib import Path

from src.core import rules as rules_module
from src.core.rules import (
    _combine_instruction_sections,
    create_default_rules,
    format_instructions_for_prompt,
    format_memory_for_prompt,
    load_all_instructions,
    load_instruction_file,
    load_project_memory,
    load_project_rules,
)


def test_load_project_rules_supports_agents_hierarchy(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    nested = repo / "src" / "feature"
    nested.mkdir(parents=True)
    (repo / ".git").mkdir()
    (repo / "AGENTS.md").write_text("root agent instructions", encoding="utf-8")
    (repo / "src" / "AGENTS.md").write_text("nested agent instructions", encoding="utf-8")

    monkeypatch.chdir(nested)

    content = load_project_rules(nested)

    assert content is not None
    assert "root agent instructions" in content
    assert "nested agent instructions" in content
    assert content.index("root agent instructions") < content.index("nested agent instructions")


def test_load_all_instructions_cache_invalidates_when_agents_changes(tmp_path, monkeypatch):
    rules_module._TEXT_CACHE.clear()
    rules_module._PROJECT_RULES_CACHE.clear()
    rules_module._ALL_INSTRUCTIONS_CACHE.clear()

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    agents = repo / "AGENTS.md"
    agents.write_text("first rules", encoding="utf-8")

    monkeypatch.chdir(repo)

    first = load_all_instructions({}, repo)
    assert "first rules" in first

    agents.write_text("second rules", encoding="utf-8")

    second = load_all_instructions({}, repo)
    assert "second rules" in second


def test_load_project_rules_returns_none_when_no_agents(tmp_path, monkeypatch):
    rules_module._TEXT_CACHE.clear()
    rules_module._PROJECT_RULES_CACHE.clear()
    rules_module._ALL_INSTRUCTIONS_CACHE.clear()
    monkeypatch.chdir(tmp_path)
    result = load_project_rules(tmp_path)
    assert result is None


def test_combine_instruction_sections_empty():
    assert _combine_instruction_sections([]) is None


def test_combine_instruction_sections_nonempty():
    result = _combine_instruction_sections([("A", "content a"), ("B", "content b")])
    assert "content a" in result
    assert "content b" in result


def test_format_instructions_for_prompt_empty():
    assert format_instructions_for_prompt("") == ""


def test_format_instructions_for_prompt_nonempty():
    result = format_instructions_for_prompt("some rules")
    assert "some rules" in result
    assert "PROJECT RULES" in result


def test_format_memory_for_prompt_empty():
    assert format_memory_for_prompt("") == ""


def test_format_memory_for_prompt_nonempty():
    result = format_memory_for_prompt("some memory")
    assert "some memory" in result
    assert "PROJECT MEMORY" in result


def test_load_instruction_file_glob_pattern(tmp_path):
    (tmp_path / "a.md").write_text("file a content")
    (tmp_path / "b.md").write_text("file b content")
    result = load_instruction_file("*.md", base_dir=tmp_path)
    assert result is not None
    assert "file a content" in result or "file b content" in result


def test_load_instruction_file_glob_no_matches(tmp_path):
    result = load_instruction_file("*.xyz", base_dir=tmp_path)
    assert result is None


def test_load_instruction_file_nonexistent():
    result = load_instruction_file("/nonexistent/path.md")
    assert result is None


def test_load_instruction_file_absolute_path(tmp_path):
    f = tmp_path / "test.md"
    f.write_text("absolute content")
    result = load_instruction_file(str(f))
    assert result == "absolute content"


def test_load_instruction_file_absolute_nonexistent():
    result = load_instruction_file("/absolutely/nonexistent/file.md")
    assert result is None


def test_load_all_instructions_with_additional_files(tmp_path, monkeypatch):
    rules_module._TEXT_CACHE.clear()
    rules_module._PROJECT_RULES_CACHE.clear()
    rules_module._ALL_INSTRUCTIONS_CACHE.clear()
    (tmp_path / ".git").mkdir()
    extra = tmp_path / "extra.md"
    extra.write_text("extra instructions")
    result = load_all_instructions({"instructions": ["extra.md"]}, tmp_path)
    assert "extra instructions" in result


def test_load_all_instructions_empty(tmp_path, monkeypatch):
    rules_module._TEXT_CACHE.clear()
    rules_module._PROJECT_RULES_CACHE.clear()
    rules_module._ALL_INSTRUCTIONS_CACHE.clear()
    monkeypatch.chdir(tmp_path)
    result = load_all_instructions({}, tmp_path)
    assert result == ""


def test_create_default_rules(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = create_default_rules(tmp_path)
    assert path.exists()
    content = path.read_text()
    assert "Project Rules" in content


def test_load_project_memory_exists(tmp_path, monkeypatch):
    rules_module._TEXT_CACHE.clear()
    (tmp_path / ".agent").mkdir()
    (tmp_path / ".agent" / "MEMORY.md").write_text("my memory")
    result = load_project_memory(tmp_path)
    assert result == "my memory"


def test_load_project_memory_missing(tmp_path):
    result = load_project_memory(tmp_path)
    assert result is None
