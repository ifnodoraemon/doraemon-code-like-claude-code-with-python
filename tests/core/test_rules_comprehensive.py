"""Additional tests for src/core/rules.py"""

import pytest

from src.core import rules as rules_module
from src.core.rules import (
    _cacheable_text_read,
    _combine_instruction_sections,
    _file_signature,
    _find_project_boundary,
    create_default_rules,
    format_instructions_for_prompt,
    format_memory_for_prompt,
    load_all_instructions,
    load_instruction_file,
    load_project_memory,
    load_project_rules,
)


@pytest.fixture(autouse=True)
def _clear_caches():
    rules_module._TEXT_CACHE.clear()
    rules_module._PROJECT_RULES_CACHE.clear()
    rules_module._ALL_INSTRUCTIONS_CACHE.clear()
    yield
    rules_module._TEXT_CACHE.clear()
    rules_module._PROJECT_RULES_CACHE.clear()
    rules_module._ALL_INSTRUCTIONS_CACHE.clear()


class TestFileSignature:
    def test_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        sig = _file_signature(f)
        assert sig is not None
        assert len(sig) == 2

    def test_missing_file(self, tmp_path):
        sig = _file_signature(tmp_path / "missing.txt")
        assert sig is None


class TestCacheableTextRead:
    def test_reads_content(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("content here")
        result = _cacheable_text_read(f)
        assert result == "content here"

    def test_caches_then_invalidates(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("original")
        first = _cacheable_text_read(f)
        assert first == "original"

        rules_module._TEXT_CACHE.clear()
        f.write_text("modified")
        second = _cacheable_text_read(f)
        assert second == "modified"

    def test_missing_file(self, tmp_path):
        result = _cacheable_text_read(tmp_path / "missing.txt")
        assert result is None


class TestFindProjectBoundary:
    def test_finds_git_root(self, tmp_path):
        (tmp_path / ".git").mkdir()
        nested = tmp_path / "src" / "pkg"
        nested.mkdir(parents=True)
        result = _find_project_boundary(nested)
        assert result == tmp_path

    def test_finds_pyproject(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]")
        nested = tmp_path / "sub"
        nested.mkdir()
        result = _find_project_boundary(nested)
        assert result == tmp_path

    def test_falls_back_to_project_dir(self, tmp_path):
        result = _find_project_boundary(tmp_path)
        assert result == tmp_path


class TestCombineInstructionSections:
    def test_empty(self):
        assert _combine_instruction_sections([]) is None

    def test_single_section(self):
        result = _combine_instruction_sections([("Label", "Content")])
        assert "Label" in result
        assert "Content" in result

    def test_multiple_sections(self):
        result = _combine_instruction_sections([("A", "a"), ("B", "b")])
        assert "A" in result
        assert "B" in result
        assert "---" in result


class TestLoadProjectRules:
    def test_no_agents_md(self, tmp_path, monkeypatch):
        (tmp_path / ".git").mkdir()
        monkeypatch.chdir(tmp_path)
        result = load_project_rules(tmp_path)
        assert result is None

    def test_with_root_agents(self, tmp_path, monkeypatch):
        (tmp_path / ".git").mkdir()
        (tmp_path / "AGENTS.md").write_text("root rules")
        monkeypatch.chdir(tmp_path)
        result = load_project_rules(tmp_path)
        assert "root rules" in result

    def test_nested_agents(self, tmp_path, monkeypatch):
        (tmp_path / ".git").mkdir()
        (tmp_path / "AGENTS.md").write_text("root")
        nested = tmp_path / "src"
        nested.mkdir()
        (nested / "AGENTS.md").write_text("nested")
        monkeypatch.chdir(nested)
        result = load_project_rules(nested)
        assert "root" in result
        assert "nested" in result


class TestLoadInstructionFile:
    def test_single_file(self, tmp_path):
        f = tmp_path / "instr.md"
        f.write_text("instruction content")
        result = load_instruction_file(str(f))
        assert "instruction content" in result

    def test_glob_pattern(self, tmp_path):
        (tmp_path / "a.md").write_text("aaa")
        (tmp_path / "b.md").write_text("bbb")
        result = load_instruction_file("*.md", base_dir=tmp_path)
        assert result is not None

    def test_missing_file(self, tmp_path):
        result = load_instruction_file("missing.md", base_dir=tmp_path)
        assert result is None

    def test_no_glob_matches(self, tmp_path):
        result = load_instruction_file("*.xyz", base_dir=tmp_path)
        assert result is None


class TestLoadAllInstructions:
    def test_no_instructions(self, tmp_path, monkeypatch):
        (tmp_path / ".git").mkdir()
        monkeypatch.chdir(tmp_path)
        result = load_all_instructions({}, tmp_path)
        assert result == ""

    def test_with_agents_md(self, tmp_path, monkeypatch):
        (tmp_path / ".git").mkdir()
        (tmp_path / "AGENTS.md").write_text("rules")
        monkeypatch.chdir(tmp_path)
        result = load_all_instructions({}, tmp_path)
        assert "rules" in result

    def test_with_config_instructions(self, tmp_path, monkeypatch):
        (tmp_path / ".git").mkdir()
        (tmp_path / "extra.md").write_text("extra content")
        monkeypatch.chdir(tmp_path)
        result = load_all_instructions({"instructions": ["extra.md"]}, tmp_path)
        assert "extra content" in result


class TestCreateDefaultRules:
    def test_creates_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = create_default_rules(tmp_path)
        assert path.exists()
        content = path.read_text()
        assert "Project Rules" in content


class TestFormatInstructionsForPrompt:
    def test_empty(self):
        assert format_instructions_for_prompt("") == ""

    def test_with_content(self):
        result = format_instructions_for_prompt("my rules")
        assert "PROJECT RULES" in result
        assert "my rules" in result


class TestFormatMemoryForPrompt:
    def test_empty(self):
        assert format_memory_for_prompt("") == ""

    def test_with_content(self):
        result = format_memory_for_prompt("remember this")
        assert "PROJECT MEMORY" in result
        assert "remember this" in result


class TestLoadProjectMemory:
    def test_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = load_project_memory(tmp_path)
        assert result is None

    def test_existing_file(self, tmp_path, monkeypatch):
        from src.core.paths import MEMORY_FILENAME

        memory_dir = tmp_path / ".agent"
        memory_dir.mkdir()
        (memory_dir / MEMORY_FILENAME).write_text("learned: use type hints")
        monkeypatch.chdir(tmp_path)
        result = load_project_memory(tmp_path)
        assert "type hints" in result
