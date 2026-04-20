"""Targeted coverage tests for core.rules - file loading edge cases."""

from pathlib import Path
from unittest.mock import patch

from src.core import rules as rules_module
from src.core.rules import (
    _cacheable_text_read,
    _file_signature,
    _find_project_boundary,
    _read_instruction_file,
    create_default_rules,
    load_instruction_file,
    load_project_memory,
)


class TestFileSignature:
    def test_nonexistent_file(self):
        assert _file_signature(Path("/nonexistent/path/file.txt")) is None

    def test_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        sig = _file_signature(f)
        assert sig is not None
        assert len(sig) == 2


class TestCacheableTextRead:
    def test_nonexistent(self):
        result = _cacheable_text_read(Path("/nonexistent/file.txt"))
        assert result is None

    def test_read_and_cache(self, tmp_path):
        rules_module._TEXT_CACHE.clear()
        f = tmp_path / "test.md"
        f.write_text("cached content")
        result = _cacheable_text_read(f)
        assert result == "cached content"

    def test_cache_hit(self, tmp_path):
        rules_module._TEXT_CACHE.clear()
        f = tmp_path / "cached.md"
        f.write_text("original")
        first = _cacheable_text_read(f)
        assert first == "original"
        f.write_text("updated")
        second = _cacheable_text_read(f)
        assert second == "updated"


class TestReadInstructionFile:
    def test_none_for_missing(self):
        result = _read_instruction_file(Path("/nonexistent"))
        assert result is None


class TestFindProjectBoundary:
    def test_finds_git_root(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".git").mkdir()
        sub = repo / "src" / "pkg"
        sub.mkdir(parents=True)
        result = _find_project_boundary(sub)
        assert result == repo

    def test_finds_pyproject_root(self, tmp_path):
        repo = tmp_path / "repo2"
        repo.mkdir()
        (repo / "pyproject.toml").write_text("")
        sub = repo / "lib"
        sub.mkdir()
        result = _find_project_boundary(sub)
        assert result == repo

    def test_no_marker_falls_back(self, tmp_path):
        d = tmp_path / "plain"
        d.mkdir()
        result = _find_project_boundary(d)
        assert result == d


class TestLoadInstructionFileGlobNoContent:
    def test_glob_matches_unreadable(self, tmp_path):
        (tmp_path / "a.md").write_text("content")
        with patch("src.core.rules._cacheable_text_read", return_value=None):
            result = load_instruction_file("*.md", base_dir=tmp_path)
            assert result is None


class TestLoadProjectMemoryEdge:
    def test_memory_read_fails(self, tmp_path):
        rules_module._TEXT_CACHE.clear()
        (tmp_path / ".agent").mkdir()
        mem = tmp_path / ".agent" / "MEMORY.md"
        mem.write_text("memory content")
        with patch("src.core.rules._cacheable_text_read", return_value=None):
            result = load_project_memory(tmp_path)
            assert result is None

    def test_memory_dir_missing(self, tmp_path):
        result = load_project_memory(tmp_path)
        assert result is None


class TestCreateDefaultRulesExisting:
    def test_creates_file_in_project_dir(self, tmp_path):
        path = create_default_rules(tmp_path)
        assert path.exists()
        content = path.read_text()
        assert "Project Rules" in content
