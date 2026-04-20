"""Tests for the simplified file-backed memory server."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

from src.servers.memory import (
    _deserialize_note,
    _keyword_search_notes,
    _load_note_by_title,
    _load_project_notes,
    _notes_dir,
    _notes_root,
    _resolve_note_path,
    _serialize_note,
    _slugify,
    _validate_note_path,
    export_notes,
    get_note_file_path,
    get_user_persona,
    memory_delete,
    memory_get,
    memory_list,
    memory_put,
    memory_search,
    update_user_persona,
)


def _chdir_temp():
    tmpdir = tempfile.TemporaryDirectory()
    original_cwd = os.getcwd()
    os.chdir(tmpdir.name)
    return tmpdir, original_cwd


class TestMemoryServer:
    def test_save_and_get_note(self):
        tmpdir, original_cwd = _chdir_temp()
        try:
            result = memory_put("Test Note", "This is test content", "default", ["python"])
            assert "saved" in result.lower()

            saved = memory_get("Test Note", "default")
            assert "Test Note" in saved
            assert "This is test content" in saved
        finally:
            os.chdir(original_cwd)
            tmpdir.cleanup()

    def test_search_notes_uses_keyword_search(self):
        tmpdir, original_cwd = _chdir_temp()
        try:
            memory_put("Release Plan", "glm5 rollout checklist", "default")
            memory_put("Other", "unrelated content", "default")

            result = memory_search("rollout", "default", 5)
            assert "Release Plan" in result
            assert "glm5 rollout checklist" in result
            assert "Other" not in result
        finally:
            os.chdir(original_cwd)
            tmpdir.cleanup()

    def test_delete_note_removes_file(self):
        tmpdir, original_cwd = _chdir_temp()
        try:
            memory_put("Disposable", "to be deleted", "default")
            path = get_note_file_path("Disposable", "default")
            assert os.path.exists(path)

            result = memory_delete("Disposable", "default")
            assert "deleted" in result.lower()
            assert not os.path.exists(path)
        finally:
            os.chdir(original_cwd)
            tmpdir.cleanup()

    def test_list_notes_reads_from_files_only(self):
        tmpdir, original_cwd = _chdir_temp()
        try:
            memory_put("One", "alpha", "default", ["a"])
            memory_put("Two", "beta", "default")

            result = memory_list("default", 10)
            assert "One" in result
            assert "Two" in result
            assert "Tags: a" in result
        finally:
            os.chdir(original_cwd)
            tmpdir.cleanup()

    def test_update_and_get_user_persona(self):
        tmpdir, original_cwd = _chdir_temp()
        try:
            result = update_user_persona("name", "John Doe")
            assert "remembered" in result.lower()

            persona = json.loads(get_user_persona())
            assert persona["name"] == "John Doe"
        finally:
            os.chdir(original_cwd)
            tmpdir.cleanup()


class TestResolveNotePath:
    def test_returns_existing_note_path(self):
        tmpdir, original_cwd = _chdir_temp()
        try:
            memory_put("My Note", "content", "default")
            path = _resolve_note_path("default", "My Note", for_write=False)
            assert path.exists()
        finally:
            os.chdir(original_cwd)
            tmpdir.cleanup()

    def test_returns_base_path_when_no_collision(self):
        tmpdir, original_cwd = _chdir_temp()
        try:
            path = _resolve_note_path("default", "New Note", for_write=False)
            assert "new-note" in str(path)
        finally:
            os.chdir(original_cwd)
            tmpdir.cleanup()

    def test_hash_collision_path(self):
        tmpdir, original_cwd = _chdir_temp()
        try:
            notes_dir = _notes_dir("default")
            notes_dir.mkdir(parents=True, exist_ok=True)
            title = "collision-xyz"
            slug = _slugify(title)
            base_path = notes_dir / f"{slug}.md"
            base_path.write_text(
                "---\n"
                '{"title": "different-title", "project": "default", "tags": [], "updated_at": ""}\n'
                "---\n\nother content"
            )
            path = _resolve_note_path("default", title, for_write=True)
            assert path != base_path
            assert "-" in path.stem and path.stem != slug
        finally:
            os.chdir(original_cwd)
            tmpdir.cleanup()

    def test_counter_collision_path(self):
        tmpdir, original_cwd = _chdir_temp()
        try:
            notes_dir = _notes_dir("default")
            notes_dir.mkdir(parents=True, exist_ok=True)
            title = "collision-abc"
            slug = _slugify(title)
            base_path = notes_dir / f"{slug}.md"
            base_path.write_text(
                "---\n"
                '{"title": "other-title", "project": "default", "tags": [], "updated_at": ""}\n'
                "---\n\ncontent"
            )
            title_hash = hashlib.sha256(title.encode("utf-8")).hexdigest()[:8]
            hash_path = notes_dir / f"{slug}-{title_hash}.md"
            hash_path.write_text("existing")
            counter2_path = notes_dir / f"{slug}-{title_hash}-2.md"
            counter2_path.write_text("existing2")
            path = _resolve_note_path("default", title, for_write=True)
            assert "-3" in str(path)
        finally:
            os.chdir(original_cwd)
            tmpdir.cleanup()


class TestDeserializeNote:
    def test_os_error_returns_none(self, tmp_path):
        result = _deserialize_note(tmp_path / "nonexistent.md")
        assert result is None

    def test_no_delimiter_fallback(self, tmp_path):
        f = tmp_path / "plain.md"
        f.write_text("just plain text")
        result = _deserialize_note(f)
        assert result is not None
        assert result["title"] == "plain"
        assert result["content"] == "just plain text"

    def test_malformed_delimiter_returns_none(self, tmp_path):
        f = tmp_path / "bad.md"
        f.write_text("---\nnot json\n---\ncontent")
        result = _deserialize_note(f)
        assert result is None

    def test_missing_closing_delimiter_returns_none(self, tmp_path):
        f = tmp_path / "noclose.md"
        f.write_text("---\n{\"title\": \"x\"}")
        result = _deserialize_note(f)
        assert result is None


class TestSerializeNote:
    def test_roundtrip(self):
        text = _serialize_note("Title", "Body", "proj", ["a", "b"])
        assert "---" in text
        assert '"Title"' in text
        assert "Body" in text
        assert '"a"' in text


class TestKeywordSearch:
    def test_empty_query_returns_first_n(self):
        tmpdir, original_cwd = _chdir_temp()
        try:
            memory_put("Alpha", "aaa", "default")
            memory_put("Beta", "bbb", "default")
            results = _keyword_search_notes("", "default", 5)
            assert len(results) == 2
        finally:
            os.chdir(original_cwd)
            tmpdir.cleanup()


class TestMemoryGetNotFound:
    def test_get_missing_note(self):
        tmpdir, original_cwd = _chdir_temp()
        try:
            result = memory_get("nonexistent", "default")
            assert "not found" in result.lower()
        finally:
            os.chdir(original_cwd)
            tmpdir.cleanup()


class TestExportNotes:
    def test_export_empty(self):
        tmpdir, original_cwd = _chdir_temp()
        try:
            result = export_notes("default")
            assert "No notes" in result
        finally:
            os.chdir(original_cwd)
            tmpdir.cleanup()

    def test_export_json(self):
        tmpdir, original_cwd = _chdir_temp()
        try:
            memory_put("Exp", "data", "default")
            result = export_notes("default", "json")
            parsed = json.loads(result)
            assert isinstance(parsed, list)
        finally:
            os.chdir(original_cwd)
            tmpdir.cleanup()

    def test_export_markdown(self):
        tmpdir, original_cwd = _chdir_temp()
        try:
            memory_put("Exp", "data", "default")
            result = export_notes("default", "markdown")
            assert "# Exp" in result
        finally:
            os.chdir(original_cwd)
            tmpdir.cleanup()

    def test_export_unsupported_format(self):
        tmpdir, original_cwd = _chdir_temp()
        try:
            memory_put("Exp", "data", "default")
            result = export_notes("default", "csv")
            assert "Unsupported" in result
        finally:
            os.chdir(original_cwd)
            tmpdir.cleanup()


class TestMemoryPutTooLong:
    def test_content_too_long(self):
        tmpdir, original_cwd = _chdir_temp()
        try:
            result = memory_put("Big", "x" * 100001, "default")
            assert "too long" in result.lower()
        finally:
            os.chdir(original_cwd)
            tmpdir.cleanup()


class TestMemorySearchNoResults:
    def test_search_no_matches(self):
        tmpdir, original_cwd = _chdir_temp()
        try:
            memory_put("Alpha", "aaa", "default")
            result = memory_search("zzz", "default", 5)
            assert "not found" in result.lower()
        finally:
            os.chdir(original_cwd)
            tmpdir.cleanup()


class TestMemoryDeleteNotFound:
    def test_delete_missing(self):
        tmpdir, original_cwd = _chdir_temp()
        try:
            result = memory_delete("ghost", "default")
            assert "not found" in result.lower()
        finally:
            os.chdir(original_cwd)
            tmpdir.cleanup()


class TestValidateNotePath:
    def test_valid_path(self):
        tmpdir, original_cwd = _chdir_temp()
        try:
            notes_dir = _notes_dir("default")
            notes_dir.mkdir(parents=True, exist_ok=True)
            target = notes_dir / "test.md"
            target.write_text("x")
            assert _validate_note_path(target, "default") is True
        finally:
            os.chdir(original_cwd)
            tmpdir.cleanup()

    def test_invalid_path(self):
        tmpdir, original_cwd = _chdir_temp()
        try:
            assert _validate_note_path(Path("/etc/passwd"), "default") is False
        finally:
            os.chdir(original_cwd)
            tmpdir.cleanup()


class TestMemoryListEmpty:
    def test_list_empty_project(self):
        tmpdir, original_cwd = _chdir_temp()
        try:
            result = memory_list("default", 10)
            assert "No notes" in result
        finally:
            os.chdir(original_cwd)
            tmpdir.cleanup()


class TestLoadProjectNotesEmpty:
    def test_no_notes_dir(self):
        tmpdir, original_cwd = _chdir_temp()
        try:
            result = _load_project_notes("nonexistent")
            assert result == []
        finally:
            os.chdir(original_cwd)
            tmpdir.cleanup()
