"""Tests for the simplified file-backed memory server."""

import json
import os
import tempfile

from src.servers.memory import (
    memory_delete,
    memory_get,
    memory_list,
    memory_put,
    memory_search,
    get_note_file_path,
    get_user_persona,
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
