"""
Comprehensive tests for the Memory Server (src/servers/memory.py).

Tests cover:
- Memory storage and retrieval functions
- Vector search and embeddings
- ChromaDB operations with mocking
- Success and error cases
- User persona management
- Collection management
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

# Import the memory server functions
from src.servers.memory import (
    delete_note,
    get_note,
    get_note_file_path,
    get_user_persona,
    save_note,
    search_notes,
    update_user_persona,
)

# ========================================
# Fixtures
# ========================================


@pytest.fixture
def temp_memory_dir():
    """Create a temporary directory for memory files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create .agent subdirectory
        agent_dir = os.path.join(tmpdir, ".agent")
        os.makedirs(agent_dir, exist_ok=True)

        original_cwd = os.getcwd()
        os.chdir(tmpdir)

        yield tmpdir

        os.chdir(original_cwd)


@pytest.fixture
def mock_embedding_function():
    """Create a mock embedding function."""
    mock_fn = MagicMock()
    mock_fn.provider = "google"
    return mock_fn


@pytest.fixture
def mock_chromadb_collection():
    """Create a mock ChromaDB collection."""
    mock_collection = MagicMock()
    mock_collection.add = MagicMock()
    mock_collection.query = MagicMock()
    mock_collection.delete = MagicMock()
    mock_collection.get = MagicMock()
    return mock_collection


@pytest.fixture
def mock_chromadb_client(mock_chromadb_collection):
    """Create a mock ChromaDB client."""
    mock_client = MagicMock()
    mock_client.get_or_create_collection = MagicMock(return_value=mock_chromadb_collection)
    return mock_client


# ========================================
# Save Note Tests
# ========================================


class TestSaveNote:
    """Tests for save_note function."""

    @patch("src.servers.memory.collection")
    def test_save_note_success(self, mock_collection):
        """Test successfully saving a note."""
        mock_collection.add = MagicMock()

        result = save_note(
            title="Test Note", content="This is test content", collection_name="default"
        )

        assert "已保存" in result or "saved" in result.lower()
        mock_collection.add.assert_called_once()

        # Verify the call arguments
        call_args = mock_collection.add.call_args
        assert call_args[1]["documents"] == ["This is test content"]
        assert call_args[1]["ids"][0].startswith("default_Test Note_")

    @patch("src.servers.memory.collection")
    def test_save_note_with_tags(self, mock_collection):
        """Test saving a note with tags."""
        mock_collection.add = MagicMock()

        result = save_note(
            title="Tagged Note",
            content="Content with tags",
            collection_name="default",
            tags=["python", "testing", "mocking"],
        )

        assert "已保存" in result or "saved" in result.lower()
        mock_collection.add.assert_called_once()

        # Verify tags are in metadata
        call_args = mock_collection.add.call_args
        metadata = call_args[1]["metadatas"][0]
        assert "python,testing,mocking" in metadata["tags"]

    @patch("src.servers.memory.collection")
    def test_save_note_with_custom_collection(self, mock_collection):
        """Test saving a note to a custom collection."""
        mock_collection.add = MagicMock()

        result = save_note(
            title="Custom Collection Note",
            content="Content for custom collection",
            collection_name="my_project",
        )

        assert "已保存" in result or "saved" in result.lower()

        # Verify collection_name is in metadata
        call_args = mock_collection.add.call_args
        metadata = call_args[1]["metadatas"][0]
        assert metadata["project"] == "my_project"

    @patch("src.servers.memory.collection", None)
    def test_save_note_without_vector_index_still_persists_file(self, temp_memory_dir):
        """Test saving a note still works when vector indexing is unavailable."""
        result = save_note(title="Test", content="Content", collection_name="default")

        assert "saved" in result.lower()
        note_file = os.path.join(temp_memory_dir, ".agent", "memory", "notes", "default", "test.md")
        assert os.path.exists(note_file)

    @patch("src.servers.memory.collection")
    def test_save_note_with_exception(self, mock_collection):
        """Test saving a note still succeeds when vector sync fails."""
        mock_collection.add.side_effect = Exception("Database error")

        result = save_note(title="Error Note", content="This will fail", collection_name="default")

        assert "已保存" in result or "saved" in result.lower()

    @patch("src.servers.memory.collection")
    def test_save_note_empty_tags(self, mock_collection):
        """Test saving a note with empty tags list."""
        mock_collection.add = MagicMock()

        result = save_note(
            title="No Tags", content="Content without tags", collection_name="default", tags=[]
        )

        assert "已保存" in result or "saved" in result.lower()

        # Verify tags are empty string
        call_args = mock_collection.add.call_args
        metadata = call_args[1]["metadatas"][0]
        assert metadata["tags"] == ""

    @patch("src.servers.memory.collection")
    def test_save_note_none_tags_defaults_to_empty(self, mock_collection):
        """Test that None tags defaults to empty list."""
        mock_collection.add = MagicMock()

        result = save_note(
            title="Default Tags", content="Content", collection_name="default", tags=None
        )

        assert "已保存" in result or "saved" in result.lower()

        # Verify tags are empty string
        call_args = mock_collection.add.call_args
        metadata = call_args[1]["metadatas"][0]
        assert metadata["tags"] == ""

    @patch("src.servers.memory.collection")
    def test_save_note_special_characters_in_title(self, mock_collection):
        """Test saving a note with special characters in title."""
        mock_collection.add = MagicMock()

        result = save_note(
            title="Note with 特殊字符 & symbols!", content="Content", collection_name="default"
        )

        assert "已保存" in result or "saved" in result.lower()
        mock_collection.add.assert_called_once()

    @patch("src.servers.memory.collection")
    def test_save_note_long_content(self, mock_collection):
        """Test saving a note with very long content."""
        mock_collection.add = MagicMock()

        long_content = "x" * 10000
        result = save_note(title="Long Note", content=long_content, collection_name="default")

        assert "已保存" in result or "saved" in result.lower()

        # Verify content is stored
        call_args = mock_collection.add.call_args
        assert call_args[1]["documents"][0] == long_content

    @patch("src.servers.memory.collection")
    def test_save_note_id_generation(self, mock_collection):
        """Test that note IDs are generated correctly."""
        mock_collection.add = MagicMock()

        save_note(title="Test", content="Content", collection_name="project1")

        call_args = mock_collection.add.call_args
        note_id = call_args[1]["ids"][0]

        # ID should start with collection_name_title_
        assert note_id.startswith("project1_Test_")
        # ID should include hash of content
        assert len(note_id) > len("project1_Test_")

    @patch("src.servers.memory.collection", None)
    def test_save_note_slug_collision_uses_distinct_files(self, temp_memory_dir):
        """Test notes with colliding slugs do not overwrite each other."""
        first = save_note(title="A/B", content="first", collection_name="default")
        second = save_note(title="A B", content="second", collection_name="default")

        assert "saved" in first.lower()
        assert "saved" in second.lower()

        notes_dir = os.path.join(temp_memory_dir, ".agent", "memory", "notes", "default")
        saved_files = sorted(os.listdir(notes_dir))
        assert len(saved_files) == 2
        assert "a-b.md" in saved_files
        assert any(name.startswith("a-b-") and name.endswith(".md") for name in saved_files)

        assert "first" in get_note("A/B", collection_name="default")
        assert "second" in get_note("A B", collection_name="default")


# ========================================
# Search Notes Tests
# ========================================


class TestSearchNotes:
    """Tests for search_notes function."""

    @patch("src.servers.memory.collection")
    def test_search_notes_success(self, mock_collection):
        """Test successfully searching for notes."""
        mock_collection.query.return_value = {
            "documents": [["Document 1", "Document 2"]],
            "metadatas": [[{"title": "Note 1"}, {"title": "Note 2"}]],
        }

        result = search_notes(query="test query", collection_name="default", n_results=3)

        assert "Note 1" in result
        assert "Document 1" in result
        mock_collection.query.assert_called_once()

    @patch("src.servers.memory.collection")
    def test_search_notes_with_custom_collection(self, mock_collection):
        """Test searching notes in a custom collection."""
        mock_collection.query.return_value = {
            "documents": [["Result"]],
            "metadatas": [[{"title": "Result"}]],
        }

        search_notes(query="search", collection_name="my_project", n_results=5)

        # Verify the where clause filters by collection_name
        call_args = mock_collection.query.call_args
        assert call_args[1]["where"]["project"] == "my_project"

    @patch("src.servers.memory.collection")
    def test_search_notes_custom_n_results(self, mock_collection):
        """Test searching with custom number of results."""
        mock_collection.query.return_value = {
            "documents": [["Result"]],
            "metadatas": [[{"title": "Result"}]],
        }

        search_notes(query="search", collection_name="default", n_results=10)

        # Verify n_results is passed correctly
        call_args = mock_collection.query.call_args
        assert call_args[1]["n_results"] == 10

    @patch("src.servers.memory.collection")
    def test_search_notes_no_results(self, mock_collection):
        """Test searching when no results are found."""
        mock_collection.query.return_value = {"documents": [[]], "metadatas": [[]]}

        result = search_notes(query="nonexistent", collection_name="default")

        assert "未找到" in result or "not found" in result.lower()

    @patch("src.servers.memory.collection")
    def test_search_notes_empty_documents(self, mock_collection):
        """Test searching when documents list is empty."""
        mock_collection.query.return_value = {"documents": [None], "metadatas": [None]}

        result = search_notes(query="search", collection_name="default")

        assert "未找到" in result or "not found" in result.lower()

    @patch("src.servers.memory.collection")
    def test_search_notes_exception(self, mock_collection):
        """Test searching degrades gracefully when vector query fails."""
        mock_collection.query.side_effect = Exception("Query error")

        result = search_notes(query="search", collection_name="default")

        assert "not found" in result.lower()

    @patch("src.servers.memory.collection", None)
    def test_search_notes_without_vector_index_uses_files(self, temp_memory_dir):
        """Test searching falls back to persisted note files without vectors."""
        save_note(title="Searchable", content="search fallback content", collection_name="default")

        result = search_notes(query="fallback", collection_name="default")

        assert "Searchable" in result
        assert "fallback content" in result

    @patch("src.servers.memory.collection")
    def test_search_notes_prefers_file_content_over_stale_vector(
        self, mock_collection, temp_memory_dir
    ):
        """Test search uses file-backed content when vector content is stale."""
        save_note(title="Release Plan", content="updated file content", collection_name="default")
        mock_collection.query.return_value = {
            "documents": [["outdated vector content"]],
            "metadatas": [[{"title": "Release Plan", "tags": "", "project": "default"}]],
        }

        result = search_notes(query="release", collection_name="default")

        assert "updated file content" in result
        assert "outdated vector content" not in result

    @patch("src.servers.memory.collection")
    def test_search_notes_multiple_results(self, mock_collection):
        """Test searching with multiple results."""
        mock_collection.query.return_value = {
            "documents": [["Document 1", "Document 2", "Document 3"]],
            "metadatas": [[{"title": "Title 1"}, {"title": "Title 2"}, {"title": "Title 3"}]],
        }

        result = search_notes(query="search", collection_name="default", n_results=3)

        assert "Title 1" in result
        assert "Title 2" in result
        assert "Title 3" in result
        assert "Document 1" in result
        assert "Document 2" in result
        assert "Document 3" in result

    @patch("src.servers.memory.collection")
    def test_search_notes_query_text_passed(self, mock_collection):
        """Test that query text is passed correctly to ChromaDB."""
        mock_collection.query.return_value = {
            "documents": [["Result"]],
            "metadatas": [[{"title": "Result"}]],
        }

        search_notes(query="my search query", collection_name="default")

        # Verify query_texts is passed correctly
        call_args = mock_collection.query.call_args
        assert call_args[1]["query_texts"] == ["my search query"]

    @patch("src.servers.memory.collection")
    def test_search_notes_result_formatting(self, mock_collection):
        """Test that search results are formatted correctly."""
        mock_collection.query.return_value = {
            "documents": [["Content of note"]],
            "metadatas": [[{"title": "My Note"}]],
        }

        result = search_notes(query="search", collection_name="default")

        # Verify formatting includes title and content
        assert "[标题: My Note]" in result
        assert "Content of note" in result


# ========================================
# Update User Persona Tests
# ========================================


class TestUpdateUserPersona:
    """Tests for update_user_persona function."""

    def test_update_user_persona_new_file(self, temp_memory_dir):
        """Test updating user persona when file doesn't exist."""
        result = update_user_persona("name", "John Doe")

        assert "已记住" in result or "remembered" in result.lower()
        assert "name = John Doe" in result

        # Verify file was created
        memory_file = os.path.join(temp_memory_dir, ".agent", "memory.json")
        assert os.path.exists(memory_file)

    def test_update_user_persona_existing_file(self, temp_memory_dir):
        """Test updating user persona when file already exists."""
        # Create initial memory file
        memory_file = os.path.join(temp_memory_dir, ".agent", "memory.json")
        initial_data = {"job": "Engineer"}
        with open(memory_file, "w") as f:
            json.dump(initial_data, f)

        # Update with new key
        result = update_user_persona("name", "Jane Doe")

        assert "已记住" in result or "remembered" in result.lower()

        # Verify both keys exist
        with open(memory_file) as f:
            data = json.load(f)
        assert data["job"] == "Engineer"
        assert data["name"] == "Jane Doe"

    def test_update_user_persona_overwrite_existing_key(self, temp_memory_dir):
        """Test overwriting an existing persona key."""
        memory_file = os.path.join(temp_memory_dir, ".agent", "memory.json")
        initial_data = {"name": "Old Name"}
        with open(memory_file, "w") as f:
            json.dump(initial_data, f)

        # Update existing key
        result = update_user_persona("name", "New Name")

        assert "已记住" in result or "remembered" in result.lower()

        # Verify key was updated
        with open(memory_file) as f:
            data = json.load(f)
        assert data["name"] == "New Name"

    def test_update_user_persona_special_characters(self, temp_memory_dir):
        """Test updating persona with special characters."""
        result = update_user_persona("preferences", "喜欢 Python & 测试 (testing)")

        assert "已记住" in result or "remembered" in result.lower()

        # Verify special characters are preserved
        memory_file = os.path.join(temp_memory_dir, ".agent", "memory.json")
        with open(memory_file) as f:
            data = json.load(f)
        assert "喜欢" in data["preferences"]
        assert "testing" in data["preferences"]

    def test_update_user_persona_empty_value(self, temp_memory_dir):
        """Test updating persona with empty value."""
        result = update_user_persona("key", "")

        assert "已记住" in result or "remembered" in result.lower()

        # Verify empty value is stored
        memory_file = os.path.join(temp_memory_dir, ".agent", "memory.json")
        with open(memory_file) as f:
            data = json.load(f)
        assert data["key"] == ""

    def test_update_user_persona_multiple_updates(self, temp_memory_dir):
        """Test multiple sequential updates."""
        update_user_persona("key1", "value1")
        update_user_persona("key2", "value2")
        update_user_persona("key3", "value3")

        memory_file = os.path.join(temp_memory_dir, ".agent", "memory.json")
        with open(memory_file) as f:
            data = json.load(f)

        assert data["key1"] == "value1"
        assert data["key2"] == "value2"
        assert data["key3"] == "value3"

    def test_update_user_persona_corrupted_json(self, temp_memory_dir):
        """Test updating persona when JSON file is corrupted."""
        memory_file = os.path.join(temp_memory_dir, ".agent", "memory.json")
        with open(memory_file, "w") as f:
            f.write("{ invalid json }")

        # Should handle gracefully and create new data
        result = update_user_persona("name", "John")

        assert "已记住" in result or "remembered" in result.lower()

        # Verify new data was written
        with open(memory_file) as f:
            data = json.load(f)
        assert data["name"] == "John"

    def test_update_user_persona_json_formatting(self, temp_memory_dir):
        """Test that JSON is formatted with proper indentation."""
        update_user_persona("key", "value")

        memory_file = os.path.join(temp_memory_dir, ".agent", "memory.json")
        with open(memory_file) as f:
            content = f.read()

        # Verify indentation (should have 2-space indent)
        assert "  " in content

    def test_update_user_persona_unicode_handling(self, temp_memory_dir):
        """Test that unicode characters are handled correctly."""
        result = update_user_persona("language", "中文, 日本語, 한국어")

        assert "已记住" in result or "remembered" in result.lower()

        memory_file = os.path.join(temp_memory_dir, ".agent", "memory.json")
        with open(memory_file) as f:
            content = f.read()

        # Verify ensure_ascii=False preserves unicode
        assert "中文" in content


# ========================================
# Get User Persona Tests
# ========================================


class TestGetUserPersona:
    """Tests for get_user_persona function."""

    def test_get_user_persona_no_file(self, temp_memory_dir):
        """Test getting persona when file doesn't exist."""
        result = get_user_persona()

        assert "暂无" in result or "no" in result.lower()

    def test_get_user_persona_existing_file(self, temp_memory_dir):
        """Test getting persona from existing file."""
        memory_file = os.path.join(temp_memory_dir, ".agent", "memory.json")
        data = {"name": "John", "job": "Engineer"}
        with open(memory_file, "w") as f:
            json.dump(data, f)

        result = get_user_persona()

        assert "John" in result
        assert "Engineer" in result

    def test_get_user_persona_empty_file(self, temp_memory_dir):
        """Test getting persona from empty JSON file."""
        memory_file = os.path.join(temp_memory_dir, ".agent", "memory.json")
        with open(memory_file, "w") as f:
            json.dump({}, f)

        result = get_user_persona()

        # Should return the file content (empty JSON)
        assert "{}" in result

    def test_get_user_persona_complex_data(self, temp_memory_dir):
        """Test getting persona with complex nested data."""
        memory_file = os.path.join(temp_memory_dir, ".agent", "memory.json")
        data = {
            "name": "John",
            "preferences": {
                "languages": ["Python", "JavaScript"],
                "frameworks": ["FastAPI", "React"],
            },
        }
        with open(memory_file, "w") as f:
            json.dump(data, f, indent=2)

        result = get_user_persona()

        assert "John" in result
        assert "Python" in result
        assert "FastAPI" in result

    def test_get_user_persona_special_characters(self, temp_memory_dir):
        """Test getting persona with special characters."""
        memory_file = os.path.join(temp_memory_dir, ".agent", "memory.json")
        data = {"preferences": "喜欢 Python & 测试"}
        with open(memory_file, "w") as f:
            json.dump(data, f, ensure_ascii=False)

        result = get_user_persona()

        assert "喜欢" in result
        assert "Python" in result

    def test_get_user_persona_returns_raw_content(self, temp_memory_dir):
        """Test that get_user_persona returns raw file content."""
        memory_file = os.path.join(temp_memory_dir, ".agent", "memory.json")
        data = {"key": "value"}
        with open(memory_file, "w") as f:
            json.dump(data, f)

        result = get_user_persona()

        # Should return the raw JSON string
        assert isinstance(result, str)
        assert "key" in result
        assert "value" in result


class TestGetAndDeleteNote:
    """Tests for note helpers that resolve exact titles."""

    @patch("src.servers.memory.collection", None)
    def test_get_note_file_path_resolves_exact_title_collision(self, temp_memory_dir):
        """Test file path lookup returns the exact note path when slugs collide."""
        save_note(title="A/B", content="first", collection_name="default")
        save_note(title="A B", content="second", collection_name="default")

        first_path = get_note_file_path("A/B", collection_name="default")
        second_path = get_note_file_path("A B", collection_name="default")

        assert first_path != second_path
        assert first_path.endswith("a-b.md")
        assert os.path.basename(second_path).startswith("a-b-")

    @patch("src.servers.memory.collection", None)
    def test_delete_note_uses_exact_title_when_slugs_collide(self, temp_memory_dir):
        """Test deleting one colliding title does not remove the other."""
        save_note(title="A/B", content="first", collection_name="default")
        save_note(title="A B", content="second", collection_name="default")

        result = delete_note("A/B", collection_name="default")

        assert "deleted" in result.lower()
        assert "not found" in get_note("A/B", collection_name="default").lower()
        assert "second" in get_note("A B", collection_name="default")


# ========================================
# Integration Tests
# ========================================


class TestMemoryIntegration:
    """Integration tests for memory functions."""

    @patch("src.servers.memory.collection")
    def test_save_and_search_workflow(self, mock_collection):
        """Test saving and then searching for notes."""
        # Setup mock for save
        mock_collection.add = MagicMock()

        # Save a note
        save_result = save_note(
            title="Integration Test",
            content="This is integration test content",
            collection_name="default",
            tags=["test", "integration"],
        )

        assert "已保存" in save_result or "saved" in save_result.lower()

        # Setup mock for search
        mock_collection.query.return_value = {
            "documents": [["This is integration test content"]],
            "metadatas": [[{"title": "Integration Test"}]],
        }

        # Search for the note
        search_result = search_notes(query="integration test", collection_name="default")

        assert "Integration Test" in search_result
        assert "integration test content" in search_result

    def test_persona_workflow(self, temp_memory_dir):
        """Test updating and retrieving user persona."""
        # Update persona
        update_result = update_user_persona("name", "Test User")
        assert "已记住" in update_result or "remembered" in update_result.lower()

        # Get persona
        get_result = get_user_persona()
        assert "Test User" in get_result

        # Update another field
        update_result2 = update_user_persona("role", "Developer")
        assert "已记住" in update_result2 or "remembered" in update_result2.lower()

        # Get updated persona
        get_result2 = get_user_persona()
        assert "Test User" in get_result2
        assert "Developer" in get_result2

    @patch("src.servers.memory.collection")
    def test_multiple_collections_isolation(self, mock_collection):
        """Test that different collections are isolated."""
        mock_collection.add = MagicMock()

        # Save to project1
        save_note(title="Note1", content="Content1", collection_name="project1")

        # Save to project2
        save_note(title="Note2", content="Content2", collection_name="project2")

        # Verify both saves happened
        assert mock_collection.add.call_count == 2

        # Verify different collection names in metadata
        calls = mock_collection.add.call_args_list
        assert calls[0][1]["metadatas"][0]["project"] == "project1"
        assert calls[1][1]["metadatas"][0]["project"] == "project2"


# ========================================
# Edge Cases and Error Handling
# ========================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @patch("src.servers.memory.collection")
    def test_save_note_with_newlines_in_content(self, mock_collection):
        """Test saving note with newlines in content."""
        mock_collection.add = MagicMock()

        content = "Line 1\nLine 2\nLine 3"
        save_note(title="Multiline", content=content, collection_name="default")

        call_args = mock_collection.add.call_args
        assert call_args[1]["documents"][0] == content

    @patch("src.servers.memory.collection")
    def test_search_notes_with_special_query(self, mock_collection):
        """Test searching with special characters in query."""
        mock_collection.query.return_value = {
            "documents": [["Result"]],
            "metadatas": [[{"title": "Result"}]],
        }

        result = search_notes(query="特殊字符 & symbols!", collection_name="default")

        assert "Result" in result

    def test_update_persona_with_very_long_value(self, temp_memory_dir):
        """Test updating persona with very long value."""
        long_value = "x" * 10000
        result = update_user_persona("long_key", long_value)

        assert "已记住" in result or "remembered" in result.lower()

        memory_file = os.path.join(temp_memory_dir, ".agent", "memory.json")
        with open(memory_file) as f:
            data = json.load(f)
        assert data["long_key"] == long_value

    @patch("src.servers.memory.collection")
    def test_search_notes_with_none_metadata_fields(self, mock_collection):
        """Test searching when metadata has missing fields."""
        mock_collection.query.return_value = {
            "documents": [["Document"]],
            "metadatas": [[{"title": None}]],
        }

        result = search_notes(query="search", collection_name="default")

        # Should handle None gracefully
        assert "Untitled" in result or "Document" in result

    @patch("src.servers.memory.collection")
    def test_save_note_hash_consistency(self, mock_collection):
        """Test that same content produces same hash in ID."""
        mock_collection.add = MagicMock()

        content = "Same content"

        # Save twice with same content
        save_note("Note1", content, "default")
        save_note("Note2", content, "default")

        calls = mock_collection.add.call_args_list
        id1 = calls[0][1]["ids"][0]
        id2 = calls[1][1]["ids"][0]

        # Extract hash parts (after the last underscore)
        hash1 = id1.split("_")[-1]
        hash2 = id2.split("_")[-1]

        # Same content should produce same hash
        assert hash1 == hash2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
