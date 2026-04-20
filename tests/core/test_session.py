"""Comprehensive tests for session.py"""

import json
import tempfile
import time
from pathlib import Path

import pytest

from src.core.session import SessionData, SessionManager, SessionMetadata


class TestSessionMetadata:
    """Tests for SessionMetadata."""

    def test_creation_with_defaults(self):
        """Test creating session metadata with defaults."""
        metadata = SessionMetadata(id="test_123")
        assert metadata.id == "test_123"
        assert metadata.name is None
        assert metadata.project == "default"
        assert metadata.message_count == 0
        assert metadata.mode == "build"
        assert metadata.parent_id is None
        assert metadata.tags == []

    def test_creation_with_all_fields(self):
        """Test creating session metadata with all fields."""
        metadata = SessionMetadata(
            id="session_456",
            name="My Session",
            project="myproject",
            message_count=10,
            total_tokens=5000,
            mode="plan",
            parent_id="parent_123",
            tags=["important", "test"],
            description="Test session",
        )
        assert metadata.name == "My Session"
        assert metadata.project == "myproject"
        assert metadata.message_count == 10
        assert metadata.mode == "plan"
        assert metadata.parent_id == "parent_123"
        assert len(metadata.tags) == 2

    def test_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = SessionMetadata(id="test_789", name="Test", project="proj", message_count=5)
        data = metadata.to_dict()
        assert data["id"] == "test_789"
        assert data["name"] == "Test"
        assert data["project"] == "proj"
        assert data["message_count"] == 5
        assert "created_at" in data
        assert "updated_at" in data

    def test_from_dict_basic(self):
        """Test creating metadata from dictionary."""
        data = {
            "id": "from_dict_123",
            "name": "From Dict",
            "project": "test_project",
            "message_count": 15,
        }
        metadata = SessionMetadata.from_dict(data)
        assert metadata.id == "from_dict_123"
        assert metadata.name == "From Dict"
        assert metadata.project == "test_project"
        assert metadata.message_count == 15

    def test_from_dict_with_defaults(self):
        """Test from_dict with missing optional fields."""
        data = {"id": "minimal_123"}
        metadata = SessionMetadata.from_dict(data)
        assert metadata.id == "minimal_123"
        assert metadata.project == "default"
        assert metadata.mode == "build"
        assert metadata.tags == []

    def test_from_dict_with_all_fields(self):
        """Test from_dict with all fields."""
        data = {
            "id": "full_123",
            "name": "Full Session",
            "project": "full_project",
            "created_at": 1234567890.0,
            "updated_at": 1234567900.0,
            "message_count": 20,
            "total_tokens": 10000,
            "mode": "plan",
            "parent_id": "parent_456",
            "tags": ["tag1", "tag2"],
            "description": "Full description",
        }
        metadata = SessionMetadata.from_dict(data)
        assert metadata.name == "Full Session"
        assert metadata.created_at == 1234567890.0
        assert metadata.updated_at == 1234567900.0
        assert metadata.total_tokens == 10000
        assert metadata.parent_id == "parent_456"
        assert metadata.description == "Full description"

    def test_get_display_name_with_name(self):
        """Test display name when name is set."""
        metadata = SessionMetadata(id="test_123", name="My Custom Name")
        assert metadata.get_display_name() == "My Custom Name"

    def test_get_display_name_without_name(self):
        """Test display name when name is not set."""
        metadata = SessionMetadata(id="test_123456789")
        display_name = metadata.get_display_name()
        assert "test_123" in display_name  # Should include ID prefix
        assert "Session" in display_name

    def test_timestamps_auto_generated(self):
        """Test that timestamps are auto-generated."""
        before = time.time()
        metadata = SessionMetadata(id="test_123")
        after = time.time()
        assert before <= metadata.created_at <= after
        assert before <= metadata.updated_at <= after


class TestSessionData:
    """Tests for SessionData."""

    def test_creation_with_defaults(self):
        """Test creating session data with defaults."""
        metadata = SessionMetadata(id="test_123")
        session = SessionData(metadata=metadata)
        assert session.metadata.id == "test_123"
        assert session.messages == []
        assert session.summaries == []
        assert session.checkpoints == []
        assert session.orchestration_state == {}

    def test_creation_with_data(self):
        """Test creating session data with messages and summaries."""
        metadata = SessionMetadata(id="test_456")
        messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
        summaries = [{"content": "Summary", "message_count": 2}]
        checkpoints = ["checkpoint_1", "checkpoint_2"]

        session = SessionData(
            metadata=metadata, messages=messages, summaries=summaries, checkpoints=checkpoints
        )
        assert len(session.messages) == 2
        assert len(session.summaries) == 1
        assert len(session.checkpoints) == 2

    def test_to_dict(self):
        """Test converting session data to dictionary."""
        metadata = SessionMetadata(id="test_789", name="Test Session")
        messages = [{"role": "user", "content": "Test"}]
        session = SessionData(metadata=metadata, messages=messages)

        data = session.to_dict()
        assert data["version"] == 1
        assert data["metadata"]["id"] == "test_789"
        assert len(data["messages"]) == 1
        assert "summaries" in data
        assert "checkpoints" in data
        assert "orchestration_state" in data

    def test_from_dict_basic(self):
        """Test creating session data from dictionary."""
        data = {
            "version": 1,
            "metadata": {"id": "from_dict_123", "name": "Test"},
            "messages": [{"role": "user", "content": "Hello"}],
            "summaries": [],
            "checkpoints": [],
        }
        session = SessionData.from_dict(data)
        assert session.metadata.id == "from_dict_123"
        assert len(session.messages) == 1
        assert session.messages[0]["content"] == "Hello"

    def test_from_dict_with_all_data(self):
        """Test from_dict with complete data."""
        data = {
            "version": 1,
            "metadata": {
                "id": "full_123",
                "name": "Full Session",
                "message_count": 5,
                "tags": ["test"],
            },
            "messages": [
                {"role": "user", "content": "Message 1"},
                {"role": "assistant", "content": "Response 1"},
            ],
            "summaries": [{"content": "Summary 1", "message_count": 2}],
            "checkpoints": ["cp_1", "cp_2", "cp_3"],
        }
        session = SessionData.from_dict(data)
        assert session.metadata.name == "Full Session"
        assert len(session.messages) == 2
        assert len(session.summaries) == 1
        assert len(session.checkpoints) == 3

    def test_from_dict_with_missing_fields(self):
        """Test from_dict with missing optional fields."""
        data = {"metadata": {"id": "minimal_123"}}
        session = SessionData.from_dict(data)
        assert session.metadata.id == "minimal_123"
        assert session.messages == []
        assert session.summaries == []
        assert session.checkpoints == []
        assert session.orchestration_state == {}

    def test_roundtrip_to_dict_from_dict(self):
        """Test that to_dict and from_dict are inverses."""
        metadata = SessionMetadata(id="roundtrip_123", name="Roundtrip Test", message_count=10)
        messages = [
            {"role": "user", "content": "Test message"},
            {"role": "assistant", "content": "Test response"},
        ]
        original = SessionData(metadata=metadata, messages=messages)

        # Convert to dict and back
        data = original.to_dict()
        restored = SessionData.from_dict(data)

        assert restored.metadata.id == original.metadata.id
        assert restored.metadata.name == original.metadata.name
        assert len(restored.messages) == len(original.messages)
        assert restored.messages[0]["content"] == original.messages[0]["content"]


class TestSessionManager:
    """Tests for SessionManager."""

    @pytest.fixture
    def temp_session_dir(self):
        """Create a temporary directory for session storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def session_manager(self, temp_session_dir):
        """Create a SessionManager instance with temp directory."""
        return SessionManager(base_dir=temp_session_dir)

    def test_initialization(self, temp_session_dir):
        """Test SessionManager initialization."""
        manager = SessionManager(base_dir=temp_session_dir)
        assert manager.base_dir == temp_session_dir
        assert temp_session_dir.exists()
        assert isinstance(manager._index, dict)

    def test_generate_id(self, session_manager):
        """Test ID generation."""
        id1 = session_manager._generate_id()
        id2 = session_manager._generate_id()
        assert len(id1) == 20
        assert len(id2) == 20
        assert id1 != id2  # IDs should be unique

    def test_get_session_path(self, session_manager):
        """Test session path generation."""
        path = session_manager._get_session_path("test_123")
        assert path.name == "test_123.json"
        assert path.parent == session_manager.base_dir

    def test_get_session_path_rejects_unsafe_identifier(self, session_manager):
        """Unsafe identifiers should not be converted into filesystem paths."""
        with pytest.raises(ValueError, match="Invalid session ID"):
            session_manager._get_session_path("../escape")

    def test_create_session_basic(self, session_manager):
        """Test creating a basic session."""
        session = session_manager.create_session()
        assert session.metadata.id is not None
        assert session.metadata.project == "default"
        assert session.metadata.mode == "build"
        assert session.messages == []

    def test_create_session_with_name(self, session_manager):
        """Test creating a session with a name."""
        session = session_manager.create_session(name="My Session")
        assert session.metadata.name == "My Session"

    def test_create_session_with_project(self, session_manager):
        """Test creating a session with a project."""
        session = session_manager.create_session(project="myproject")
        assert session.metadata.project == "myproject"

    def test_create_session_with_all_params(self, session_manager):
        """Test creating a session with all parameters."""
        session = session_manager.create_session(
            project="testproj",
            name="Test Session",
            mode="plan",
            description="Test description",
            tags=["tag1", "tag2"],
        )
        assert session.metadata.project == "testproj"
        assert session.metadata.name == "Test Session"
        assert session.metadata.mode == "plan"
        assert session.metadata.description == "Test description"
        assert session.metadata.tags == ["tag1", "tag2"]

    def test_create_session_persists_to_disk(self, session_manager, temp_session_dir):
        """Test that created sessions are persisted to disk."""
        session = session_manager.create_session(name="Persist Test")
        session_id = session.metadata.id

        # Check that session file exists
        session_path = temp_session_dir / f"{session_id}.json"
        assert session_path.exists()

        # Check that index file exists
        index_path = temp_session_dir / "index.json"
        assert index_path.exists()

    def test_load_session(self, session_manager):
        """Test loading a session."""
        created = session_manager.create_session(name="Load Test")
        session_id = created.metadata.id

        loaded = session_manager.load_session(session_id)
        assert loaded is not None
        assert loaded.metadata.id == session_id
        assert loaded.metadata.name == "Load Test"

    def test_load_nonexistent_session(self, session_manager):
        """Test loading a nonexistent session."""
        loaded = session_manager.load_session("nonexistent_id")
        assert loaded is None

    def test_load_session_rejects_unsafe_identifier(self, session_manager):
        """Unsafe identifiers should be rejected before any file access."""
        loaded = session_manager.load_session("../escape")
        assert loaded is None

    def test_save_session(self, session_manager):
        """Test saving a session."""
        session = session_manager.create_session(name="Save Test")
        session.messages.append({"role": "user", "content": "Test message"})
        session.metadata.message_count = 1

        session_manager.save_session(session)

        # Reload and verify
        loaded = session_manager.load_session(session.metadata.id)
        assert len(loaded.messages) == 1
        assert loaded.messages[0]["content"] == "Test message"

    def test_resume_session_by_id(self, session_manager):
        """Test resuming a session by ID."""
        created = session_manager.create_session(name="Resume Test")
        session_id = created.metadata.id

        resumed = session_manager.resume_session(session_id)
        assert resumed is not None
        assert resumed.metadata.id == session_id

    def test_resume_session_by_name(self, session_manager):
        """Test resuming a session by name."""
        session_manager.create_session(name="Named Session")

        resumed = session_manager.resume_session("Named Session")
        assert resumed is not None
        assert resumed.metadata.name == "Named Session"

    def test_resume_nonexistent_session(self, session_manager):
        """Test resuming a nonexistent session."""
        resumed = session_manager.resume_session("nonexistent")
        assert resumed is None

    def test_fork_session_basic(self, session_manager):
        """Test forking a session."""
        original = session_manager.create_session(name="Original")
        original.messages = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
        ]
        session_manager.save_session(original)

        forked = session_manager.fork_session(original.metadata.id)
        assert forked is not None
        assert forked.metadata.parent_id == original.metadata.id
        assert len(forked.messages) == 2
        assert forked.messages[0]["content"] == "Message 1"

    def test_fork_session_with_name(self, session_manager):
        """Test forking a session with a custom name."""
        original = session_manager.create_session(name="Original")
        forked = session_manager.fork_session(original.metadata.id, name="Custom Fork")
        assert forked.metadata.name == "Custom Fork"

    def test_fork_session_at_message(self, session_manager):
        """Test forking a session at a specific message."""
        original = session_manager.create_session(name="Original")
        original.messages = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2"},
            {"role": "assistant", "content": "Response 2"},
        ]
        session_manager.save_session(original)

        forked = session_manager.fork_session(original.metadata.id, at_message=2)
        assert len(forked.messages) == 2
        assert forked.messages[-1]["content"] == "Response 1"

    def test_fork_nonexistent_session(self, session_manager):
        """Test forking a nonexistent session."""
        forked = session_manager.fork_session("nonexistent")
        assert forked is None

    def test_fork_preserves_metadata(self, session_manager):
        """Test that forking preserves relevant metadata."""
        original = session_manager.create_session(
            project="testproj", mode="plan", tags=["important", "test"]
        )
        forked = session_manager.fork_session(original.metadata.id)
        assert forked.metadata.project == original.metadata.project
        assert forked.metadata.mode == original.metadata.mode
        assert forked.metadata.tags == original.metadata.tags

    def test_list_sessions_empty(self, session_manager):
        """Test listing sessions when none exist."""
        sessions = session_manager.list_sessions()
        assert sessions == []

    def test_list_sessions_basic(self, session_manager):
        """Test listing sessions."""
        session_manager.create_session(name="Session 1")
        session_manager.create_session(name="Session 2")

        sessions = session_manager.list_sessions()
        assert len(sessions) == 2
        assert any(s.name == "Session 1" for s in sessions)
        assert any(s.name == "Session 2" for s in sessions)

    def test_list_sessions_by_project(self, session_manager):
        """Test listing sessions filtered by project."""
        session_manager.create_session(project="proj1", name="Session 1")
        session_manager.create_session(project="proj2", name="Session 2")
        session_manager.create_session(project="proj1", name="Session 3")

        proj1_sessions = session_manager.list_sessions(project="proj1")
        assert len(proj1_sessions) == 2
        assert all(s.project == "proj1" for s in proj1_sessions)

    def test_list_sessions_limit(self, session_manager):
        """Test listing sessions with limit."""
        for i in range(5):
            session_manager.create_session(name=f"Session {i}")

        sessions = session_manager.list_sessions(limit=3)
        assert len(sessions) == 3

    def test_list_sessions_exclude_forks(self, session_manager):
        """Test listing sessions excluding forks."""
        original = session_manager.create_session(name="Original")
        session_manager.fork_session(original.metadata.id, name="Fork 1")
        session_manager.fork_session(original.metadata.id, name="Fork 2")

        all_sessions = session_manager.list_sessions(include_forks=True)
        main_sessions = session_manager.list_sessions(include_forks=False)

        assert len(all_sessions) == 3
        assert len(main_sessions) == 1

    def test_list_sessions_sorted_by_updated_at(self, session_manager):
        """Test that sessions are sorted by updated_at descending."""
        session1 = session_manager.create_session(name="Session 1")
        time.sleep(0.01)
        session2 = session_manager.create_session(name="Session 2")

        sessions = session_manager.list_sessions()
        assert sessions[0].id == session2.metadata.id
        assert sessions[1].id == session1.metadata.id

    def test_search_sessions_by_name(self, session_manager):
        """Test searching sessions by name."""
        session_manager.create_session(name="Python Project")
        session_manager.create_session(name="JavaScript Project")
        session_manager.create_session(name="Python Testing")

        results = session_manager.search_sessions("Python")
        assert len(results) == 2
        assert all("Python" in s.name for s in results)

    def test_search_sessions_by_description(self, session_manager):
        """Test searching sessions by description."""
        session_manager.create_session(name="Session 1", description="This is about authentication")
        session_manager.create_session(name="Session 2", description="This is about database")

        results = session_manager.search_sessions("authentication")
        assert len(results) == 1
        assert "authentication" in results[0].description

    def test_search_sessions_by_tags(self, session_manager):
        """Test searching sessions by tags."""
        session_manager.create_session(name="Session 1", tags=["bug-fix", "urgent"])
        session_manager.create_session(name="Session 2", tags=["feature", "enhancement"])

        results = session_manager.search_sessions("bug-fix")
        assert len(results) == 1
        assert "bug-fix" in results[0].tags

    def test_search_sessions_case_insensitive(self, session_manager):
        """Test that search is case insensitive."""
        session_manager.create_session(name="Python Project")

        results = session_manager.search_sessions("python")
        assert len(results) == 1

    def test_search_sessions_with_project_filter(self, session_manager):
        """Test searching sessions with project filter."""
        session_manager.create_session(project="proj1", name="Python Project")
        session_manager.create_session(project="proj2", name="Python Testing")

        results = session_manager.search_sessions("Python", project="proj1")
        assert len(results) == 1
        assert results[0].project == "proj1"

    def test_search_sessions_empty_results(self, session_manager):
        """Test searching with no matches."""
        session_manager.create_session(name="Session 1")

        results = session_manager.search_sessions("nonexistent")
        assert results == []

    def test_search_sessions_relevance_sorting(self, session_manager):
        """Test that search results are sorted by relevance."""
        session_manager.create_session(name="Python", description="About something else")
        session_manager.create_session(name="JavaScript", description="Python testing framework")

        results = session_manager.search_sessions("Python")
        # Name match should come first
        assert results[0].name == "Python"

    def test_rename_session(self, session_manager):
        """Test renaming a session."""
        session = session_manager.create_session(name="Old Name")
        session_id = session.metadata.id

        success = session_manager.rename_session(session_id, "New Name")
        assert success is True

        # Verify rename
        renamed = session_manager.load_session(session_id)
        assert renamed.metadata.name == "New Name"

    def test_rename_nonexistent_session(self, session_manager):
        """Test renaming a nonexistent session."""
        success = session_manager.rename_session("nonexistent", "New Name")
        assert success is False

    def test_delete_session(self, session_manager, temp_session_dir):
        """Test deleting a session."""
        session = session_manager.create_session(name="To Delete")
        session_id = session.metadata.id
        session_path = temp_session_dir / f"{session_id}.json"

        assert session_path.exists()

        success = session_manager.delete_session(session_id)
        assert success is True
        assert not session_path.exists()

    def test_delete_nonexistent_session(self, session_manager):
        """Test deleting a nonexistent session."""
        success = session_manager.delete_session("nonexistent")
        assert success is False

    def test_delete_removes_from_index(self, session_manager):
        """Test that deleting removes session from index."""
        session = session_manager.create_session(name="To Delete")
        session_id = session.metadata.id

        assert session_id in session_manager._index

        session_manager.delete_session(session_id)
        assert session_id not in session_manager._index

    def test_export_session_json(self, session_manager):
        """Test exporting a session as JSON."""
        session = session_manager.create_session(name="Export Test")
        session.messages = [{"role": "user", "content": "Test"}]
        session_manager.save_session(session)

        exported = session_manager.export_session(session.metadata.id, format="json")
        data = json.loads(exported)
        assert data["metadata"]["name"] == "Export Test"
        assert len(data["messages"]) == 1

    def test_export_session_markdown(self, session_manager):
        """Test exporting a session as Markdown."""
        session = session_manager.create_session(name="Export Test")
        session.messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        session_manager.save_session(session)

        exported = session_manager.export_session(session.metadata.id, format="markdown")
        assert "# Export Test" in exported
        assert "### User" in exported
        assert "### Assistant" in exported
        assert "Hello" in exported

    def test_export_session_text(self, session_manager):
        """Test exporting a session as plain text."""
        session = session_manager.create_session(name="Export Test")
        session.messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        session_manager.save_session(session)

        exported = session_manager.export_session(session.metadata.id, format="text")
        assert "Export Test" in exported
        assert "[USER]" in exported
        assert "[ASSISTANT]" in exported

    def test_export_session_to_file(self, session_manager, temp_session_dir):
        """Test exporting a session to a file."""
        session = session_manager.create_session(name="Export Test")
        session_manager.save_session(session)

        export_path = temp_session_dir / "export.json"
        session_manager.export_session(session.metadata.id, format="json", path=export_path)

        assert export_path.exists()
        content = export_path.read_text()
        data = json.loads(content)
        assert data["metadata"]["name"] == "Export Test"

    def test_export_nonexistent_session(self, session_manager):
        """Test exporting a nonexistent session."""
        with pytest.raises(ValueError):
            session_manager.export_session("nonexistent", format="json")

    def test_export_invalid_format(self, session_manager):
        """Test exporting with invalid format."""
        session = session_manager.create_session(name="Export Test")
        session_manager.save_session(session)

        with pytest.raises(ValueError):
            session_manager.export_session(session.metadata.id, format="invalid")

    def test_get_recent_sessions(self, session_manager):
        """Test getting recent sessions."""
        for i in range(3):
            session_manager.create_session(name=f"Session {i}")

        recent = session_manager.get_recent_sessions(limit=2)
        assert len(recent) == 2
        assert all("id" in s for s in recent)
        assert all("name" in s for s in recent)
        assert all("project" in s for s in recent)
        assert all("updated_at" in s for s in recent)
        assert all("message_count" in s for s in recent)

    def test_get_recent_sessions_empty(self, session_manager):
        """Test getting recent sessions when none exist."""
        recent = session_manager.get_recent_sessions()
        assert recent == []

    def test_index_persistence(self, session_manager, temp_session_dir):
        """Test that index is persisted and reloaded."""
        session_manager.create_session(name="Session 1")
        session_manager.create_session(name="Session 2")

        # Create new manager instance
        new_manager = SessionManager(base_dir=temp_session_dir)
        sessions = new_manager.list_sessions()

        assert len(sessions) == 2
        assert any(s.name == "Session 1" for s in sessions)
        assert any(s.name == "Session 2" for s in sessions)

    def test_session_metadata_updates(self, session_manager):
        """Test that session metadata is updated on save."""
        session = session_manager.create_session(name="Update Test")
        original_updated_at = session.metadata.updated_at

        time.sleep(0.01)
        session.messages.append({"role": "user", "content": "New message"})
        session_manager.save_session(session)

        loaded = session_manager.load_session(session.metadata.id)
        assert loaded.metadata.updated_at > original_updated_at

    def test_export_markdown_with_metadata(self, session_manager):
        """Test markdown export includes all metadata."""
        session = session_manager.create_session(
            name="Full Test", description="Test description", tags=["tag1", "tag2"]
        )
        session.metadata.total_tokens = 1000
        session.messages = [{"role": "user", "content": "Test"}]
        session_manager.save_session(session)

        exported = session_manager.export_session(session.metadata.id, format="markdown")
        assert "Full Test" in exported
        assert "Test description" in exported
        assert "tag1" in exported
        assert "tag2" in exported
        assert "1,000" in exported  # Formatted tokens

    def test_export_text_with_multiple_messages(self, session_manager):
        """Test text export with multiple messages."""
        session = session_manager.create_session(name="Multi Test")
        session.messages = [
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Question 2"},
            {"role": "assistant", "content": "Answer 2"},
        ]
        session_manager.save_session(session)

        exported = session_manager.export_session(session.metadata.id, format="text")
        assert exported.count("[USER]") == 2
        assert exported.count("[ASSISTANT]") == 2

    def test_fork_with_empty_messages(self, session_manager):
        """Test forking a session with no messages."""
        original = session_manager.create_session(name="Empty Original")
        forked = session_manager.fork_session(original.metadata.id)

        assert len(forked.messages) == 0
        assert forked.metadata.parent_id == original.metadata.id

    def test_multiple_forks_from_same_parent(self, session_manager):
        """Test creating multiple forks from the same parent."""
        original = session_manager.create_session(name="Parent")
        original.messages = [{"role": "user", "content": "Message"}]
        session_manager.save_session(original)

        fork1 = session_manager.fork_session(original.metadata.id, name="Fork 1")
        fork2 = session_manager.fork_session(original.metadata.id, name="Fork 2")

        assert fork1.metadata.parent_id == original.metadata.id
        assert fork2.metadata.parent_id == original.metadata.id
        assert fork1.metadata.id != fork2.metadata.id

    def test_session_with_summaries(self, session_manager):
        """Test session with summaries."""
        session = session_manager.create_session(name="Summary Test")
        session.summaries = [
            {"content": "Summary 1", "message_count": 5},
            {"content": "Summary 2", "message_count": 3},
        ]
        session_manager.save_session(session)

        loaded = session_manager.load_session(session.metadata.id)
        assert len(loaded.summaries) == 2
        assert loaded.summaries[0]["content"] == "Summary 1"

    def test_session_with_checkpoints(self, session_manager):
        """Test session with checkpoints."""
        session = session_manager.create_session(name="Checkpoint Test")
        session.checkpoints = ["cp_1", "cp_2", "cp_3"]
        session_manager.save_session(session)

        loaded = session_manager.load_session(session.metadata.id)
        assert len(loaded.checkpoints) == 3
        assert "cp_2" in loaded.checkpoints

    def test_session_with_orchestration_state(self, session_manager):
        """Test session with persisted orchestration state."""
        session = session_manager.create_session(name="Orchestration Test")
        session.orchestration_state = {
            "run_id": "run-1",
            "success": True,
            "summary": "planned: auth flow",
            "task_graph": [{"id": "root", "title": "Root", "status": "completed", "ready": True}],
            "worker_assignments": {
                "task-1": {
                    "role": "inspect",
                    "worker_session_id": "worker-1",
                    "allowed_tool_names": ["read", "search"],
                }
            },
        }
        session.orchestration_runs = [dict(session.orchestration_state)]
        session.active_orchestration_run_id = "run-1"
        session_manager.save_session(session)

        loaded = session_manager.load_session(session.metadata.id)
        assert loaded.active_orchestration_run_id == "run-1"
        assert loaded.orchestration_runs[0]["run_id"] == "run-1"
        assert loaded.orchestration_state["summary"] == "planned: auth flow"
        assert loaded.orchestration_state["task_graph"][0]["id"] == "root"
        assert loaded.orchestration_state["worker_assignments"]["task-1"]["role"] == "inspect"

    def test_export_json_preserves_all_data(self, session_manager):
        """Test that JSON export preserves all session data."""
        session = session_manager.create_session(
            name="Full Data Test",
            project="testproj",
            mode="plan",
            description="Test",
            tags=["tag1"],
        )
        session.messages = [{"role": "user", "content": "Test"}]
        session.summaries = [{"content": "Summary"}]
        session.checkpoints = ["cp_1"]
        session_manager.save_session(session)

        exported = session_manager.export_session(session.metadata.id, format="json")
        data = json.loads(exported)

        assert data["metadata"]["name"] == "Full Data Test"
        assert data["metadata"]["project"] == "testproj"
        assert data["metadata"]["mode"] == "plan"
        assert len(data["messages"]) == 1
        assert len(data["summaries"]) == 1
        assert len(data["checkpoints"]) == 1
        assert data["orchestration_state"] == {}
        assert data["orchestration_runs"] == []
        assert data["active_orchestration_run_id"] is None

    # Additional comprehensive tests for improved coverage

    def test_create_session_generates_unique_ids(self, session_manager):
        """Test that each created session has a unique ID."""
        sessions = [session_manager.create_session() for _ in range(5)]
        ids = [s.metadata.id for s in sessions]
        assert len(ids) == len(set(ids))  # All IDs should be unique

    def test_create_session_with_empty_tags(self, session_manager):
        """Test creating session with empty tags list."""
        session = session_manager.create_session(tags=[])
        assert session.metadata.tags == []

    def test_create_session_with_none_tags(self, session_manager):
        """Test creating session with None tags defaults to empty list."""
        session = session_manager.create_session(tags=None)
        assert session.metadata.tags == []

    def test_session_metadata_to_dict_contains_all_fields(self):
        """Test that metadata.to_dict() includes all required fields."""
        metadata = SessionMetadata(
            id="test_id",
            name="Test",
            project="proj",
            message_count=5,
            total_tokens=1000,
            mode="plan",
            parent_id="parent",
            tags=["tag1"],
            description="desc",
        )
        data = metadata.to_dict()
        required_fields = [
            "id",
            "name",
            "project",
            "created_at",
            "updated_at",
            "message_count",
            "total_tokens",
            "mode",
            "parent_id",
            "tags",
            "description",
        ]
        for field in required_fields:
            assert field in data

    def test_session_data_to_dict_version(self):
        """Test that SessionData.to_dict() includes version."""
        metadata = SessionMetadata(id="test")
        session = SessionData(metadata=metadata)
        data = session.to_dict()
        assert data["version"] == 1

    def test_load_session_with_corrupted_json(self, session_manager, temp_session_dir):
        """Test loading a session with corrupted JSON file."""
        session = session_manager.create_session(name="Test")
        session_id = session.metadata.id
        session_path = temp_session_dir / f"{session_id}.json"

        # Corrupt the JSON file
        session_path.write_text("{ invalid json }", encoding="utf-8")

        # Should return None on load error
        loaded = session_manager.load_session(session_id)
        assert loaded is None

    def test_load_index_with_corrupted_index_file(self, temp_session_dir):
        """Test loading index with corrupted index.json."""
        index_path = temp_session_dir / "index.json"
        index_path.write_text("{ invalid json }", encoding="utf-8")

        # Should handle gracefully
        manager = SessionManager(base_dir=temp_session_dir)
        assert manager._index == {}

    def test_save_session_updates_timestamp(self, session_manager):
        """Test that saving a session updates the updated_at timestamp."""
        session = session_manager.create_session(name="Timestamp Test")
        original_time = session.metadata.updated_at

        time.sleep(0.01)
        session.messages.append({"role": "user", "content": "New"})
        session_manager.save_session(session)

        loaded = session_manager.load_session(session.metadata.id)
        assert loaded.metadata.updated_at > original_time

    def test_resume_session_prefers_id_over_name(self, session_manager):
        """Test that resume_session tries ID first before name."""
        session1 = session_manager.create_session(name="Session A")
        session_manager.create_session(name="Session B")

        # Resume by ID should work
        resumed = session_manager.resume_session(session1.metadata.id)
        assert resumed.metadata.id == session1.metadata.id

    def test_fork_session_copies_messages_not_references(self, session_manager):
        """Test that forking creates a copy of messages, not references."""
        original = session_manager.create_session(name="Original")
        original.messages = [{"role": "user", "content": "Original"}]
        session_manager.save_session(original)

        forked = session_manager.fork_session(original.metadata.id)

        # Modify forked messages
        forked.messages[0]["content"] = "Modified"

        # Original should be unchanged
        reloaded_original = session_manager.load_session(original.metadata.id)
        assert reloaded_original.messages[0]["content"] == "Original"

    def test_fork_session_copies_summaries_not_references(self, session_manager):
        """Test that forking creates a copy of summaries, not references."""
        original = session_manager.create_session(name="Original")
        original.summaries = [{"content": "Summary"}]
        session_manager.save_session(original)

        forked = session_manager.fork_session(original.metadata.id)

        # Modify forked summaries
        forked.summaries[0]["content"] = "Modified"

        # Original should be unchanged
        reloaded_original = session_manager.load_session(original.metadata.id)
        assert reloaded_original.summaries[0]["content"] == "Summary"

    def test_fork_session_copies_tags_not_references(self, session_manager):
        """Test that forking creates a copy of tags, not references."""
        original = session_manager.create_session(name="Original", tags=["tag1", "tag2"])
        forked = session_manager.fork_session(original.metadata.id)

        # Modify forked tags
        forked.metadata.tags.append("tag3")

        # Original should be unchanged
        reloaded_original = session_manager.load_session(original.metadata.id)
        assert len(reloaded_original.metadata.tags) == 2

    def test_fork_session_at_message_zero(self, session_manager):
        """Test forking at message index 0."""
        original = session_manager.create_session(name="Original")
        original.messages = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
        ]
        session_manager.save_session(original)

        forked = session_manager.fork_session(original.metadata.id, at_message=0)
        assert len(forked.messages) == 0

    def test_fork_session_at_message_beyond_length(self, session_manager):
        """Test forking at message index beyond message list length."""
        original = session_manager.create_session(name="Original")
        original.messages = [{"role": "user", "content": "Message 1"}]
        session_manager.save_session(original)

        forked = session_manager.fork_session(original.metadata.id, at_message=100)
        assert len(forked.messages) == 1

    def test_fork_session_filters_future_orchestration_runs(self, session_manager):
        """Forking at an earlier message should drop later run snapshots."""
        original = session_manager.create_session(name="Original")
        original.messages = [
            {"role": "user", "content": "Goal 1"},
            {"role": "assistant", "content": "Run 1 done"},
            {"role": "user", "content": "Goal 2"},
            {"role": "assistant", "content": "Run 2 done"},
        ]
        original.orchestration_runs = [
            {
                "run_id": "run-1",
                "goal": "Goal 1",
                "message_start_index": 0,
                "message_end_index": 1,
                "root_task_id": "root-1",
                "success": True,
                "summary": "Run 1 done",
                "task_graph": [],
                "worker_assignments": {},
            },
            {
                "run_id": "run-2",
                "goal": "Goal 2",
                "message_start_index": 2,
                "message_end_index": 3,
                "root_task_id": "root-2",
                "success": True,
                "summary": "Run 2 done",
                "task_graph": [],
                "worker_assignments": {},
            },
        ]
        original.orchestration_state = dict(original.orchestration_runs[-1])
        original.active_orchestration_run_id = "run-2"
        session_manager.save_session(original)

        forked = session_manager.fork_session(original.metadata.id, at_message=2)

        assert [run["run_id"] for run in forked.orchestration_runs] == ["run-1"]
        assert forked.active_orchestration_run_id == "run-1"
        assert forked.orchestration_state["run_id"] == "run-1"

    def test_list_sessions_sorted_descending(self, session_manager):
        """Test that list_sessions returns sessions in descending order by updated_at."""
        sessions = []
        for i in range(3):
            s = session_manager.create_session(name=f"Session {i}")
            sessions.append(s)
            time.sleep(0.01)

        listed = session_manager.list_sessions()
        # Most recent should be first
        assert listed[0].id == sessions[-1].metadata.id
        assert listed[-1].id == sessions[0].metadata.id

    def test_list_sessions_with_project_and_limit(self, session_manager):
        """Test list_sessions with both project filter and limit."""
        for i in range(5):
            session_manager.create_session(project="proj1", name=f"Session {i}")

        listed = session_manager.list_sessions(project="proj1", limit=2)
        assert len(listed) == 2
        assert all(s.project == "proj1" for s in listed)

    def test_search_sessions_multiple_matches_in_name(self, session_manager):
        """Test search with multiple matches in name."""
        session_manager.create_session(name="Python Backend")
        session_manager.create_session(name="Python Frontend")
        session_manager.create_session(name="JavaScript Backend")

        results = session_manager.search_sessions("Python")
        assert len(results) == 2

    def test_search_sessions_partial_match(self, session_manager):
        """Test search with partial word match."""
        session_manager.create_session(name="Authentication System")
        session_manager.create_session(name="Authorization Module")

        results = session_manager.search_sessions("auth")
        assert len(results) == 2

    def test_search_sessions_tag_partial_match(self, session_manager):
        """Test search with partial tag match."""
        session_manager.create_session(name="Session 1", tags=["bug-fix-urgent"])
        session_manager.create_session(name="Session 2", tags=["feature-request"])

        results = session_manager.search_sessions("bug")
        assert len(results) == 1

    def test_search_sessions_no_project_filter_returns_all(self, session_manager):
        """Test search without project filter returns from all projects."""
        session_manager.create_session(project="proj1", name="Python")
        session_manager.create_session(project="proj2", name="Python")

        results = session_manager.search_sessions("Python")
        assert len(results) == 2

    def test_rename_session_updates_index(self, session_manager):
        """Test that renaming updates the index."""
        session = session_manager.create_session(name="Old")
        session_id = session.metadata.id

        session_manager.rename_session(session_id, "New")

        assert session_manager._index[session_id].name == "New"

    def test_rename_session_updates_timestamp(self, session_manager):
        """Test that renaming updates the updated_at timestamp."""
        session = session_manager.create_session(name="Old")
        session_id = session.metadata.id
        original_time = session.metadata.updated_at

        time.sleep(0.01)
        session_manager.rename_session(session_id, "New")

        assert session_manager._index[session_id].updated_at > original_time

    def test_delete_session_updates_index(self, session_manager):
        """Test that deleting removes from index."""
        session = session_manager.create_session(name="To Delete")
        session_id = session.metadata.id

        assert session_id in session_manager._index
        session_manager.delete_session(session_id)
        assert session_id not in session_manager._index

    def test_delete_session_nonexistent_file(self, session_manager):
        """Test deleting session when file doesn't exist but index does."""
        session = session_manager.create_session(name="Test")
        session_id = session.metadata.id

        # Remove the file but keep in index
        session_path = session_manager._get_session_path(session_id)
        session_path.unlink()

        # Should still succeed
        success = session_manager.delete_session(session_id)
        assert success is True

    def test_export_session_markdown_with_no_messages(self, session_manager):
        """Test markdown export with no messages."""
        session = session_manager.create_session(name="Empty")
        session_manager.save_session(session)

        exported = session_manager.export_session(session.metadata.id, format="markdown")
        assert "# Empty" in exported
        assert "## Conversation" in exported

    def test_export_session_text_with_no_messages(self, session_manager):
        """Test text export with no messages."""
        session = session_manager.create_session(name="Empty")
        session_manager.save_session(session)

        exported = session_manager.export_session(session.metadata.id, format="text")
        assert "Empty" in exported

    def test_export_session_markdown_with_unknown_role(self, session_manager):
        """Test markdown export with unknown message role."""
        session = session_manager.create_session(name="Test")
        session.messages = [{"role": "unknown", "content": "Content"}]
        session_manager.save_session(session)

        exported = session_manager.export_session(session.metadata.id, format="markdown")
        # Should handle unknown role gracefully
        assert "Content" in exported

    def test_export_session_text_with_unknown_role(self, session_manager):
        """Test text export with unknown message role."""
        session = session_manager.create_session(name="Test")
        session.messages = [{"role": "unknown", "content": "Content"}]
        session_manager.save_session(session)

        exported = session_manager.export_session(session.metadata.id, format="text")
        # Should handle unknown role gracefully
        assert "Content" in exported

    def test_export_session_markdown_with_missing_content(self, session_manager):
        """Test markdown export with message missing content."""
        session = session_manager.create_session(name="Test")
        session.messages = [{"role": "user"}]  # No content
        session_manager.save_session(session)

        exported = session_manager.export_session(session.metadata.id, format="markdown")
        assert "### User" in exported

    def test_export_session_text_with_missing_content(self, session_manager):
        """Test text export with message missing content."""
        session = session_manager.create_session(name="Test")
        session.messages = [{"role": "user"}]  # No content
        session_manager.save_session(session)

        exported = session_manager.export_session(session.metadata.id, format="text")
        assert "[USER]" in exported

    def test_export_session_to_file_with_existing_parent(self, session_manager, temp_session_dir):
        """Test that export works when parent directory exists."""
        session = session_manager.create_session(name="Test")
        session_manager.save_session(session)

        export_path = temp_session_dir / "export.json"
        session_manager.export_session(session.metadata.id, format="json", path=export_path)

        # File should be created
        assert export_path.exists()

    def test_get_recent_sessions_returns_dict_format(self, session_manager):
        """Test that get_recent_sessions returns proper dict format."""
        session_manager.create_session(name="Test")

        recent = session_manager.get_recent_sessions(limit=1)
        assert len(recent) == 1

        item = recent[0]
        assert "id" in item
        assert "name" in item
        assert "project" in item
        assert "updated_at" in item
        assert "message_count" in item

    def test_get_recent_sessions_respects_limit(self, session_manager):
        """Test that get_recent_sessions respects the limit parameter."""
        for i in range(10):
            session_manager.create_session(name=f"Session {i}")

        recent = session_manager.get_recent_sessions(limit=3)
        assert len(recent) == 3

    def test_from_dict_orchestration_run_id_fallback(self):
        """Test line 115: active_orchestration_run_id fallback from orchestration_state."""
        data = {
            "metadata": {"id": "orch_test"},
            "orchestration_state": {"run_id": "run-fallback"},
            "orchestration_runs": None,
        }
        session = SessionData.from_dict(data)
        assert session.active_orchestration_run_id == "run-fallback"

    def test_from_dict_orchestration_runs_none_fallback(self):
        """Test line 111: orchestration_runs None fallback."""
        data = {
            "metadata": {"id": "runs_none_test"},
            "orchestration_state": {},
            "orchestration_runs": None,
        }
        session = SessionData.from_dict(data)
        assert session.orchestration_runs == []

    def test_from_dict_orchestration_runs_none_with_state(self):
        """Test line 111 with non-empty orchestration_state when runs is None."""
        data = {
            "metadata": {"id": "runs_state_test"},
            "orchestration_state": {"run_id": "r1"},
            "orchestration_runs": None,
        }
        session = SessionData.from_dict(data)
        assert len(session.orchestration_runs) == 1
        assert session.orchestration_runs[0]["run_id"] == "r1"

    def test_get_recent_sessions_default_limit(self, session_manager):
        """Test that get_recent_sessions uses default limit of 5."""
        for i in range(10):
            session_manager.create_session(name=f"Session {i}")

        recent = session_manager.get_recent_sessions()
        assert len(recent) == 5

    def test_session_manager_base_dir_creation(self, temp_session_dir):
        """Test that SessionManager creates base_dir if it doesn't exist."""
        new_dir = temp_session_dir / "new_sessions"
        assert not new_dir.exists()

        SessionManager(base_dir=new_dir)
        assert new_dir.exists()

    def test_session_manager_with_string_path(self, temp_session_dir):
        """Test SessionManager initialization with string path."""
        manager = SessionManager(base_dir=str(temp_session_dir))
        assert isinstance(manager.base_dir, Path)

    def test_session_manager_with_path_object(self, temp_session_dir):
        """Test SessionManager initialization with Path object."""
        manager = SessionManager(base_dir=temp_session_dir)
        assert isinstance(manager.base_dir, Path)

    def test_fork_session_default_name_format(self, session_manager):
        """Test that fork without name uses default format."""
        original = session_manager.create_session(name="Original Session")
        forked = session_manager.fork_session(original.metadata.id)

        assert "Fork of" in forked.metadata.name
        assert "Original Session" in forked.metadata.name

    def test_fork_session_description_format(self, session_manager):
        """Test that forked session has proper description."""
        original = session_manager.create_session(name="Original")
        forked = session_manager.fork_session(original.metadata.id)

        assert "Forked from" in forked.metadata.description

    def test_session_metadata_display_name_format(self):
        """Test display name format when name is not set."""
        metadata = SessionMetadata(id="abc123def456")
        display_name = metadata.get_display_name()

        assert "abc123" in display_name
        assert "Session" in display_name

    def test_session_metadata_display_name_includes_timestamp(self):
        """Test that display name includes timestamp when name not set."""
        metadata = SessionMetadata(id="test123")
        display_name = metadata.get_display_name()

        # Should include year-month-day format
        assert "-" in display_name  # Date separator

    def test_export_markdown_includes_project(self, session_manager):
        """Test that markdown export includes project name."""
        session = session_manager.create_session(name="Test", project="myproject")
        session_manager.save_session(session)

        exported = session_manager.export_session(session.metadata.id, format="markdown")
        assert "myproject" in exported

    def test_export_markdown_includes_created_date(self, session_manager):
        """Test that markdown export includes creation date."""
        session = session_manager.create_session(name="Test")
        session_manager.save_session(session)

        exported = session_manager.export_session(session.metadata.id, format="markdown")
        assert "Created:" in exported

    def test_export_markdown_includes_message_count(self, session_manager):
        """Test that markdown export includes message count."""
        session = session_manager.create_session(name="Test")
        session.metadata.message_count = 5
        session_manager.save_session(session)

        exported = session_manager.export_session(session.metadata.id, format="markdown")
        assert "Messages:" in exported

    def test_export_markdown_includes_token_count(self, session_manager):
        """Test that markdown export includes token count."""
        session = session_manager.create_session(name="Test")
        session.metadata.total_tokens = 5000
        session_manager.save_session(session)

        exported = session_manager.export_session(session.metadata.id, format="markdown")
        assert "Tokens:" in exported
        assert "5,000" in exported  # Formatted with comma

    def test_export_text_includes_project(self, session_manager):
        """Test that text export includes project."""
        session = session_manager.create_session(name="Test", project="myproject")
        session_manager.save_session(session)

        exported = session_manager.export_session(session.metadata.id, format="text")
        assert "myproject" in exported

    def test_export_text_includes_created_date(self, session_manager):
        """Test that text export includes creation date."""
        session = session_manager.create_session(name="Test")
        session_manager.save_session(session)

        exported = session_manager.export_session(session.metadata.id, format="text")
        assert "Created:" in exported

    def test_export_text_separator_line(self, session_manager):
        """Test that text export includes separator line."""
        session = session_manager.create_session(name="Test")
        session_manager.save_session(session)

        exported = session_manager.export_session(session.metadata.id, format="text")
        assert "-" * 50 in exported

    def test_session_data_from_dict_backward_compatibility(self):
        """Test SessionData.from_dict handles old format (metadata as root)."""
        # Old format where metadata fields were at root level
        data = {"id": "test_123", "name": "Test", "project": "proj"}
        session = SessionData.from_dict(data)
        assert session.metadata.id == "test_123"

    def test_multiple_sessions_independent_state(self, session_manager):
        """Test that multiple sessions maintain independent state."""
        session1 = session_manager.create_session(name="Session 1")
        session2 = session_manager.create_session(name="Session 2")

        session1.messages.append({"role": "user", "content": "Message 1"})
        session_manager.save_session(session1)

        loaded2 = session_manager.load_session(session2.metadata.id)
        assert len(loaded2.messages) == 0

    def test_session_manager_index_consistency(self, session_manager):
        """Test that index remains consistent after operations."""
        session1 = session_manager.create_session(name="Session 1")
        session_manager.create_session(name="Session 2")

        assert len(session_manager._index) == 2

        session_manager.delete_session(session1.metadata.id)
        assert len(session_manager._index) == 1

        session_manager.create_session(name="Session 3")
        assert len(session_manager._index) == 2
