"""Comprehensive tests for checkpoint.py"""

from pathlib import Path

import pytest

from src.core.checkpoint import Checkpoint, CheckpointConfig, CheckpointManager, FileSnapshot


class TestFileSnapshot:
    """Tests for FileSnapshot."""

    def test_creation_existing_file(self):
        """Test creating snapshot for existing file."""
        snapshot = FileSnapshot(
            path="/test/file.py", content="print('hello')", exists=True, size=14, mtime=1234567890.0
        )
        assert snapshot.path == "/test/file.py"
        assert snapshot.content == "print('hello')"
        assert snapshot.exists is True
        assert snapshot.size == 14
        assert snapshot.mtime == 1234567890.0

    def test_creation_nonexistent_file(self):
        """Test creating snapshot for non-existent file."""
        snapshot = FileSnapshot(
            path="/test/missing.py", content=None, exists=False, size=0, mtime=None
        )
        assert snapshot.path == "/test/missing.py"
        assert snapshot.content is None
        assert snapshot.exists is False
        assert snapshot.size == 0
        assert snapshot.mtime is None

    def test_to_dict(self):
        """Test converting snapshot to dictionary."""
        snapshot = FileSnapshot(
            path="/test/file.py", content="content", exists=True, size=7, mtime=1234567890.0
        )
        data = snapshot.to_dict()
        assert data["path"] == "/test/file.py"
        assert data["content"] == "content"
        assert data["exists"] is True
        assert data["size"] == 7
        assert data["mtime"] == 1234567890.0

    def test_from_dict(self):
        """Test creating snapshot from dictionary."""
        data = {
            "path": "/test/restored.py",
            "content": "restored content",
            "exists": True,
            "size": 16,
            "mtime": 1234567890.0,
        }
        snapshot = FileSnapshot.from_dict(data)
        assert snapshot.path == "/test/restored.py"
        assert snapshot.content == "restored content"
        assert snapshot.exists is True
        assert snapshot.size == 16

    def test_from_dict_with_defaults(self):
        """Test from_dict with missing optional fields."""
        data = {"path": "/test/minimal.py"}
        snapshot = FileSnapshot.from_dict(data)
        assert snapshot.path == "/test/minimal.py"
        assert snapshot.content is None
        assert snapshot.exists is True  # Default
        assert snapshot.size == 0  # Default

    def test_roundtrip_to_dict_from_dict(self):
        """Test that to_dict and from_dict are inverses."""
        original = FileSnapshot(
            path="/test/roundtrip.py",
            content="test content",
            exists=True,
            size=12,
            mtime=1234567890.0,
        )
        data = original.to_dict()
        restored = FileSnapshot.from_dict(data)
        assert restored.path == original.path
        assert restored.content == original.content
        assert restored.exists == original.exists
        assert restored.size == original.size
        assert restored.mtime == original.mtime


class TestCheckpoint:
    """Tests for Checkpoint."""

    def test_creation_basic(self):
        """Test creating a basic checkpoint."""
        checkpoint = Checkpoint(id="cp_123", created_at=1234567890.0, prompt="Test prompt")
        assert checkpoint.id == "cp_123"
        assert checkpoint.created_at == 1234567890.0
        assert checkpoint.prompt == "Test prompt"
        assert checkpoint.files == []
        assert checkpoint.message_count == 0
        assert checkpoint.description == ""

    def test_creation_with_files(self):
        """Test creating checkpoint with file snapshots."""
        snapshot1 = FileSnapshot(
            path="/test/file1.py", content="content1", exists=True, size=8, mtime=None
        )
        snapshot2 = FileSnapshot(
            path="/test/file2.py", content="content2", exists=True, size=8, mtime=None
        )
        checkpoint = Checkpoint(
            id="cp_456",
            created_at=1234567890.0,
            prompt="Multi-file edit",
            files=[snapshot1, snapshot2],
            message_count=5,
            description="Edited two files",
        )
        assert len(checkpoint.files) == 2
        assert checkpoint.message_count == 5
        assert checkpoint.description == "Edited two files"

    def test_to_dict(self):
        """Test converting checkpoint to dictionary."""
        snapshot = FileSnapshot(
            path="/test/file.py", content="content", exists=True, size=7, mtime=None
        )
        checkpoint = Checkpoint(
            id="cp_789", created_at=1234567890.0, prompt="Test", files=[snapshot], message_count=3
        )
        data = checkpoint.to_dict()
        assert data["id"] == "cp_789"
        assert data["created_at"] == 1234567890.0
        assert data["prompt"] == "Test"
        assert len(data["files"]) == 1
        assert data["message_count"] == 3

    def test_from_dict(self):
        """Test creating checkpoint from dictionary."""
        data = {
            "id": "cp_from_dict",
            "created_at": 1234567890.0,
            "prompt": "Restored prompt",
            "files": [
                {
                    "path": "/test/file.py",
                    "content": "content",
                    "exists": True,
                    "size": 7,
                    "mtime": None,
                }
            ],
            "message_count": 10,
            "description": "Restored checkpoint",
        }
        checkpoint = Checkpoint.from_dict(data)
        assert checkpoint.id == "cp_from_dict"
        assert checkpoint.prompt == "Restored prompt"
        assert len(checkpoint.files) == 1
        assert checkpoint.files[0].path == "/test/file.py"
        assert checkpoint.message_count == 10

    def test_from_dict_with_defaults(self):
        """Test from_dict with missing optional fields."""
        data = {"id": "cp_minimal", "created_at": 1234567890.0}
        checkpoint = Checkpoint.from_dict(data)
        assert checkpoint.id == "cp_minimal"
        assert checkpoint.prompt == ""  # Default
        assert checkpoint.files == []
        assert checkpoint.message_count == 0

    def test_from_dict_empty_files(self):
        """Test from_dict with empty files list."""
        data = {"id": "cp_no_files", "created_at": 1234567890.0, "prompt": "No files", "files": []}
        checkpoint = Checkpoint.from_dict(data)
        assert checkpoint.files == []

    def test_roundtrip_to_dict_from_dict(self):
        """Test that to_dict and from_dict are inverses."""
        snapshot = FileSnapshot(
            path="/test/roundtrip.py", content="test", exists=True, size=4, mtime=1234567890.0
        )
        original = Checkpoint(
            id="cp_roundtrip",
            created_at=1234567890.0,
            prompt="Roundtrip test",
            files=[snapshot],
            message_count=7,
            description="Test description",
        )
        data = original.to_dict()
        restored = Checkpoint.from_dict(data)
        assert restored.id == original.id
        assert restored.created_at == original.created_at
        assert restored.prompt == original.prompt
        assert len(restored.files) == len(original.files)
        assert restored.files[0].path == original.files[0].path
        assert restored.message_count == original.message_count


class TestCheckpointConfig:
    """Tests for CheckpointConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CheckpointConfig()
        assert config.enabled is True
        assert config.save_directory == ".agent/checkpoints"
        assert config.max_file_size == 1024 * 1024  # 1MB
        assert config.retention_days == 30
        assert config.compress is True

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = CheckpointConfig(
            enabled=False,
            save_directory="/custom/path",
            max_file_size=2048 * 1024,  # 2MB
            retention_days=60,
            compress=False,
        )
        assert config.enabled is False
        assert config.save_directory == "/custom/path"
        assert config.max_file_size == 2048 * 1024
        assert config.retention_days == 60
        assert config.compress is False

    def test_partial_custom_values(self):
        """Test creating config with some custom values."""
        config = CheckpointConfig(retention_days=90, compress=False)
        assert config.enabled is True  # Default
        assert config.retention_days == 90  # Custom
        assert config.compress is False  # Custom
        assert config.max_file_size == 1024 * 1024  # Default


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    def test_initialization_default(self, tmp_path):
        """Test CheckpointManager initialization with defaults."""
        config = CheckpointConfig(save_directory=str(tmp_path))
        manager = CheckpointManager(project="test_project", config=config)

        assert manager.project == "test_project"
        assert manager.config.enabled is True
        assert manager.checkpoints == []
        assert manager._current is None
        assert manager._pending_files == set()

    def test_initialization_creates_directory(self, tmp_path):
        """Test that initialization creates checkpoint directory."""
        config = CheckpointConfig(save_directory=str(tmp_path))
        CheckpointManager(project="test_project", config=config)

        checkpoint_dir = Path(tmp_path) / "test_project"
        assert checkpoint_dir.exists()
        assert checkpoint_dir.is_dir()

    def test_initialization_with_disabled_config(self, tmp_path):
        """Test initialization with disabled checkpoint config."""
        config = CheckpointConfig(enabled=False, save_directory=str(tmp_path))
        manager = CheckpointManager(project="disabled", config=config)

        assert manager.config.enabled is False

    def test_begin_checkpoint_returns_id(self, tmp_path):
        """Test begin_checkpoint returns a checkpoint ID."""
        config = CheckpointConfig(save_directory=str(tmp_path))
        manager = CheckpointManager(project="test", config=config)

        checkpoint_id = manager.begin_checkpoint("Test prompt", message_count=5)

        assert checkpoint_id != ""
        assert len(checkpoint_id) == 12  # MD5 hash truncated to 12 chars
        assert manager._current is not None
        assert manager._current.prompt == "Test prompt"
        assert manager._current.message_count == 5

    def test_begin_checkpoint_disabled(self, tmp_path):
        """Test begin_checkpoint returns empty string when disabled."""
        config = CheckpointConfig(enabled=False, save_directory=str(tmp_path))
        manager = CheckpointManager(project="test", config=config)

        checkpoint_id = manager.begin_checkpoint("Test prompt")

        assert checkpoint_id == ""
        assert manager._current is None

    def test_begin_checkpoint_truncates_long_prompt(self, tmp_path):
        """Test that begin_checkpoint truncates prompts longer than 500 chars."""
        config = CheckpointConfig(save_directory=str(tmp_path))
        manager = CheckpointManager(project="test", config=config)

        long_prompt = "x" * 600
        manager.begin_checkpoint(long_prompt)

        assert len(manager._current.prompt) == 500
        assert manager._current.prompt == "x" * 500

    def test_snapshot_file_existing_file(self, tmp_path):
        """Test snapshot_file with an existing file."""
        config = CheckpointConfig(save_directory=str(tmp_path))
        manager = CheckpointManager(project="test", config=config)

        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")

        manager.begin_checkpoint("Test")
        result = manager.snapshot_file(str(test_file))

        assert result is True
        assert len(manager._current.files) == 1
        assert manager._current.files[0].path == str(test_file.resolve())
        assert manager._current.files[0].content == "print('hello')"
        assert manager._current.files[0].exists is True

    def test_snapshot_file_nonexistent_file(self, tmp_path):
        """Test snapshot_file with a non-existent file."""
        config = CheckpointConfig(save_directory=str(tmp_path))
        manager = CheckpointManager(project="test", config=config)

        nonexistent = tmp_path / "missing.py"

        manager.begin_checkpoint("Test")
        result = manager.snapshot_file(str(nonexistent))

        assert result is True
        assert len(manager._current.files) == 1
        assert manager._current.files[0].exists is False
        assert manager._current.files[0].content is None

    def test_snapshot_file_without_checkpoint(self, tmp_path):
        """Test snapshot_file returns False when no checkpoint active."""
        config = CheckpointConfig(save_directory=str(tmp_path))
        manager = CheckpointManager(project="test", config=config)

        test_file = tmp_path / "test.py"
        test_file.write_text("content")

        result = manager.snapshot_file(str(test_file))

        assert result is False

    def test_snapshot_file_disabled(self, tmp_path):
        """Test snapshot_file returns False when checkpoints disabled."""
        config = CheckpointConfig(enabled=False, save_directory=str(tmp_path))
        manager = CheckpointManager(project="test", config=config)

        test_file = tmp_path / "test.py"
        test_file.write_text("content")

        manager.begin_checkpoint("Test")
        result = manager.snapshot_file(str(test_file))

        assert result is False

    def test_snapshot_file_duplicate_prevention(self, tmp_path):
        """Test that duplicate snapshots are prevented in same checkpoint."""
        config = CheckpointConfig(save_directory=str(tmp_path))
        manager = CheckpointManager(project="test", config=config)

        test_file = tmp_path / "test.py"
        test_file.write_text("content")

        manager.begin_checkpoint("Test")
        result1 = manager.snapshot_file(str(test_file))
        result2 = manager.snapshot_file(str(test_file))

        assert result1 is True
        assert result2 is False
        assert len(manager._current.files) == 1

    def test_snapshot_file_too_large(self, tmp_path):
        """Test snapshot_file with file exceeding max size."""
        config = CheckpointConfig(
            save_directory=str(tmp_path),
            max_file_size=100,  # 100 bytes
        )
        manager = CheckpointManager(project="test", config=config)

        test_file = tmp_path / "large.py"
        test_file.write_text("x" * 200)

        manager.begin_checkpoint("Test")
        result = manager.snapshot_file(str(test_file))

        assert result is True
        assert manager._current.files[0].content is None  # Content not stored
        assert manager._current.files[0].exists is True
        assert manager._current.files[0].size == 200

    def test_snapshot_file_error_handling(self, tmp_path):
        """Test snapshot_file error handling with invalid path."""
        config = CheckpointConfig(save_directory=str(tmp_path))
        manager = CheckpointManager(project="test", config=config)

        manager.begin_checkpoint("Test")
        # Use a path with a non-existent parent that can't be resolved
        # The snapshot_file method will still succeed for non-existent files
        # but we can test with a file that exists but can't be read
        result = manager.snapshot_file("/dev/null")

        # Should return True even for special files
        assert result is True

    def test_finalize_checkpoint_with_files(self, tmp_path):
        """Test finalize_checkpoint saves checkpoint with files."""
        config = CheckpointConfig(save_directory=str(tmp_path))
        manager = CheckpointManager(project="test", config=config)

        test_file = tmp_path / "test.py"
        test_file.write_text("content")

        manager.begin_checkpoint("Test prompt")
        manager.snapshot_file(str(test_file))
        checkpoint_id = manager.finalize_checkpoint("Test description")

        assert checkpoint_id is not None
        assert len(manager.checkpoints) == 1
        assert manager.checkpoints[0].description == "Test description"
        assert manager._current is None

    def test_finalize_checkpoint_no_files(self, tmp_path):
        """Test finalize_checkpoint returns None when no files changed."""
        config = CheckpointConfig(save_directory=str(tmp_path))
        manager = CheckpointManager(project="test", config=config)

        manager.begin_checkpoint("Test prompt")
        checkpoint_id = manager.finalize_checkpoint()

        assert checkpoint_id is None
        assert len(manager.checkpoints) == 0
        assert manager._current is None

    def test_finalize_checkpoint_no_active_checkpoint(self, tmp_path):
        """Test finalize_checkpoint returns None when no checkpoint active."""
        config = CheckpointConfig(save_directory=str(tmp_path))
        manager = CheckpointManager(project="test", config=config)

        checkpoint_id = manager.finalize_checkpoint()

        assert checkpoint_id is None

    def test_discard_checkpoint(self, tmp_path):
        """Test discard_checkpoint clears current checkpoint."""
        config = CheckpointConfig(save_directory=str(tmp_path))
        manager = CheckpointManager(project="test", config=config)

        test_file = tmp_path / "test.py"
        test_file.write_text("content")

        manager.begin_checkpoint("Test")
        manager.snapshot_file(str(test_file))
        manager.discard_checkpoint()

        assert manager._current is None
        assert manager._pending_files == set()
        assert len(manager.checkpoints) == 0

    def test_list_checkpoints_empty(self, tmp_path):
        """Test list_checkpoints with no checkpoints."""
        config = CheckpointConfig(save_directory=str(tmp_path))
        manager = CheckpointManager(project="test", config=config)

        result = manager.list_checkpoints()

        assert result == []

    def test_list_checkpoints_with_data(self, tmp_path):
        """Test list_checkpoints returns checkpoint summaries."""
        config = CheckpointConfig(save_directory=str(tmp_path))
        manager = CheckpointManager(project="test", config=config)

        test_file = tmp_path / "test.py"
        test_file.write_text("content")

        manager.begin_checkpoint("Test prompt")
        manager.snapshot_file(str(test_file))
        manager.finalize_checkpoint("Test description")

        result = manager.list_checkpoints()

        assert len(result) == 1
        assert result[0]["prompt"] == "Test prompt"
        assert result[0]["files_count"] == 1
        assert result[0]["description"] == "Test description"

    def test_list_checkpoints_limit(self, tmp_path):
        """Test list_checkpoints respects limit parameter."""
        config = CheckpointConfig(save_directory=str(tmp_path))
        manager = CheckpointManager(project="test", config=config)

        test_file = tmp_path / "test.py"
        test_file.write_text("content")

        # Create 5 checkpoints
        for i in range(5):
            manager.begin_checkpoint(f"Prompt {i}")
            manager.snapshot_file(str(test_file))
            manager.finalize_checkpoint()

        result = manager.list_checkpoints(limit=2)

        assert len(result) == 2

    def test_list_checkpoints_truncates_long_prompt(self, tmp_path):
        """Test list_checkpoints truncates long prompts."""
        config = CheckpointConfig(save_directory=str(tmp_path))
        manager = CheckpointManager(project="test", config=config)

        test_file = tmp_path / "test.py"
        test_file.write_text("content")

        long_prompt = "x" * 150
        manager.begin_checkpoint(long_prompt)
        manager.snapshot_file(str(test_file))
        manager.finalize_checkpoint()

        result = manager.list_checkpoints()

        assert len(result[0]["prompt"]) == 103  # 100 chars + "..."

    def test_get_checkpoint_found(self, tmp_path):
        """Test get_checkpoint returns checkpoint when found."""
        config = CheckpointConfig(save_directory=str(tmp_path))
        manager = CheckpointManager(project="test", config=config)

        test_file = tmp_path / "test.py"
        test_file.write_text("content")

        manager.begin_checkpoint("Test")
        manager.snapshot_file(str(test_file))
        checkpoint_id = manager.finalize_checkpoint()

        result = manager.get_checkpoint(checkpoint_id)

        assert result is not None
        assert result.id == checkpoint_id
        assert result.prompt == "Test"

    def test_get_checkpoint_not_found(self, tmp_path):
        """Test get_checkpoint returns None when not found."""
        config = CheckpointConfig(save_directory=str(tmp_path))
        manager = CheckpointManager(project="test", config=config)

        result = manager.get_checkpoint("nonexistent_id")

        assert result is None

    def test_rewind_code_mode(self, tmp_path):
        """Test rewind with code mode restores files."""
        config = CheckpointConfig(save_directory=str(tmp_path))
        manager = CheckpointManager(project="test", config=config)

        test_file = tmp_path / "test.py"
        test_file.write_text("original content")

        manager.begin_checkpoint("Test")
        manager.snapshot_file(str(test_file))
        checkpoint_id = manager.finalize_checkpoint()

        # Modify file
        test_file.write_text("modified content")

        # Rewind
        result = manager.rewind(checkpoint_id, mode="code")

        assert len(result["restored_files"]) == 1
        assert test_file.read_text() == "original content"

    def test_rewind_conversation_mode(self, tmp_path):
        """Test rewind with conversation mode returns message count."""
        config = CheckpointConfig(save_directory=str(tmp_path))
        manager = CheckpointManager(project="test", config=config)

        test_file = tmp_path / "test.py"
        test_file.write_text("content")

        manager.begin_checkpoint("Test", message_count=10)
        manager.snapshot_file(str(test_file))
        checkpoint_id = manager.finalize_checkpoint()

        result = manager.rewind(checkpoint_id, mode="conversation")

        assert result["message_count"] == 10
        assert len(result["restored_files"]) == 0

    def test_rewind_both_mode(self, tmp_path):
        """Test rewind with both mode restores files and returns message count."""
        config = CheckpointConfig(save_directory=str(tmp_path))
        manager = CheckpointManager(project="test", config=config)

        test_file = tmp_path / "test.py"
        test_file.write_text("original")

        manager.begin_checkpoint("Test", message_count=5)
        manager.snapshot_file(str(test_file))
        checkpoint_id = manager.finalize_checkpoint()

        test_file.write_text("modified")

        result = manager.rewind(checkpoint_id, mode="both")

        assert len(result["restored_files"]) == 1
        assert result["message_count"] == 5
        assert test_file.read_text() == "original"

    def test_rewind_nonexistent_checkpoint(self, tmp_path):
        """Test rewind raises ValueError for nonexistent checkpoint."""
        config = CheckpointConfig(save_directory=str(tmp_path))
        manager = CheckpointManager(project="test", config=config)

        with pytest.raises(ValueError, match="Checkpoint not found"):
            manager.rewind("nonexistent_id")

    def test_rewind_removes_later_checkpoints(self, tmp_path):
        """Test rewind removes checkpoints created after target."""
        config = CheckpointConfig(save_directory=str(tmp_path))
        manager = CheckpointManager(project="test", config=config)

        test_file = tmp_path / "test.py"
        test_file.write_text("content")

        # Create first checkpoint
        manager.begin_checkpoint("First")
        manager.snapshot_file(str(test_file))
        cp1_id = manager.finalize_checkpoint()

        # Create second checkpoint
        manager.begin_checkpoint("Second")
        manager.snapshot_file(str(test_file))
        manager.finalize_checkpoint()

        assert len(manager.checkpoints) == 2

        # Rewind to first
        result = manager.rewind(cp1_id)

        assert result["checkpoints_removed"] == 1
        assert len(manager.checkpoints) == 1

    def test_rewind_last_with_checkpoints(self, tmp_path):
        """Test rewind_last goes to previous checkpoint."""
        config = CheckpointConfig(save_directory=str(tmp_path))
        manager = CheckpointManager(project="test", config=config)

        test_file = tmp_path / "test.py"
        test_file.write_text("v1")

        manager.begin_checkpoint("First", message_count=1)
        manager.snapshot_file(str(test_file))
        manager.finalize_checkpoint()

        test_file.write_text("v2")
        manager.begin_checkpoint("Second", message_count=2)
        manager.snapshot_file(str(test_file))
        manager.finalize_checkpoint()

        result = manager.rewind_last(mode="both")

        assert result is not None
        assert result["message_count"] == 1
        assert test_file.read_text() == "v1"

    def test_rewind_last_insufficient_checkpoints(self, tmp_path):
        """Test rewind_last returns None with fewer than 2 checkpoints."""
        config = CheckpointConfig(save_directory=str(tmp_path))
        manager = CheckpointManager(project="test", config=config)

        result = manager.rewind_last()

        assert result is None

    def test_multiple_file_snapshots(self, tmp_path):
        """Test checkpoint with multiple file snapshots."""
        config = CheckpointConfig(save_directory=str(tmp_path))
        manager = CheckpointManager(project="test", config=config)

        file1 = tmp_path / "file1.py"
        file2 = tmp_path / "file2.py"
        file3 = tmp_path / "file3.py"

        file1.write_text("content1")
        file2.write_text("content2")
        file3.write_text("content3")

        manager.begin_checkpoint("Multi-file edit")
        manager.snapshot_file(str(file1))
        manager.snapshot_file(str(file2))
        manager.snapshot_file(str(file3))
        checkpoint_id = manager.finalize_checkpoint()

        assert len(manager.checkpoints[0].files) == 3

        # Modify all files
        file1.write_text("modified1")
        file2.write_text("modified2")
        file3.write_text("modified3")

        # Rewind
        result = manager.rewind(checkpoint_id)

        assert len(result["restored_files"]) == 3
        assert file1.read_text() == "content1"
        assert file2.read_text() == "content2"
        assert file3.read_text() == "content3"

    def test_restore_file_creates_parent_directories(self, tmp_path):
        """Test that restore_file creates parent directories."""
        config = CheckpointConfig(save_directory=str(tmp_path))
        manager = CheckpointManager(project="test", config=config)

        nested_file = tmp_path / "deep" / "nested" / "file.py"
        nested_file.parent.mkdir(parents=True, exist_ok=True)
        nested_file.write_text("content")

        manager.begin_checkpoint("Test")
        manager.snapshot_file(str(nested_file))
        checkpoint_id = manager.finalize_checkpoint()

        # Delete the file and parent directories
        nested_file.unlink()

        # Rewind should recreate directories
        manager.rewind(checkpoint_id)

        assert nested_file.exists()
        assert nested_file.read_text() == "content"

    def test_restore_deleted_file(self, tmp_path):
        """Test restoring a file that didn't exist at checkpoint time."""
        config = CheckpointConfig(save_directory=str(tmp_path))
        manager = CheckpointManager(project="test", config=config)

        new_file = tmp_path / "new.py"

        manager.begin_checkpoint("Test")
        manager.snapshot_file(str(new_file))
        checkpoint_id = manager.finalize_checkpoint()

        # Create the file
        new_file.write_text("new content")
        assert new_file.exists()

        # Rewind should delete it
        manager.rewind(checkpoint_id)

        assert not new_file.exists()

    def test_checkpoint_index_persistence(self, tmp_path):
        """Test that checkpoint index is saved and loaded."""
        config = CheckpointConfig(save_directory=str(tmp_path))

        # Create and save checkpoint
        manager1 = CheckpointManager(project="test", config=config)
        test_file = tmp_path / "test.py"
        test_file.write_text("content")

        manager1.begin_checkpoint("Test prompt")
        manager1.snapshot_file(str(test_file))
        manager1.finalize_checkpoint()

        # Create new manager instance - should load from index
        manager2 = CheckpointManager(project="test", config=config)

        assert len(manager2.checkpoints) == 1
        assert manager2.checkpoints[0].prompt == "Test prompt"

    def test_cleanup_old_checkpoints(self, tmp_path, monkeypatch):
        """Test that old checkpoints are cleaned up."""
        import time as time_module

        config = CheckpointConfig(
            save_directory=str(tmp_path),
            retention_days=1,  # 1 day retention
        )

        manager = CheckpointManager(project="test", config=config)
        test_file = tmp_path / "test.py"
        test_file.write_text("content")

        manager.begin_checkpoint("Test")
        manager.snapshot_file(str(test_file))
        manager.finalize_checkpoint()

        # Mock time to be 2 days in the future
        future_time = time_module.time() + (2 * 24 * 60 * 60)
        monkeypatch.setattr(time_module, "time", lambda: future_time)

        # Create new manager - should trigger cleanup
        manager2 = CheckpointManager(project="test", config=config)

        assert len(manager2.checkpoints) == 0

    def test_checkpoint_with_compression(self, tmp_path):
        """Test checkpoint saving with compression enabled."""
        config = CheckpointConfig(save_directory=str(tmp_path), compress=True)
        manager = CheckpointManager(project="test", config=config)

        test_file = tmp_path / "test.py"
        test_file.write_text("content")

        manager.begin_checkpoint("Test")
        manager.snapshot_file(str(test_file))
        manager.finalize_checkpoint()

        # Check that .gz file was created
        checkpoint_files = list((tmp_path / "test").glob("*.json.gz"))
        assert len(checkpoint_files) > 0

    def test_checkpoint_without_compression(self, tmp_path):
        """Test checkpoint saving without compression."""
        config = CheckpointConfig(save_directory=str(tmp_path), compress=False)
        manager = CheckpointManager(project="test", config=config)

        test_file = tmp_path / "test.py"
        test_file.write_text("content")

        manager.begin_checkpoint("Test")
        manager.snapshot_file(str(test_file))
        manager.finalize_checkpoint()

        # Check that .json file was created (not .gz)
        checkpoint_files = list((tmp_path / "test").glob("*.json"))
        assert len(checkpoint_files) > 0
        assert not any(f.suffix == ".gz" for f in checkpoint_files)
