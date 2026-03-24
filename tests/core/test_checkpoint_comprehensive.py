"""
Comprehensive test suite for src/core/checkpoint.py

Tests cover:
- FileSnapshot class
- Checkpoint class
- CheckpointConfig class
- CheckpointManager initialization
- Checkpoint lifecycle (begin, snapshot, finalize)
- File restoration and rewind
- Cleanup and retention
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from src.core.checkpoint import (
    Checkpoint,
    CheckpointConfig,
    CheckpointManager,
    FileSnapshot,
)


class TestFileSnapshot:
    """Tests for FileSnapshot dataclass."""

    def test_file_snapshot_creation_existing_file(self):
        """Test creating snapshot for existing file."""
        snapshot = FileSnapshot(
            path="/path/to/file.py",
            content="print('hello')",
            exists=True,
            size=14,
            mtime=1234567890.0,
        )

        assert snapshot.path == "/path/to/file.py"
        assert snapshot.content == "print('hello')"
        assert snapshot.exists is True
        assert snapshot.size == 14
        assert snapshot.mtime == 1234567890.0

    def test_file_snapshot_creation_nonexistent_file(self):
        """Test creating snapshot for non-existent file."""
        snapshot = FileSnapshot(
            path="/path/to/new.py",
            content=None,
            exists=False,
            size=0,
            mtime=None,
        )

        assert snapshot.path == "/path/to/new.py"
        assert snapshot.content is None
        assert snapshot.exists is False
        assert snapshot.size == 0
        assert snapshot.mtime is None

    def test_file_snapshot_to_dict(self):
        """Test serializing snapshot to dict."""
        snapshot = FileSnapshot(
            path="/test.py",
            content="code",
            exists=True,
            size=4,
            mtime=1000.0,
        )

        result = snapshot.to_dict()

        assert result == {
            "path": "/test.py",
            "content": "code",
            "exists": True,
            "size": 4,
            "mtime": 1000.0,
        }

    def test_file_snapshot_from_dict(self):
        """Test deserializing snapshot from dict."""
        data = {
            "path": "/test.py",
            "content": "code",
            "exists": True,
            "size": 4,
            "mtime": 1000.0,
        }

        snapshot = FileSnapshot.from_dict(data)

        assert snapshot.path == "/test.py"
        assert snapshot.content == "code"
        assert snapshot.exists is True
        assert snapshot.size == 4
        assert snapshot.mtime == 1000.0

    def test_file_snapshot_from_dict_with_defaults(self):
        """Test deserializing with missing optional fields."""
        data = {
            "path": "/test.py",
        }

        snapshot = FileSnapshot.from_dict(data)

        assert snapshot.path == "/test.py"
        assert snapshot.content is None
        assert snapshot.exists is True  # Default
        assert snapshot.size == 0  # Default
        assert snapshot.mtime is None

    def test_file_snapshot_round_trip(self):
        """Test serialization round-trip."""
        original = FileSnapshot(
            path="/file.py",
            content="test",
            exists=True,
            size=4,
            mtime=2000.0,
        )

        data = original.to_dict()
        restored = FileSnapshot.from_dict(data)

        assert restored.path == original.path
        assert restored.content == original.content
        assert restored.exists == original.exists
        assert restored.size == original.size
        assert restored.mtime == original.mtime


class TestCheckpoint:
    """Tests for Checkpoint dataclass."""

    def test_checkpoint_creation_minimal(self):
        """Test creating checkpoint with minimal fields."""
        checkpoint = Checkpoint(
            id="abc123",
            created_at=1000.0,
            prompt="test prompt",
        )

        assert checkpoint.id == "abc123"
        assert checkpoint.created_at == 1000.0
        assert checkpoint.prompt == "test prompt"
        assert checkpoint.files == []
        assert checkpoint.message_count == 0
        assert checkpoint.description == ""

    def test_checkpoint_creation_full(self):
        """Test creating checkpoint with all fields."""
        snapshot = FileSnapshot("/test.py", "code", True, 4, 1000.0)

        checkpoint = Checkpoint(
            id="xyz789",
            created_at=2000.0,
            prompt="full test",
            files=[snapshot],
            message_count=5,
            description="Test checkpoint",
        )

        assert checkpoint.id == "xyz789"
        assert checkpoint.created_at == 2000.0
        assert checkpoint.prompt == "full test"
        assert len(checkpoint.files) == 1
        assert checkpoint.files[0] == snapshot
        assert checkpoint.message_count == 5
        assert checkpoint.description == "Test checkpoint"

    def test_checkpoint_to_dict(self):
        """Test serializing checkpoint to dict."""
        snapshot = FileSnapshot("/test.py", "code", True, 4, 1000.0)
        checkpoint = Checkpoint(
            id="test123",
            created_at=3000.0,
            prompt="serialize test",
            files=[snapshot],
            message_count=3,
            description="desc",
        )

        result = checkpoint.to_dict()

        assert result["id"] == "test123"
        assert result["created_at"] == 3000.0
        assert result["prompt"] == "serialize test"
        assert len(result["files"]) == 1
        assert result["files"][0]["path"] == "/test.py"
        assert result["message_count"] == 3
        assert result["description"] == "desc"

    def test_checkpoint_from_dict(self):
        """Test deserializing checkpoint from dict."""
        data = {
            "id": "test456",
            "created_at": 4000.0,
            "prompt": "deserialize test",
            "files": [
                {
                    "path": "/file.py",
                    "content": "data",
                    "exists": True,
                    "size": 4,
                    "mtime": 1500.0,
                }
            ],
            "message_count": 7,
            "description": "test desc",
        }

        checkpoint = Checkpoint.from_dict(data)

        assert checkpoint.id == "test456"
        assert checkpoint.created_at == 4000.0
        assert checkpoint.prompt == "deserialize test"
        assert len(checkpoint.files) == 1
        assert checkpoint.files[0].path == "/file.py"
        assert checkpoint.message_count == 7
        assert checkpoint.description == "test desc"

    def test_checkpoint_from_dict_with_defaults(self):
        """Test deserializing with missing optional fields."""
        data = {
            "id": "min123",
            "created_at": 5000.0,
        }

        checkpoint = Checkpoint.from_dict(data)

        assert checkpoint.id == "min123"
        assert checkpoint.created_at == 5000.0
        assert checkpoint.prompt == ""
        assert checkpoint.files == []
        assert checkpoint.message_count == 0
        assert checkpoint.description == ""

    def test_checkpoint_round_trip(self):
        """Test checkpoint serialization round-trip."""
        snapshot1 = FileSnapshot("/a.py", "a", True, 1, 100.0)
        snapshot2 = FileSnapshot("/b.py", "b", True, 1, 200.0)

        original = Checkpoint(
            id="round123",
            created_at=6000.0,
            prompt="round trip",
            files=[snapshot1, snapshot2],
            message_count=10,
            description="test",
        )

        data = original.to_dict()
        restored = Checkpoint.from_dict(data)

        assert restored.id == original.id
        assert restored.created_at == original.created_at
        assert restored.prompt == original.prompt
        assert len(restored.files) == len(original.files)
        assert restored.message_count == original.message_count
        assert restored.description == original.description


class TestCheckpointConfig:
    """Tests for CheckpointConfig dataclass."""

    def test_checkpoint_config_defaults(self):
        """Test default configuration values."""
        config = CheckpointConfig()

        assert config.enabled is True
        assert config.save_directory == ".agent/checkpoints"
        assert config.max_file_size == 1024 * 1024
        assert config.retention_days == 30
        assert config.compress is True

    def test_checkpoint_config_custom_values(self):
        """Test custom configuration values."""
        config = CheckpointConfig(
            enabled=False,
            save_directory="/custom/path",
            max_file_size=500000,
            retention_days=7,
            compress=False,
        )

        assert config.enabled is False
        assert config.save_directory == "/custom/path"
        assert config.max_file_size == 500000
        assert config.retention_days == 7
        assert config.compress is False

    def test_checkpoint_config_partial_override(self):
        """Test overriding only some config values."""
        config = CheckpointConfig(
            retention_days=60,
            compress=False,
        )

        assert config.enabled is True  # Default
        assert config.save_directory == ".agent/checkpoints"  # Default
        assert config.max_file_size == 1024 * 1024  # Default
        assert config.retention_days == 60  # Custom
        assert config.compress is False  # Custom


class TestCheckpointManagerInit:
    """Tests for CheckpointManager initialization."""

    def test_manager_initialization_default(self):
        """Test manager initialization with defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir)
            manager = CheckpointManager(project="test", config=config)

            assert manager.project == "test"
            assert manager.config == config
            assert manager.checkpoints == []
            assert manager._current is None
            assert manager._pending_files == set()

    def test_manager_creates_directory(self):
        """Test that manager creates checkpoint directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir) / "checkpoints"
            config = CheckpointConfig(save_directory=str(save_dir))

            CheckpointManager(project="test", config=config)

            expected_dir = save_dir / "test"
            assert expected_dir.exists()
            assert expected_dir.is_dir()

    def test_manager_with_custom_config(self):
        """Test manager with custom configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(
                enabled=True,
                save_directory=tmpdir,
                max_file_size=500000,
                retention_days=7,
                compress=False,
            )

            manager = CheckpointManager(project="custom", config=config)

            assert manager.config.max_file_size == 500000
            assert manager.config.retention_days == 7
            assert manager.config.compress is False

    def test_manager_default_project_name(self):
        """Test manager with default project name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir)
            manager = CheckpointManager(config=config)

            assert manager.project == "default"

    @patch("src.core.checkpoint.CheckpointManager._load_index")
    @patch("src.core.checkpoint.CheckpointManager._cleanup_old_checkpoints")
    def test_manager_calls_initialization_methods(self, mock_cleanup, mock_load):
        """Test that manager calls load and cleanup on init."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir)
            CheckpointManager(project="test", config=config)

            mock_load.assert_called_once()
            mock_cleanup.assert_called_once()


class TestCheckpointLifecycle:
    """Tests for checkpoint lifecycle operations."""

    def test_begin_checkpoint(self):
        """Test beginning a new checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir)
            manager = CheckpointManager(project="test", config=config)

            checkpoint_id = manager.begin_checkpoint("test prompt", message_count=5)

            assert checkpoint_id != ""
            assert manager._current is not None
            assert manager._current.id == checkpoint_id
            assert manager._current.prompt == "test prompt"
            assert manager._current.message_count == 5
            assert manager._pending_files == set()

    def test_begin_checkpoint_truncates_long_prompt(self):
        """Test that long prompts are truncated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir)
            manager = CheckpointManager(project="test", config=config)

            long_prompt = "x" * 1000
            manager.begin_checkpoint(long_prompt, message_count=0)

            assert len(manager._current.prompt) == 500

    def test_begin_checkpoint_when_disabled(self):
        """Test that begin_checkpoint returns empty string when disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(enabled=False, save_directory=tmpdir)
            manager = CheckpointManager(project="test", config=config)

            checkpoint_id = manager.begin_checkpoint("test", message_count=0)

            assert checkpoint_id == ""
            assert manager._current is None

    def test_finalize_checkpoint_with_files(self):
        """Test finalizing checkpoint with file changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir, compress=False)
            manager = CheckpointManager(project="test", config=config)

            # Create a test file
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("content")

            # Begin checkpoint and snapshot file
            checkpoint_id = manager.begin_checkpoint("test", message_count=0)
            manager.snapshot_file(str(test_file))

            # Finalize
            result_id = manager.finalize_checkpoint("test description")

            assert result_id == checkpoint_id
            assert len(manager.checkpoints) == 1
            assert manager.checkpoints[0].id == checkpoint_id
            assert manager.checkpoints[0].description == "test description"
            assert manager._current is None
            assert manager._pending_files == set()

    def test_finalize_checkpoint_without_files(self):
        """Test that checkpoint without files is discarded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir)
            manager = CheckpointManager(project="test", config=config)

            manager.begin_checkpoint("test", message_count=0)
            result_id = manager.finalize_checkpoint()

            assert result_id is None
            assert len(manager.checkpoints) == 0
            assert manager._current is None

    def test_finalize_checkpoint_when_no_current(self):
        """Test finalizing when no checkpoint is active."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir)
            manager = CheckpointManager(project="test", config=config)

            result_id = manager.finalize_checkpoint()

            assert result_id is None

    def test_discard_checkpoint(self):
        """Test discarding current checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir)
            manager = CheckpointManager(project="test", config=config)

            # Create a test file
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("content")

            manager.begin_checkpoint("test", message_count=0)
            manager.snapshot_file(str(test_file))

            # Discard
            manager.discard_checkpoint()

            assert manager._current is None
            assert manager._pending_files == set()
            assert len(manager.checkpoints) == 0


class TestFileSnapshots:
    """Tests for file snapshot operations."""

    def test_snapshot_existing_file(self):
        """Test taking snapshot of existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir)
            manager = CheckpointManager(project="test", config=config)

            # Create test file
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("print('hello')")

            manager.begin_checkpoint("test", message_count=0)
            result = manager.snapshot_file(str(test_file))

            assert result is True
            assert len(manager._current.files) == 1

            snapshot = manager._current.files[0]
            assert snapshot.path == str(test_file.resolve())
            assert snapshot.content == "print('hello')"
            assert snapshot.exists is True
            assert snapshot.size == 14

    def test_snapshot_nonexistent_file(self):
        """Test taking snapshot of non-existent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir)
            manager = CheckpointManager(project="test", config=config)

            nonexistent = Path(tmpdir) / "nonexistent.py"

            manager.begin_checkpoint("test", message_count=0)
            result = manager.snapshot_file(str(nonexistent))

            assert result is True
            assert len(manager._current.files) == 1

            snapshot = manager._current.files[0]
            assert snapshot.path == str(nonexistent.resolve())
            assert snapshot.content is None
            assert snapshot.exists is False
            assert snapshot.size == 0
            assert snapshot.mtime is None

    def test_snapshot_duplicate_file_ignored(self):
        """Test that duplicate snapshots are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir)
            manager = CheckpointManager(project="test", config=config)

            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("content")

            manager.begin_checkpoint("test", message_count=0)

            result1 = manager.snapshot_file(str(test_file))
            result2 = manager.snapshot_file(str(test_file))

            assert result1 is True
            assert result2 is False
            assert len(manager._current.files) == 1

    def test_snapshot_large_file(self):
        """Test snapshot of file exceeding max size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(
                save_directory=tmpdir,
                max_file_size=100,  # Small limit
            )
            manager = CheckpointManager(project="test", config=config)

            # Create large file
            test_file = Path(tmpdir) / "large.txt"
            test_file.write_text("x" * 200)

            manager.begin_checkpoint("test", message_count=0)
            result = manager.snapshot_file(str(test_file))

            assert result is True
            snapshot = manager._current.files[0]
            assert snapshot.content is None  # Content not saved
            assert snapshot.exists is True
            assert snapshot.size == 200

    def test_snapshot_when_disabled(self):
        """Test that snapshot returns False when disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(enabled=False, save_directory=tmpdir)
            manager = CheckpointManager(project="test", config=config)

            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("content")

            result = manager.snapshot_file(str(test_file))

            assert result is False

    def test_snapshot_when_no_current_checkpoint(self):
        """Test snapshot without active checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir)
            manager = CheckpointManager(project="test", config=config)

            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("content")

            result = manager.snapshot_file(str(test_file))

            assert result is False

    def test_snapshot_multiple_files(self):
        """Test snapshotting multiple files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir)
            manager = CheckpointManager(project="test", config=config)

            file1 = Path(tmpdir) / "file1.py"
            file2 = Path(tmpdir) / "file2.py"
            file1.write_text("content1")
            file2.write_text("content2")

            manager.begin_checkpoint("test", message_count=0)
            manager.snapshot_file(str(file1))
            manager.snapshot_file(str(file2))

            assert len(manager._current.files) == 2
            assert manager._current.files[0].content == "content1"
            assert manager._current.files[1].content == "content2"


class TestRewindOperations:
    """Tests for checkpoint rewind functionality."""

    def test_rewind_code_only(self):
        """Test rewinding code changes only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir, compress=False)
            manager = CheckpointManager(project="test", config=config)

            # Create and modify file
            test_file = Path(tmpdir) / "code.py"
            test_file.write_text("original")

            manager.begin_checkpoint("checkpoint 1", message_count=5)
            manager.snapshot_file(str(test_file))
            manager.finalize_checkpoint()

            # Modify file
            test_file.write_text("modified")

            # Rewind
            result = manager.rewind(manager.checkpoints[0].id, mode="code")

            assert test_file.read_text() == "original"
            assert len(result["restored_files"]) == 1
            assert result["message_count"] is None
            assert result["checkpoints_removed"] == 0

    def test_rewind_conversation_only(self):
        """Test rewinding conversation only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir, compress=False)
            manager = CheckpointManager(project="test", config=config)

            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("original")

            manager.begin_checkpoint("checkpoint", message_count=10)
            manager.snapshot_file(str(test_file))
            manager.finalize_checkpoint()

            # Modify file
            test_file.write_text("modified")

            # Rewind conversation only
            result = manager.rewind(manager.checkpoints[0].id, mode="conversation")

            # File should not be restored
            assert test_file.read_text() == "modified"
            assert len(result["restored_files"]) == 0
            assert result["message_count"] == 10

    def test_rewind_both(self):
        """Test rewinding both code and conversation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir, compress=False)
            manager = CheckpointManager(project="test", config=config)

            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("original")

            manager.begin_checkpoint("checkpoint", message_count=15)
            manager.snapshot_file(str(test_file))
            manager.finalize_checkpoint()

            test_file.write_text("modified")

            result = manager.rewind(manager.checkpoints[0].id, mode="both")

            assert test_file.read_text() == "original"
            assert len(result["restored_files"]) == 1
            assert result["message_count"] == 15

    def test_rewind_nonexistent_checkpoint(self):
        """Test rewinding to non-existent checkpoint raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir)
            manager = CheckpointManager(project="test", config=config)

            with pytest.raises(ValueError, match="Checkpoint not found"):
                manager.rewind("nonexistent123")

    def test_rewind_removes_later_checkpoints(self):
        """Test that rewind removes checkpoints after target."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir, compress=False)
            manager = CheckpointManager(project="test", config=config)

            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("v1")

            # Create 3 checkpoints
            manager.begin_checkpoint("cp1", message_count=1)
            manager.snapshot_file(str(test_file))
            cp1_id = manager.finalize_checkpoint()

            test_file.write_text("v2")
            manager.begin_checkpoint("cp2", message_count=2)
            manager.snapshot_file(str(test_file))
            manager.finalize_checkpoint()

            test_file.write_text("v3")
            manager.begin_checkpoint("cp3", message_count=3)
            manager.snapshot_file(str(test_file))
            manager.finalize_checkpoint()

            assert len(manager.checkpoints) == 3

            # Rewind to first checkpoint
            result = manager.rewind(cp1_id, mode="code")

            assert len(manager.checkpoints) == 1
            assert result["checkpoints_removed"] == 2
            assert test_file.read_text() == "v1"

    def test_rewind_last(self):
        """Test rewinding to previous checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir, compress=False)
            manager = CheckpointManager(project="test", config=config)

            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("v1")

            manager.begin_checkpoint("cp1", message_count=1)
            manager.snapshot_file(str(test_file))
            manager.finalize_checkpoint()

            test_file.write_text("v2")
            manager.begin_checkpoint("cp2", message_count=2)
            manager.snapshot_file(str(test_file))
            manager.finalize_checkpoint()

            # Rewind to previous
            result = manager.rewind_last(mode="code")

            assert result is not None
            assert test_file.read_text() == "v1"
            assert len(manager.checkpoints) == 1

    def test_rewind_last_with_no_checkpoints(self):
        """Test rewind_last returns None with no checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir)
            manager = CheckpointManager(project="test", config=config)

            result = manager.rewind_last()

            assert result is None

    def test_rewind_last_with_single_checkpoint(self):
        """Test rewind_last returns None with only one checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir, compress=False)
            manager = CheckpointManager(project="test", config=config)

            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("content")

            manager.begin_checkpoint("cp1", message_count=1)
            manager.snapshot_file(str(test_file))
            manager.finalize_checkpoint()

            result = manager.rewind_last()

            assert result is None

    def test_restore_deleted_file(self):
        """Test restoring a file that was deleted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir, compress=False)
            manager = CheckpointManager(project="test", config=config)

            # Snapshot non-existent file
            test_file = Path(tmpdir) / "new.py"

            manager.begin_checkpoint("cp1", message_count=1)
            manager.snapshot_file(str(test_file))
            manager.finalize_checkpoint()

            # Create the file
            test_file.write_text("new content")
            assert test_file.exists()

            # Rewind should delete it
            manager.rewind(manager.checkpoints[0].id, mode="code")

            assert not test_file.exists()

    def test_rewind_with_failed_file_restoration(self):
        """Test rewind handles file restoration failures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir, compress=False)
            manager = CheckpointManager(project="test", config=config)

            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("content")

            manager.begin_checkpoint("cp1", message_count=1)
            manager.snapshot_file(str(test_file))
            manager.finalize_checkpoint()

            # Corrupt the snapshot
            manager.checkpoints[0].files[0].content = None
            manager.checkpoints[0].files[0].exists = True

            result = manager.rewind(manager.checkpoints[0].id, mode="code")

            assert len(result["failed_files"]) == 1
            assert len(result["restored_files"]) == 0


class TestCheckpointListing:
    """Tests for listing and retrieving checkpoints."""

    def test_list_checkpoints_empty(self):
        """Test listing when no checkpoints exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir)
            manager = CheckpointManager(project="test", config=config)

            result = manager.list_checkpoints()

            assert result == []

    def test_list_checkpoints_single(self):
        """Test listing single checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir, compress=False)
            manager = CheckpointManager(project="test", config=config)

            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("content")

            manager.begin_checkpoint("test prompt", message_count=5)
            manager.snapshot_file(str(test_file))
            manager.finalize_checkpoint("test description")

            result = manager.list_checkpoints()

            assert len(result) == 1
            assert "id" in result[0]
            assert "test prompt" in result[0]["prompt"]
            assert result[0]["files_count"] == 1
            assert result[0]["message_count"] == 5
            assert result[0]["description"] == "test description"

    def test_list_checkpoints_multiple(self):
        """Test listing multiple checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir, compress=False)
            manager = CheckpointManager(project="test", config=config)

            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("v1")

            # Create 3 checkpoints
            for i in range(3):
                manager.begin_checkpoint(f"prompt {i}", message_count=i)
                manager.snapshot_file(str(test_file))
                manager.finalize_checkpoint(f"desc {i}")
                test_file.write_text(f"v{i + 2}")
                time.sleep(0.01)  # Ensure different timestamps

            result = manager.list_checkpoints()

            assert len(result) == 3
            # Should be in reverse order (newest first)
            assert "prompt 2" in result[0]["prompt"]
            assert "prompt 0" in result[2]["prompt"]

    def test_list_checkpoints_with_limit(self):
        """Test listing with limit parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir, compress=False)
            manager = CheckpointManager(project="test", config=config)

            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("content")

            # Create 5 checkpoints
            for i in range(5):
                manager.begin_checkpoint(f"prompt {i}", message_count=i)
                manager.snapshot_file(str(test_file))
                manager.finalize_checkpoint()
                time.sleep(0.01)

            result = manager.list_checkpoints(limit=3)

            assert len(result) == 3

    def test_list_checkpoints_truncates_long_prompt(self):
        """Test that long prompts are truncated in listing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir, compress=False)
            manager = CheckpointManager(project="test", config=config)

            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("content")

            long_prompt = "x" * 200
            manager.begin_checkpoint(long_prompt, message_count=0)
            manager.snapshot_file(str(test_file))
            manager.finalize_checkpoint()

            result = manager.list_checkpoints()

            assert len(result[0]["prompt"]) <= 103  # 100 + "..."

    def test_get_checkpoint_by_id(self):
        """Test retrieving checkpoint by ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir, compress=False)
            manager = CheckpointManager(project="test", config=config)

            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("content")

            manager.begin_checkpoint("test", message_count=5)
            manager.snapshot_file(str(test_file))
            cp_id = manager.finalize_checkpoint()

            checkpoint = manager.get_checkpoint(cp_id)

            assert checkpoint is not None
            assert checkpoint.id == cp_id
            assert checkpoint.prompt == "test"
            assert checkpoint.message_count == 5

    def test_get_checkpoint_nonexistent(self):
        """Test getting non-existent checkpoint returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir)
            manager = CheckpointManager(project="test", config=config)

            checkpoint = manager.get_checkpoint("nonexistent")

            assert checkpoint is None


class TestPersistence:
    """Tests for checkpoint persistence and loading."""

    def test_save_and_load_checkpoint_uncompressed(self):
        """Test saving and loading uncompressed checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir, compress=False)
            manager = CheckpointManager(project="test", config=config)

            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("content")

            manager.begin_checkpoint("test", message_count=5)
            manager.snapshot_file(str(test_file))
            cp_id = manager.finalize_checkpoint("description")

            # Load checkpoint
            loaded = manager._load_checkpoint(cp_id)

            assert loaded is not None
            assert loaded.id == cp_id
            assert loaded.prompt == "test"
            assert loaded.message_count == 5
            assert loaded.description == "description"
            assert len(loaded.files) == 1

    def test_save_and_load_checkpoint_compressed(self):
        """Test saving and loading compressed checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir, compress=True)
            manager = CheckpointManager(project="test", config=config)

            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("content")

            manager.begin_checkpoint("test", message_count=5)
            manager.snapshot_file(str(test_file))
            cp_id = manager.finalize_checkpoint()

            # Verify compressed file exists
            cp_path = manager._get_checkpoint_path(cp_id)
            assert cp_path.suffix == ".gz"
            assert cp_path.exists()

            # Load checkpoint
            loaded = manager._load_checkpoint(cp_id)

            assert loaded is not None
            assert loaded.id == cp_id

    def test_load_nonexistent_checkpoint(self):
        """Test loading non-existent checkpoint returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir)
            manager = CheckpointManager(project="test", config=config)

            loaded = manager._load_checkpoint("nonexistent")

            assert loaded is None

    def test_save_and_load_index(self):
        """Test saving and loading checkpoint index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir, compress=False)
            manager = CheckpointManager(project="test", config=config)

            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("content")

            # Create checkpoints
            manager.begin_checkpoint("cp1", message_count=1)
            manager.snapshot_file(str(test_file))
            manager.finalize_checkpoint()

            manager.begin_checkpoint("cp2", message_count=2)
            manager.snapshot_file(str(test_file))
            manager.finalize_checkpoint()

            # Create new manager to load from index
            manager2 = CheckpointManager(project="test", config=config)

            assert len(manager2.checkpoints) == 2
            assert manager2.checkpoints[0].prompt == "cp1"
            assert manager2.checkpoints[1].prompt == "cp2"

    def test_load_index_with_missing_checkpoint_files(self):
        """Test loading index when checkpoint files are missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir, compress=False)
            manager = CheckpointManager(project="test", config=config)

            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("content")

            manager.begin_checkpoint("test", message_count=5)
            manager.snapshot_file(str(test_file))
            cp_id = manager.finalize_checkpoint()

            # Delete checkpoint file but keep index
            cp_path = manager._get_checkpoint_path(cp_id)
            cp_path.unlink()

            # Load should create minimal checkpoint from index
            manager2 = CheckpointManager(project="test", config=config)

            assert len(manager2.checkpoints) == 1
            assert manager2.checkpoints[0].id == cp_id
            assert manager2.checkpoints[0].prompt == "test"

    def test_delete_checkpoint_file(self):
        """Test deleting checkpoint file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir, compress=False)
            manager = CheckpointManager(project="test", config=config)

            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("content")

            manager.begin_checkpoint("test", message_count=0)
            manager.snapshot_file(str(test_file))
            cp_id = manager.finalize_checkpoint()

            cp_path = manager._get_checkpoint_path(cp_id)
            assert cp_path.exists()

            manager._delete_checkpoint_file(cp_id)

            assert not cp_path.exists()

    def test_delete_checkpoint_file_both_formats(self):
        """Test deleting checkpoint file tries both compressed and uncompressed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir, compress=True)
            manager = CheckpointManager(project="test", config=config)

            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("content")

            manager.begin_checkpoint("test", message_count=0)
            manager.snapshot_file(str(test_file))
            cp_id = manager.finalize_checkpoint()

            # Should delete .json.gz file
            manager._delete_checkpoint_file(cp_id)

            gz_path = manager._base_dir / f"{cp_id}.json.gz"
            json_path = manager._base_dir / f"{cp_id}.json"

            assert not gz_path.exists()
            assert not json_path.exists()


class TestCleanup:
    """Tests for checkpoint cleanup and retention."""

    def test_cleanup_old_checkpoints(self):
        """Test cleanup removes old checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(
                save_directory=tmpdir,
                retention_days=1,
                compress=False,
            )
            manager = CheckpointManager(project="test", config=config)

            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("content")

            # Create old checkpoint
            old_time = time.time() - (2 * 24 * 60 * 60)  # 2 days ago
            manager.begin_checkpoint("old", message_count=0)
            manager.snapshot_file(str(test_file))
            manager._current.created_at = old_time
            old_id = manager.finalize_checkpoint()

            # Create recent checkpoint
            manager.begin_checkpoint("recent", message_count=0)
            manager.snapshot_file(str(test_file))
            manager.finalize_checkpoint()

            # Manually trigger cleanup
            manager._cleanup_old_checkpoints()

            # Old checkpoint should be removed
            assert len(manager.checkpoints) == 1
            assert manager.checkpoints[0].prompt == "recent"

            # Old checkpoint file should be deleted
            old_path = manager._get_checkpoint_path(old_id)
            assert not old_path.exists()

    def test_cleanup_with_zero_retention(self):
        """Test that zero retention days disables cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(
                save_directory=tmpdir,
                retention_days=0,
                compress=False,
            )
            manager = CheckpointManager(project="test", config=config)

            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("content")

            # Create very old checkpoint
            old_time = time.time() - (365 * 24 * 60 * 60)  # 1 year ago
            manager.begin_checkpoint("old", message_count=0)
            manager.snapshot_file(str(test_file))
            manager._current.created_at = old_time
            manager.finalize_checkpoint()

            manager._cleanup_old_checkpoints()

            # Should not be removed
            assert len(manager.checkpoints) == 1

    def test_cleanup_preserves_recent_checkpoints(self):
        """Test that cleanup preserves recent checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(
                save_directory=tmpdir,
                retention_days=7,
                compress=False,
            )
            manager = CheckpointManager(project="test", config=config)

            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("content")

            # Create recent checkpoints
            for i in range(3):
                manager.begin_checkpoint(f"cp{i}", message_count=i)
                manager.snapshot_file(str(test_file))
                manager.finalize_checkpoint()
                time.sleep(0.01)

            manager._cleanup_old_checkpoints()

            # All should be preserved
            assert len(manager.checkpoints) == 3


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_generate_id_uniqueness(self):
        """Test that generated IDs are unique."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir)
            manager = CheckpointManager(project="test", config=config)

            ids = set()
            for _ in range(10):
                id = manager._generate_id()
                ids.add(id)
                time.sleep(0.001)

            assert len(ids) == 10

    def test_snapshot_file_with_unicode_content(self):
        """Test snapshotting file with unicode content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir, compress=False)
            manager = CheckpointManager(project="test", config=config)

            test_file = Path(tmpdir) / "unicode.py"
            test_file.write_text("# 中文注释\nprint('你好')", encoding="utf-8")

            manager.begin_checkpoint("test", message_count=0)
            result = manager.snapshot_file(str(test_file))

            assert result is True
            assert "中文" in manager._current.files[0].content

    def test_restore_file_creates_parent_directories(self):
        """Test that restore creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir, compress=False)
            manager = CheckpointManager(project="test", config=config)

            # Create file in subdirectory
            subdir = Path(tmpdir) / "sub" / "dir"
            subdir.mkdir(parents=True)
            test_file = subdir / "test.py"
            test_file.write_text("content")

            manager.begin_checkpoint("test", message_count=0)
            manager.snapshot_file(str(test_file))
            manager.finalize_checkpoint()

            # Delete subdirectory
            test_file.unlink()
            subdir.rmdir()
            subdir.parent.rmdir()

            # Restore should recreate directories
            manager.rewind(manager.checkpoints[0].id, mode="code")

            assert test_file.exists()
            assert test_file.read_text() == "content"

    def test_checkpoint_with_empty_prompt(self):
        """Test checkpoint with empty prompt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir, compress=False)
            manager = CheckpointManager(project="test", config=config)

            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("content")

            manager.begin_checkpoint("", message_count=0)
            manager.snapshot_file(str(test_file))
            cp_id = manager.finalize_checkpoint()

            checkpoint = manager.get_checkpoint(cp_id)
            assert checkpoint.prompt == ""

    def test_multiple_managers_same_project(self):
        """Test multiple managers for same project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir, compress=False)

            manager1 = CheckpointManager(project="shared", config=config)
            manager2 = CheckpointManager(project="shared", config=config)

            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("content")

            # Create checkpoint with manager1
            manager1.begin_checkpoint("test", message_count=0)
            manager1.snapshot_file(str(test_file))
            manager1.finalize_checkpoint()

            # Manager2 should see it after reload
            manager2._load_index()

            assert len(manager2.checkpoints) == 1

    def test_checkpoint_path_with_compression(self):
        """Test checkpoint path generation with compression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir, compress=True)
            manager = CheckpointManager(project="test", config=config)

            path = manager._get_checkpoint_path("test123")

            assert path.suffix == ".gz"
            assert "test123.json.gz" in str(path)

    def test_checkpoint_path_without_compression(self):
        """Test checkpoint path generation without compression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir, compress=False)
            manager = CheckpointManager(project="test", config=config)

            path = manager._get_checkpoint_path("test123")

            assert path.suffix == ".json"
            assert "test123.json" in str(path)

    @patch("src.core.checkpoint.logger")
    def test_snapshot_file_logs_error_on_exception(self, mock_logger):
        """Test that snapshot_file logs errors on exception."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir)
            manager = CheckpointManager(project="test", config=config)

            manager.begin_checkpoint("test", message_count=0)

            # Try to snapshot a directory (should fail)
            result = manager.snapshot_file(tmpdir)

            assert result is False
            mock_logger.error.assert_called()

    @patch("src.core.checkpoint.logger")
    def test_load_checkpoint_logs_error_on_corruption(self, mock_logger):
        """Test that load_checkpoint logs error on corrupted data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_directory=tmpdir, compress=False)
            manager = CheckpointManager(project="test", config=config)

            # Create corrupted checkpoint file
            cp_path = manager._base_dir / "corrupt123.json"
            cp_path.write_text("invalid json{{{")

            result = manager._load_checkpoint("corrupt123")

            assert result is None
            mock_logger.error.assert_called()
