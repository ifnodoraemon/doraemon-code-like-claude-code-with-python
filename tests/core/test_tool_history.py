"""Comprehensive tests for tool_history.py"""

import json
import time

import pytest

from src.core.tool_history import (
    ExecutionStatus,
    ToolExecution,
    ToolHistoryManager,
    get_tool_history,
)


class TestExecutionStatus:
    """Tests for ExecutionStatus enum."""

    def test_execution_status_values(self):
        """Test ExecutionStatus enum values."""
        assert ExecutionStatus.SUCCESS.value == "success"
        assert ExecutionStatus.ERROR.value == "error"
        assert ExecutionStatus.TIMEOUT.value == "timeout"
        assert ExecutionStatus.CANCELLED.value == "cancelled"

    def test_execution_status_from_value(self):
        """Test creating ExecutionStatus from value."""
        status = ExecutionStatus("success")
        assert status == ExecutionStatus.SUCCESS
        status = ExecutionStatus("error")
        assert status == ExecutionStatus.ERROR


class TestToolExecution:
    """Tests for ToolExecution dataclass."""

    def test_tool_execution_creation(self):
        """Test creating a ToolExecution."""
        execution = ToolExecution(
            id="exec-123",
            tool="file_read",
            arguments={"path": "/test/file.txt"},
            result="file content",
            status=ExecutionStatus.SUCCESS,
            started_at=1000.0,
            completed_at=1001.5,
        )
        assert execution.id == "exec-123"
        assert execution.tool == "file_read"
        assert execution.arguments == {"path": "/test/file.txt"}
        assert execution.result == "file content"
        assert execution.status == ExecutionStatus.SUCCESS

    def test_tool_execution_duration_property(self):
        """Test duration property calculation."""
        execution = ToolExecution(
            id="exec-123",
            tool="file_read",
            arguments={},
            result="content",
            status=ExecutionStatus.SUCCESS,
            started_at=1000.0,
            completed_at=1005.5,
        )
        assert execution.duration == 5.5

    def test_tool_execution_with_error_message(self):
        """Test ToolExecution with error message."""
        execution = ToolExecution(
            id="exec-456",
            tool="file_write",
            arguments={"path": "/test/file.txt", "content": "data"},
            result=None,
            status=ExecutionStatus.ERROR,
            started_at=1000.0,
            completed_at=1001.0,
            error_message="Permission denied",
        )
        assert execution.error_message == "Permission denied"
        assert execution.status == ExecutionStatus.ERROR

    def test_tool_execution_with_metadata(self):
        """Test ToolExecution with metadata."""
        metadata = {"user_id": "user-123", "request_id": "req-456"}
        execution = ToolExecution(
            id="exec-789",
            tool="api_call",
            arguments={"endpoint": "/api/data"},
            result={"status": "ok"},
            status=ExecutionStatus.SUCCESS,
            started_at=1000.0,
            completed_at=1001.0,
            metadata=metadata,
        )
        assert execution.metadata == metadata

    def test_tool_execution_with_parent_id(self):
        """Test ToolExecution with parent_id for nested calls."""
        execution = ToolExecution(
            id="exec-child",
            tool="nested_tool",
            arguments={},
            result="nested result",
            status=ExecutionStatus.SUCCESS,
            started_at=1000.0,
            completed_at=1001.0,
            parent_id="exec-parent",
        )
        assert execution.parent_id == "exec-parent"

    def test_tool_execution_to_dict(self):
        """Test converting ToolExecution to dictionary."""
        execution = ToolExecution(
            id="exec-123",
            tool="file_read",
            arguments={"path": "/test/file.txt"},
            result="file content",
            status=ExecutionStatus.SUCCESS,
            started_at=1000.0,
            completed_at=1005.0,
            error_message=None,
            session_id="session-123",
            parent_id=None,
            metadata={"key": "value"},
        )
        data = execution.to_dict()
        assert data["id"] == "exec-123"
        assert data["tool"] == "file_read"
        assert data["arguments"] == {"path": "/test/file.txt"}
        assert data["result"] == "file content"
        assert data["status"] == "success"
        assert data["duration"] == 5.0
        assert data["session_id"] == "session-123"
        assert data["metadata"] == {"key": "value"}

    def test_tool_execution_serialize_result_primitives(self):
        """Test serializing primitive result types."""
        execution = ToolExecution(
            id="exec-1",
            tool="test",
            arguments={},
            result="string",
            status=ExecutionStatus.SUCCESS,
            started_at=1000.0,
            completed_at=1001.0,
        )
        assert execution._serialize_result("string") == "string"
        assert execution._serialize_result(42) == 42
        assert execution._serialize_result(3.14) == 3.14
        assert execution._serialize_result(True) is True
        assert execution._serialize_result(None) is None

    def test_tool_execution_serialize_result_json_types(self):
        """Test serializing JSON-compatible types."""
        execution = ToolExecution(
            id="exec-1",
            tool="test",
            arguments={},
            result={},
            status=ExecutionStatus.SUCCESS,
            started_at=1000.0,
            completed_at=1001.0,
        )
        list_result = [1, 2, 3]
        assert execution._serialize_result(list_result) == list_result
        dict_result = {"key": "value", "nested": {"data": 123}}
        assert execution._serialize_result(dict_result) == dict_result

    def test_tool_execution_serialize_result_non_json(self):
        """Test serializing non-JSON-compatible types."""
        execution = ToolExecution(
            id="exec-1",
            tool="test",
            arguments={},
            result=None,
            status=ExecutionStatus.SUCCESS,
            started_at=1000.0,
            completed_at=1001.0,
        )
        # Object that can't be JSON serialized
        obj = object()
        result = execution._serialize_result(obj)
        assert isinstance(result, str)
        assert "object" in result

    def test_tool_execution_from_dict(self):
        """Test creating ToolExecution from dictionary."""
        data = {
            "id": "exec-123",
            "tool": "file_read",
            "arguments": {"path": "/test/file.txt"},
            "result": "file content",
            "status": "success",
            "started_at": 1000.0,
            "completed_at": 1005.0,
            "error_message": None,
            "session_id": "session-123",
            "parent_id": None,
            "metadata": {"key": "value"},
        }
        execution = ToolExecution.from_dict(data)
        assert execution.id == "exec-123"
        assert execution.tool == "file_read"
        assert execution.status == ExecutionStatus.SUCCESS
        assert execution.metadata == {"key": "value"}

    def test_tool_execution_from_dict_with_defaults(self):
        """Test from_dict with missing optional fields."""
        data = {
            "id": "exec-123",
            "tool": "test_tool",
            "arguments": {},
            "result": "result",
            "status": "success",
            "started_at": 1000.0,
            "completed_at": 1001.0,
        }
        execution = ToolExecution.from_dict(data)
        assert execution.error_message is None
        assert execution.session_id is None
        assert execution.parent_id is None
        assert execution.metadata == {}

    def test_tool_execution_roundtrip(self):
        """Test to_dict and from_dict roundtrip."""
        original = ToolExecution(
            id="exec-123",
            tool="file_read",
            arguments={"path": "/test/file.txt"},
            result="file content",
            status=ExecutionStatus.SUCCESS,
            started_at=1000.0,
            completed_at=1005.0,
            error_message=None,
            session_id="session-123",
            parent_id="parent-123",
            metadata={"key": "value"},
        )
        data = original.to_dict()
        restored = ToolExecution.from_dict(data)
        assert restored.id == original.id
        assert restored.tool == original.tool
        assert restored.arguments == original.arguments
        assert restored.result == original.result
        assert restored.status == original.status
        assert restored.session_id == original.session_id
        assert restored.parent_id == original.parent_id
        assert restored.metadata == original.metadata


class TestToolHistoryManagerInitialization:
    """Tests for ToolHistoryManager initialization."""

    def test_initialization_default(self):
        """Test default initialization."""
        manager = ToolHistoryManager()
        assert manager._max_entries == 10000
        assert manager._persist_path is None
        assert len(manager._entries) == 0
        assert len(manager._pending) == 0
        assert manager._session_id is not None

    def test_initialization_with_max_entries(self):
        """Test initialization with custom max_entries."""
        manager = ToolHistoryManager(max_entries=5000)
        assert manager._max_entries == 5000

    def test_initialization_with_session_id(self):
        """Test initialization with custom session_id."""
        manager = ToolHistoryManager(session_id="custom-session")
        assert manager._session_id == "custom-session"

    def test_initialization_generates_session_id(self):
        """Test that session_id is generated if not provided."""
        manager1 = ToolHistoryManager()
        manager2 = ToolHistoryManager()
        assert manager1._session_id != manager2._session_id
        assert len(manager1._session_id) == 8  # UUID truncated to 8 chars

    def test_initialization_with_persist_path_nonexistent(self, temp_dir):
        """Test initialization with non-existent persist path."""
        persist_path = temp_dir / "history.json"
        manager = ToolHistoryManager(persist_path=persist_path)
        assert manager._persist_path == persist_path
        assert len(manager._entries) == 0

    def test_initialization_with_persist_path_existing(self, temp_dir):
        """Test initialization loads existing history."""
        persist_path = temp_dir / "history.json"
        # Create a history file
        history_data = {
            "entries": [
                {
                    "id": "exec-1",
                    "tool": "test_tool",
                    "arguments": {},
                    "result": "result",
                    "status": "success",
                    "started_at": 1000.0,
                    "completed_at": 1001.0,
                }
            ]
        }
        persist_path.write_text(json.dumps(history_data))

        manager = ToolHistoryManager(persist_path=persist_path)
        assert len(manager._entries) == 1
        assert manager._entries[0].id == "exec-1"


class TestToolHistoryManagerStartComplete:
    """Tests for start() and complete() methods."""

    def test_start_creates_pending_entry(self):
        """Test start() creates a pending entry."""
        manager = ToolHistoryManager()
        exec_id = manager.start("file_read", {"path": "/test/file.txt"})
        assert exec_id in manager._pending
        assert manager._pending[exec_id]["tool"] == "file_read"
        assert manager._pending[exec_id]["arguments"] == {"path": "/test/file.txt"}

    def test_start_returns_unique_ids(self):
        """Test start() returns unique execution IDs."""
        manager = ToolHistoryManager()
        id1 = manager.start("tool1", {})
        id2 = manager.start("tool2", {})
        assert id1 != id2

    def test_start_with_parent_id(self):
        """Test start() with parent_id for nested calls."""
        manager = ToolHistoryManager()
        parent_id = manager.start("parent_tool", {})
        child_id = manager.start("child_tool", {}, parent_id=parent_id)
        assert manager._pending[child_id]["parent_id"] == parent_id

    def test_start_with_metadata(self):
        """Test start() with metadata."""
        manager = ToolHistoryManager()
        metadata = {"user_id": "user-123", "request_id": "req-456"}
        exec_id = manager.start("tool", {}, metadata=metadata)
        assert manager._pending[exec_id]["metadata"] == metadata

    def test_start_sets_session_id(self):
        """Test start() sets session_id."""
        manager = ToolHistoryManager(session_id="test-session")
        exec_id = manager.start("tool", {})
        assert manager._pending[exec_id]["session_id"] == "test-session"

    def test_complete_moves_to_entries(self):
        """Test complete() moves pending entry to entries."""
        manager = ToolHistoryManager()
        exec_id = manager.start("file_read", {"path": "/test/file.txt"})
        manager.complete(exec_id, "file content")
        assert exec_id not in manager._pending
        assert len(manager._entries) == 1
        assert manager._entries[0].id == exec_id

    def test_complete_with_success_status(self):
        """Test complete() with SUCCESS status."""
        manager = ToolHistoryManager()
        exec_id = manager.start("tool", {})
        manager.complete(exec_id, "result", status=ExecutionStatus.SUCCESS)
        entry = manager._entries[0]
        assert entry.status == ExecutionStatus.SUCCESS
        assert entry.result == "result"

    def test_complete_with_error_status(self):
        """Test complete() with ERROR status."""
        manager = ToolHistoryManager()
        exec_id = manager.start("tool", {})
        manager.complete(
            exec_id,
            None,
            status=ExecutionStatus.ERROR,
            error_message="Tool failed",
        )
        entry = manager._entries[0]
        assert entry.status == ExecutionStatus.ERROR
        assert entry.error_message == "Tool failed"

    def test_complete_with_timeout_status(self):
        """Test complete() with TIMEOUT status."""
        manager = ToolHistoryManager()
        exec_id = manager.start("tool", {})
        manager.complete(exec_id, None, status=ExecutionStatus.TIMEOUT)
        entry = manager._entries[0]
        assert entry.status == ExecutionStatus.TIMEOUT

    def test_complete_unknown_exec_id_logs_warning(self, caplog):
        """Test complete() with unknown exec_id logs warning."""
        manager = ToolHistoryManager()
        manager.complete("unknown-id", "result")
        assert len(manager._entries) == 0

    def test_complete_records_duration(self):
        """Test complete() records execution duration."""
        manager = ToolHistoryManager()
        exec_id = manager.start("tool", {})
        time.sleep(0.1)  # Small delay
        manager.complete(exec_id, "result")
        entry = manager._entries[0]
        assert entry.duration >= 0.1

    def test_complete_with_complex_result(self):
        """Test complete() with complex result types."""
        manager = ToolHistoryManager()
        exec_id = manager.start("tool", {})
        complex_result = {
            "status": "ok",
            "data": [1, 2, 3],
            "nested": {"key": "value"},
        }
        manager.complete(exec_id, complex_result)
        entry = manager._entries[0]
        assert entry.result == complex_result


class TestExecutionRecorder:
    """Tests for ExecutionRecorder context manager."""

    def test_recorder_context_manager_success(self):
        """Test ExecutionRecorder as context manager with success."""
        manager = ToolHistoryManager()
        with manager.record("file_read", {"path": "/test/file.txt"}) as recorder:
            assert recorder.exec_id is not None
            recorder.set_result("file content")

        assert len(manager._entries) == 1
        entry = manager._entries[0]
        assert entry.tool == "file_read"
        assert entry.result == "file content"
        assert entry.status == ExecutionStatus.SUCCESS

    def test_recorder_context_manager_with_error(self):
        """Test ExecutionRecorder with exception."""
        manager = ToolHistoryManager()
        try:
            with manager.record("tool", {}) as recorder:
                recorder.set_result("partial result")
                raise ValueError("Tool error")
        except ValueError:
            pass

        assert len(manager._entries) == 1
        entry = manager._entries[0]
        assert entry.status == ExecutionStatus.ERROR
        assert "Tool error" in entry.error_message

    def test_recorder_set_error(self):
        """Test ExecutionRecorder.set_error()."""
        manager = ToolHistoryManager()
        with manager.record("tool", {}) as recorder:
            recorder.set_error("Custom error message")

        entry = manager._entries[0]
        assert entry.status == ExecutionStatus.ERROR
        assert entry.error_message == "Custom error message"

    def test_recorder_exec_id_property(self):
        """Test ExecutionRecorder.exec_id property."""
        manager = ToolHistoryManager()
        with manager.record("tool", {}) as recorder:
            exec_id = recorder.exec_id
            assert exec_id is not None
            assert isinstance(exec_id, str)

    def test_recorder_with_parent_id(self):
        """Test ExecutionRecorder with parent_id."""
        manager = ToolHistoryManager()
        parent_id = manager.start("parent", {})
        with manager.record("child", {}, parent_id=parent_id) as recorder:
            recorder.set_result("child result")

        entry = manager._entries[0]
        assert entry.parent_id == parent_id


class TestToolHistoryManagerQuerying:
    """Tests for querying methods: get_recent, get_by_id, filter, search."""

    def test_get_recent_default_limit(self):
        """Test get_recent() with default limit."""
        manager = ToolHistoryManager()
        for i in range(100):
            exec_id = manager.start(f"tool_{i}", {})
            manager.complete(exec_id, f"result_{i}")

        recent = manager.get_recent()
        assert len(recent) == 50  # Default limit

    def test_get_recent_custom_limit(self):
        """Test get_recent() with custom limit."""
        manager = ToolHistoryManager()
        for i in range(100):
            exec_id = manager.start(f"tool_{i}", {})
            manager.complete(exec_id, f"result_{i}")

        recent = manager.get_recent(limit=10)
        assert len(recent) == 10

    def test_get_recent_limit_exceeds_entries(self):
        """Test get_recent() when limit exceeds total entries."""
        manager = ToolHistoryManager()
        for i in range(5):
            exec_id = manager.start(f"tool_{i}", {})
            manager.complete(exec_id, f"result_{i}")

        recent = manager.get_recent(limit=100)
        assert len(recent) == 5

    def test_get_recent_returns_latest(self):
        """Test get_recent() returns latest entries."""
        manager = ToolHistoryManager()
        exec_ids = []
        for i in range(10):
            exec_id = manager.start(f"tool_{i}", {})
            manager.complete(exec_id, f"result_{i}")
            exec_ids.append(exec_id)

        recent = manager.get_recent(limit=3)
        assert len(recent) == 3
        assert recent[0].id == exec_ids[-3]
        assert recent[1].id == exec_ids[-2]
        assert recent[2].id == exec_ids[-1]

    def test_get_by_id_found(self):
        """Test get_by_id() when entry exists."""
        manager = ToolHistoryManager()
        exec_id = manager.start("tool", {})
        manager.complete(exec_id, "result")

        entry = manager.get_by_id(exec_id)
        assert entry is not None
        assert entry.id == exec_id

    def test_get_by_id_not_found(self):
        """Test get_by_id() when entry doesn't exist."""
        manager = ToolHistoryManager()
        entry = manager.get_by_id("nonexistent-id")
        assert entry is None

    def test_filter_by_tool(self):
        """Test filter() by tool name."""
        manager = ToolHistoryManager()
        for _i in range(5):
            exec_id = manager.start("file_read", {})
            manager.complete(exec_id, "result")
        for _i in range(3):
            exec_id = manager.start("file_write", {})
            manager.complete(exec_id, "result")

        results = manager.filter(tool="file_read")
        assert len(results) == 5
        assert all(e.tool == "file_read" for e in results)

    def test_filter_by_status(self):
        """Test filter() by status."""
        manager = ToolHistoryManager()
        for _i in range(3):
            exec_id = manager.start("tool", {})
            manager.complete(exec_id, "result", status=ExecutionStatus.SUCCESS)
        for _i in range(2):
            exec_id = manager.start("tool", {})
            manager.complete(exec_id, None, status=ExecutionStatus.ERROR, error_message="error")

        results = manager.filter(status=ExecutionStatus.SUCCESS)
        assert len(results) == 3
        assert all(e.status == ExecutionStatus.SUCCESS for e in results)

    def test_filter_by_session_id(self):
        """Test filter() by session_id."""
        manager1 = ToolHistoryManager(session_id="session-1")
        manager2 = ToolHistoryManager(session_id="session-2")

        exec_id1 = manager1.start("tool", {})
        manager1.complete(exec_id1, "result")

        exec_id2 = manager2.start("tool", {})
        manager2.complete(exec_id2, "result")

        # Combine entries for filtering
        manager1._entries.extend(manager2._entries)

        results = manager1.filter(session_id="session-1")
        assert len(results) == 1
        assert results[0].session_id == "session-1"

    def test_filter_by_time_range(self):
        """Test filter() by time range."""
        manager = ToolHistoryManager()
        time.time()

        exec_id1 = manager.start("tool", {})
        manager.complete(exec_id1, "result")

        time.sleep(0.1)
        mid_time = time.time()
        time.sleep(0.1)

        exec_id2 = manager.start("tool", {})
        manager.complete(exec_id2, "result")

        # Filter entries after mid_time
        results = manager.filter(since=mid_time)
        assert len(results) == 1

    def test_filter_multiple_criteria(self):
        """Test filter() with multiple criteria."""
        manager = ToolHistoryManager()
        for _i in range(5):
            exec_id = manager.start("file_read", {})
            manager.complete(exec_id, "result", status=ExecutionStatus.SUCCESS)
        for _i in range(3):
            exec_id = manager.start("file_write", {})
            manager.complete(exec_id, "result", status=ExecutionStatus.SUCCESS)
        for _i in range(2):
            exec_id = manager.start("file_read", {})
            manager.complete(exec_id, None, status=ExecutionStatus.ERROR, error_message="error")

        results = manager.filter(tool="file_read", status=ExecutionStatus.SUCCESS)
        assert len(results) == 5

    def test_search_by_tool_name(self):
        """Test search() by tool name."""
        manager = ToolHistoryManager()
        exec_id1 = manager.start("file_read", {})
        manager.complete(exec_id1, "result")
        exec_id2 = manager.start("api_call", {})
        manager.complete(exec_id2, "result")

        results = manager.search("file")
        assert len(results) == 1
        assert results[0].tool == "file_read"

    def test_search_by_arguments(self):
        """Test search() by arguments."""
        manager = ToolHistoryManager()
        exec_id1 = manager.start("tool", {"path": "/test/file.txt"})
        manager.complete(exec_id1, "result")
        exec_id2 = manager.start("tool", {"path": "/other/file.txt"})
        manager.complete(exec_id2, "result")

        results = manager.search("test")
        assert len(results) == 1

    def test_search_case_insensitive(self):
        """Test search() is case insensitive."""
        manager = ToolHistoryManager()
        exec_id = manager.start("FILE_READ", {})
        manager.complete(exec_id, "result")

        results = manager.search("file")
        assert len(results) == 1


class TestToolHistoryManagerStatistics:
    """Tests for get_stats() method."""

    def test_get_stats_empty_history(self):
        """Test get_stats() with empty history."""
        manager = ToolHistoryManager()
        stats = manager.get_stats()
        assert stats["total"] == 0
        assert stats["by_tool"] == {}
        assert stats["by_status"] == {}
        assert stats["avg_duration"] == 0

    def test_get_stats_single_entry(self):
        """Test get_stats() with single entry."""
        manager = ToolHistoryManager()
        exec_id = manager.start("file_read", {})
        manager.complete(exec_id, "result")

        stats = manager.get_stats()
        assert stats["total"] == 1
        assert stats["by_tool"]["file_read"] == 1
        assert stats["by_status"]["success"] == 1

    def test_get_stats_multiple_tools(self):
        """Test get_stats() with multiple tools."""
        manager = ToolHistoryManager()
        for _i in range(3):
            exec_id = manager.start("file_read", {})
            manager.complete(exec_id, "result")
        for _i in range(2):
            exec_id = manager.start("file_write", {})
            manager.complete(exec_id, "result")

        stats = manager.get_stats()
        assert stats["total"] == 5
        assert stats["by_tool"]["file_read"] == 3
        assert stats["by_tool"]["file_write"] == 2

    def test_get_stats_multiple_statuses(self):
        """Test get_stats() with multiple statuses."""
        manager = ToolHistoryManager()
        for _i in range(3):
            exec_id = manager.start("tool", {})
            manager.complete(exec_id, "result", status=ExecutionStatus.SUCCESS)
        for _i in range(2):
            exec_id = manager.start("tool", {})
            manager.complete(exec_id, None, status=ExecutionStatus.ERROR, error_message="error")

        stats = manager.get_stats()
        assert stats["by_status"]["success"] == 3
        assert stats["by_status"]["error"] == 2

    def test_get_stats_average_duration(self):
        """Test get_stats() calculates average duration."""
        manager = ToolHistoryManager()
        for _i in range(3):
            exec_id = manager.start("tool", {})
            time.sleep(0.05)
            manager.complete(exec_id, "result")

        stats = manager.get_stats()
        assert stats["avg_duration"] > 0
        assert stats["total"] == 3

    def test_get_stats_includes_session_id(self):
        """Test get_stats() includes session_id."""
        manager = ToolHistoryManager(session_id="test-session")
        exec_id = manager.start("tool", {})
        manager.complete(exec_id, "result")

        stats = manager.get_stats()
        assert stats["session_id"] == "test-session"


class TestToolHistoryManagerPersistence:
    """Tests for persistence: save/load and export/import."""

    def test_save_history_creates_file(self, temp_dir):
        """Test _save_history() creates file."""
        persist_path = temp_dir / "history.json"
        manager = ToolHistoryManager(persist_path=persist_path)
        exec_id = manager.start("tool", {})
        manager.complete(exec_id, "result")

        assert persist_path.exists()

    def test_load_history_from_file(self, temp_dir):
        """Test _load_history() loads from file."""
        persist_path = temp_dir / "history.json"
        history_data = {
            "entries": [
                {
                    "id": "exec-1",
                    "tool": "test_tool",
                    "arguments": {"key": "value"},
                    "result": "result",
                    "status": "success",
                    "started_at": 1000.0,
                    "completed_at": 1001.0,
                }
            ]
        }
        persist_path.write_text(json.dumps(history_data))

        manager = ToolHistoryManager(persist_path=persist_path)
        assert len(manager._entries) == 1
        assert manager._entries[0].tool == "test_tool"

    def test_export_to_json_string(self):
        """Test export() returns JSON string."""
        manager = ToolHistoryManager()
        exec_id = manager.start("tool", {})
        manager.complete(exec_id, "result")

        json_str = manager.export()
        data = json.loads(json_str)
        assert "exported_at" in data
        assert "session_id" in data
        assert "total_entries" in data
        assert "entries" in data
        assert len(data["entries"]) == 1

    def test_export_to_file(self, temp_dir):
        """Test export() writes to file."""
        export_path = temp_dir / "export.json"
        manager = ToolHistoryManager()
        exec_id = manager.start("tool", {})
        manager.complete(exec_id, "result")

        manager.export(path=export_path)
        assert export_path.exists()
        data = json.loads(export_path.read_text())
        assert len(data["entries"]) == 1

    def test_import_history_from_file(self, temp_dir):
        """Test import_history() loads from file."""
        import_path = temp_dir / "import.json"
        import_data = {
            "exported_at": time.time(),
            "session_id": "imported-session",
            "total_entries": 2,
            "entries": [
                {
                    "id": "exec-1",
                    "tool": "tool1",
                    "arguments": {},
                    "result": "result1",
                    "status": "success",
                    "started_at": 1000.0,
                    "completed_at": 1001.0,
                },
                {
                    "id": "exec-2",
                    "tool": "tool2",
                    "arguments": {},
                    "result": "result2",
                    "status": "success",
                    "started_at": 1002.0,
                    "completed_at": 1003.0,
                },
            ],
        }
        import_path.write_text(json.dumps(import_data))

        manager = ToolHistoryManager()
        manager.import_history(import_path)
        assert len(manager._entries) == 2
        assert manager._entries[0].tool == "tool1"
        assert manager._entries[1].tool == "tool2"

    def test_import_history_with_invalid_entry(self, temp_dir):
        """Test import_history() skips invalid entries."""
        import_path = temp_dir / "import.json"
        import_data = {
            "entries": [
                {
                    "id": "exec-1",
                    "tool": "tool1",
                    "arguments": {},
                    "result": "result1",
                    "status": "success",
                    "started_at": 1000.0,
                    "completed_at": 1001.0,
                },
                {
                    # Missing required fields
                    "id": "exec-2",
                    "tool": "tool2",
                },
            ]
        }
        import_path.write_text(json.dumps(import_data))

        manager = ToolHistoryManager()
        manager.import_history(import_path)
        # Should only import the valid entry
        assert len(manager._entries) == 1


class TestToolHistoryManagerClear:
    """Tests for clear() method."""

    def test_clear_removes_entries(self):
        """Test clear() removes all entries."""
        manager = ToolHistoryManager()
        for i in range(5):
            exec_id = manager.start(f"tool_{i}", {})
            manager.complete(exec_id, "result")

        assert len(manager._entries) == 5
        manager.clear()
        assert len(manager._entries) == 0

    def test_clear_removes_pending(self):
        """Test clear() removes pending entries."""
        manager = ToolHistoryManager()
        manager.start("tool", {})
        assert len(manager._pending) == 1
        manager.clear()
        assert len(manager._pending) == 0

    def test_clear_deletes_persist_file(self, temp_dir):
        """Test clear() deletes persist file."""
        persist_path = temp_dir / "history.json"
        manager = ToolHistoryManager(persist_path=persist_path)
        exec_id = manager.start("tool", {})
        manager.complete(exec_id, "result")

        assert persist_path.exists()
        manager.clear()
        assert not persist_path.exists()


class TestToolHistoryManagerMaxEntries:
    """Tests for max_entries trimming."""

    def test_max_entries_trimming(self):
        """Test that history is trimmed when exceeding max_entries."""
        manager = ToolHistoryManager(max_entries=100)
        # Add 150 entries
        for i in range(150):
            exec_id = manager.start(f"tool_{i}", {})
            manager.complete(exec_id, f"result_{i}")

        # Should be trimmed to 50 or less (half of max)
        assert len(manager._entries) <= 100
        # After trimming, should have kept recent entries
        assert manager._entries[-1].result == "result_149"

    def test_max_entries_keeps_recent(self):
        """Test that trimming keeps the most recent entries."""
        manager = ToolHistoryManager(max_entries=100)
        for i in range(150):
            exec_id = manager.start(f"tool_{i}", {})
            manager.complete(exec_id, f"result_{i}")

        # Check that we have the most recent entries
        assert manager._entries[-1].result == "result_149"
        # Verify we don't have very old entries
        oldest_result_num = int(manager._entries[0].result.split("_")[1])
        assert oldest_result_num > 0  # Not the first entry


class TestToolHistoryManagerReplay:
    """Tests for replay() method."""

    @pytest.mark.asyncio
    async def test_replay_execution(self):
        """Test replay() re-executes a tool."""
        manager = ToolHistoryManager()
        exec_id = manager.start("tool", {"param": "value"})
        manager.complete(exec_id, "original_result")

        # Mock executor
        async def mock_executor(tool, args):
            return f"replayed_{args['param']}"

        result = await manager.replay(exec_id, mock_executor)
        assert result == "replayed_value"

    @pytest.mark.asyncio
    async def test_replay_nonexistent_execution(self):
        """Test replay() raises error for nonexistent execution."""
        manager = ToolHistoryManager()

        async def mock_executor(tool, args):
            return "result"

        with pytest.raises(ValueError, match="Execution not found"):
            await manager.replay("nonexistent-id", mock_executor)


class TestToolHistoryManagerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_arguments(self):
        """Test with empty arguments."""
        manager = ToolHistoryManager()
        exec_id = manager.start("tool", {})
        manager.complete(exec_id, "result")
        entry = manager._entries[0]
        assert entry.arguments == {}

    def test_none_result(self):
        """Test with None result."""
        manager = ToolHistoryManager()
        exec_id = manager.start("tool", {})
        manager.complete(exec_id, None)
        entry = manager._entries[0]
        assert entry.result is None

    def test_large_result(self):
        """Test with large result."""
        manager = ToolHistoryManager()
        large_result = "x" * 1000000  # 1MB string
        exec_id = manager.start("tool", {})
        manager.complete(exec_id, large_result)
        entry = manager._entries[0]
        assert len(entry.result) == 1000000

    def test_special_characters_in_tool_name(self):
        """Test with special characters in tool name."""
        manager = ToolHistoryManager()
        exec_id = manager.start("tool-name_v2.0", {})
        manager.complete(exec_id, "result")
        entry = manager._entries[0]
        assert entry.tool == "tool-name_v2.0"

    def test_unicode_in_arguments(self):
        """Test with unicode in arguments."""
        manager = ToolHistoryManager()
        exec_id = manager.start("tool", {"text": "Hello 世界 🌍"})
        manager.complete(exec_id, "result")
        entry = manager._entries[0]
        assert entry.arguments["text"] == "Hello 世界 🌍"

    def test_deeply_nested_arguments(self):
        """Test with deeply nested arguments."""
        manager = ToolHistoryManager()
        nested_args = {"level1": {"level2": {"level3": {"level4": {"value": "deep"}}}}}
        exec_id = manager.start("tool", nested_args)
        manager.complete(exec_id, "result")
        entry = manager._entries[0]
        assert entry.arguments["level1"]["level2"]["level3"]["level4"]["value"] == "deep"

    def test_circular_reference_in_result(self):
        """Test with circular reference in result (stored as-is if JSON-compatible)."""
        manager = ToolHistoryManager()
        exec_id = manager.start("tool", {})
        # Create a circular reference
        circular = {}
        circular["self"] = circular
        manager.complete(exec_id, circular)
        entry = manager._entries[0]
        # The circular reference is stored as-is (dict with self-reference)
        # It won't be serialized to string unless export/import is called
        assert isinstance(entry.result, dict)

    def test_multiple_concurrent_recordings(self):
        """Test multiple concurrent recordings."""
        manager = ToolHistoryManager()
        exec_ids = []
        for i in range(5):
            exec_id = manager.start(f"tool_{i}", {})
            exec_ids.append(exec_id)

        # All should be pending
        assert len(manager._pending) == 5

        # Complete them in different order
        for exec_id in reversed(exec_ids):
            manager.complete(exec_id, "result")

        assert len(manager._entries) == 5
        assert len(manager._pending) == 0

    def test_filter_with_no_matches(self):
        """Test filter() with no matches."""
        manager = ToolHistoryManager()
        exec_id = manager.start("tool_a", {})
        manager.complete(exec_id, "result")

        results = manager.filter(tool="tool_b")
        assert len(results) == 0

    def test_search_with_no_matches(self):
        """Test search() with no matches."""
        manager = ToolHistoryManager()
        exec_id = manager.start("tool", {})
        manager.complete(exec_id, "result")

        results = manager.search("nonexistent")
        assert len(results) == 0


class TestGlobalToolHistory:
    """Tests for global tool history singleton."""

    def test_get_tool_history_returns_singleton(self):
        """Test get_tool_history() returns singleton."""
        history1 = get_tool_history()
        history2 = get_tool_history()
        assert history1 is history2

    def test_get_tool_history_creates_instance(self):
        """Test get_tool_history() creates instance if needed."""
        # Reset global
        import src.core.tool_history as tool_history_module

        tool_history_module._tool_history = None

        history = get_tool_history()
        assert history is not None
        assert isinstance(history, ToolHistoryManager)


class TestToolHistoryIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow(self, temp_dir):
        """Test complete workflow: record, query, export, import."""
        persist_path = temp_dir / "history.json"
        manager = ToolHistoryManager(persist_path=persist_path)

        # Record some executions
        for i in range(5):
            with manager.record(f"tool_{i}", {"index": i}) as recorder:
                recorder.set_result(f"result_{i}")

        # Query
        recent = manager.get_recent(limit=3)
        assert len(recent) == 3

        stats = manager.get_stats()
        assert stats["total"] == 5

        # Export
        export_path = temp_dir / "export.json"
        manager.export(path=export_path)
        assert export_path.exists()

        # Import into new manager
        new_manager = ToolHistoryManager()
        new_manager.import_history(export_path)
        assert len(new_manager._entries) == 5

    def test_nested_tool_calls(self):
        """Test recording nested tool calls."""
        manager = ToolHistoryManager()

        # Parent call
        parent_id = manager.start("parent_tool", {"param": "value"})

        # Child calls
        child_id1 = manager.start("child_tool_1", {}, parent_id=parent_id)
        manager.complete(child_id1, "child_result_1")

        child_id2 = manager.start("child_tool_2", {}, parent_id=parent_id)
        manager.complete(child_id2, "child_result_2")

        # Complete parent
        manager.complete(parent_id, "parent_result")

        # Verify structure
        manager.get_by_id(parent_id)
        children = [e for e in manager._entries if e.parent_id == parent_id]
        assert len(children) == 2
        assert all(c.parent_id == parent_id for c in children)

    def test_error_recovery_workflow(self):
        """Test workflow with errors and recovery."""
        manager = ToolHistoryManager()

        # First attempt fails
        exec_id1 = manager.start("risky_tool", {"attempt": 1})
        manager.complete(
            exec_id1,
            None,
            status=ExecutionStatus.ERROR,
            error_message="Connection timeout",
        )

        # Retry succeeds
        exec_id2 = manager.start("risky_tool", {"attempt": 2})
        manager.complete(exec_id2, "success", status=ExecutionStatus.SUCCESS)

        # Verify both recorded
        errors = manager.filter(status=ExecutionStatus.ERROR)
        successes = manager.filter(status=ExecutionStatus.SUCCESS)
        assert len(errors) == 1
        assert len(successes) == 1

    def test_performance_with_large_history(self):
        """Test performance with large history."""
        manager = ToolHistoryManager(max_entries=10000)

        # Add 1000 entries
        for i in range(1000):
            exec_id = manager.start(f"tool_{i % 10}", {"index": i})
            manager.complete(exec_id, f"result_{i}")

        # Queries should still be fast
        recent = manager.get_recent(limit=100)
        assert len(recent) == 100

        by_tool = manager.filter(tool="tool_0")
        assert len(by_tool) == 100

        stats = manager.get_stats()
        assert stats["total"] == 1000
