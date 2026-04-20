"""Tests for src/webui/routes/sessions.py"""

import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.webui.routes.sessions import (
    SessionResponse,
    UndoRequest,
    _ACTIVE_STREAMS,
    _serialize_run,
    mark_stream_active,
    mark_stream_finished,
)


class TestSerializeRun:
    def test_include_details(self):
        run = {"task_graph": {"a": 1}, "task_summaries": [1, 2], "other": "val"}
        result = _serialize_run(run, include_details=True)
        assert result == run

    def test_summary_mode_omits_large_fields(self):
        run = {
            "task_graph": {"a": 1, "b": 2},
            "task_summaries": [1, 2, 3],
            "worker_assignments": ["w1"],
            "executed_task_ids": ["e1", "e2"],
            "completed_task_ids": ["c1"],
            "failed_task_ids": ["f1"],
            "other": "val",
        }
        result = _serialize_run(run, include_details=False)
        assert "task_graph" not in result
        assert "task_summaries" not in result
        assert result["task_graph_count"] == 2
        assert result["task_summary_count"] == 3
        assert result["worker_assignment_count"] == 1
        assert result["executed_task_count"] == 2
        assert result["completed_task_count"] == 1
        assert result["failed_task_count"] == 1
        assert result["other"] == "val"

    def test_summary_mode_non_dict_list_fields(self):
        run = {"task_graph": "not a dict", "other": 42}
        result = _serialize_run(run, include_details=False)
        assert result["other"] == 42
        assert "task_graph_count" not in result

    def test_summary_mode_missing_fields(self):
        run = {"other": "val"}
        result = _serialize_run(run, include_details=False)
        assert result["other"] == "val"

    def test_summary_mode_with_list_field(self):
        run = {"task_graph": [1, 2, 3], "other": "x"}
        result = _serialize_run(run, include_details=False)
        assert result["task_graph_count"] == 3
        assert "task_graph" not in result


class TestMarkStream:
    def test_mark_active(self):
        mark_stream_active("sess1")
        from src.webui.routes.sessions import _ACTIVE_STREAMS
        assert "sess1" in _ACTIVE_STREAMS
        mark_stream_finished("sess1")

    def test_mark_finished_removes(self):
        mark_stream_active("sess2")
        mark_stream_finished("sess2")
        from src.webui.routes.sessions import _ACTIVE_STREAMS
        assert "sess2" not in _ACTIVE_STREAMS

    def test_mark_finished_idempotent(self):
        mark_stream_finished("nonexistent")
        from src.webui.routes.sessions import _ACTIVE_STREAMS
        assert "nonexistent" not in _ACTIVE_STREAMS


class TestSessionResponse:
    def test_creation(self):
        sr = SessionResponse(id="abc", name="test", message_count=5, updated_at=1.0)
        assert sr.id == "abc"
        assert sr.message_count == 5

    def test_name_none(self):
        sr = SessionResponse(id="x", name=None, message_count=0, updated_at=0.0)
        assert sr.name is None


class TestUndoRequest:
    def test_defaults(self):
        req = UndoRequest()
        assert req.checkpoint_id is None
        assert req.mode == "code"
        assert req.dry_run is False

    def test_custom(self):
        req = UndoRequest(checkpoint_id="cp1", mode="full", dry_run=True)
        assert req.checkpoint_id == "cp1"
        assert req.dry_run is True


class TestSessionIdValidation:
    def test_valid_ids(self):
        pattern = r"^[a-zA-Z0-9_-]+$"
        assert re.match(pattern, "abc123")
        assert re.match(pattern, "session-1_test")
        assert re.match(pattern, "a")

    def test_invalid_ids(self):
        pattern = r"^[a-zA-Z0-9_-]+$"
        assert not re.match(pattern, "")
        assert not re.match(pattern, "abc def")
        assert not re.match(pattern, "abc/def")
        assert not re.match(pattern, "abc.def")


class TestListSessionsRoute:
    @pytest.mark.asyncio
    async def test_list_sessions(self):
        from src.webui.routes.sessions import list_sessions
        mock_mgr = MagicMock()
        mock_session = MagicMock()
        mock_session.id = "s1"
        mock_session.name = "test"
        mock_session.message_count = 5
        mock_session.updated_at = 123.0
        mock_mgr.list_sessions.return_value = [mock_session]

        with patch("src.webui.routes.sessions.SessionManager", return_value=mock_mgr):
            result = await list_sessions(project="default", limit=10)
            assert len(result) == 1
            assert result[0].id == "s1"

    @pytest.mark.asyncio
    async def test_list_sessions_limit_capped(self):
        from src.webui.routes.sessions import list_sessions
        mock_mgr = MagicMock()
        mock_mgr.list_sessions.return_value = []

        with patch("src.webui.routes.sessions.SessionManager", return_value=mock_mgr):
            result = await list_sessions(limit=200)
            mock_mgr.list_sessions.assert_called_once_with(project="default", limit=100)


class TestGetSessionRoute:
    @pytest.mark.asyncio
    async def test_invalid_session_id(self):
        from src.webui.routes.sessions import get_session
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await get_session(session_id="bad.id")
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_session_not_found(self):
        from src.webui.routes.sessions import get_session
        from fastapi import HTTPException
        mock_mgr = MagicMock()
        mock_mgr.resume_session.return_value = None
        with patch("src.webui.routes.sessions.SessionManager", return_value=mock_mgr):
            with pytest.raises(HTTPException) as exc_info:
                await get_session(session_id="valid-id")
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_session_success(self):
        from src.webui.routes.sessions import get_session
        mock_mgr = MagicMock()
        mock_session = MagicMock()
        mock_session.metadata.id = "s1"
        mock_session.metadata.name = "test"
        mock_session.messages = [MagicMock(), MagicMock()]
        mock_session.orchestration_state = {"run_id": "r1"}
        mock_session.orchestration_runs = []
        mock_session.active_orchestration_run_id = None
        mock_mgr.resume_session.return_value = mock_session

        with patch("src.webui.routes.sessions.SessionManager", return_value=mock_mgr):
            result = await get_session(session_id="s1")
            assert result["id"] == "s1"
            assert result["message_count"] == 2

    @pytest.mark.asyncio
    async def test_get_session_without_messages(self):
        from src.webui.routes.sessions import get_session
        mock_mgr = MagicMock()
        mock_session = MagicMock()
        mock_session.metadata.id = "s1"
        mock_session.metadata.name = "test"
        mock_session.messages = [MagicMock()]
        mock_session.orchestration_state = {}
        mock_session.orchestration_runs = []
        mock_session.active_orchestration_run_id = None
        mock_mgr.resume_session.return_value = mock_session

        with patch("src.webui.routes.sessions.SessionManager", return_value=mock_mgr):
            result = await get_session(session_id="s1", include_messages=False)
            assert result["messages"] == []
            assert result["message_count"] == 1

    @pytest.mark.asyncio
    async def test_get_session_with_message_limit(self):
        from src.webui.routes.sessions import get_session
        mock_mgr = MagicMock()
        mock_session = MagicMock()
        mock_session.metadata.id = "s1"
        mock_session.metadata.name = "test"
        mock_session.messages = list(range(10))
        mock_session.orchestration_state = {}
        mock_session.orchestration_runs = []
        mock_session.active_orchestration_run_id = None
        mock_mgr.resume_session.return_value = mock_session

        with patch("src.webui.routes.sessions.SessionManager", return_value=mock_mgr):
            result = await get_session(session_id="s1", message_limit=3)
            assert len(result["messages"]) == 3

    @pytest.mark.asyncio
    async def test_get_session_with_message_offset(self):
        from src.webui.routes.sessions import get_session
        mock_mgr = MagicMock()
        mock_session = MagicMock()
        mock_session.metadata.id = "s1"
        mock_session.metadata.name = "test"
        mock_session.messages = list(range(10))
        mock_session.orchestration_state = {}
        mock_session.orchestration_runs = []
        mock_session.active_orchestration_run_id = None
        mock_mgr.resume_session.return_value = mock_session

        with patch("src.webui.routes.sessions.SessionManager", return_value=mock_mgr):
            result = await get_session(session_id="s1", message_offset=5, message_limit=2)
            assert len(result["messages"]) == 2
            assert result["message_offset"] == 5


class TestGetSessionRunRoute:
    @pytest.mark.asyncio
    async def test_invalid_session_id(self):
        from src.webui.routes.sessions import get_session_run
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await get_session_run(session_id="bad.id", run_id="r1")
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_invalid_run_id(self):
        from src.webui.routes.sessions import get_session_run
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await get_session_run(session_id="s1", run_id="bad.id")
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_session_not_found(self):
        from src.webui.routes.sessions import get_session_run
        from fastapi import HTTPException
        mock_mgr = MagicMock()
        mock_mgr.resume_session.return_value = None
        with patch("src.webui.routes.sessions.SessionManager", return_value=mock_mgr):
            with pytest.raises(HTTPException) as exc_info:
                await get_session_run(session_id="s1", run_id="r1")
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_run_found_in_runs(self):
        from src.webui.routes.sessions import get_session_run
        mock_mgr = MagicMock()
        mock_session = MagicMock()
        mock_session.orchestration_runs = [{"run_id": "r1", "data": "test"}]
        mock_session.orchestration_state = {}
        mock_mgr.resume_session.return_value = mock_session

        with patch("src.webui.routes.sessions.SessionManager", return_value=mock_mgr):
            result = await get_session_run(session_id="s1", run_id="r1")
            assert result["run"]["run_id"] == "r1"
            assert result["run"]["data"] == "test"

    @pytest.mark.asyncio
    async def test_run_found_in_state(self):
        from src.webui.routes.sessions import get_session_run
        mock_mgr = MagicMock()
        mock_session = MagicMock()
        mock_session.orchestration_runs = []
        mock_session.orchestration_state = {"run_id": "r1", "data": "from_state"}
        mock_mgr.resume_session.return_value = mock_session

        with patch("src.webui.routes.sessions.SessionManager", return_value=mock_mgr):
            result = await get_session_run(session_id="s1", run_id="r1")
            assert result["run"]["data"] == "from_state"

    @pytest.mark.asyncio
    async def test_run_not_found(self):
        from src.webui.routes.sessions import get_session_run
        from fastapi import HTTPException
        mock_mgr = MagicMock()
        mock_session = MagicMock()
        mock_session.orchestration_runs = []
        mock_session.orchestration_state = {}
        mock_mgr.resume_session.return_value = mock_session

        with patch("src.webui.routes.sessions.SessionManager", return_value=mock_mgr):
            with pytest.raises(HTTPException) as exc_info:
                await get_session_run(session_id="s1", run_id="r1")
            assert exc_info.value.status_code == 404


class TestGetSessionDiffRoute:
    @pytest.mark.asyncio
    async def test_invalid_session_id(self):
        from src.webui.routes.sessions import get_session_diff
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await get_session_diff(session_id="bad.id")
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_session_not_found(self):
        from src.webui.routes.sessions import get_session_diff
        from fastapi import HTTPException
        mock_mgr = MagicMock()
        mock_mgr.resume_session.return_value = None
        with patch("src.webui.routes.sessions.SessionManager", return_value=mock_mgr):
            with pytest.raises(HTTPException) as exc_info:
                await get_session_diff(session_id="s1")
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_diff_with_no_checkpoints(self):
        from src.webui.routes.sessions import get_session_diff
        mock_mgr = MagicMock()
        mock_session = MagicMock()
        mock_session.metadata.project = "default"
        mock_session.metadata.checkpoints = None
        mock_session.metadata.created_at = 1000
        mock_mgr.resume_session.return_value = mock_session

        mock_cp_mgr = MagicMock()
        mock_cp_mgr.checkpoints = []

        with patch("src.webui.routes.sessions.SessionManager", return_value=mock_mgr):
            with patch("src.core.checkpoint.CheckpointManager", return_value=mock_cp_mgr):
                result = await get_session_diff(session_id="s1")
                assert result["session_id"] == "s1"
                assert result["checkpoint_count"] == 0


class TestUndoSessionRoute:
    @pytest.mark.asyncio
    async def test_invalid_session_id(self):
        from src.webui.routes.sessions import undo_session
        from fastapi import HTTPException
        req = UndoRequest()
        with pytest.raises(HTTPException) as exc_info:
            await undo_session(session_id="bad.id", request=req)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_active_stream_blocks_undo(self):
        from src.webui.routes.sessions import undo_session, mark_stream_active
        from fastapi import HTTPException
        mark_stream_active("active-sess")
        req = UndoRequest()
        try:
            with pytest.raises(HTTPException) as exc_info:
                await undo_session(session_id="active-sess", request=req)
            assert exc_info.value.status_code == 409
        finally:
            mark_stream_finished("active-sess")

    @pytest.mark.asyncio
    async def test_session_not_found(self):
        from src.webui.routes.sessions import undo_session
        from fastapi import HTTPException
        mock_mgr = MagicMock()
        mock_mgr.resume_session.return_value = None
        req = UndoRequest()
        with patch("src.webui.routes.sessions.SessionManager", return_value=mock_mgr):
            with pytest.raises(HTTPException) as exc_info:
                await undo_session(session_id="s1", request=req)
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_dry_run(self):
        from src.webui.routes.sessions import undo_session
        mock_mgr = MagicMock()
        mock_session = MagicMock()
        mock_session.metadata.project = "default"
        mock_session.metadata.checkpoints = ["cp1"]
        mock_mgr.resume_session.return_value = mock_session

        mock_cp_mgr = MagicMock()
        mock_cp = MagicMock()
        mock_cp.files = [MagicMock(path="test.py")]
        mock_cp_mgr.get_checkpoint.return_value = mock_cp

        req = UndoRequest(dry_run=True)
        with patch("src.webui.routes.sessions.SessionManager", return_value=mock_mgr):
            with patch("src.core.checkpoint.CheckpointManager", return_value=mock_cp_mgr):
                result = await undo_session(session_id="s1", request=req)
                assert result["dry_run"] is True
                assert "test.py" in result["restored_files"]

    @pytest.mark.asyncio
    async def test_undo_no_checkpoints_available(self):
        from src.webui.routes.sessions import undo_session
        from fastapi import HTTPException
        mock_mgr = MagicMock()
        mock_session = MagicMock()
        mock_session.metadata.project = "default"
        mock_session.metadata.checkpoints = None
        mock_mgr.resume_session.return_value = mock_session

        mock_cp_mgr = MagicMock()
        mock_cp_mgr.checkpoints = []

        req = UndoRequest()
        with patch("src.webui.routes.sessions.SessionManager", return_value=mock_mgr):
            with patch("src.core.checkpoint.CheckpointManager", return_value=mock_cp_mgr):
                with pytest.raises(HTTPException) as exc_info:
                    await undo_session(session_id="s1", request=req)
                assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_undo_success(self):
        from src.webui.routes.sessions import undo_session
        mock_mgr = MagicMock()
        mock_session = MagicMock()
        mock_session.metadata.project = "default"
        mock_session.metadata.checkpoints = ["cp1"]
        mock_mgr.resume_session.return_value = mock_session

        mock_cp_mgr = MagicMock()
        mock_result = MagicMock()
        mock_result.restored_files = ["test.py"]
        mock_result.deleted_files = ["old.py"]
        mock_result.failed_files = []
        mock_cp_mgr.rewind.return_value = mock_result

        req = UndoRequest()
        with patch("src.webui.routes.sessions.SessionManager", return_value=mock_mgr):
            with patch("src.core.checkpoint.CheckpointManager", return_value=mock_cp_mgr):
                result = await undo_session(session_id="s1", request=req)
                assert result["dry_run"] is False
                assert result["restored_files"] == ["test.py"]
                assert result["deleted_files"] == ["old.py"]


class TestGetSessionDiffWithCheckpoints:
    @pytest.mark.asyncio
    async def test_diff_with_checkpoint_files(self, tmp_path):
        from src.webui.routes.sessions import get_session_diff
        from fastapi import HTTPException

        mock_mgr = MagicMock()
        mock_session = MagicMock()
        mock_session.metadata.project = "default"
        mock_session.metadata.checkpoints = ["cp1"]
        mock_mgr.resume_session.return_value = mock_session

        mock_snap = MagicMock()
        mock_snap.path = str(tmp_path / "testfile.py")
        mock_snap.exists = True
        mock_snap.content = "before content"
        mock_snap.size = 100
        mock_snap.mtime = 1234.0

        mock_cp = MagicMock()
        mock_cp.id = "cp1"
        mock_cp.created_at = 1000
        mock_cp.prompt = "test prompt"
        mock_cp.description = "test desc"
        mock_cp.files = [mock_snap]

        mock_cp_mgr = MagicMock()
        mock_cp_mgr.get_checkpoint.return_value = mock_cp

        test_file = tmp_path / "testfile.py"
        test_file.write_text("after content", encoding="utf-8")

        with patch("src.webui.routes.sessions.SessionManager", return_value=mock_mgr):
            with patch("src.core.checkpoint.CheckpointManager", return_value=mock_cp_mgr):
                result = await get_session_diff(session_id="s1", include_content=True)
                assert result["checkpoint_count"] == 1
                cp_data = result["checkpoints"][0]
                assert cp_data["files"][0]["before"]["content"] == "before content"
                assert cp_data["files"][0]["after"]["exists"] is True
                assert cp_data["files"][0]["after"]["content"] == "after content"

    @pytest.mark.asyncio
    async def test_diff_with_deleted_file(self, tmp_path):
        from src.webui.routes.sessions import get_session_diff

        mock_mgr = MagicMock()
        mock_session = MagicMock()
        mock_session.metadata.project = "default"
        mock_session.metadata.checkpoints = ["cp1"]
        mock_mgr.resume_session.return_value = mock_session

        mock_snap = MagicMock()
        mock_snap.path = str(tmp_path / "deleted_file.py")
        mock_snap.exists = True
        mock_snap.content = "old content"
        mock_snap.size = 50
        mock_snap.mtime = 1000.0

        mock_cp = MagicMock()
        mock_cp.id = "cp1"
        mock_cp.created_at = None
        mock_cp.prompt = None
        mock_cp.description = ""
        mock_cp.files = [mock_snap]

        mock_cp_mgr = MagicMock()
        mock_cp_mgr.get_checkpoint.return_value = mock_cp

        with patch("src.webui.routes.sessions.SessionManager", return_value=mock_mgr):
            with patch("src.core.checkpoint.CheckpointManager", return_value=mock_cp_mgr):
                result = await get_session_diff(session_id="s1", include_content=True)
                file_data = result["checkpoints"][0]["files"][0]
                assert file_data["before"]["content"] == "old content"
                assert file_data["after"]["exists"] is False

    @pytest.mark.asyncio
    async def test_undo_rewind_exception(self):
        from src.webui.routes.sessions import undo_session
        from fastapi import HTTPException

        mock_mgr = MagicMock()
        mock_session = MagicMock()
        mock_session.metadata.project = "default"
        mock_session.metadata.checkpoints = ["cp1"]
        mock_mgr.resume_session.return_value = mock_session

        mock_cp_mgr = MagicMock()
        mock_cp_mgr.rewind.side_effect = RuntimeError("rewind failed")

        req = UndoRequest()
        with patch("src.webui.routes.sessions.SessionManager", return_value=mock_mgr):
            with patch("src.core.checkpoint.CheckpointManager", return_value=mock_cp_mgr):
                with pytest.raises(HTTPException) as exc_info:
                    await undo_session(session_id="s1", request=req)
                assert exc_info.value.status_code == 500
