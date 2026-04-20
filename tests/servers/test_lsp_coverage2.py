"""Targeted coverage for servers/lsp.py uncovered lines: 138-161,188-216,414-415,472,570-571."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.servers.lsp import (
    CompletionItem,
    PylspClient,
    _validate_py_file,
    lsp_completions,
    lsp_definition,
    lsp_diagnostics,
    lsp_hover,
    lsp_references,
    lsp_rename,
)


class TestPylspClientSendRequestIO:
    @pytest.mark.asyncio
    async def test_send_request_with_no_process(self):
        client = PylspClient()
        client._process = None
        result = await client._send_request("initialize", {})
        assert result is None

    @pytest.mark.asyncio
    async def test_send_request_io_error(self):
        client = PylspClient()
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stdin.write.side_effect = BrokenPipeError("pipe closed")
        client._process = mock_proc
        result = await client._send_request("initialize", {})
        assert result is None


class TestPylspClientSendNotification:
    @pytest.mark.asyncio
    async def test_notification_no_process(self):
        client = PylspClient()
        client._process = None
        await client._send_notification("initialized", {})

    @pytest.mark.asyncio
    async def test_notification_write_error(self):
        client = PylspClient()
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write.side_effect = BrokenPipeError("pipe closed")
        mock_proc.stdin.flush = MagicMock()
        client._process = mock_proc
        await client._send_notification("initialized", {})


class TestPylspClientGetDiagnostics:
    @pytest.mark.asyncio
    async def test_diagnostics_not_initialized(self):
        client = PylspClient()
        client._initialized = False
        result = await client.get_diagnostics("test.py")
        assert result == []

    @pytest.mark.asyncio
    async def test_diagnostics_initialized_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1\n")
        client = PylspClient()
        client._initialized = True
        client._send_notification = AsyncMock()
        result = await client.get_diagnostics(str(py_file))
        assert result == []


class TestPylspClientGetCompletionsNotInit:
    @pytest.mark.asyncio
    async def test_completions_not_initialized(self):
        client = PylspClient()
        client._initialized = False
        result = await client.get_completions("test.py", 0, 0)
        assert result == []


class TestPylspClientGetHoverNotInit:
    @pytest.mark.asyncio
    async def test_hover_not_initialized(self):
        client = PylspClient()
        client._initialized = False
        result = await client.get_hover("test.py", 0, 0)
        assert result is None


class TestPylspClientStop:
    def test_stop_with_process(self):
        client = PylspClient()
        mock_proc = MagicMock()
        client._process = mock_proc
        client._initialized = True
        client.stop()
        mock_proc.terminate.assert_called_once()
        assert client._process is None
        assert client._initialized is False

    def test_stop_no_process(self):
        client = PylspClient()
        client._process = None
        client.stop()
        assert client._process is None


class TestValidatePyFile:
    def test_validate_permission_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        with patch("src.servers.lsp.validate_path", side_effect=PermissionError("denied")):
            path, err = _validate_py_file("/denied")
            assert err is not None
            assert "Error" in err

    def test_validate_nonexistent(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        path, err = _validate_py_file(str(tmp_path / "nope.py"))
        assert "not found" in err.lower()


class TestLspRenameApplyWithFile:
    @pytest.mark.asyncio
    async def test_rename_apply_with_exception(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        py_file = tmp_path / "mod.py"
        py_file.write_text("old_val = 1\n")
        with patch("src.servers.lsp.validate_path", return_value=str(py_file)):
            with patch("builtins.open", side_effect=PermissionError("write denied")):
                result = await lsp_rename(str(py_file), "old_val", "new_val", preview=False)
                assert "Error" in result or "Failed" in result or "0" in result


class TestLspDefinitionAllPatternsFail:
    @pytest.mark.asyncio
    async def test_definition_not_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        with patch("src.servers.lsp.validate_path", return_value=str(tmp_path)):
            with patch("src.servers.filesystem.grep_search", return_value="No matches found"):
                result = await lsp_definition(str(tmp_path), "MissingSymbol")
                assert "not found" in result.lower()


class TestLspDiagnosticsIncludeMypy:
    @pytest.mark.asyncio
    async def test_diagnostics_include_mypy(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        py_file = tmp_path / "test.py"
        py_file.write_text("x: int = 'bad'\n")
        with patch("src.servers.lsp._run_ruff", return_value=[]):
            with patch("src.servers.lsp._run_mypy", return_value=[
                {"line": 1, "severity": "error", "message": "Incompatible types"}
            ]):
                result = await lsp_diagnostics(str(py_file), include_mypy=True)
                assert "mypy" in result


class TestLspReferencesFilesystemUnavailable:
    @pytest.mark.asyncio
    async def test_references_import_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        with patch("src.servers.lsp.validate_path", return_value=str(tmp_path)):
            with patch("src.servers.filesystem.grep_search", side_effect=ImportError("no module")):
                result = await lsp_references(str(tmp_path), "symbol")
                assert "Error" in result
