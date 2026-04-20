"""Targeted coverage tests for servers.lsp - references, rename, definition, hover completions."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.servers.lsp import (
    CompletionItem,
    PylspClient,
    lsp_completions,
    lsp_definition,
    lsp_hover,
    lsp_references,
    lsp_rename,
)


class TestLspReferencesWithPathValidation:
    @pytest.mark.asyncio
    async def test_references_with_path_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        with patch("src.servers.lsp.validate_path", side_effect=PermissionError("denied")):
            result = await lsp_references("/denied", "symbol")
            assert "Error" in result


class TestLspRenameApply:
    @pytest.mark.asyncio
    async def test_rename_file_with_occurrences(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        py_file = tmp_path / "mod.py"
        py_file.write_text("old_name = 1\nold_name = 2\n")
        with patch("src.servers.lsp.validate_path", return_value=str(py_file)):
            result = await lsp_rename(str(py_file), "old_name", "new_name", preview=False)
            assert "Renamed" in result
            content = py_file.read_text()
            assert "new_name" in content
            assert "old_name" not in content


class TestLspDefinitionLoop:
    @pytest.mark.asyncio
    async def test_definition_tries_multiple_patterns(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        patterns_tried = []

        def mock_grep(pattern, **kw):
            patterns_tried.append(pattern)
            if "def MyObj" in pattern:
                return "file.py:10: def MyObj():"
            return "No matches found"

        with patch("src.servers.lsp.validate_path", return_value=str(tmp_path)):
            with patch("src.servers.filesystem.grep_search", side_effect=mock_grep):
                result = await lsp_definition(str(tmp_path), "MyObj")
                assert "Definition" in result
                assert any("def" in p for p in patterns_tried)


class TestLspHoverWithNoInfo:
    @pytest.mark.asyncio
    async def test_hover_with_no_info_at_position(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1\n")
        mock_client = MagicMock()
        mock_client.get_hover = AsyncMock(return_value=None)
        with patch("src.servers.lsp._get_lsp_client", new_callable=AsyncMock, return_value=mock_client):
            result = await lsp_hover(str(py_file), 1, 1)
            assert "No information" in result


class TestLspCompletionsTruncation:
    @pytest.mark.asyncio
    async def test_completions_shows_many(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1\n")
        mock_client = MagicMock()
        mock_client.get_completions = AsyncMock(return_value=[
            CompletionItem(label=f"item_{i}", kind="variable") for i in range(20)
        ])
        with patch("src.servers.lsp._get_lsp_client", new_callable=AsyncMock, return_value=mock_client):
            result = await lsp_completions(str(py_file), 1, 1)
            assert "Completions" in result
            assert "item_0" in result


class TestGetLspClient:
    @pytest.mark.asyncio
    async def test_get_lsp_client_start_fails(self):
        with patch("src.servers.lsp.PylspClient") as MockClient:
            instance = MockClient.return_value
            instance.start = AsyncMock(return_value=False)
            from src.servers.lsp import _get_lsp_client
            import src.servers.lsp as lsp_mod
            lsp_mod._lsp_client = None
            result = await _get_lsp_client()
            assert result is None
