"""Comprehensive tests for src/servers/lsp.py"""

import json
import os
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.servers.lsp import (
    CompletionItem,
    Diagnostic,
    PylspClient,
    Position,
    Range,
    _require_lsp,
    _run_mypy,
    _run_ruff,
    _validate_py_file,
    lsp_completions,
    lsp_definition,
    lsp_diagnostics,
    lsp_hover,
    lsp_references,
    lsp_rename,
)


class TestPosition:
    def test_fields(self):
        p = Position(line=5, character=10)
        assert p.line == 5
        assert p.character == 10


class TestRange:
    def test_fields(self):
        r = Range(start=Position(line=0, character=0), end=Position(line=10, character=5))
        assert r.start.line == 0
        assert r.end.character == 5


class TestDiagnostic:
    def test_fields(self):
        d = Diagnostic(
            message="undefined variable",
            severity="error",
            range=Range(start=Position(line=1, character=0), end=Position(line=1, character=5)),
            source="pylsp",
        )
        assert d.severity == "error"
        assert d.source == "pylsp"


class TestCompletionItem:
    def test_defaults(self):
        ci = CompletionItem(label="foo", kind="function")
        assert ci.detail is None
        assert ci.documentation is None

    def test_with_extras(self):
        ci = CompletionItem(label="bar", kind="class", detail="class Bar", documentation="A bar class")
        assert ci.detail == "class Bar"
        assert ci.documentation == "A bar class"


class TestPylspClientInit:
    def test_init(self):
        client = PylspClient()
        assert client._process is None
        assert client._initialized is False
        assert client._request_id == 0

    def test_stop_without_process(self):
        client = PylspClient()
        client.stop()
        assert client._initialized is False

    def test_stop_with_process(self):
        client = PylspClient()
        mock_proc = MagicMock()
        client._process = mock_proc
        client._initialized = True
        client.stop()
        mock_proc.terminate.assert_called_once()
        assert client._process is None
        assert client._initialized is False


class TestPylspClientStart:
    @pytest.mark.asyncio
    async def test_start_file_not_found(self):
        client = PylspClient()
        with patch("src.servers.lsp.subprocess.Popen", side_effect=FileNotFoundError):
            result = await client.start()
            assert result is False

    @pytest.mark.asyncio
    async def test_start_generic_exception(self):
        client = PylspClient()
        with patch("src.servers.lsp.subprocess.Popen", side_effect=OSError("fail")):
            result = await client.start()
            assert result is False

    @pytest.mark.asyncio
    async def test_start_success(self):
        client = PylspClient()
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()

        with patch("src.servers.lsp.subprocess.Popen", return_value=mock_proc):
            with patch.object(client, "_initialize", new_callable=AsyncMock):
                result = await client.start()
                assert result is True
                assert client._process is mock_proc


class TestPylspClientSendRequest:
    @pytest.mark.asyncio
    async def test_send_request_no_process(self):
        client = PylspClient()
        result = await client._send_request("initialize", {})
        assert result is None

    @pytest.mark.asyncio
    async def test_send_request_no_stdin(self):
        client = PylspClient()
        client._process = MagicMock()
        client._process.stdin = None
        result = await client._send_request("initialize", {})
        assert result is None


class TestPylspClientSendNotification:
    @pytest.mark.asyncio
    async def test_send_notification_no_process(self):
        client = PylspClient()
        await client._send_notification("initialized", {})

    @pytest.mark.asyncio
    async def test_send_notification_no_stdin(self):
        client = PylspClient()
        client._process = MagicMock()
        client._process.stdin = None
        await client._send_notification("initialized", {})

    @pytest.mark.asyncio
    async def test_send_notification_success(self):
        client = PylspClient()
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        client._process = mock_proc
        await client._send_notification("initialized", {})
        mock_proc.stdin.write.assert_called()
        mock_proc.stdin.flush.assert_called()

    @pytest.mark.asyncio
    async def test_send_notification_io_error(self):
        client = PylspClient()
        mock_proc = MagicMock()
        mock_proc.stdin.write.side_effect = BrokenPipeError("broken")
        client._process = mock_proc
        await client._send_notification("initialized", {})


class TestPylspClientGetDiagnostics:
    @pytest.mark.asyncio
    async def test_not_initialized(self):
        client = PylspClient()
        client._initialized = False
        result = await client.get_diagnostics("test.py")
        assert result == []


class TestPylspClientGetCompletions:
    @pytest.mark.asyncio
    async def test_not_initialized(self):
        client = PylspClient()
        client._initialized = False
        result = await client.get_completions("test.py", 0, 0)
        assert result == []

    @pytest.mark.asyncio
    async def test_completions_with_response(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        py_file = tmp_path / "test.py"
        py_file.write_text("import os\n")

        client = PylspClient()
        client._initialized = True
        client._process = MagicMock()
        client._process.stdin = MagicMock()

        response = {
            "result": {
                "items": [
                    {"label": "os", "kind": 9, "detail": "module os"},
                    {"label": "path", "kind": 6, "detail": "variable"},
                ]
            }
        }

        with patch.object(client, "_send_notification", new_callable=AsyncMock):
            with patch.object(client, "_send_request", new_callable=AsyncMock, return_value=response):
                result = await client.get_completions(str(py_file), 0, 7)
                assert len(result) == 2
                assert result[0].label == "os"
                assert result[0].kind == "module"
                assert result[1].label == "path"
                assert result[1].kind == "variable"

    @pytest.mark.asyncio
    async def test_completions_no_result(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1\n")

        client = PylspClient()
        client._initialized = True
        client._process = MagicMock()

        with patch.object(client, "_send_notification", new_callable=AsyncMock):
            with patch.object(client, "_send_request", new_callable=AsyncMock, return_value=None):
                result = await client.get_completions(str(py_file), 0, 0)
                assert result == []

    @pytest.mark.asyncio
    async def test_completions_list_result(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1\n")

        client = PylspClient()
        client._initialized = True
        client._process = MagicMock()

        response = {
            "result": [{"label": "foo", "kind": 3}]
        }

        with patch.object(client, "_send_notification", new_callable=AsyncMock):
            with patch.object(client, "_send_request", new_callable=AsyncMock, return_value=response):
                result = await client.get_completions(str(py_file), 0, 0)
                assert len(result) == 1
                assert result[0].label == "foo"
                assert result[0].kind == "function"


class TestPylspClientGetHover:
    @pytest.mark.asyncio
    async def test_not_initialized(self):
        client = PylspClient()
        client._initialized = False
        result = await client.get_hover("test.py", 0, 0)
        assert result is None

    @pytest.mark.asyncio
    async def test_hover_dict_contents(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1\n")

        client = PylspClient()
        client._initialized = True
        client._process = MagicMock()

        response = {"result": {"contents": {"value": "int"}}}

        with patch.object(client, "_send_notification", new_callable=AsyncMock):
            with patch.object(client, "_send_request", new_callable=AsyncMock, return_value=response):
                result = await client.get_hover(str(py_file), 0, 0)
                assert result == "int"

    @pytest.mark.asyncio
    async def test_hover_list_contents(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1\n")

        client = PylspClient()
        client._initialized = True
        client._process = MagicMock()

        response = {"result": {"contents": [{"value": "type1"}, "plain"]}}

        with patch.object(client, "_send_notification", new_callable=AsyncMock):
            with patch.object(client, "_send_request", new_callable=AsyncMock, return_value=response):
                result = await client.get_hover(str(py_file), 0, 0)
                assert "type1" in result
                assert "plain" in result

    @pytest.mark.asyncio
    async def test_hover_string_contents(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1\n")

        client = PylspClient()
        client._initialized = True
        client._process = MagicMock()

        response = {"result": {"contents": "just a string"}}

        with patch.object(client, "_send_notification", new_callable=AsyncMock):
            with patch.object(client, "_send_request", new_callable=AsyncMock, return_value=response):
                result = await client.get_hover(str(py_file), 0, 0)
                assert result == "just a string"

    @pytest.mark.asyncio
    async def test_hover_no_result(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1\n")

        client = PylspClient()
        client._initialized = True
        client._process = MagicMock()

        with patch.object(client, "_send_notification", new_callable=AsyncMock):
            with patch.object(client, "_send_request", new_callable=AsyncMock, return_value=None):
                result = await client.get_hover(str(py_file), 0, 0)
                assert result is None

    @pytest.mark.asyncio
    async def test_hover_empty_result(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1\n")

        client = PylspClient()
        client._initialized = True
        client._process = MagicMock()

        response = {"result": None}

        with patch.object(client, "_send_notification", new_callable=AsyncMock):
            with patch.object(client, "_send_request", new_callable=AsyncMock, return_value=response):
                result = await client.get_hover(str(py_file), 0, 0)
                assert result is None


class TestValidatePyFile:
    def test_nonexistent(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        path, err = _validate_py_file(str(tmp_path / "missing.py"))
        assert err is not None
        assert "not found" in err

    def test_non_py_extension(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        f = tmp_path / "test.txt"
        f.write_text("hello")
        path, err = _validate_py_file(str(f))
        assert err is not None
        assert ".py" in err

    def test_valid_py_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        f = tmp_path / "test.py"
        f.write_text("x = 1")
        path, err = _validate_py_file(str(f))
        assert err is None
        assert path.endswith(".py")


class TestRunRuff:
    def test_returns_empty_on_import_error(self):
        with patch.dict("sys.modules", {"src.servers.lint": None}):
            result = _run_ruff("some.py")
            assert result == []

    def test_returns_parsed_json(self):
        mock_data = [{"location": {"row": 1}, "code": "F841", "message": "local var unused"}]
        with patch("src.servers.lint._run_command", return_value=(0, json.dumps(mock_data), "")):
            result = _run_ruff("test.py")
            assert result == mock_data

    def test_returns_empty_on_bad_json(self):
        with patch("src.servers.lint._run_command", return_value=(0, "bad json", "")):
            result = _run_ruff("test.py")
            assert result == []


class TestRunMypy:
    def test_returns_empty_on_import_error(self):
        with patch.dict("sys.modules", {"src.servers.lint": None}):
            result = _run_mypy("some.py")
            assert result == []

    def test_parses_mypy_output(self):
        mypy_output = "file.py:10: error: Incompatible types [assignment]"
        with patch("src.servers.lint._run_command", return_value=(1, mypy_output, "")):
            result = _run_mypy("test.py")
            assert len(result) == 1
            assert result[0]["line"] == 10
            assert result[0]["severity"] == "error"

    def test_ignores_empty_lines(self):
        with patch("src.servers.lint._run_command", return_value=(0, "", "")):
            result = _run_mypy("test.py")
            assert result == []


class TestLspDiagnostics:
    @pytest.mark.asyncio
    async def test_invalid_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = await lsp_diagnostics(str(tmp_path / "missing.py"))
        assert "Error" in result or "not found" in result

    @pytest.mark.asyncio
    async def test_no_issues(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        py_file = tmp_path / "clean.py"
        py_file.write_text("x = 1\n")
        with patch("src.servers.lsp._run_ruff", return_value=[]):
            result = await lsp_diagnostics(str(py_file))
            assert "No issues" in result

    @pytest.mark.asyncio
    async def test_with_ruff_issues(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        py_file = tmp_path / "messy.py"
        py_file.write_text("x = 1\n")
        mock_diags = [{"location": {"row": 1}, "code": "F841", "message": "unused var"}]
        with patch("src.servers.lsp._run_ruff", return_value=mock_diags):
            result = await lsp_diagnostics(str(py_file))
            assert "F841" in result
            assert "L1" in result

    @pytest.mark.asyncio
    async def test_with_mypy(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        py_file = tmp_path / "typed.py"
        py_file.write_text("x: int = 'bad'\n")
        mock_mypy = [{"line": 1, "severity": "error", "message": "Incompatible types"}]
        with patch("src.servers.lsp._run_ruff", return_value=[]):
            with patch("src.servers.lsp._run_mypy", return_value=mock_mypy):
                result = await lsp_diagnostics(str(py_file), include_mypy=True)
                assert "mypy:error" in result

    @pytest.mark.asyncio
    async def test_non_py_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("hello")
        result = await lsp_diagnostics(str(txt_file))
        assert "Error" in result


class TestLspCompletions:
    @pytest.mark.asyncio
    async def test_invalid_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = await lsp_completions(str(tmp_path / "missing.py"), 1, 1)
        assert "Error" in result or "not found" in result

    @pytest.mark.asyncio
    async def test_lsp_not_available(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1\n")
        with patch("src.servers.lsp._get_lsp_client", new_callable=AsyncMock, return_value=None):
            result = await lsp_completions(str(py_file), 1, 1)
            assert "not available" in result


class TestLspHover:
    @pytest.mark.asyncio
    async def test_invalid_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = await lsp_hover(str(tmp_path / "missing.py"), 1, 1)
        assert "Error" in result or "not found" in result

    @pytest.mark.asyncio
    async def test_lsp_not_available(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1\n")
        with patch("src.servers.lsp._get_lsp_client", new_callable=AsyncMock, return_value=None):
            result = await lsp_hover(str(py_file), 1, 1)
            assert "not available" in result


class TestLspReferences:
    @pytest.mark.asyncio
    async def test_invalid_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        with patch("src.servers.lsp.validate_path", side_effect=ValueError("bad path")):
            result = await lsp_references("/bad/path", "symbol")
            assert "Error" in result

    @pytest.mark.asyncio
    async def test_grep_search_import_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        with patch("src.servers.lsp.validate_path", return_value=str(tmp_path)):
            with patch.dict("sys.modules", {"src.servers.filesystem": None}):
                result = await lsp_references(str(tmp_path), "symbol")
                assert "not available" in result

    @pytest.mark.asyncio
    async def test_grep_search_success(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        with patch("src.servers.lsp.validate_path", return_value=str(tmp_path)):
            with patch("src.servers.filesystem.grep_search", return_value="file.py:1: symbol here"):
                result = await lsp_references(str(tmp_path), "symbol")
                assert "References" in result

    @pytest.mark.asyncio
    async def test_grep_no_matches(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        with patch("src.servers.lsp.validate_path", return_value=str(tmp_path)):
            with patch("src.servers.filesystem.grep_search", return_value="No matches found"):
                result = await lsp_references(str(tmp_path), "symbol")
                assert "No references" in result


class TestLspRename:
    @pytest.mark.asyncio
    async def test_invalid_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        with patch("src.servers.lsp.validate_path", side_effect=ValueError("bad")):
            result = await lsp_rename("/bad", "old", "new")
            assert "Error" in result

    @pytest.mark.asyncio
    async def test_path_not_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        with patch("src.servers.lsp.validate_path", return_value=str(tmp_path / "nope")):
            result = await lsp_rename(str(tmp_path / "nope"), "old", "new")
            assert "not found" in result

    @pytest.mark.asyncio
    async def test_rename_preview(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        py_file = tmp_path / "mod.py"
        py_file.write_text("old_name = 1\n")
        with patch("src.servers.lsp.validate_path", return_value=str(py_file)):
            result = await lsp_rename(str(py_file), "old_name", "new_name", preview=True)
            assert "Preview" in result
            assert "old_name" in result
            assert "new_name" in result

    @pytest.mark.asyncio
    async def test_rename_apply(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        py_file = tmp_path / "mod.py"
        py_file.write_text("old_name = 1\nold_name = 2\n")
        with patch("src.servers.lsp.validate_path", return_value=str(py_file)):
            result = await lsp_rename(str(py_file), "old_name", "new_name", preview=False)
            assert "Renamed" in result
            content = py_file.read_text()
            assert "new_name" in content

    @pytest.mark.asyncio
    async def test_rename_no_occurrences(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        py_file = tmp_path / "mod.py"
        py_file.write_text("x = 1\n")
        with patch("src.servers.lsp.validate_path", return_value=str(py_file)):
            result = await lsp_rename(str(py_file), "nonexistent", "new_name", preview=False)
            assert "No occurrences" in result

    @pytest.mark.asyncio
    async def test_rename_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        sub = tmp_path / "pkg"
        sub.mkdir()
        py_file = sub / "mod.py"
        py_file.write_text("target_var = 1\n")
        with patch("src.servers.lsp.validate_path", return_value=str(sub)):
            with patch("src.servers.lsp.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout=str(py_file) + "\n", returncode=0)
                result = await lsp_rename(str(sub), "target_var", "renamed_var", preview=True)
                assert "Preview" in result


class TestLspDefinition:
    @pytest.mark.asyncio
    async def test_invalid_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        with patch("src.servers.lsp.validate_path", side_effect=ValueError("bad")):
            result = await lsp_definition("/bad", "MyClass")
            assert "Error" in result

    @pytest.mark.asyncio
    async def test_found_class_definition(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        with patch("src.servers.lsp.validate_path", return_value=str(tmp_path)):
            with patch("src.servers.filesystem.grep_search", return_value="file.py:5: class MyClass:"):
                result = await lsp_definition(str(tmp_path), "MyClass")
                assert "Definition" in result

    @pytest.mark.asyncio
    async def test_not_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        with patch("src.servers.lsp.validate_path", return_value=str(tmp_path)):
            with patch("src.servers.filesystem.grep_search", return_value="No matches found"):
                result = await lsp_definition(str(tmp_path), "MissingSymbol")
                assert "not found" in result

    @pytest.mark.asyncio
    async def test_import_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        with patch("src.servers.lsp.validate_path", return_value=str(tmp_path)):
            with patch.dict("sys.modules", {"src.servers.filesystem": None}):
                result = await lsp_definition(str(tmp_path), "MyClass")
                assert "not available" in result


class TestRequireLsp:
    @pytest.mark.asyncio
    async def test_lsp_not_available(self):
        with patch("src.servers.lsp._get_lsp_client", new_callable=AsyncMock, return_value=None):
            client, err = await _require_lsp()
            assert client is None
            assert err is not None

    @pytest.mark.asyncio
    async def test_lsp_available(self):
        mock_client = MagicMock()
        with patch("src.servers.lsp._get_lsp_client", new_callable=AsyncMock, return_value=mock_client):
            client, err = await _require_lsp()
            assert client is mock_client
            assert err is None


class TestPylspClientInitialize:
    @pytest.mark.asyncio
    async def test_initialize_success(self):
        client = PylspClient()
        client._process = MagicMock()
        client._process.stdin = MagicMock()
        with patch.object(client, "_send_request", new_callable=AsyncMock, return_value={"id": 1}):
            with patch.object(client, "_send_notification", new_callable=AsyncMock):
                await client._initialize()
                assert client._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_no_response(self):
        client = PylspClient()
        client._process = MagicMock()
        client._process.stdin = MagicMock()
        with patch.object(client, "_send_request", new_callable=AsyncMock, return_value=None):
            await client._initialize()
            assert client._initialized is False


class TestPylspClientSendRequestWithProcess:
    @pytest.mark.asyncio
    async def test_send_request_with_process(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        client = PylspClient()
        client._process = MagicMock()
        client._process.stdin = MagicMock()
        client._process.stdout = MagicMock()
        with patch("src.servers.lsp.asyncio.get_running_loop", side_effect=RuntimeError("no loop")):
            try:
                result = await client._send_request("test", {})
            except RuntimeError:
                pass


class TestLspRenameTimeout:
    @pytest.mark.asyncio
    async def test_rename_timeout(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        with patch("src.servers.lsp.validate_path", return_value=str(tmp_path)):
            with patch("src.servers.lsp.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="grep", timeout=30)):
                result = await lsp_rename(str(tmp_path), "old", "new", preview=False)
                assert "timed out" in result.lower() or "Error" in result


class TestLspRenameGenericException:
    @pytest.mark.asyncio
    async def test_rename_generic_exception(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        with patch("src.servers.lsp.validate_path", return_value=str(tmp_path)):
            with patch("src.servers.lsp.subprocess.run", side_effect=OSError("fail")):
                result = await lsp_rename(str(tmp_path), "old", "new", preview=False)
                assert "Error" in result


class TestLspDefinitionAllPatternsFail:
    @pytest.mark.asyncio
    async def test_definition_all_patterns_fail(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        with patch("src.servers.lsp.validate_path", return_value=str(tmp_path)):
            with patch("src.servers.filesystem.grep_search", side_effect=Exception("grep fail")):
                result = await lsp_definition(str(tmp_path), "Missing")
                assert "not found" in result.lower()


class TestLspReferencesException:
    @pytest.mark.asyncio
    async def test_grep_search_exception(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        with patch("src.servers.lsp.validate_path", return_value=str(tmp_path)):
            with patch("src.servers.filesystem.grep_search", side_effect=Exception("search fail")):
                result = await lsp_references(str(tmp_path), "symbol")
                assert "Error" in result


class TestLspCompletionsWithClient:
    @pytest.mark.asyncio
    async def test_completions_with_lsp_client(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1\n")
        mock_client = MagicMock()
        mock_client.get_completions = AsyncMock(return_value=[
            CompletionItem(label="foo", kind="function", detail="a func"),
        ] * 20)
        with patch("src.servers.lsp._get_lsp_client", new_callable=AsyncMock, return_value=mock_client):
            result = await lsp_completions(str(py_file), 1, 1)
            assert "Completions" in result


class TestLspHoverWithClient:
    @pytest.mark.asyncio
    async def test_hover_with_lsp_client(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1\n")
        mock_client = MagicMock()
        mock_client.get_hover = AsyncMock(return_value="int")
        with patch("src.servers.lsp._get_lsp_client", new_callable=AsyncMock, return_value=mock_client):
            result = await lsp_hover(str(py_file), 1, 1)
            assert "Info" in result

    @pytest.mark.asyncio
    async def test_hover_no_info(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1\n")
        mock_client = MagicMock()
        mock_client.get_hover = AsyncMock(return_value=None)
        with patch("src.servers.lsp._get_lsp_client", new_callable=AsyncMock, return_value=mock_client):
            result = await lsp_hover(str(py_file), 1, 1)
            assert "No information" in result
