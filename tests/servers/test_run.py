from unittest.mock import MagicMock, patch

import pytest

from src.servers.run import (
    DEFAULT_LIMITS,
    ResourceLimits,
    _create_sandbox_preexec,
    _get_sandbox_wrapper_code,
    _indent_code,
    _run_background,
    _run_install,
    _run_python,
    _run_shell,
    _validate_package_name,
    run,
)


class _FakeStdout:
    def __init__(self, process, lines):
        self._process = process
        self._lines = list(lines)

    def readline(self):
        if self._lines:
            return self._lines.pop(0)
        return ""

    def close(self):
        self._process.done = True


class _FakeProcess:
    def __init__(self, lines):
        self.done = False
        self.returncode = 0
        self.stdout = _FakeStdout(self, lines)

    def poll(self):
        return self.returncode if self.done else None

    def kill(self):
        self.done = True


def test_run_shell_does_not_truncate_output(monkeypatch, tmp_path):
    large_output = "x" * 40000

    monkeypatch.setattr("src.servers.run._is_command_blocked", lambda command: False)
    monkeypatch.setattr("src.servers.run._check_git_safety", lambda command: None)
    monkeypatch.setattr("src.servers.run.validate_path", lambda path: path)
    monkeypatch.setattr(
        "src.servers.run.subprocess.Popen",
        lambda *args, **kwargs: _FakeProcess([large_output]),
    )

    result = _run_shell("echo large", timeout=5, working_dir=str(tmp_path))

    assert result == large_output
    assert "truncated" not in result.lower()


class TestRun:
    def test_unknown_mode(self):
        result = run("cmd", mode="unknown")
        assert "Unknown mode" in result

    @patch("src.servers.run._run_shell", return_value="shell ok")
    def test_shell_mode(self, mock_shell):
        result = run("echo hi", mode="shell")
        assert result == "shell ok"

    @patch("src.servers.run._run_python", return_value="py ok")
    def test_python_mode(self, mock_py):
        result = run("print(1)", mode="python")
        assert result == "py ok"

    @patch("src.servers.run._run_background", return_value="bg ok")
    def test_background_mode(self, mock_bg):
        result = run("sleep 1", mode="background")
        assert result == "bg ok"

    @patch("src.servers.run._run_install", return_value="inst ok")
    def test_install_mode(self, mock_inst):
        result = run("requests", mode="install")
        assert result == "inst ok"


class TestRunShellBlocked:
    def test_blocked_command(self, monkeypatch):
        monkeypatch.setattr("src.servers.run._is_command_blocked", lambda c: True)
        result = _run_shell("rm -rf /", 10, None)
        assert "blocked" in result.lower()

    def test_git_safety_fail(self, monkeypatch):
        monkeypatch.setattr("src.servers.run._is_command_blocked", lambda c: False)
        monkeypatch.setattr(
            "src.servers.run._check_git_safety", lambda c: "Git safety: force push blocked"
        )
        result = _run_shell("git push --force", 10, None)
        assert "Git safety" in result

    def test_invalid_working_dir(self, monkeypatch):
        monkeypatch.setattr("src.servers.run._is_command_blocked", lambda c: False)
        monkeypatch.setattr("src.servers.run._check_git_safety", lambda c: None)

        def raise_val(p):
            raise ValueError("bad dir")

        monkeypatch.setattr("src.servers.run.validate_path", raise_val)
        result = _run_shell("echo hi", 10, "/bad")
        assert "Error" in result

    def test_nonexistent_working_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr("src.servers.run._is_command_blocked", lambda c: False)
        monkeypatch.setattr("src.servers.run._check_git_safety", lambda c: None)
        monkeypatch.setattr("src.servers.run.validate_path", lambda p: p)
        result = _run_shell("echo hi", 10, str(tmp_path / "nonexistent"))
        assert "does not exist" in result

    def test_nonzero_exit_code(self, monkeypatch, tmp_path):
        monkeypatch.setattr("src.servers.run._is_command_blocked", lambda c: False)
        monkeypatch.setattr("src.servers.run._check_git_safety", lambda c: None)
        monkeypatch.setattr("src.servers.run.validate_path", lambda p: p)
        proc = _FakeProcess(["error\n"])
        proc.returncode = 1
        proc.done = True
        monkeypatch.setattr("src.servers.run.subprocess.Popen", lambda *a, **kw: proc)
        result = _run_shell("bad_cmd", 5, str(tmp_path))
        assert "Exit code: 1" in result


class TestRunPython:
    def test_simple_code(self):
        result = _run_python("print('hello')", timeout=10)
        assert "hello" in result

    def test_timeout(self, monkeypatch):
        import subprocess

        def raise_timeout(*a, **kw):
            raise subprocess.TimeoutExpired(cmd="x", timeout=1)

        monkeypatch.setattr("src.servers.run.subprocess.run", raise_timeout)
        result = _run_python("import time; time.sleep(999)", timeout=1)
        assert "timed out" in result.lower()

    def test_error_handling(self):
        result = _run_python("raise ValueError('test')", timeout=10)
        assert "ValueError" in result or "Error" in result


class TestRunBackground:
    def test_blocked_command(self, monkeypatch):
        monkeypatch.setattr("src.servers.run._is_command_blocked", lambda c: True)
        result = _run_background("rm -rf /", None)
        assert "blocked" in result.lower()

    def test_git_safety_fail(self, monkeypatch):
        monkeypatch.setattr("src.servers.run._is_command_blocked", lambda c: False)
        monkeypatch.setattr("src.servers.run._check_git_safety", lambda c: "blocked")
        result = _run_background("git push --force", None)
        assert "blocked" in result

    def test_invalid_dir(self, monkeypatch):
        monkeypatch.setattr("src.servers.run._is_command_blocked", lambda c: False)
        monkeypatch.setattr("src.servers.run._check_git_safety", lambda c: None)

        def raise_val(p):
            raise ValueError("bad")

        monkeypatch.setattr("src.servers.run.validate_path", raise_val)
        result = _run_background("echo hi", "/bad")
        assert "Error" in result

    def test_success(self, monkeypatch, tmp_path):
        monkeypatch.setattr("src.servers.run._is_command_blocked", lambda c: False)
        monkeypatch.setattr("src.servers.run._check_git_safety", lambda c: None)
        monkeypatch.setattr("src.servers.run.validate_path", lambda p: p)
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        monkeypatch.setattr("src.servers.run.subprocess.Popen", lambda *a, **kw: mock_proc)
        monkeypatch.setattr("src.servers.run._register_background_process", lambda p, c, d: 12345)
        result = _run_background("sleep 1", str(tmp_path))
        assert "PID: 12345" in result


class TestRunInstall:
    def test_invalid_package_name(self):
        result = _run_install("../../etc/passwd")
        assert "Error" in result

    def test_empty_package_name(self):
        result = _run_install("")
        assert "Error" in result

    def test_package_with_semicolon(self):
        result = _run_install("pkg;rm -rf /")
        assert "Error" in result

    def test_invalid_pattern(self):
        result = _run_install("123invalid")
        assert "Error" in result


class TestValidatePackageName:
    def test_valid(self):
        ok, msg = _validate_package_name("requests")
        assert ok is True

    def test_empty(self):
        ok, msg = _validate_package_name("")
        assert ok is False

    def test_dangerous_chars(self):
        ok, msg = _validate_package_name("pkg;evil")
        assert ok is False

    def test_invalid_start(self):
        ok, msg = _validate_package_name(".hidden")
        assert ok is False


class TestResourceLimits:
    def test_defaults(self):
        assert DEFAULT_LIMITS.max_memory_mb == 512
        assert DEFAULT_LIMITS.max_cpu_time_seconds == 30

    def test_custom(self):
        r = ResourceLimits(max_memory_mb=1024)
        assert r.max_memory_mb == 1024


class TestIndentCode:
    def test_basic(self):
        result = _indent_code("line1\nline2", 4)
        assert result == "    line1\n    line2"

    def test_blank_lines_preserved(self):
        result = _indent_code("line1\n\nline2", 2)
        assert "\n\n" in result


class TestSandboxWrapper:
    def test_wrapper_includes_user_code(self):
        code = _get_sandbox_wrapper_code("x = 1")
        assert "x = 1" in code


class TestCreateSandboxPreexec:
    def test_preexec_returns_callable(self):
        limits = ResourceLimits()
        fn = _create_sandbox_preexec(limits)
        assert callable(fn)

    @patch("platform.system", return_value="Windows")
    def test_preexec_windows_noop(self, mock_sys):
        limits = ResourceLimits()
        fn = _create_sandbox_preexec(limits)
        fn()


class TestRunShellSuccess:
    def test_successful_command_output(self, monkeypatch, tmp_path):
        monkeypatch.setattr("src.servers.run._is_command_blocked", lambda c: False)
        monkeypatch.setattr("src.servers.run._check_git_safety", lambda c: None)
        monkeypatch.setattr("src.servers.run.validate_path", lambda p: p)
        proc = _FakeProcess(["hello world\n"])
        monkeypatch.setattr("src.servers.run.subprocess.Popen", lambda *a, **kw: proc)
        result = _run_shell("echo hello", 5, str(tmp_path))
        assert "hello world" in result

    def test_empty_output_success(self, monkeypatch, tmp_path):
        monkeypatch.setattr("src.servers.run._is_command_blocked", lambda c: False)
        monkeypatch.setattr("src.servers.run._check_git_safety", lambda c: None)
        monkeypatch.setattr("src.servers.run.validate_path", lambda p: p)
        proc = _FakeProcess([""])
        monkeypatch.setattr("src.servers.run.subprocess.Popen", lambda *a, **kw: proc)
        result = _run_shell("true", 5, str(tmp_path))
        assert "successfully" in result.lower()


class TestRunPythonExitCode137:
    def test_memory_exceeded(self, monkeypatch):
        import subprocess

        def mock_run(*a, **kw):
            return MagicMock(returncode=137, stdout="", stderr="")

        monkeypatch.setattr("src.servers.run.subprocess.run", mock_run)
        result = _run_python("x=1", timeout=10)
        assert "137" in result
        assert "Exceeded memory" in result


class TestRunPythonGeneralException:
    def test_general_exception(self, monkeypatch):
        def mock_run(*a, **kw):
            raise OSError("broken")

        monkeypatch.setattr("src.servers.run.subprocess.run", mock_run)
        result = _run_python("x=1", timeout=10)
        assert "Error" in result


class TestRunBackgroundBlocked:
    def test_blocked_background_command(self, monkeypatch):
        monkeypatch.setattr("src.servers.run._is_command_blocked", lambda c: True)
        result = _run_background("rm -rf /", None)
        assert "blocked" in result.lower()


class TestRunBackgroundNonexistentDir:
    def test_nonexistent_working_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr("src.servers.run._is_command_blocked", lambda c: False)
        monkeypatch.setattr("src.servers.run._check_git_safety", lambda c: None)
        monkeypatch.setattr("src.servers.run.validate_path", lambda p: p)
        result = _run_background("echo hi", str(tmp_path / "nonexistent"))
        assert "does not exist" in result


class TestRunInstallPyPIUnreachable:
    def test_pypi_unreachable(self, monkeypatch):
        from urllib.error import URLError

        def mock_urlopen(*a, **kw):
            raise URLError("unreachable")

        monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)
        result = _run_install("requests")
        assert "refused" in result.lower() or "unavailable" in result.lower()


class TestRunInstallPypiNotFound:
    def test_pypi_not_found(self, monkeypatch):
        class FakeResp:
            status = 404

        def mock_urlopen(*a, **kw):
            return FakeResp()

        monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)
        result = _run_install("nonexistent_pkg_xyz")
        assert "not found" in result.lower()


class TestRunInstallSuccess:
    def test_successful_install(self, monkeypatch):
        class FakeResp:
            status = 200

        def mock_urlopen(*a, **kw):
            return FakeResp()

        def mock_pip_run(*a, **kw):
            return MagicMock(returncode=0, stdout="ok", stderr="")

        monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)
        monkeypatch.setattr("src.servers.run.subprocess.run", mock_pip_run)
        result = _run_install("requests")
        assert "Successfully" in result


class TestRunInstallPipFail:
    def test_pip_failure(self, monkeypatch):
        class FakeResp:
            status = 200

        def mock_urlopen(*a, **kw):
            return FakeResp()

        def mock_pip_run(*a, **kw):
            return MagicMock(returncode=1, stdout="", stderr="pip error")

        monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)
        monkeypatch.setattr("src.servers.run.subprocess.run", mock_pip_run)
        result = _run_install("badpkg")
        assert "Failed" in result


class TestRunInstallException:
    def test_install_exception(self, monkeypatch):
        class FakeResp:
            status = 200

        def mock_urlopen(*a, **kw):
            return FakeResp()

        def mock_pip_run(*a, **kw):
            raise RuntimeError("boom")

        monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)
        monkeypatch.setattr("src.servers.run.subprocess.run", mock_pip_run)
        result = _run_install("pkg")
        assert "Error" in result
