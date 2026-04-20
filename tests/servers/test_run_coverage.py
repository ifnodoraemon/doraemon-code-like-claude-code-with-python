"""Targeted coverage tests for servers.run - shell blocked, background, install validation."""

from unittest.mock import MagicMock, patch

import pytest

from src.servers.run import (
    _run_background,
    _run_install,
    _run_shell,
    _validate_package_name,
    run,
)


class TestRunShellAdditional:
    def test_shell_exception_handling(self, monkeypatch, tmp_path):
        monkeypatch.setattr("src.servers.run._is_command_blocked", lambda c: False)
        monkeypatch.setattr("src.servers.run._check_git_safety", lambda c: None)
        monkeypatch.setattr("src.servers.run.validate_path", lambda p: str(tmp_path))

        def bad_popen(*a, **kw):
            raise OSError("spawn failed")

        monkeypatch.setattr("src.servers.run.subprocess.Popen", bad_popen)
        result = _run_shell("echo hi", 5, str(tmp_path))
        assert "Error" in result


class TestRunBackgroundAdditional:
    def test_background_success_with_pid(self, monkeypatch, tmp_path):
        monkeypatch.setattr("src.servers.run._is_command_blocked", lambda c: False)
        monkeypatch.setattr("src.servers.run._check_git_safety", lambda c: None)
        monkeypatch.setattr("src.servers.run.validate_path", lambda p: p)
        mock_proc = MagicMock()
        mock_proc.pid = 99999
        monkeypatch.setattr("src.servers.run.subprocess.Popen", lambda *a, **kw: mock_proc)
        monkeypatch.setattr("src.servers.run._register_background_process", lambda p, c, d: 99999)
        result = _run_background("sleep 1", str(tmp_path))
        assert "99999" in result

    def test_background_exception(self, monkeypatch, tmp_path):
        monkeypatch.setattr("src.servers.run._is_command_blocked", lambda c: False)
        monkeypatch.setattr("src.servers.run._check_git_safety", lambda c: None)
        monkeypatch.setattr("src.servers.run.validate_path", lambda p: p)

        def bad_popen(*a, **kw):
            raise OSError("bg fail")

        monkeypatch.setattr("src.servers.run.subprocess.Popen", bad_popen)
        result = _run_background("echo hi", str(tmp_path))
        assert "Error" in result


class TestRunInstallAdditional:
    def test_valid_package_name_with_hyphen(self):
        ok, _ = _validate_package_name("my-package")
        assert ok is True

    def test_package_name_with_dot(self):
        ok, _ = _validate_package_name("zope.interface")
        assert ok is True

    def test_package_name_backtick(self):
        ok, _ = _validate_package_name("pkg`evil")
        assert ok is False

    def test_install_pypi_non_200(self, monkeypatch):
        class FakeResp:
            status = 403

        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: FakeResp())
        result = _run_install("forbidden-pkg")
        assert "not found" in result.lower()


class TestRunModeUnknown:
    def test_unknown_mode_returns_error(self):
        result = run("cmd", mode="docker")
        assert "Unknown mode" in result


class TestShellBlockedGitSafety:
    def test_git_force_push_blocked(self, monkeypatch, tmp_path):
        monkeypatch.setattr("src.servers.run._is_command_blocked", lambda c: False)
        monkeypatch.setattr("src.servers.run._check_git_safety", lambda c: "Git safety: force push blocked")
        result = _run_shell("git push --force", 10, None)
        assert "Git safety" in result
