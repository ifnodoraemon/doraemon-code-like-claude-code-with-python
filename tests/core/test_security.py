import os
from pathlib import Path

import pytest

from src.core.security.security import (
    is_sensitive_path,
    validate_path,
    SENSITIVE_PATHS,
    SENSITIVE_PATTERNS,
)


class TestIsSensitivePath:
    def test_etc_passwd(self):
        assert is_sensitive_path("/etc/passwd") is True

    def test_etc_shadow(self):
        assert is_sensitive_path("/etc/shadow") is True

    def test_root_ssh(self):
        assert is_sensitive_path("/root/.ssh") is True

    def test_proc(self):
        assert is_sensitive_path("/proc") is True

    def test_normal_path(self):
        assert is_sensitive_path("/home/user/project") is False

    def test_ssh_pattern(self):
        assert is_sensitive_path("/home/user/.ssh/config") is True

    def test_pem_pattern(self):
        assert is_sensitive_path("/home/user/id_rsa") is True

    def test_bash_history_pattern(self):
        assert is_sensitive_path("/home/user/.bash_history") is True

    def test_id_rsa_pattern(self):
        assert is_sensitive_path("/home/user/id_rsa") is True

    def test_normal_file_not_sensitive(self):
        assert is_sensitive_path("/home/user/project/main.py") is False

    def test_dot_ssh_dir_pattern(self):
        assert is_sensitive_path("/home/user/.ssh/config") is True


class TestValidatePath:
    def test_valid_path_within_sandbox(self, tmp_path):
        base = str(tmp_path)
        (tmp_path / "test.py").write_text("hello")
        result = validate_path(str(tmp_path / "test.py"), base)
        assert str(tmp_path.resolve()) in result

    def test_empty_path_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_path("")

    def test_whitespace_path_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_path("   ")

    def test_sensitive_path_raises(self):
        with pytest.raises(PermissionError, match="sensitive"):
            validate_path("/etc/passwd")

    def test_escape_sandbox_raises(self, tmp_path):
        with pytest.raises(PermissionError, match="outside"):
            validate_path("../../../etc/passwd", str(tmp_path))

    def test_absolute_path_within_sandbox(self, tmp_path):
        target = tmp_path / "subdir"
        target.mkdir()
        result = validate_path(str(target), str(tmp_path))
        assert str(target.resolve()) == result

    def test_symlink_escape_raises(self, tmp_path):
        link = tmp_path / "escape_link"
        try:
            link.symlink_to("/etc/passwd")
        except OSError:
            pytest.skip("Cannot create symlink")
        with pytest.raises(PermissionError):
            validate_path(str(link), str(tmp_path))

    def test_home_expansion(self, tmp_path):
        base = str(tmp_path)
        result = validate_path(str(tmp_path / "file.txt"), base)
        assert result.endswith("file.txt")
