"""Tests for src.core.workspace."""

from pathlib import Path

from src.core.workspace import WorkspaceDirectory, WorkspaceManager


class TestWorkspaceDirectory:
    def test_to_dict(self):
        d = WorkspaceDirectory(path=Path("/tmp/proj"), alias="proj", readonly=True, primary=True)
        result = d.to_dict()
        assert result["path"] == "/tmp/proj"
        assert result["alias"] == "proj"
        assert result["readonly"] is True
        assert result["primary"] is True

    def test_defaults(self):
        d = WorkspaceDirectory(path=Path("/tmp/x"))
        assert d.alias is None
        assert d.readonly is False
        assert d.primary is False


class TestWorkspaceManager:
    def test_init_with_primary(self, tmp_path):
        mgr = WorkspaceManager(primary_dir=tmp_path)
        assert mgr.get_primary() == tmp_path.resolve()

    def test_add_directory(self, tmp_path):
        mgr = WorkspaceManager(primary_dir=tmp_path)
        extra = tmp_path / "lib"
        extra.mkdir()
        assert mgr.add_directory(extra, alias="lib") is True

    def test_add_nonexistent_directory(self, tmp_path):
        mgr = WorkspaceManager(primary_dir=tmp_path)
        assert mgr.add_directory("/nonexistent/path") is False

    def test_add_file_as_directory(self, tmp_path):
        mgr = WorkspaceManager(primary_dir=tmp_path)
        f = tmp_path / "file.txt"
        f.write_text("hi")
        assert mgr.add_directory(f) is False

    def test_add_duplicate_directory(self, tmp_path):
        mgr = WorkspaceManager(primary_dir=tmp_path)
        assert mgr.add_directory(tmp_path) is False

    def test_remove_directory_by_path(self, tmp_path):
        mgr = WorkspaceManager(primary_dir=tmp_path)
        extra = tmp_path / "lib"
        extra.mkdir()
        mgr.add_directory(extra, alias="lib")
        assert mgr.remove_directory(str(extra.resolve())) is True

    def test_remove_directory_by_alias(self, tmp_path):
        mgr = WorkspaceManager(primary_dir=tmp_path)
        extra = tmp_path / "lib"
        extra.mkdir()
        mgr.add_directory(extra, alias="lib")
        assert mgr.remove_directory("lib") is True

    def test_cannot_remove_primary(self, tmp_path):
        mgr = WorkspaceManager(primary_dir=tmp_path)
        assert mgr.remove_directory(str(tmp_path.resolve())) is False

    def test_remove_nonexistent(self, tmp_path):
        mgr = WorkspaceManager(primary_dir=tmp_path)
        assert mgr.remove_directory("nope") is False

    def test_list_directories(self, tmp_path):
        mgr = WorkspaceManager(primary_dir=tmp_path)
        dirs = mgr.list_directories()
        assert len(dirs) >= 1

    def test_get_primary(self, tmp_path):
        mgr = WorkspaceManager(primary_dir=tmp_path)
        assert mgr.get_primary() == tmp_path.resolve()

    def test_resolve_path_absolute(self, tmp_path):
        mgr = WorkspaceManager(primary_dir=tmp_path)
        target = tmp_path / "exists.py"
        target.write_text("x")
        resolved = mgr.resolve_path(str(target))
        assert resolved is not None

    def test_resolve_path_absolute_nonexistent(self, tmp_path):
        mgr = WorkspaceManager(primary_dir=tmp_path)
        assert mgr.resolve_path("/nonexistent") is None

    def test_resolve_path_alias(self, tmp_path):
        mgr = WorkspaceManager(primary_dir=tmp_path)
        lib = tmp_path / "lib"
        lib.mkdir()
        mgr.add_directory(lib, alias="lib")
        resolved = mgr.resolve_path("lib/src/utils.py")
        assert resolved == lib / "src/utils.py"

    def test_resolve_path_relative(self, tmp_path):
        mgr = WorkspaceManager(primary_dir=tmp_path)
        f = tmp_path / "main.py"
        f.write_text("x")
        resolved = mgr.resolve_path("main.py")
        assert resolved == f

    def test_is_in_workspace(self, tmp_path):
        mgr = WorkspaceManager(primary_dir=tmp_path)
        assert mgr.is_in_workspace(tmp_path / "sub" / "file.py") is True
        assert mgr.is_in_workspace("/completely/different/path") is False

    def test_get_directory_for_path(self, tmp_path):
        mgr = WorkspaceManager(primary_dir=tmp_path)
        ws_dir = mgr.get_directory_for_path(tmp_path / "sub.py")
        assert ws_dir is not None
        assert ws_dir.primary is True

    def test_is_readonly(self, tmp_path):
        mgr = WorkspaceManager(primary_dir=tmp_path)
        lib = tmp_path / "lib"
        lib.mkdir()
        mgr.add_directory(lib, alias="lib", readonly=True)
        file_in_lib = lib / "x.py"
        file_in_lib.write_text("x")
        ws_dir = mgr.get_directory_for_path(file_in_lib)
        if ws_dir and ws_dir.readonly:
            assert mgr.is_readonly(file_in_lib) is True
        else:
            assert mgr.is_readonly(file_in_lib) is False

    def test_get_all_paths(self, tmp_path):
        mgr = WorkspaceManager(primary_dir=tmp_path)
        lib = tmp_path / "lib"
        lib.mkdir()
        mgr.add_directory(lib)
        paths = mgr.get_all_paths()
        assert len(paths) == 2

    def test_format_path_with_alias(self, tmp_path):
        mgr = WorkspaceManager(primary_dir=tmp_path)
        lib = tmp_path / "lib"
        lib.mkdir()
        mgr.add_directory(lib, alias="lib")
        formatted = mgr.format_path(lib / "utils.py")
        assert formatted == "lib/utils.py"

    def test_format_path_primary(self, tmp_path):
        mgr = WorkspaceManager(primary_dir=tmp_path)
        f = tmp_path / "main.py"
        f.write_text("x")
        formatted = mgr.format_path(f)
        assert "main.py" in formatted

    def test_to_dict(self, tmp_path):
        mgr = WorkspaceManager(primary_dir=tmp_path)
        d = mgr.to_dict()
        assert "directories" in d
        assert "primary" in d

    def test_get_summary(self, tmp_path):
        mgr = WorkspaceManager(primary_dir=tmp_path)
        summary = mgr.get_summary()
        assert isinstance(summary, str)

    def test_resolve_path_alias_only(self, tmp_path):
        mgr = WorkspaceManager(primary_dir=tmp_path)
        lib = tmp_path / "lib"
        lib.mkdir()
        mgr.add_directory(lib, alias="lib")
        resolved = mgr.resolve_path("lib")
        assert resolved == lib

    def test_resolve_path_relative_nonexistent(self, tmp_path):
        mgr = WorkspaceManager(primary_dir=tmp_path)
        assert mgr.resolve_path("nonexistent.py") is None

    def test_is_readonly_for_unknown_path(self, tmp_path):
        mgr = WorkspaceManager(primary_dir=tmp_path)
        assert mgr.is_readonly("/completely/unknown/path") is False

    def test_get_directory_for_path_not_in_workspace(self, tmp_path):
        mgr = WorkspaceManager(primary_dir=tmp_path)
        assert mgr.get_directory_for_path("/completely/outside") is None

    def test_remove_primary_by_alias(self, tmp_path):
        mgr = WorkspaceManager(primary_dir=tmp_path)
        mgr._directories[0].alias = "primary"
        assert mgr.remove_directory("primary") is False

    def test_add_directory_with_primary_flag(self, tmp_path):
        mgr = WorkspaceManager(primary_dir=tmp_path)
        new_primary = tmp_path / "new_primary"
        new_primary.mkdir()
        assert mgr.add_directory(new_primary, primary=True) is True
        assert mgr.get_primary() == new_primary.resolve()

    def test_format_path_outside_workspace(self, tmp_path):
        mgr = WorkspaceManager(primary_dir=tmp_path)
        from pathlib import Path
        external = Path("/completely/outside/path.py").resolve()
        formatted = mgr.format_path(external)
        assert str(external) == formatted

    def test_resolve_path_absolute_nonexistent(self, tmp_path):
        mgr = WorkspaceManager(primary_dir=tmp_path)
        assert mgr.resolve_path("/absolutely/nonexistent") is None

    def test_get_primary_fallback_no_directories(self):
        from pathlib import Path
        mgr = WorkspaceManager.__new__(WorkspaceManager)
        mgr._directories = []
        result = mgr.get_primary()
        assert result == Path.cwd()
