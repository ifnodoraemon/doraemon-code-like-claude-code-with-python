"""
Unit tests for IsolatedEnvironment class.

Tests the isolated environment functionality for evaluation trials.
"""

import os
import sqlite3
import subprocess
import tempfile
from pathlib import Path

import pytest

from tests.evals.isolated_environment import (
    IsolatedEnvironment,
    EnvironmentSnapshot,
    EnvironmentDiff,
    FileSnapshot,
    FileDiff,
    isolated_trial,
    create_fixture_template,
)


class TestIsolatedEnvironmentBasic:
    """Basic tests for IsolatedEnvironment."""

    def test_context_manager_creates_temp_dir(self):
        """Test that entering context creates a temporary directory."""
        with IsolatedEnvironment() as env:
            assert env.root_dir.exists()
            assert env.root_dir.is_dir()
            assert env.is_active

    def test_context_manager_cleans_up(self):
        """Test that exiting context cleans up the temporary directory."""
        with IsolatedEnvironment() as env:
            temp_path = env.root_dir
            assert temp_path.exists()

        # After context exit, directory should be cleaned up
        assert not temp_path.exists()

    def test_cleanup_disabled(self):
        """Test that cleanup can be disabled."""
        with IsolatedEnvironment(cleanup=False) as env:
            temp_path = env.root_dir

        # Directory should still exist
        assert temp_path.exists()

        # Manual cleanup
        import shutil
        shutil.rmtree(temp_path)

    def test_base_dir_option(self):
        """Test that base_dir option works."""
        with tempfile.TemporaryDirectory() as base:
            with IsolatedEnvironment(base_dir=base) as env:
                assert str(env.root_dir).startswith(base)

    def test_is_active_property(self):
        """Test is_active property."""
        env = IsolatedEnvironment()
        assert not env.is_active

        with env:
            assert env.is_active

        assert not env.is_active

    def test_root_dir_raises_outside_context(self):
        """Test that accessing root_dir outside context raises error."""
        env = IsolatedEnvironment()
        with pytest.raises(RuntimeError, match="Environment not initialized"):
            _ = env.root_dir


class TestFileOperations:
    """Tests for file operations in IsolatedEnvironment."""

    def test_create_file(self):
        """Test creating a file."""
        with IsolatedEnvironment() as env:
            file_path = env.create_file("test.txt", "Hello, World!")

            assert file_path.exists()
            assert file_path.read_text() == "Hello, World!"

    def test_create_file_in_subdirectory(self):
        """Test creating a file in a subdirectory."""
        with IsolatedEnvironment() as env:
            file_path = env.create_file("subdir/nested/test.txt", "content")

            assert file_path.exists()
            assert file_path.parent.name == "nested"

    def test_create_directory(self):
        """Test creating a directory."""
        with IsolatedEnvironment() as env:
            dir_path = env.create_directory("my_dir/nested")

            assert dir_path.exists()
            assert dir_path.is_dir()

    def test_read_file(self):
        """Test reading a file."""
        with IsolatedEnvironment() as env:
            env.create_file("test.txt", "Test content")
            content = env.read_file("test.txt")

            assert content == "Test content"

    def test_file_exists(self):
        """Test file_exists method."""
        with IsolatedEnvironment() as env:
            assert not env.file_exists("test.txt")

            env.create_file("test.txt", "content")
            assert env.file_exists("test.txt")

    def test_get_path(self):
        """Test get_path method."""
        with IsolatedEnvironment() as env:
            path = env.get_path("subdir/file.txt")

            assert path == env.root_dir / "subdir" / "file.txt"

    def test_get_all_files(self):
        """Test get_all_files method."""
        with IsolatedEnvironment() as env:
            env.create_file("file1.txt", "content1")
            env.create_file("dir/file2.txt", "content2")
            env.create_file("dir/subdir/file3.txt", "content3")

            files = env.get_all_files()

            assert len(files) == 3
            assert "file1.txt" in files
            assert "dir/file2.txt" in files
            assert "dir/subdir/file3.txt" in files


class TestFixtureManagement:
    """Tests for fixture management."""

    def test_load_fixture_python_project(self):
        """Test loading python_project fixture."""
        with IsolatedEnvironment(fixtures=["python_project"]) as env:
            assert env.file_exists("pyproject.toml")
            assert env.file_exists("src/main.py")
            assert env.file_exists("src/utils.py")
            assert env.file_exists("tests/test_main.py")

    def test_load_fixture_git_repo(self):
        """Test loading git_repo fixture."""
        with IsolatedEnvironment(fixtures=["git_repo"]) as env:
            assert env.file_exists("README.md")
            assert env.file_exists(".gitignore")
            assert env.file_exists("src/app.py")
            assert env.file_exists("config/settings.yaml")

    def test_load_fixture_api_project(self):
        """Test loading api_project fixture."""
        with IsolatedEnvironment(fixtures=["api_project"]) as env:
            assert env.file_exists("pyproject.toml")
            assert env.file_exists("app/main.py")
            assert env.file_exists("app/routes.py")
            assert env.file_exists("tests/test_routes.py")

    def test_load_multiple_fixtures(self):
        """Test loading multiple fixtures."""
        with IsolatedEnvironment(fixtures=["python_project", "git_repo"]) as env:
            # Files from python_project
            assert env.file_exists("pyproject.toml")
            assert env.file_exists("src/main.py")

            # Files from git_repo (may overwrite some)
            assert env.file_exists(".gitignore")
            assert env.file_exists("config/settings.yaml")

    def test_load_nonexistent_fixture(self):
        """Test loading a non-existent fixture raises error."""
        with pytest.raises(FileNotFoundError, match="Fixture not found"):
            with IsolatedEnvironment(fixtures=["nonexistent_fixture"]) as env:
                pass

    def test_copy_from_template(self):
        """Test copy_from_template method."""
        with IsolatedEnvironment() as env:
            env.copy_from_template("python_project", "my_project")

            assert env.file_exists("my_project/pyproject.toml")
            assert env.file_exists("my_project/src/main.py")


class TestStateValidation:
    """Tests for state validation and snapshots."""

    def test_snapshot_captures_files(self):
        """Test that snapshot captures file state."""
        with IsolatedEnvironment() as env:
            env.create_file("test.txt", "content")
            snapshot = env.snapshot()

            assert "test.txt" in snapshot.files
            assert snapshot.files["test.txt"].exists
            assert snapshot.files["test.txt"].content_hash is not None

    def test_snapshot_to_dict(self):
        """Test snapshot to_dict method."""
        with IsolatedEnvironment() as env:
            env.create_file("test.txt", "content")
            snapshot = env.snapshot()
            data = snapshot.to_dict()

            assert "timestamp" in data
            assert "root_dir" in data
            assert "files" in data
            assert "test.txt" in data["files"]

    def test_diff_detects_created_files(self):
        """Test that diff detects created files."""
        with IsolatedEnvironment() as env:
            initial = env.snapshot()

            env.create_file("new_file.txt", "content")

            diff = env.diff(initial)
            assert "new_file.txt" in diff.created_files

    def test_diff_detects_modified_files(self):
        """Test that diff detects modified files."""
        with IsolatedEnvironment() as env:
            env.create_file("test.txt", "original")
            initial = env.snapshot()

            # Modify the file
            (env.root_dir / "test.txt").write_text("modified")

            diff = env.diff(initial)
            assert "test.txt" in diff.modified_files

    def test_diff_detects_deleted_files(self):
        """Test that diff detects deleted files."""
        with IsolatedEnvironment() as env:
            env.create_file("test.txt", "content")
            initial = env.snapshot()

            # Delete the file
            (env.root_dir / "test.txt").unlink()

            diff = env.diff(initial)
            assert "test.txt" in diff.deleted_files

    def test_get_created_files(self):
        """Test get_created_files method."""
        with IsolatedEnvironment() as env:
            env.create_file("new_file.txt", "content")

            created = env.get_created_files()
            assert "new_file.txt" in created

    def test_get_modified_files(self):
        """Test get_modified_files method."""
        with IsolatedEnvironment(fixtures=["python_project"]) as env:
            # Modify an existing file
            (env.root_dir / "pyproject.toml").write_text("modified content")

            modified = env.get_modified_files()
            assert "pyproject.toml" in modified

    def test_get_deleted_files(self):
        """Test get_deleted_files method."""
        with IsolatedEnvironment(fixtures=["python_project"]) as env:
            # Delete a file
            (env.root_dir / "pyproject.toml").unlink()

            deleted = env.get_deleted_files()
            assert "pyproject.toml" in deleted

    def test_diff_to_dict(self):
        """Test EnvironmentDiff to_dict method."""
        diff = EnvironmentDiff(
            created_files=["new.txt"],
            modified_files=["changed.txt"],
            deleted_files=["removed.txt"],
        )
        data = diff.to_dict()

        assert data["created_files"] == ["new.txt"]
        assert data["modified_files"] == ["changed.txt"]
        assert data["deleted_files"] == ["removed.txt"]


class TestEnvironmentVariables:
    """Tests for environment variable isolation."""

    def test_env_vars_are_set(self):
        """Test that custom env vars are set."""
        with IsolatedEnvironment(env_vars={"MY_VAR": "my_value"}) as env:
            assert os.environ.get("MY_VAR") == "my_value"

    def test_env_vars_are_restored(self):
        """Test that env vars are restored after context exit."""
        original = os.environ.get("MY_VAR")

        with IsolatedEnvironment(env_vars={"MY_VAR": "temp_value"}) as env:
            assert os.environ.get("MY_VAR") == "temp_value"

        assert os.environ.get("MY_VAR") == original

    def test_isolation_env_vars_are_set(self):
        """Test that isolation env vars are automatically set."""
        with IsolatedEnvironment() as env:
            assert os.environ.get("ISOLATED_ENV") == "1"
            assert os.environ.get("ISOLATED_ENV_ROOT") == str(env.root_dir)

    def test_isolation_env_vars_are_restored(self):
        """Test that isolation env vars are restored."""
        original_isolated = os.environ.get("ISOLATED_ENV")
        original_root = os.environ.get("ISOLATED_ENV_ROOT")

        with IsolatedEnvironment() as env:
            pass

        assert os.environ.get("ISOLATED_ENV") == original_isolated
        assert os.environ.get("ISOLATED_ENV_ROOT") == original_root


class TestWorkingDirectory:
    """Tests for working directory management."""

    def test_cwd_changes_to_root_dir(self):
        """Test that cwd changes to root_dir by default."""
        original_cwd = Path.cwd()

        with IsolatedEnvironment() as env:
            assert Path.cwd() == env.root_dir

        assert Path.cwd() == original_cwd

    def test_cwd_not_changed_when_disabled(self):
        """Test that cwd is not changed when change_cwd=False."""
        original_cwd = Path.cwd()

        with IsolatedEnvironment(change_cwd=False) as env:
            assert Path.cwd() == original_cwd

    def test_original_cwd_property(self):
        """Test original_cwd property."""
        original_cwd = Path.cwd()

        with IsolatedEnvironment() as env:
            assert env.original_cwd == original_cwd


class TestGitIntegration:
    """Tests for git integration."""

    def test_git_init(self):
        """Test git repository initialization."""
        with IsolatedEnvironment(init_git=True) as env:
            git_dir = env.root_dir / ".git"
            assert git_dir.exists()

    def test_git_status(self):
        """Test git_status method."""
        with IsolatedEnvironment(init_git=True) as env:
            env.create_file("new_file.txt", "content")
            status = env.git_status()

            assert "new_file.txt" in status

    def test_git_commit(self):
        """Test git_commit method."""
        with IsolatedEnvironment(init_git=True) as env:
            env.create_file("test.txt", "content")
            env.git_commit("Add test file")

            # Status should be clean after commit
            status = env.git_status()
            assert status.strip() == ""

    def test_git_diff(self):
        """Test git_diff method."""
        with IsolatedEnvironment(init_git=True) as env:
            env.create_file("test.txt", "original")
            env.git_commit("Initial commit")

            # Modify the file
            (env.root_dir / "test.txt").write_text("modified")

            diff = env.git_diff()
            assert "original" in diff or "modified" in diff

    def test_git_not_initialized_raises(self):
        """Test that git methods raise when git not initialized."""
        with IsolatedEnvironment(init_git=False) as env:
            with pytest.raises(RuntimeError, match="Git not initialized"):
                env.git_status()

    def test_git_with_fixtures(self):
        """Test git initialization with fixtures creates initial commit."""
        with IsolatedEnvironment(fixtures=["python_project"], init_git=True) as env:
            # Should have an initial commit
            result = subprocess.run(
                ["git", "log", "--oneline"],
                cwd=env.root_dir,
                capture_output=True,
                text=True,
            )
            assert "Initial commit" in result.stdout


class TestDatabaseIntegration:
    """Tests for database integration."""

    def test_db_init(self):
        """Test database initialization."""
        with IsolatedEnvironment(init_db=True) as env:
            assert env.db_path.exists()

    def test_db_path_property(self):
        """Test db_path property."""
        with IsolatedEnvironment(init_db=True) as env:
            assert env.db_path == env.root_dir / "test.db"

    def test_db_path_none_when_not_initialized(self):
        """Test db_path is None when db not initialized."""
        with IsolatedEnvironment(init_db=False) as env:
            assert env.db_path is None

    def test_execute_sql(self):
        """Test execute_sql method."""
        schema = """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        );
        """
        with IsolatedEnvironment(init_db=True, db_schema=schema) as env:
            env.execute_sql("INSERT INTO users (name) VALUES (?)", ("Alice",))
            result = env.query_sql("SELECT name FROM users")

            assert result == [("Alice",)]

    def test_query_sql(self):
        """Test query_sql method."""
        schema = """
        CREATE TABLE items (
            id INTEGER PRIMARY KEY,
            name TEXT
        );
        INSERT INTO items (name) VALUES ('Item 1');
        INSERT INTO items (name) VALUES ('Item 2');
        """
        with IsolatedEnvironment(init_db=True, db_schema=schema) as env:
            results = env.query_sql("SELECT name FROM items ORDER BY id")

            assert len(results) == 2
            assert results[0] == ("Item 1",)
            assert results[1] == ("Item 2",)

    def test_db_not_initialized_raises(self):
        """Test that db methods raise when db not initialized."""
        with IsolatedEnvironment(init_db=False) as env:
            with pytest.raises(RuntimeError, match="Database not initialized"):
                env.execute_sql("SELECT 1")


class TestCommandExecution:
    """Tests for command execution."""

    def test_run_command(self):
        """Test run_command method."""
        with IsolatedEnvironment() as env:
            result = env.run_command(["echo", "hello"])

            assert result.returncode == 0
            assert "hello" in result.stdout.decode()

    def test_run_command_in_root_dir(self):
        """Test that commands run in root_dir."""
        with IsolatedEnvironment() as env:
            env.create_file("test.txt", "content")
            result = env.run_command(["ls"])

            assert "test.txt" in result.stdout.decode()


class TestIsolatedTrialContextManager:
    """Tests for isolated_trial convenience function."""

    def test_isolated_trial_basic(self):
        """Test basic isolated_trial usage."""
        with isolated_trial() as env:
            assert env.is_active
            assert env.root_dir.exists()

    def test_isolated_trial_with_fixtures(self):
        """Test isolated_trial with fixtures."""
        with isolated_trial(fixtures=["python_project"]) as env:
            assert env.file_exists("pyproject.toml")

    def test_isolated_trial_with_git(self):
        """Test isolated_trial with git."""
        with isolated_trial(init_git=True) as env:
            assert (env.root_dir / ".git").exists()

    def test_isolated_trial_with_db(self):
        """Test isolated_trial with database."""
        with isolated_trial(init_db=True) as env:
            assert env.db_path.exists()


class TestCreateFixtureTemplate:
    """Tests for create_fixture_template function."""

    def test_create_fixture_template(self):
        """Test creating a fixture template."""
        with tempfile.TemporaryDirectory() as base:
            base_path = Path(base)
            fixture_path = create_fixture_template(
                "test_fixture",
                {
                    "file1.txt": "content1",
                    "dir/file2.txt": "content2",
                },
                base_dir=base_path,
            )

            assert fixture_path.exists()
            assert (fixture_path / "file1.txt").read_text() == "content1"
            assert (fixture_path / "dir" / "file2.txt").read_text() == "content2"


class TestFileSnapshot:
    """Tests for FileSnapshot class."""

    def test_from_path_existing_file(self):
        """Test creating snapshot from existing file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            f.flush()

            snapshot = FileSnapshot.from_path(Path(f.name))

            assert snapshot.exists
            assert snapshot.content_hash is not None
            assert snapshot.size > 0

            Path(f.name).unlink()

    def test_from_path_nonexistent_file(self):
        """Test creating snapshot from non-existent file."""
        snapshot = FileSnapshot.from_path(Path("/nonexistent/file.txt"))

        assert not snapshot.exists
        assert snapshot.content_hash is None


class TestCleanupNow:
    """Tests for manual cleanup."""

    def test_cleanup_now(self):
        """Test cleanup_now method."""
        env = IsolatedEnvironment()
        env._setup_environment()
        temp_path = env.root_dir

        assert temp_path.exists()

        env.cleanup_now()

        assert not temp_path.exists()
        assert not env.is_active


class TestTrialIsolation:
    """Integration tests for trial isolation."""

    def test_trials_are_isolated(self):
        """Test that multiple trials are isolated from each other."""
        # First trial
        with IsolatedEnvironment() as env1:
            env1.create_file("trial1.txt", "trial 1 content")
            path1 = env1.root_dir

        # Second trial
        with IsolatedEnvironment() as env2:
            # Should not see files from first trial
            assert not env2.file_exists("trial1.txt")
            env2.create_file("trial2.txt", "trial 2 content")
            path2 = env2.root_dir

        # Paths should be different
        assert path1 != path2

    def test_env_vars_isolated_between_trials(self):
        """Test that env vars are isolated between trials."""
        with IsolatedEnvironment(env_vars={"TRIAL_VAR": "trial1"}) as env1:
            assert os.environ.get("TRIAL_VAR") == "trial1"

        with IsolatedEnvironment(env_vars={"TRIAL_VAR": "trial2"}) as env2:
            assert os.environ.get("TRIAL_VAR") == "trial2"

        # After both trials, var should be restored
        assert os.environ.get("TRIAL_VAR") is None

    def test_cwd_isolated_between_trials(self):
        """Test that cwd is isolated between trials."""
        original_cwd = Path.cwd()

        with IsolatedEnvironment() as env1:
            cwd1 = Path.cwd()

        with IsolatedEnvironment() as env2:
            cwd2 = Path.cwd()

        # Each trial should have different cwd
        assert cwd1 != cwd2

        # After trials, cwd should be restored
        assert Path.cwd() == original_cwd


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
