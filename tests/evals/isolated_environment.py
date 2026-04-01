"""
Isolated Environment for Evaluation Trials

Provides isolated execution environments for evaluation trials to prevent
state pollution between trials. This is a key requirement from Anthropic's
evaluation best practices.

Features:
- Temporary directory management (auto-create and cleanup)
- Test fixture setup (copy template files)
- Git repository initialization (optional)
- Database initialization (optional, SQLite)
- Environment variable isolation
- Working directory switching
- State tracking and validation
"""

import hashlib
import os
import shutil
import sqlite3
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class FileSnapshot:
    """Snapshot of a file's state."""

    path: str
    exists: bool
    content_hash: str | None = None
    size: int | None = None
    mtime: float | None = None

    @classmethod
    def from_path(cls, path: Path) -> "FileSnapshot":
        """Create a snapshot from a file path."""
        if not path.exists():
            return cls(path=str(path), exists=False)

        content_hash = None
        if path.is_file():
            content_hash = hashlib.md5(path.read_bytes()).hexdigest()

        stat = path.stat()
        return cls(
            path=str(path),
            exists=True,
            content_hash=content_hash,
            size=stat.st_size,
            mtime=stat.st_mtime,
        )


@dataclass
class EnvironmentSnapshot:
    """Snapshot of the entire environment state."""

    timestamp: str
    root_dir: str
    files: dict[str, FileSnapshot] = field(default_factory=dict)
    env_vars: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert snapshot to dictionary."""
        return {
            "timestamp": self.timestamp,
            "root_dir": self.root_dir,
            "files": {
                path: {
                    "exists": snap.exists,
                    "content_hash": snap.content_hash,
                    "size": snap.size,
                    "mtime": snap.mtime,
                }
                for path, snap in self.files.items()
            },
            "env_vars": self.env_vars,
        }


@dataclass
class FileDiff:
    """Difference between two file states."""

    path: str
    change_type: str  # "created", "modified", "deleted", "unchanged"
    old_hash: str | None = None
    new_hash: str | None = None


@dataclass
class EnvironmentDiff:
    """Difference between two environment snapshots."""

    created_files: list[str] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)
    deleted_files: list[str] = field(default_factory=list)
    env_var_changes: dict[str, dict[str, str | None]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert diff to dictionary."""
        return {
            "created_files": self.created_files,
            "modified_files": self.modified_files,
            "deleted_files": self.deleted_files,
            "env_var_changes": self.env_var_changes,
        }


class IsolatedEnvironment:
    """
    Isolated environment for evaluation trials.

    Ensures each evaluation trial runs in complete isolation, preventing
    state pollution between trials. This is critical for reproducible
    and reliable evaluation results.

    Usage:
        with IsolatedEnvironment(fixtures=["python_project"]) as env:
            # Run evaluation trial
            env.create_file("test.py", "print('hello')")
            result = run_agent_task(env.root_dir)

            # Verify state changes
            created = env.get_created_files()
            modified = env.get_modified_files()

    Attributes:
        root_dir: Path to the isolated temporary directory
        original_cwd: Original working directory before entering context
        original_env: Original environment variables
    """

    # Default fixtures directory relative to this file
    FIXTURES_DIR = Path(__file__).parent / "fixtures"

    def __init__(
        self,
        base_dir: str | None = None,
        fixtures: list[str] | None = None,
        cleanup: bool = True,
        init_git: bool = False,
        init_db: bool = False,
        db_schema: str | None = None,
        env_vars: dict[str, str] | None = None,
        change_cwd: bool = True,
    ):
        """
        Initialize the isolated environment.

        Args:
            base_dir: Base directory for creating temp directory. If None,
                     uses system temp directory.
            fixtures: List of fixture names to load (e.g., ["python_project"])
            cleanup: Whether to cleanup temp directory on exit (default: True)
            init_git: Whether to initialize a git repository (default: False)
            init_db: Whether to initialize a SQLite database (default: False)
            db_schema: SQL schema to execute if init_db is True
            env_vars: Environment variables to set in the isolated environment
            change_cwd: Whether to change working directory to root_dir (default: True)
        """
        self.base_dir = Path(base_dir) if base_dir else None
        self.fixtures = fixtures or []
        self.cleanup = cleanup
        self.init_git = init_git
        self.init_db = init_db
        self.db_schema = db_schema
        self.env_vars = env_vars or {}
        self.change_cwd = change_cwd

        # State tracking
        self._temp_dir: tempfile.TemporaryDirectory | None = None
        self._root_dir: Path | None = None
        self._original_cwd: Path | None = None
        self._original_env: dict[str, str | None] = {}
        self._initial_snapshot: EnvironmentSnapshot | None = None
        self._db_connection: sqlite3.Connection | None = None
        self._created_files: set[str] = set()
        self._is_active: bool = False

    @property
    def root_dir(self) -> Path:
        """Get the root directory of the isolated environment."""
        if self._root_dir is None:
            raise RuntimeError("Environment not initialized. Use 'with' statement.")
        return self._root_dir

    @property
    def original_cwd(self) -> Path | None:
        """Get the original working directory."""
        return self._original_cwd

    @property
    def db_path(self) -> Path | None:
        """Get the database path if initialized."""
        if self.init_db and self._root_dir:
            return self._root_dir / "test.db"
        return None

    @property
    def is_active(self) -> bool:
        """Check if the environment is currently active."""
        return self._is_active

    def __enter__(self) -> "IsolatedEnvironment":
        """Enter the isolated environment context."""
        self._setup_environment()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the isolated environment context."""
        self._teardown_environment()
        return None

    def _setup_environment(self) -> None:
        """Set up the isolated environment."""
        # Create temporary directory
        self._temp_dir = tempfile.TemporaryDirectory(
            prefix="eval_isolated_",
            dir=str(self.base_dir) if self.base_dir else None,
        )
        self._root_dir = Path(self._temp_dir.name)

        # Save original working directory
        self._original_cwd = Path.cwd()

        # Set up environment variables
        self._setup_env_vars()

        # Load fixtures
        for fixture_name in self.fixtures:
            self.load_fixture(fixture_name)

        # Initialize git repository if requested
        if self.init_git:
            self._init_git_repo()

        # Initialize database if requested
        if self.init_db:
            self._init_database()

        # Change working directory if requested
        if self.change_cwd:
            os.chdir(self._root_dir)

        # Take initial snapshot
        self._initial_snapshot = self.snapshot()
        self._is_active = True

    def _teardown_environment(self) -> None:
        """Tear down the isolated environment."""
        self._is_active = False

        # Close database connection
        if self._db_connection:
            self._db_connection.close()
            self._db_connection = None

        # Restore working directory
        if self._original_cwd and self.change_cwd:
            os.chdir(self._original_cwd)

        # Restore environment variables
        self._restore_env_vars()

        # Cleanup temporary directory
        if self.cleanup and self._temp_dir:
            self._temp_dir.cleanup()
            self._temp_dir = None
            self._root_dir = None

    def _setup_env_vars(self) -> None:
        """Set up isolated environment variables."""
        # Save original values and set new ones
        for key, value in self.env_vars.items():
            self._original_env[key] = os.environ.get(key)
            os.environ[key] = value

        # Set some default isolation variables
        isolation_vars = {
            "ISOLATED_ENV": "1",
            "ISOLATED_ENV_ROOT": str(self._root_dir),
        }
        for key, value in isolation_vars.items():
            if key not in self._original_env:
                self._original_env[key] = os.environ.get(key)
            os.environ[key] = value

    def _restore_env_vars(self) -> None:
        """Restore original environment variables."""
        for key, original_value in self._original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value
        self._original_env.clear()

    def _init_git_repo(self) -> None:
        """Initialize a git repository in the root directory."""
        subprocess.run(
            ["git", "init"],
            cwd=self._root_dir,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=self._root_dir,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=self._root_dir,
            capture_output=True,
            check=True,
        )

        # Create initial commit if there are files (excluding .git directory)
        has_files = any(item for item in self._root_dir.iterdir() if item.name != ".git")
        if has_files:
            subprocess.run(
                ["git", "add", "-A"],
                cwd=self._root_dir,
                capture_output=True,
                check=True,
            )
            # Check if there's anything staged before committing
            result = subprocess.run(
                ["git", "diff", "--cached", "--quiet"],
                cwd=self._root_dir,
                capture_output=True,
            )
            # Non-zero exit means there are staged changes
            if result.returncode != 0:
                subprocess.run(
                    ["git", "commit", "-m", "Initial commit"],
                    cwd=self._root_dir,
                    capture_output=True,
                    check=True,
                )

    def _init_database(self) -> None:
        """Initialize a SQLite database."""
        db_path = self._root_dir / "test.db"
        self._db_connection = sqlite3.connect(str(db_path))

        if self.db_schema:
            self._db_connection.executescript(self.db_schema)
            self._db_connection.commit()

    # ==================== Fixture Management ====================

    def load_fixture(self, name: str) -> Path:
        """
        Load a predefined fixture into the environment.

        Args:
            name: Name of the fixture (e.g., "python_project", "git_repo")

        Returns:
            Path to the loaded fixture directory

        Raises:
            FileNotFoundError: If fixture does not exist
        """
        fixture_path = self.FIXTURES_DIR / name

        if not fixture_path.exists():
            raise FileNotFoundError(f"Fixture not found: {name} (looked in {fixture_path})")

        # Copy fixture contents to root directory
        if fixture_path.is_dir():
            for item in fixture_path.iterdir():
                dest = self._root_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)
        else:
            shutil.copy2(fixture_path, self._root_dir / fixture_path.name)

        return self._root_dir

    def create_file(self, path: str | Path, content: str = "") -> Path:
        """
        Create a test file in the isolated environment.

        Args:
            path: Relative path within the environment
            content: File content (default: empty string)

        Returns:
            Absolute path to the created file
        """
        file_path = self._root_dir / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        self._created_files.add(str(file_path.relative_to(self._root_dir)))
        return file_path

    def create_dir(self, path: str | Path) -> Path:
        """
        Create a directory in the isolated environment.

        Args:
            path: Relative path within the environment

        Returns:
            Absolute path to the created directory
        """
        directory_path = self._root_dir / path
        directory_path.mkdir(parents=True, exist_ok=True)
        return directory_path

    def copy_from_template(self, template_name: str, dest: str | None = None) -> Path:
        """
        Copy files from a template directory.

        Args:
            template_name: Name of the template in fixtures directory
            dest: Destination path within environment (default: root)

        Returns:
            Path to the destination directory
        """
        template_path = self.FIXTURES_DIR / template_name

        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_name}")

        dest_path = self._root_dir / dest if dest else self._root_dir
        dest_path.mkdir(parents=True, exist_ok=True)

        if template_path.is_dir():
            for item in template_path.iterdir():
                item_dest = dest_path / item.name
                if item.is_dir():
                    shutil.copytree(item, item_dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, item_dest)
        else:
            shutil.copy2(template_path, dest_path / template_path.name)

        return dest_path

    def read_path(self, path: str | Path) -> str:
        """
        Read a file from the isolated environment.

        Args:
            path: Relative path within the environment

        Returns:
            File content as string
        """
        file_path = self._root_dir / path
        return file_path.read_text()

    def file_exists(self, path: str | Path) -> bool:
        """
        Check if a file exists in the isolated environment.

        Args:
            path: Relative path within the environment

        Returns:
            True if file exists, False otherwise
        """
        return (self._root_dir / path).exists()

    # ==================== State Validation ====================

    def snapshot(self) -> EnvironmentSnapshot:
        """
        Create a snapshot of the current environment state.

        Returns:
            EnvironmentSnapshot containing file states and env vars
        """
        files = {}
        for path in self._root_dir.rglob("*"):
            if path.is_file():
                rel_path = str(path.relative_to(self._root_dir))
                # Skip git internal files
                if not rel_path.startswith(".git/"):
                    files[rel_path] = FileSnapshot.from_path(path)

        return EnvironmentSnapshot(
            timestamp=datetime.now().isoformat(),
            root_dir=str(self._root_dir),
            files=files,
            env_vars=dict(os.environ),
        )

    def diff(self, snapshot: EnvironmentSnapshot) -> EnvironmentDiff:
        """
        Compare current state with a previous snapshot.

        Args:
            snapshot: Previous environment snapshot to compare against

        Returns:
            EnvironmentDiff containing all changes
        """
        current = self.snapshot()

        created = []
        modified = []
        deleted = []

        # Find created and modified files
        for path, current_snap in current.files.items():
            if path not in snapshot.files:
                created.append(path)
            elif current_snap.content_hash != snapshot.files[path].content_hash:
                modified.append(path)

        # Find deleted files
        for path in snapshot.files:
            if path not in current.files:
                deleted.append(path)

        # Find env var changes
        env_changes = {}
        all_keys = set(snapshot.env_vars.keys()) | set(current.env_vars.keys())
        for key in all_keys:
            old_val = snapshot.env_vars.get(key)
            new_val = current.env_vars.get(key)
            if old_val != new_val:
                env_changes[key] = {"old": old_val, "new": new_val}

        return EnvironmentDiff(
            created_files=created,
            modified_files=modified,
            deleted_files=deleted,
            env_var_changes=env_changes,
        )

    def get_created_files(self) -> list[str]:
        """
        Get list of files created since environment initialization.

        Returns:
            List of relative file paths
        """
        if not self._initial_snapshot:
            return []

        diff = self.diff(self._initial_snapshot)
        return diff.created_files

    def get_modified_files(self) -> list[str]:
        """
        Get list of files modified since environment initialization.

        Returns:
            List of relative file paths
        """
        if not self._initial_snapshot:
            return []

        diff = self.diff(self._initial_snapshot)
        return diff.modified_files

    def get_deleted_files(self) -> list[str]:
        """
        Get list of files deleted since environment initialization.

        Returns:
            List of relative file paths
        """
        if not self._initial_snapshot:
            return []

        diff = self.diff(self._initial_snapshot)
        return diff.deleted_files

    def get_all_files(self) -> list[str]:
        """
        Get list of all files in the environment.

        Returns:
            List of relative file paths
        """
        files = []
        for path in self._root_dir.rglob("*"):
            if path.is_file():
                rel_path = str(path.relative_to(self._root_dir))
                if not rel_path.startswith(".git/"):
                    files.append(rel_path)
        return sorted(files)

    # ==================== Database Operations ====================

    def execute_sql(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """
        Execute SQL on the isolated database.

        Args:
            sql: SQL statement to execute
            params: Parameters for the SQL statement

        Returns:
            Cursor with results

        Raises:
            RuntimeError: If database not initialized
        """
        if not self._db_connection:
            raise RuntimeError("Database not initialized. Set init_db=True.")

        cursor = self._db_connection.execute(sql, params)
        self._db_connection.commit()
        return cursor

    def query_sql(self, sql: str, params: tuple = ()) -> list[tuple]:
        """
        Execute a query and return all results.

        Args:
            sql: SQL query to execute
            params: Parameters for the query

        Returns:
            List of result tuples
        """
        cursor = self.execute_sql(sql, params)
        return cursor.fetchall()

    # ==================== Git Operations ====================

    def git_status(self) -> str:
        """
        Get git status of the environment.

        Returns:
            Git status output

        Raises:
            RuntimeError: If git not initialized
        """
        if not self.init_git:
            raise RuntimeError("Git not initialized. Set init_git=True.")

        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=self._root_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout

    def git_commit(self, message: str) -> None:
        """
        Create a git commit with all changes.

        Args:
            message: Commit message
        """
        if not self.init_git:
            raise RuntimeError("Git not initialized. Set init_git=True.")

        subprocess.run(
            ["git", "add", "-A"],
            cwd=self._root_dir,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=self._root_dir,
            capture_output=True,
            check=True,
        )

    def git_diff(self) -> str:
        """
        Get git diff of uncommitted changes.

        Returns:
            Git diff output
        """
        if not self.init_git:
            raise RuntimeError("Git not initialized. Set init_git=True.")

        result = subprocess.run(
            ["git", "diff"],
            cwd=self._root_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout

    # ==================== Utility Methods ====================

    def run_command(
        self,
        command: list[str],
        capture_output: bool = True,
        check: bool = True,
        **kwargs,
    ) -> subprocess.CompletedProcess:
        """
        Run a command in the isolated environment.

        Args:
            command: Command and arguments as list
            capture_output: Whether to capture stdout/stderr
            check: Whether to raise on non-zero exit
            **kwargs: Additional arguments for subprocess.run

        Returns:
            CompletedProcess instance
        """
        return subprocess.run(
            command,
            cwd=self._root_dir,
            capture_output=capture_output,
            check=check,
            **kwargs,
        )

    def get_path(self, relative_path: str | Path) -> Path:
        """
        Get absolute path for a relative path in the environment.

        Args:
            relative_path: Path relative to environment root

        Returns:
            Absolute path
        """
        return self._root_dir / relative_path

    def cleanup_now(self) -> None:
        """
        Manually cleanup the environment before context exit.

        Useful for freeing resources early.
        """
        if self._temp_dir:
            self._teardown_environment()


@contextmanager
def isolated_trial(
    fixtures: list[str] | None = None,
    init_git: bool = False,
    init_db: bool = False,
    **kwargs,
):
    """
    Convenience context manager for isolated evaluation trials.

    Args:
        fixtures: List of fixture names to load
        init_git: Whether to initialize git
        init_db: Whether to initialize database
        **kwargs: Additional arguments for IsolatedEnvironment

    Yields:
        IsolatedEnvironment instance

    Example:
        with isolated_trial(fixtures=["python_project"], init_git=True) as env:
            # Run evaluation
            pass
    """
    with IsolatedEnvironment(
        fixtures=fixtures,
        init_git=init_git,
        init_db=init_db,
        **kwargs,
    ) as env:
        yield env


def create_fixture_template(
    name: str,
    files: dict[str, str],
    base_dir: Path | None = None,
) -> Path:
    """
    Create a new fixture template.

    Args:
        name: Name of the fixture
        files: Dictionary of {relative_path: content}
        base_dir: Base directory for fixtures (default: FIXTURES_DIR)

    Returns:
        Path to the created fixture directory
    """
    fixtures_dir = base_dir or IsolatedEnvironment.FIXTURES_DIR
    fixture_path = fixtures_dir / name
    fixture_path.mkdir(parents=True, exist_ok=True)

    for rel_path, content in files.items():
        file_path = fixture_path / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)

    return fixture_path
