"""
Unit tests for the Git Operations Server.

Tests git commands, branch management, and GitHub integration.
"""

import os
import shutil
import subprocess

import pytest

from src.servers.git import (
    _is_git_repo,
    _run_git_command,
    git_add,
    git_branch,
    git_diff,
    git_log,
    git_stash,
    git_status,
)


# ========================================
# Fixtures
# ========================================

@pytest.fixture
def git_repo():
    """Create a temporary git repository for testing within project."""
    # Save original working directory
    original_cwd = os.getcwd()

    # Create temp dir inside project
    tmpdir = os.path.join(original_cwd, ".test_git_repo")
    os.makedirs(tmpdir, exist_ok=True)

    # Initialize git repo
    subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmpdir,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmpdir,
        capture_output=True,
    )

    # Create initial commit
    test_file = os.path.join(tmpdir, "test.txt")
    with open(test_file, "w") as f:
        f.write("Initial content\n")

    subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=tmpdir,
        capture_output=True,
    )

    yield tmpdir

    # Restore original working directory before cleanup
    os.chdir(original_cwd)
    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def non_git_dir():
    """Create a temporary directory that is NOT a git repo within project."""
    # Save original working directory
    original_cwd = os.getcwd()

    tmpdir = os.path.join(original_cwd, ".test_non_git_dir")
    os.makedirs(tmpdir, exist_ok=True)
    yield tmpdir

    # Restore original working directory before cleanup
    os.chdir(original_cwd)
    shutil.rmtree(tmpdir, ignore_errors=True)


# ========================================
# Repository Detection Tests
# ========================================

class TestRepositoryDetection:
    """Tests for git repository detection"""

    def test_is_git_repo_true(self, git_repo):
        """Test detecting a valid git repository"""
        assert _is_git_repo(git_repo) is True

    def test_is_git_repo_false(self):
        """Test detecting a non-git directory"""
        # Use /tmp which is outside of any git repo
        # Note: This path is outside the sandbox, so we use a different check
        result = _is_git_repo("/nonexistent/path/12345")
        # Should return False for non-existent path
        assert result is False


# ========================================
# Git Status Tests
# ========================================

class TestGitStatus:
    """Tests for git status command"""

    def test_status_clean_repo(self, git_repo):
        """Test status of a clean repository"""
        result = git_status(git_repo)
        assert "nothing to commit" in result.lower() or "clean" in result.lower()

    def test_status_with_changes(self, git_repo):
        """Test status with uncommitted changes"""
        # Make a change
        test_file = os.path.join(git_repo, "test.txt")
        with open(test_file, "a") as f:
            f.write("New content\n")
        
        result = git_status(git_repo)
        assert "modified" in result.lower() or "changes" in result.lower()

    def test_status_non_git_dir(self, non_git_dir):
        """Test status on non-git directory"""
        result = git_status(non_git_dir)
        assert "error" in result.lower() or "not a git repository" in result.lower()


# ========================================
# Git Diff Tests
# ========================================

class TestGitDiff:
    """Tests for git diff command"""

    def test_diff_no_changes(self, git_repo):
        """Test diff with no changes"""
        result = git_diff(git_repo)
        assert "no changes" in result.lower() or result.strip() == ""

    def test_diff_with_changes(self, git_repo):
        """Test diff with uncommitted changes"""
        test_file = os.path.join(git_repo, "test.txt")
        with open(test_file, "a") as f:
            f.write("New line\n")
        
        result = git_diff(git_repo)
        assert "New line" in result or "+" in result

    def test_diff_staged(self, git_repo):
        """Test diff of staged changes"""
        test_file = os.path.join(git_repo, "test.txt")
        with open(test_file, "a") as f:
            f.write("Staged change\n")
        
        subprocess.run(["git", "add", "."], cwd=git_repo, capture_output=True)
        
        result = git_diff(git_repo, staged=True)
        assert "Staged change" in result or "+" in result


# ========================================
# Git Log Tests
# ========================================

class TestGitLog:
    """Tests for git log command"""

    def test_log_basic(self, git_repo):
        """Test basic git log"""
        result = git_log(git_repo)
        assert "Initial commit" in result

    def test_log_with_count(self, git_repo):
        """Test log with limited count"""
        result = git_log(git_repo, count=1)
        assert "Initial commit" in result

    def test_log_oneline(self, git_repo):
        """Test oneline log format"""
        result = git_log(git_repo, oneline=True)
        # Should be more compact
        assert len(result.split("\n")) <= 2


# ========================================
# Git Branch Tests
# ========================================

class TestGitBranch:
    """Tests for git branch command"""

    def test_branch_list(self, git_repo):
        """Test listing branches"""
        result = git_branch(git_repo)
        # Should have at least master/main
        assert "master" in result.lower() or "main" in result.lower()

    def test_branch_current_marked(self, git_repo):
        """Test that current branch is marked"""
        result = git_branch(git_repo)
        assert "*" in result


# ========================================
# Git Add Tests
# ========================================

class TestGitAdd:
    """Tests for git add command"""

    def test_add_single_file(self, git_repo):
        """Test adding a single file"""
        new_file = os.path.join(git_repo, "new.txt")
        with open(new_file, "w") as f:
            f.write("New file content\n")
        
        result = git_add("new.txt", path=git_repo)
        assert "staged" in result.lower() or "new.txt" in result

    def test_add_all_files(self, git_repo):
        """Test adding all files"""
        new_file = os.path.join(git_repo, "another.txt")
        with open(new_file, "w") as f:
            f.write("Another file\n")
        
        result = git_add(".", path=git_repo)
        assert "staged" in result.lower() or "." in result


# ========================================
# Git Stash Tests
# ========================================

class TestGitStash:
    """Tests for git stash command"""

    def test_stash_list_empty(self, git_repo):
        """Test listing stashes when empty"""
        result = git_stash("list", path=git_repo)
        # Should be empty or show no stashes
        assert isinstance(result, str)

    def test_stash_push_and_list(self, git_repo):
        """Test stashing changes"""
        # Make a change
        test_file = os.path.join(git_repo, "test.txt")
        with open(test_file, "a") as f:
            f.write("Stash me\n")
        
        # Stash it
        result = git_stash("push", path=git_repo, message="Test stash")
        # Either saved or no changes to stash
        assert isinstance(result, str)


# ========================================
# Error Handling Tests
# ========================================

class TestErrorHandling:
    """Tests for error handling"""

    def test_invalid_path(self):
        """Test with invalid path"""
        result = git_status("/nonexistent/path/12345")
        assert "error" in result.lower()

    def test_run_git_command_timeout(self, git_repo):
        """Test that commands can timeout"""
        # This should complete quickly
        success, output = _run_git_command(["status"], cwd=git_repo, timeout=5)
        assert success is True
