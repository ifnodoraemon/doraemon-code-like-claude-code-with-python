"""
Git Hooks Integration

Install and manage Git hooks for automation.

Features:
- Auto-install hooks
- Pre-commit validation
- Post-commit actions
- Pre-push checks
- Custom hook scripts
"""

import logging
import os
import stat
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class GitHookType(Enum):
    """Git hook types."""

    PRE_COMMIT = "pre-commit"
    POST_COMMIT = "post-commit"
    PRE_PUSH = "pre-push"
    POST_MERGE = "post-merge"
    PRE_REBASE = "pre-rebase"
    COMMIT_MSG = "commit-msg"
    PREPARE_COMMIT_MSG = "prepare-commit-msg"


@dataclass
class HookScript:
    """A hook script configuration."""

    hook_type: GitHookType
    name: str
    script: str  # Script content or path
    is_file: bool = False  # True if script is a file path
    enabled: bool = True
    priority: int = 0  # Lower = runs first


@dataclass
class HookResult:
    """Result of hook execution."""

    success: bool
    hook_type: GitHookType
    output: str = ""
    error: str = ""
    duration: float = 0


class GitHooksManager:
    """
    Manages Git hooks installation and execution.

    Usage:
        hooks = GitHooksManager()

        # Install hooks
        hooks.install()

        # Add custom pre-commit check
        hooks.add_hook(HookScript(
            hook_type=GitHookType.PRE_COMMIT,
            name="lint",
            script="ruff check ."
        ))

        # Run hooks manually
        result = hooks.run_hook(GitHookType.PRE_COMMIT)

        # Uninstall
        hooks.uninstall()
    """

    # Default hook scripts
    DEFAULT_HOOKS = {
        GitHookType.PRE_COMMIT: '''#!/bin/bash
# Doraemon pre-commit hook

# Run linting
if command -v ruff &> /dev/null; then
    echo "Running ruff..."
    ruff check . --fix || exit 1
fi

# Run type checking
if command -v mypy &> /dev/null; then
    echo "Running mypy..."
    mypy src/ --ignore-missing-imports || exit 1
fi

# Run tests (optional, can be slow)
# pytest tests/ -q || exit 1

exit 0
''',
        GitHookType.POST_COMMIT: '''#!/bin/bash
# Doraemon post-commit hook

# Notify completion (if notification available)
if command -v notify-send &> /dev/null; then
    notify-send "Git" "Commit successful"
fi

exit 0
''',
        GitHookType.PRE_PUSH: '''#!/bin/bash
# Doraemon pre-push hook

# Run full test suite before push
if command -v pytest &> /dev/null; then
    echo "Running tests before push..."
    pytest tests/ -q || exit 1
fi

exit 0
''',
        GitHookType.COMMIT_MSG: r'''#!/bin/bash
# Doraemon commit-msg hook

# Check commit message format
commit_msg_file=$1
commit_msg=$(cat "$commit_msg_file")

# Check minimum length
if [ ${#commit_msg} -lt 10 ]; then
    echo "Error: Commit message too short (min 10 characters)"
    exit 1
fi

# Check for conventional commit format (optional)
# if ! echo "$commit_msg" | grep -qE "^(feat|fix|docs|style|refactor|test|chore)(\(.+\))?: .+"; then
#     echo "Error: Commit message should follow conventional commits format"
#     exit 1
# fi

exit 0
''',
    }

    def __init__(self, repo_path: Path | None = None):
        """
        Initialize Git hooks manager.

        Args:
            repo_path: Path to Git repository
        """
        self._repo_path = repo_path or Path.cwd()
        self._hooks_dir = self._repo_path / ".git" / "hooks"
        self._custom_hooks: dict[GitHookType, list[HookScript]] = {}
        self._callbacks: dict[GitHookType, list[Callable]] = {}

    def is_git_repo(self) -> bool:
        """Check if current directory is a Git repository."""
        return (self._repo_path / ".git").exists()

    def install(self, hooks: list[GitHookType] | None = None) -> dict[str, bool]:
        """
        Install Git hooks.

        Args:
            hooks: Specific hooks to install (None = all defaults)

        Returns:
            Dict of hook name -> success
        """
        if not self.is_git_repo():
            logger.error("Not a Git repository")
            return {}

        self._hooks_dir.mkdir(parents=True, exist_ok=True)

        hooks_to_install = hooks or list(self.DEFAULT_HOOKS.keys())
        results = {}

        for hook_type in hooks_to_install:
            if hook_type in self.DEFAULT_HOOKS:
                success = self._install_hook(hook_type, self.DEFAULT_HOOKS[hook_type])
                results[hook_type.value] = success

        return results

    def _install_hook(self, hook_type: GitHookType, script: str) -> bool:
        """Install a single hook."""
        hook_path = self._hooks_dir / hook_type.value

        try:
            # Backup existing hook
            if hook_path.exists():
                backup_path = hook_path.with_suffix(".backup")
                hook_path.rename(backup_path)
                logger.info(f"Backed up existing hook: {hook_type.value}")

            # Write new hook
            hook_path.write_text(script, encoding="utf-8")

            # Make executable
            hook_path.chmod(hook_path.stat().st_mode | stat.S_IEXEC)

            logger.info(f"Installed hook: {hook_type.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to install hook {hook_type.value}: {e}")
            return False

    def uninstall(self, hooks: list[GitHookType] | None = None):
        """
        Uninstall Git hooks.

        Args:
            hooks: Specific hooks to uninstall (None = all)
        """
        hooks_to_remove = hooks or list(GitHookType)

        for hook_type in hooks_to_remove:
            hook_path = self._hooks_dir / hook_type.value

            if hook_path.exists():
                hook_path.unlink()
                logger.info(f"Removed hook: {hook_type.value}")

                # Restore backup if exists
                backup_path = hook_path.with_suffix(".backup")
                if backup_path.exists():
                    backup_path.rename(hook_path)
                    logger.info(f"Restored backup: {hook_type.value}")

    def add_hook(self, hook: HookScript):
        """
        Add a custom hook script.

        Args:
            hook: Hook script configuration
        """
        if hook.hook_type not in self._custom_hooks:
            self._custom_hooks[hook.hook_type] = []

        self._custom_hooks[hook.hook_type].append(hook)
        self._custom_hooks[hook.hook_type].sort(key=lambda h: h.priority)

    def remove_hook(self, hook_type: GitHookType, name: str):
        """Remove a custom hook by name."""
        if hook_type in self._custom_hooks:
            self._custom_hooks[hook_type] = [
                h for h in self._custom_hooks[hook_type] if h.name != name
            ]

    def on_hook(self, hook_type: GitHookType, callback: Callable):
        """
        Register a callback for a hook type.

        Args:
            hook_type: Hook type
            callback: Function to call when hook runs
        """
        if hook_type not in self._callbacks:
            self._callbacks[hook_type] = []
        self._callbacks[hook_type].append(callback)

    def run_hook(
        self,
        hook_type: GitHookType,
        args: list[str] | None = None,
    ) -> HookResult:
        """
        Run a hook manually.

        Args:
            hook_type: Hook type to run
            args: Arguments to pass to hook

        Returns:
            HookResult
        """
        import time

        start_time = time.time()
        output_parts = []
        error_parts = []
        success = True

        # Run custom hooks
        if hook_type in self._custom_hooks:
            for hook in self._custom_hooks[hook_type]:
                if not hook.enabled:
                    continue

                try:
                    if hook.is_file:
                        result = subprocess.run(
                            [hook.script] + (args or []),
                            capture_output=True,
                            text=True,
                            cwd=self._repo_path,
                        )
                    else:
                        result = subprocess.run(
                            hook.script,
                            shell=True,
                            capture_output=True,
                            text=True,
                            cwd=self._repo_path,
                        )

                    if result.returncode != 0:
                        success = False
                        error_parts.append(f"[{hook.name}] {result.stderr}")
                    else:
                        output_parts.append(f"[{hook.name}] {result.stdout}")

                except Exception as e:
                    success = False
                    error_parts.append(f"[{hook.name}] Error: {e}")

        # Run callbacks
        if hook_type in self._callbacks:
            for callback in self._callbacks[hook_type]:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Hook callback error: {e}")

        duration = time.time() - start_time

        return HookResult(
            success=success,
            hook_type=hook_type,
            output="\n".join(output_parts),
            error="\n".join(error_parts),
            duration=duration,
        )

    def is_installed(self, hook_type: GitHookType) -> bool:
        """Check if a hook is installed."""
        hook_path = self._hooks_dir / hook_type.value
        return hook_path.exists()

    def get_installed_hooks(self) -> list[GitHookType]:
        """Get list of installed hooks."""
        installed = []
        for hook_type in GitHookType:
            if self.is_installed(hook_type):
                installed.append(hook_type)
        return installed

    def get_hook_content(self, hook_type: GitHookType) -> str | None:
        """Get content of an installed hook."""
        hook_path = self._hooks_dir / hook_type.value
        if hook_path.exists():
            return hook_path.read_text(encoding="utf-8")
        return None

    def update_hook(self, hook_type: GitHookType, script: str) -> bool:
        """
        Update an installed hook.

        Args:
            hook_type: Hook type
            script: New script content

        Returns:
            True if updated
        """
        return self._install_hook(hook_type, script)

    def validate_hooks(self) -> dict[str, Any]:
        """Validate installed hooks."""
        results = {}

        for hook_type in self.get_installed_hooks():
            hook_path = self._hooks_dir / hook_type.value

            # Check executable
            is_executable = os.access(hook_path, os.X_OK)

            # Check shebang
            content = hook_path.read_text(encoding="utf-8")
            has_shebang = content.startswith("#!")

            results[hook_type.value] = {
                "exists": True,
                "executable": is_executable,
                "has_shebang": has_shebang,
                "valid": is_executable and has_shebang,
            }

        return results

    def get_summary(self) -> dict[str, Any]:
        """Get hooks summary."""
        installed = self.get_installed_hooks()
        return {
            "is_git_repo": self.is_git_repo(),
            "hooks_dir": str(self._hooks_dir),
            "installed": [h.value for h in installed],
            "custom_hooks": {
                k.value: len(v) for k, v in self._custom_hooks.items()
            },
        }


# Global instance
_hooks_manager: GitHooksManager | None = None


def get_git_hooks_manager() -> GitHooksManager:
    """Get the global Git hooks manager."""
    global _hooks_manager
    if _hooks_manager is None:
        _hooks_manager = GitHooksManager()
    return _hooks_manager
