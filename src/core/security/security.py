"""
Security Utilities

Provides path validation, sanitization, and security checks.
"""

import os
from pathlib import Path

_CACHED_BASE_DIR: str | None = None


def set_base_dir(base_dir: str) -> str:
    """Set the base directory for the workspace sandbox.

    Call this once at application startup to pin the sandbox root,
    so that subsequent os.chdir() calls do not change the sandbox boundary.
    If never called, validate_path uses os.getcwd() as before.
    """
    global _CACHED_BASE_DIR
    _CACHED_BASE_DIR = os.path.abspath(base_dir)
    return _CACHED_BASE_DIR

SENSITIVE_PATHS = {
    "/etc/passwd",
    "/etc/shadow",
    "/etc/sudoers",
    "/etc/ssh/sshd_config",
    "/root/.ssh",
    "/root/.bash_history",
    "/var/log",
    "/proc",
    "/sys",
    "C:\\Windows\\System32",
    "C:\\Windows\\System32\\config",
}

SENSITIVE_PATTERNS = [
    ".ssh",
    ".gnupg",
    ".bash_history",
    ".pypirc",
    ".netrc",
    "_netrc",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    ".pem",
    ".key",
]


def is_sensitive_path(path: str) -> bool:
    """Check if path points to sensitive system files or directories."""
    abs_path = os.path.abspath(os.path.expanduser(path))

    for sensitive in SENSITIVE_PATHS:
        if abs_path.startswith(sensitive) or sensitive.startswith(abs_path):
            return True

    path_lower = path.lower()
    for pattern in SENSITIVE_PATTERNS:
        pl = pattern.lower()
        if pl.startswith("."):
            if f"/{pl}" in path_lower or f"\\{pl}" in path_lower:
                return True
            if path_lower.startswith(pl):
                return True
        elif f"/{pl}" in path_lower or f"\\{pl}" in path_lower:
            return True
        elif path_lower.endswith(pl):
            return True

    return False


def validate_path(path: str, base_dir: str | None = None) -> str:
    """
    Validate that path is within the workspace sandbox.

    This function:
    - Blocks access to sensitive system files and directories
    - Resolves symbolic links to prevent symlink attacks
    - Uses proper path comparison to prevent prefix attacks
    - Handles edge cases like '/home/user123' vs '/home/user1'
    - Expands user home directory (~)

    Args:
        path: The path to validate
        base_dir: The base directory for the sandbox (defaults to cwd)

    Returns:
        The absolute path if valid.

    Raises:
        PermissionError: If path is outside sandbox or points to sensitive location.
        ValueError: If path is empty or invalid.
    """
    if not path or not path.strip():
        raise ValueError("Path cannot be empty")

    if is_sensitive_path(path):
        raise PermissionError(
            f"Access Denied: Path '{path}' points to a sensitive system location."
        )

    if base_dir is None:
        if _CACHED_BASE_DIR is not None:
            base_dir = _CACHED_BASE_DIR
        else:
            base_dir = os.getcwd()

    try:
        expanded_path = os.path.expanduser(path)
        abs_path = Path(expanded_path).resolve()
        abs_base = Path(base_dir).resolve()

        try:
            abs_path.relative_to(abs_base)
        except ValueError as e:
            raise PermissionError(
                f"Access Denied: Path '{path}' resolves to '{abs_path}' "
                f"which is outside of the workspace sandbox ({abs_base})."
            ) from e

        return str(abs_path)

    except OSError as e:
        raise PermissionError(f"Access Denied: Cannot resolve path '{path}': {e}") from e
