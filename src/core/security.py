import os
from pathlib import Path


def validate_path(path: str, base_dir: str | None = None) -> str:
    """
    Validate that path is within the workspace sandbox.
    Defaults to current working directory if base_dir is not provided.

    This function:
    - Resolves symbolic links to prevent symlink attacks
    - Uses proper path comparison to prevent prefix attacks
    - Handles edge cases like '/home/user123' vs '/home/user1'

    Args:
        path: The path to validate
        base_dir: The base directory for the sandbox (defaults to cwd)

    Returns:
        The absolute path if valid.

    Raises:
        PermissionError: If path is outside sandbox.
        ValueError: If path is empty or invalid.
    """
    if not path or not path.strip():
        raise ValueError("Path cannot be empty")

    if base_dir is None:
        base_dir = os.getcwd()

    try:
        # Resolve symbolic links and normalize the path
        abs_path = Path(path).resolve()
        abs_base = Path(base_dir).resolve()

        # Use is_relative_to for proper path comparison (Python 3.9+)
        # This correctly handles cases like '/home/user123' vs '/home/user1'
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
