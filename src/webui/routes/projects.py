"""
Project context API routes.

Expose the effective startup directory used by the Web UI runtime.
"""

from pathlib import Path

from fastapi import APIRouter

router = APIRouter()


@router.get("", include_in_schema=False)
@router.get("/")
async def get_project_context():
    """Return the effective project directory for the current Web UI process."""
    cwd = Path.cwd().resolve()
    return {
        "project_name": cwd.name or str(cwd),
    }
