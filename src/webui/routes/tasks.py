"""
Tasks API Routes

Expose the shared runtime task graph for UI consumers.
"""

from pathlib import Path

from fastapi import APIRouter

from src.runtime.bootstrap import bootstrap_runtime

router = APIRouter()


@router.get("/")
async def list_tasks(
    project: str = "default",
    mode: str = "build",
    ready_only: bool = False,
):
    """List runtime tasks and the current task tree."""
    runtime = await bootstrap_runtime(
        mode=mode,
        project=project,
        project_dir=Path.cwd(),
        create_model_client=False,
    )
    try:
        task_manager = runtime.task_manager
        if ready_only:
            ready_tasks = [task.to_dict() for task in task_manager.list_ready_tasks()]
            return {"tasks": ready_tasks}

        tasks = [task.to_dict() for task in task_manager.list_tasks()]
        return {
            "tasks": tasks,
            "tree": task_manager.get_task_tree(),
        }
    finally:
        await runtime.aclose()
