"""
Evaluation Dashboard API

Backend API endpoints for the evaluation dashboard.
Provides access to evaluation results, trends, and task statistics.
"""

import asyncio
import json
import logging
import os
import re
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, field_validator

# Setup logging
logger = logging.getLogger(__name__)

# Router setup
router = APIRouter()

# Templates setup (with autoescape enabled for security)
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))
templates.env.autoescape = True

# Default eval results directory
EVAL_RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "eval_results"


def _empty_trends() -> dict[str, Any]:
    return {
        "success_rate_trend": [],
        "latency_trend": [],
        "task_count_trend": [],
    }


def _empty_task_statistics() -> dict[str, Any]:
    return {
        "total_evaluations": 0,
        "total_tasks": 0,
        "total_success": 0,
        "overall_success_rate": 0,
        "by_category": {},
        "by_difficulty": {},
    }


def _dashboard_page_requires_client_auth(request: Request) -> bool:
    api_key = os.getenv("AGENT_WEBUI_API_KEY") or None
    if not api_key:
        return False

    authorization = request.headers.get("Authorization")
    if not authorization:
        return True

    submitted_key = authorization[7:] if authorization.startswith("Bearer ") else authorization
    return not secrets.compare_digest(submitted_key, api_key)


class EvaluationRequest(BaseModel):
    """Request model for triggering a new evaluation."""

    task_set: str = "default"
    n_trials: int = 1
    max_workers: int = 2
    model: str | None = None

    @field_validator("task_set")
    @classmethod
    def validate_task_set(cls, v: str) -> str:
        allowed = {"default", "quick", "full"}
        if v not in allowed:
            raise ValueError(f"task_set must be one of {allowed}")
        return v

    @field_validator("n_trials")
    @classmethod
    def validate_n_trials(cls, v: int) -> int:
        if not 1 <= v <= 10:
            raise ValueError("n_trials must be between 1 and 10")
        return v

    @field_validator("max_workers")
    @classmethod
    def validate_max_workers(cls, v: int) -> int:
        if not 1 <= v <= 8:
            raise ValueError("max_workers must be between 1 and 8")
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str | None) -> str | None:
        if v is not None:
            if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9._:/-]*$", v):
                raise ValueError("Invalid model name format")
            if len(v) > 128:
                raise ValueError("Model name too long")
        return v


class EvaluationProgress(BaseModel):
    """Model for evaluation progress tracking."""

    id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: float  # 0.0 to 1.0
    current_task: str | None = None
    total_tasks: int = 0
    completed_tasks: int = 0
    start_time: str | None = None
    end_time: str | None = None
    error: str | None = None


# In-memory storage for running evaluations
_running_evaluations: dict[str, EvaluationProgress] = {}


def get_eval_results_dir() -> Path:
    """Get the evaluation results directory."""
    return EVAL_RESULTS_DIR


def list_evaluation_files() -> list[dict[str, Any]]:
    """
    List all evaluation result files.

    Returns:
        List of evaluation file metadata.
    """
    results_dir = get_eval_results_dir()
    evaluations = []

    if not results_dir.exists():
        return evaluations

    # Scan all subdirectories
    for subdir in results_dir.iterdir():
        if not subdir.is_dir():
            continue

        # Find summary files
        for summary_file in subdir.glob("summary_*.json"):
            try:
                with open(summary_file, encoding="utf-8") as f:
                    summary = json.load(f)

                # Extract timestamp from filename
                timestamp_str = summary_file.stem.replace("summary_", "")

                # Find corresponding results file
                results_file = subdir / f"results_{timestamp_str}.json"

                evaluations.append(
                    {
                        "id": f"{subdir.name}_{timestamp_str}",
                        "name": subdir.name,
                        "timestamp": summary.get("timestamp", timestamp_str),
                        "total_tasks": summary.get("total_tasks", 0),
                        "success_rate": summary.get("success_rate", 0),
                        "total_time": summary.get("total_time", 0),
                        "summary_file": str(summary_file),
                        "results_file": str(results_file) if results_file.exists() else None,
                        "category": subdir.name,
                    }
                )
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read %s: %s", summary_file, e)
                continue

    # Sort by timestamp (newest first)
    evaluations.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return evaluations


def load_evaluation_details(eval_id: str) -> dict[str, Any]:
    """
    Load detailed evaluation results.

    Args:
        eval_id: Evaluation ID in format "category_timestamp"

    Returns:
        Detailed evaluation data including summary and individual results.
    """
    parts = eval_id.rsplit("_", 2)
    if len(parts) < 3:
        raise ValueError(f"Invalid evaluation ID format: {eval_id}")

    category = parts[0]
    timestamp = f"{parts[1]}_{parts[2]}"

    # Validate category to prevent path traversal
    if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$", category):
        raise ValueError(f"Invalid category in evaluation ID: {category}")
    if ".." in category or "/" in category:
        raise ValueError("Invalid characters in evaluation ID")

    results_dir = get_eval_results_dir() / category
    # Ensure resolved path is within eval results directory
    try:
        results_dir.resolve().relative_to(get_eval_results_dir().resolve())
    except ValueError:
        raise ValueError("Path traversal detected in evaluation ID") from None

    summary_file = results_dir / f"summary_{timestamp}.json"
    results_file = results_dir / f"results_{timestamp}.json"

    if not summary_file.exists():
        raise FileNotFoundError(f"Evaluation not found: {eval_id}")

    with open(summary_file, encoding="utf-8") as f:
        summary = json.load(f)

    results = []
    if results_file.exists():
        with open(results_file, encoding="utf-8") as f:
            results = json.load(f)

    return {
        "id": eval_id,
        "summary": summary,
        "results": results,
        "category": category,
    }


def calculate_trends() -> dict[str, Any]:
    """
    Calculate trend data from historical evaluations.

    Returns:
        Trend data including success rates over time, latency trends, etc.
    """
    evaluations = list_evaluation_files()

    if not evaluations:
        return {
            "success_rate_trend": [],
            "latency_trend": [],
            "task_count_trend": [],
        }

    # Sort by timestamp (oldest first for trends)
    evaluations.sort(key=lambda x: x.get("timestamp", ""))

    success_rate_trend = []
    latency_trend = []
    task_count_trend = []

    for eval_data in evaluations:
        timestamp = eval_data.get("timestamp", "")
        success_rate = eval_data.get("success_rate", 0)
        total_time = eval_data.get("total_time", 0)
        total_tasks = eval_data.get("total_tasks", 0)

        # Parse timestamp for display
        try:
            if "T" in timestamp:
                dt = datetime.fromisoformat(timestamp)
                label = dt.strftime("%m/%d %H:%M")
            else:
                label = timestamp[:10]
        except (ValueError, TypeError):
            label = timestamp[:10] if timestamp else "Unknown"

        success_rate_trend.append(
            {
                "label": label,
                "value": success_rate * 100,  # Convert to percentage
                "category": eval_data.get("category", "unknown"),
            }
        )

        if total_tasks > 0:
            avg_latency = total_time / total_tasks
            latency_trend.append(
                {
                    "label": label,
                    "value": avg_latency,
                    "category": eval_data.get("category", "unknown"),
                }
            )

        task_count_trend.append(
            {
                "label": label,
                "value": total_tasks,
                "category": eval_data.get("category", "unknown"),
            }
        )

    return {
        "success_rate_trend": success_rate_trend,
        "latency_trend": latency_trend,
        "task_count_trend": task_count_trend,
    }


def calculate_task_statistics() -> dict[str, Any]:
    """
    Calculate aggregated task statistics across all evaluations.

    Returns:
        Task statistics including category breakdown, difficulty analysis, etc.
    """
    evaluations = list_evaluation_files()

    category_stats: dict[str, dict[str, Any]] = {}
    difficulty_stats: dict[str, dict[str, Any]] = {}
    total_tasks = 0
    total_success = 0

    for eval_data in evaluations:
        try:
            details = load_evaluation_details(eval_data["id"])
            summary = details.get("summary", {})

            # Aggregate by category
            by_category = summary.get("by_category", {})
            for cat, stats in by_category.items():
                if cat not in category_stats:
                    category_stats[cat] = {"total": 0, "success": 0}
                category_stats[cat]["total"] += stats.get("total", 0)
                category_stats[cat]["success"] += stats.get("success", 0)

            # Aggregate by difficulty
            by_difficulty = summary.get("by_difficulty", {})
            for diff, stats in by_difficulty.items():
                if diff not in difficulty_stats:
                    difficulty_stats[diff] = {"total": 0, "success": 0}
                difficulty_stats[diff]["total"] += stats.get("total", 0)
                difficulty_stats[diff]["success"] += stats.get("success", 0)

            total_tasks += summary.get("total_tasks", 0)
            total_success += summary.get("successful_tasks", 0)

        except (FileNotFoundError, ValueError) as e:
            logger.warning("Failed to load evaluation %s: %s", eval_data['id'], e)
            continue

    # Calculate success rates
    for cat in category_stats:
        total = category_stats[cat]["total"]
        success = category_stats[cat]["success"]
        category_stats[cat]["success_rate"] = success / total if total > 0 else 0

    for diff in difficulty_stats:
        total = difficulty_stats[diff]["total"]
        success = difficulty_stats[diff]["success"]
        difficulty_stats[diff]["success_rate"] = success / total if total > 0 else 0

    return {
        "total_evaluations": len(evaluations),
        "total_tasks": total_tasks,
        "total_success": total_success,
        "overall_success_rate": total_success / total_tasks if total_tasks > 0 else 0,
        "by_category": category_stats,
        "by_difficulty": difficulty_stats,
    }


async def run_evaluation_async(
    eval_id: str,
    task_set: str,
    n_trials: int,
    max_workers: int,
    model: str | None,
) -> None:
    """
    Run evaluation asynchronously.

    Args:
        eval_id: Unique evaluation ID
        task_set: Task set to evaluate
        n_trials: Number of trials per task
        max_workers: Maximum parallel workers
        model: Model to use (optional)
    """
    progress = _running_evaluations.get(eval_id)
    if not progress:
        return

    progress.status = "running"
    progress.start_time = datetime.now().isoformat()

    try:
        # Build command
        cmd = [
            "python",
            "-m",
            "tests.evals.parallel_runner",
            "--output-dir",
            str(EVAL_RESULTS_DIR / "dashboard"),
            "--n-trials",
            str(n_trials),
            "--max-workers",
            str(max_workers),
        ]

        if model:
            cmd.extend(["--model", model])

        # Run evaluation
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            progress.status = "completed"
            progress.progress = 1.0
        else:
            progress.status = "failed"
            progress.error = stderr.decode() if stderr else "Unknown error"

    except Exception as e:
        progress.status = "failed"
        progress.error = str(e)
        logger.error("Evaluation %s failed: %s", eval_id, e)

    finally:
        progress.end_time = datetime.now().isoformat()


# =============================================================================
# API Endpoints
# =============================================================================


@router.get("/", response_class=HTMLResponse)
async def dashboard_index(request: Request):
    """
    Render the dashboard index page.
    """
    requires_auth = _dashboard_page_requires_client_auth(request)
    if requires_auth:
        evaluations = []
        trends = _empty_trends()
        task_stats = _empty_task_statistics()
        running_evaluations = []
    else:
        evaluations = list_evaluation_files()
        trends = calculate_trends()
        task_stats = calculate_task_statistics()
        running_evaluations = list(_running_evaluations.values())

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "evaluations": evaluations[:10],  # Latest 10
            "trends": trends,
            "task_stats": task_stats,
            "running_evaluations": running_evaluations,
            "requires_auth": requires_auth,
        },
    )


@router.get("/api/evaluations")
async def get_evaluations(
    limit: int = 20,
    offset: int = 0,
    category: str | None = None,
) -> dict[str, Any]:
    """
    Get list of evaluations.

    Args:
        limit: Maximum number of results
        offset: Offset for pagination
        category: Filter by category

    Returns:
        List of evaluation summaries.
    """
    limit = min(max(1, limit), 100)
    offset = max(0, offset)
    evaluations = list_evaluation_files()

    # Filter by category if specified
    if category:
        evaluations = [e for e in evaluations if e.get("category") == category]

    total = len(evaluations)
    evaluations = evaluations[offset : offset + limit]

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "evaluations": evaluations,
    }


@router.get("/api/evaluations/{eval_id}")
async def get_evaluation_details(eval_id: str) -> dict[str, Any]:
    """
    Get detailed evaluation results.

    Args:
        eval_id: Evaluation ID

    Returns:
        Detailed evaluation data.
    """
    try:
        return load_evaluation_details(eval_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/api/trends")
async def get_trends() -> dict[str, Any]:
    """
    Get trend data for charts.

    Returns:
        Trend data including success rates, latency, etc.
    """
    return calculate_trends()


@router.get("/api/tasks")
async def get_task_statistics() -> dict[str, Any]:
    """
    Get aggregated task statistics.

    Returns:
        Task statistics across all evaluations.
    """
    return calculate_task_statistics()


@router.post("/api/evaluate")
async def trigger_evaluation(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """
    Trigger a new evaluation.

    Args:
        request: Evaluation configuration
        background_tasks: FastAPI background tasks

    Returns:
        Evaluation ID and status.
    """
    # Generate unique ID
    eval_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create progress tracker
    progress = EvaluationProgress(
        id=eval_id,
        status="pending",
        progress=0.0,
        total_tasks=0,
        completed_tasks=0,
    )
    _running_evaluations[eval_id] = progress

    # Start evaluation in background
    background_tasks.add_task(
        run_evaluation_async,
        eval_id,
        request.task_set,
        request.n_trials,
        request.max_workers,
        request.model,
    )

    return {
        "id": eval_id,
        "status": "pending",
        "message": "Evaluation started",
    }


@router.get("/api/evaluate/{eval_id}/progress")
async def get_evaluation_progress(eval_id: str) -> dict[str, Any]:
    """
    Get progress of a running evaluation.

    Args:
        eval_id: Evaluation ID

    Returns:
        Current progress status.
    """
    progress = _running_evaluations.get(eval_id)
    if not progress:
        raise HTTPException(status_code=404, detail=f"Evaluation not found: {eval_id}")

    return {
        "id": progress.id,
        "status": progress.status,
        "progress": progress.progress,
        "current_task": progress.current_task,
        "total_tasks": progress.total_tasks,
        "completed_tasks": progress.completed_tasks,
        "start_time": progress.start_time,
        "end_time": progress.end_time,
        "error": progress.error,
    }


@router.get("/api/categories")
async def get_categories() -> dict[str, Any]:
    """
    Get list of evaluation categories.

    Returns:
        List of available categories.
    """
    results_dir = get_eval_results_dir()
    categories = []

    if results_dir.exists():
        for subdir in results_dir.iterdir():
            if subdir.is_dir():
                # Count evaluations in this category
                count = len(list(subdir.glob("summary_*.json")))
                if count > 0:
                    categories.append(
                        {
                            "name": subdir.name,
                            "count": count,
                        }
                    )

    return {"categories": categories}


@router.get("/api/models/compare")
async def compare_models() -> dict[str, Any]:
    """
    Compare performance across different models.

    Returns:
        Model comparison data.
    """
    evaluations = list_evaluation_files()
    model_stats: dict[str, dict[str, Any]] = {}

    for eval_data in evaluations:
        try:
            details = load_evaluation_details(eval_data["id"])
            summary = details.get(
                "summary",
            )

            # Try to extract model name from summary or category
            model_name = summary.get("model_name", eval_data.get("category", "unknown"))

            if model_name not in model_stats:
                model_stats[model_name] = {
                    "total_tasks": 0,
                    "successful_tasks": 0,
                    "total_time": 0,
                    "evaluations": 0,
                }

            model_stats[model_name]["total_tasks"] += summary.get("total_tasks", 0)
            model_stats[model_name]["successful_tasks"] += summary.get("successful_tasks", 0)
            model_stats[model_name]["total_time"] += summary.get("total_time", 0)
            model_stats[model_name]["evaluations"] += 1

        except (FileNotFoundError, ValueError):
            continue

    # Calculate averages
    for model in model_stats:
        stats = model_stats[model]
        total = stats["total_tasks"]
        stats["success_rate"] = stats["successful_tasks"] / total if total > 0 else 0
        stats["avg_time_per_task"] = stats["total_time"] / total if total > 0 else 0

    return {"models": model_stats}
