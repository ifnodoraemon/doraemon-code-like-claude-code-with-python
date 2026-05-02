#!/usr/bin/env python3
"""Run repo-style benchmark suites against the real Doraemon agent."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SUITES = {
    "real_repo": PROJECT_ROOT / "benchmarks" / "real_repo_tasks.json",
    "repo_patch": PROJECT_ROOT / "benchmarks" / "samples" / "repo_patch_sample.json",
    "humaneval_plus": PROJECT_ROOT / "benchmarks" / "samples" / "humaneval_plus_sample.json",
    "terminal_bench": PROJECT_ROOT / "benchmarks" / "samples" / "terminal_bench_sample.json",
}


def resolve_suite_path(suite: str | None, task_file: str | None) -> Path:
    if task_file:
        return Path(task_file).expanduser().resolve()
    if not suite:
        raise ValueError("Specify --suite or --task-file")
    try:
        return SUITES[suite]
    except KeyError as exc:
        valid = ", ".join(sorted(SUITES))
        raise ValueError(f"Unknown suite '{suite}'. Valid suites: {valid}") from exc


def load_tasks(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    tasks = data if isinstance(data, list) else [data]
    if limit > 0:
        return tasks[:limit]
    return tasks


def materialize_files(task: dict[str, Any], sandbox_dir: Path) -> None:
    for relative_path, content in task.get("files", {}).items():
        file_path = sandbox_dir / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")


def run_verify(task: dict[str, Any], sandbox_dir: Path, timeout: int) -> dict[str, Any]:
    verify = task.get("verify") or {}
    if verify.get("type") != "command":
        return {"success": True, "skipped": True, "output": "No command verifier configured"}

    command = verify.get("command")
    if not command:
        return {"success": False, "output": "Verifier command is empty"}

    completed = subprocess.run(
        command,
        shell=True,
        cwd=sandbox_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    return {
        "success": completed.returncode == 0,
        "returncode": completed.returncode,
        "output": output.strip(),
    }


def run_task(agent: Any, task: dict[str, Any], verify_timeout: int) -> dict[str, Any]:
    task_id = task.get("instance_id") or task.get("id") or "unknown"
    prompt = task.get("problem_statement") or task.get("prompt")
    if not prompt:
        return {"task_id": task_id, "success": False, "error": "Task has no prompt"}

    start = time.time()
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix=f"bench_{task_id}_") as sandbox:
        sandbox_dir = Path(sandbox)
        materialize_files(task, sandbox_dir)
        try:
            os.chdir(sandbox_dir)
            response = agent.execute(prompt)
            agent_success = bool(getattr(response, "success", True))
            errors = list(getattr(response, "errors", []) or [])
            verify_result = run_verify(task, sandbox_dir, verify_timeout)
            success = agent_success and bool(verify_result.get("success"))
            return {
                "task_id": task_id,
                "success": success,
                "agent_success": agent_success,
                "verify": verify_result,
                "errors": errors,
                "execution_time": time.time() - start,
            }
        except Exception as exc:
            return {
                "task_id": task_id,
                "success": False,
                "error": str(exc),
                "execution_time": time.time() - start,
            }
        finally:
            os.chdir(original_cwd)


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    successful = sum(1 for result in results if result.get("success"))
    return {
        "total_tasks": total,
        "successful_tasks": successful,
        "success_rate": successful / total if total else 0.0,
        "avg_execution_time": (
            sum(float(result.get("execution_time", 0.0)) for result in results) / total
            if total
            else 0.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Doraemon benchmark suites")
    parser.add_argument("--suite", choices=sorted(SUITES), help="Named benchmark suite")
    parser.add_argument("--task-file", help="Path to a benchmark JSON file")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of tasks")
    parser.add_argument("--verify-timeout", type=int, default=120, help="Verifier timeout seconds")
    parser.add_argument("--output", default="eval_results/benchmarks", help="Output directory")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument("--list", action="store_true", help="List available suites and exit")
    args = parser.parse_args()

    if args.list:
        for name, path in sorted(SUITES.items()):
            print(f"{name}\t{path.relative_to(PROJECT_ROOT)}")
        return 0

    try:
        task_path = resolve_suite_path(args.suite, args.task_file)
        tasks = load_tasks(task_path, args.limit)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    from tests.evals.agent_adapter import create_doraemon_agent

    agent = create_doraemon_agent()
    try:
        results = []
        for task in tasks:
            task_id = task.get("instance_id") or task.get("id") or "unknown"
            if not args.json:
                print(f"Running {task_id}...")
            result = run_task(agent, task, args.verify_timeout)
            results.append(result)
            if not args.json:
                status = "PASS" if result.get("success") else "FAIL"
                print(f"  {status} ({result.get('execution_time', 0.0):.2f}s)")
    finally:
        close = getattr(agent, "close", None)
        if callable(close):
            close()

    report = {
        "suite": args.suite,
        "task_file": str(task_path),
        "summary": summarize(results),
        "results": results,
    }

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{args.suite or task_path.stem}_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        summary = report["summary"]
        print("\nBenchmark complete")
        print(f"  Tasks: {summary['successful_tasks']}/{summary['total_tasks']}")
        print(f"  Success rate: {summary['success_rate']:.1%}")
        print(f"  Avg time: {summary['avg_execution_time']:.2f}s")
        print(f"  Report: {report_path}")

    return 0 if report["summary"]["success_rate"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
