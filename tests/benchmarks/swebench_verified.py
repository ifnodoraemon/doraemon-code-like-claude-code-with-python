"""
SWE-bench Verified Benchmark Integration

SWE-bench Verified is a dataset of 500 human-verified GitHub issues
that test an agent's ability to resolve real-world software engineering tasks.

Dataset: https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified
"""

import asyncio
import json
import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SWEBenchTask:
    """A single SWE-bench task."""

    instance_id: str
    repo: str
    problem_statement: str
    base_commit: str
    hints_text: str
    test_patch: str
    patch: str
    version: str

    @classmethod
    def from_dict(cls, data: dict) -> "SWEBenchTask":
        return cls(
            instance_id=data["instance_id"],
            repo=data["repo"],
            problem_statement=data["problem_statement"],
            base_commit=data["base_commit"],
            hints_text=data.get("hints_text", ""),
            test_patch=data["test_patch"],
            patch=data.get("patch", ""),
            version=data.get("version", ""),
        )


@dataclass
class SWEBenchResult:
    """Result of evaluating a SWE-bench task."""

    instance_id: str
    resolved: bool
    patch_generated: str
    error: str | None = None
    duration: float = 0.0
    tool_calls: int = 0


class SWEBenchRunner:
    """
    SWE-bench Verified runner.

    This is a simplified runner that:
    1. Clones the repo at the base commit
    2. Lets the agent work on the issue
    3. Applies the test patch
    4. Runs tests to verify

    Usage:
        runner = SWEBenchRunner()
        results = await runner.run(agent, n_tasks=5)
        print(f"Resolved: {results['resolved_rate']:.1%}")
    """

    def __init__(self, work_dir: Path | None = None):
        self.work_dir = work_dir or Path(tempfile.mkdtemp())
        self._tasks: list[SWEBenchTask] = []

    async def load_dataset(
        self,
        n_tasks: int | None = None,
        difficulty: str | None = None,
    ) -> list[SWEBenchTask]:
        """Load SWE-bench Verified dataset."""
        if self._tasks:
            return self._tasks[:n_tasks] if n_tasks else self._tasks

        try:
            from datasets import load_dataset

            dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

            self._tasks = [SWEBenchTask.from_dict(item) for item in dataset]

            if difficulty:
                self._tasks = [t for t in self._tasks if self._get_difficulty(t) == difficulty]

            return self._tasks[:n_tasks] if n_tasks else self._tasks

        except ImportError:
            logger.warning("datasets library not installed, using sample tasks")
            return self._get_sample_tasks()[:n_tasks] if n_tasks else self._get_sample_tasks()

    def _get_sample_tasks(self) -> list[SWEBenchTask]:
        """Get sample tasks for testing without dataset download."""
        return [
            SWEBenchTask(
                instance_id="django__django-12345",
                repo="django/django",
                problem_statement="Fix URL routing bug in Django 4.0",
                base_commit="abc123",
                hints_text="Check the urlresolvers module",
                test_patch="",
                patch="",
                version="4.0",
            ),
        ]

    def _get_difficulty(self, task: SWEBenchTask) -> str:
        """Estimate task difficulty."""
        stmt_len = len(task.problem_statement)
        if stmt_len < 500:
            return "easy"
        elif stmt_len < 1500:
            return "medium"
        return "hard"

    async def run(
        self,
        agent: DoraemonAgent,
        n_tasks: int = 5,
        timeout: float = 600.0,
        setup_repo: bool = True,
    ) -> dict[str, Any]:
        """
        Run benchmark on agent.

        Args:
            agent: Agent to evaluate
            n_tasks: Number of tasks to run
            timeout: Timeout per task (10 min default)
            setup_repo: Whether to clone repos (disable for dry run)

        Returns:
            Dict with resolved_rate, results, etc.
        """
        tasks = await self.load_dataset(n_tasks)
        results = []

        for i, task in enumerate(tasks):
            logger.info(f"Running task {i + 1}/{len(tasks)}: {task.instance_id}")

            result = await self._run_task(agent, task, timeout, setup_repo)
            results.append(result)

            if result.resolved:
                logger.info(f"  ✓ Resolved")
            else:
                logger.info(f"  ✗ Failed: {result.error}")

        resolved = sum(1 for r in results if r.resolved)
        total = len(results)

        return {
            "resolved_rate": resolved / total if total > 0 else 0,
            "resolved": resolved,
            "total": total,
            "results": [r.__dict__ for r in results],
            "avg_duration": sum(r.duration for r in results) / total if total > 0 else 0,
        }

    async def _run_task(
        self,
        agent: DoraemonAgent,
        task: SWEBenchTask,
        timeout: float,
        setup_repo: bool,
    ) -> SWEBenchResult:
        """Run a single task."""
        start_time = time.time()

        prompt = f"""You are working on the repository: {task.repo}

## Issue
{task.problem_statement}

## Instructions
1. Explore the codebase to understand the issue
2. Make the necessary changes to fix the issue
3. Ensure your changes don't break existing tests

{"## Hints\\n" + task.hints_text if task.hints_text else ""}
"""

        work_dir = None

        try:
            if setup_repo:
                work_dir = await self._setup_repo(task)

            agent.reset()
            result = await asyncio.wait_for(
                agent.run(prompt),
                timeout=timeout,
            )

            if work_dir:
                resolved, error = await self._verify_solution(work_dir, task)
            else:
                resolved = False
                error = "Repo not setup (dry run)"

            return SWEBenchResult(
                instance_id=task.instance_id,
                resolved=resolved,
                patch_generated="",  # TODO: extract patch
                error=error,
                duration=time.time() - start_time,
                tool_calls=len(result.tool_calls),
            )

        except asyncio.TimeoutError:
            return SWEBenchResult(
                instance_id=task.instance_id,
                resolved=False,
                patch_generated="",
                error="Timeout",
                duration=time.time() - start_time,
            )
        except Exception as e:
            return SWEBenchResult(
                instance_id=task.instance_id,
                resolved=False,
                patch_generated="",
                error=str(e),
                duration=time.time() - start_time,
            )
        finally:
            if work_dir:
                await self._cleanup_repo(work_dir)

    async def _setup_repo(self, task: SWEBenchTask) -> Path:
        """Clone and setup repo for task."""
        repo_dir = self.work_dir / task.instance_id

        if not repo_dir.exists():
            repo_url = f"https://github.com/{task.repo}"

            proc = await asyncio.create_subprocess_exec(
                "git",
                "clone",
                "--depth",
                "1",
                repo_url,
                str(repo_dir),
            )
            await proc.communicate()

        return repo_dir

    async def _verify_solution(
        self,
        repo_dir: Path,
        task: SWEBenchTask,
    ) -> tuple[bool, str | None]:
        """Verify solution by running tests."""
        if task.test_patch:
            patch_file = repo_dir / "test_patch.diff"
            patch_file.write_text(task.test_patch)

            proc = await asyncio.create_subprocess_exec(
                "git",
                "apply",
                str(patch_file),
                cwd=repo_dir,
            )
            await proc.communicate()

        proc = await asyncio.create_subprocess_exec(
            "python",
            "-m",
            "pytest",
            "-x",
            cwd=repo_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=120.0,
            )
        except asyncio.TimeoutError:
            return False, "Test timeout"

        if proc.returncode == 0:
            return True, None
        else:
            return False, stderr.decode()[:500] or "Tests failed"

    async def _cleanup_repo(self, repo_dir: Path) -> None:
        """Cleanup repo after task."""
        import shutil

        try:
            shutil.rmtree(repo_dir)
        except Exception:
            pass


async def run_swebench_quick(
    agent: DoraemonAgent,
    n_tasks: int = 3,
    setup_repo: bool = False,
) -> dict[str, Any]:
    """Quick SWE-bench run for CI (no repo setup by default)."""
    runner = SWEBenchRunner()
    return await runner.run(agent, n_tasks=n_tasks, setup_repo=setup_repo)
