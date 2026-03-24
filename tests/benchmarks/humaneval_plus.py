"""
HumanEval+ Benchmark Integration

HumanEval+ is an improved version of HumanEval with 80x more tests per problem.
Dataset: https://github.com/ganler/codegen
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.agent import AgentResult, AgentState, DoraemonAgent, ToolDefinition

logger = logging.getLogger(__name__)

HUMANEVAL_PLUS_URL = "https://github.com/ganler/codegen/raw/main/HumanEvalPlus.json"


@dataclass
class HumanEvalTask:
    """A single HumanEval+ task."""

    task_id: str
    prompt: str
    canonical_solution: str
    test: str
    entry_point: str

    @classmethod
    def from_dict(cls, data: dict) -> "HumanEvalTask":
        return cls(
            task_id=data["task_id"],
            prompt=data["prompt"],
            canonical_solution=data.get("canonical_solution", ""),
            test=data["test"],
            entry_point=data["entry_point"],
        )


@dataclass
class HumanEvalResult:
    """Result of evaluating a HumanEval+ task."""

    task_id: str
    passed: bool
    generated_code: str
    error: str | None = None
    duration: float = 0.0
    tokens_used: int = 0


class HumanEvalBenchmark:
    """
    HumanEval+ benchmark runner.

    Usage:
        benchmark = HumanEvalBenchmark()
        results = await benchmark.run(agent, n_tasks=20)
        print(f"pass@1: {results['pass_rate']:.1%}")
    """

    def __init__(self, dataset_path: Path | None = None):
        self.dataset_path = dataset_path
        self._tasks: list[HumanEvalTask] = []

    async def load_dataset(self, n_tasks: int | None = None) -> list[HumanEvalTask]:
        """Load HumanEval+ dataset."""
        if self._tasks:
            return self._tasks[:n_tasks] if n_tasks else self._tasks

        if self.dataset_path and self.dataset_path.exists():
            data = json.loads(self.dataset_path.read_text())
        else:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(HUMANEVAL_PLUS_URL)
                data = response.json()

        self._tasks = [HumanEvalTask.from_dict(t) for t in data]
        return self._tasks[:n_tasks] if n_tasks else self._tasks

    async def run(
        self,
        agent: DoraemonAgent,
        n_tasks: int = 20,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """
        Run benchmark on agent.

        Args:
            agent: Agent to evaluate
            n_tasks: Number of tasks to run (None = all)
            timeout: Timeout per task

        Returns:
            Dict with pass_rate, results, etc.
        """
        tasks = await self.load_dataset(n_tasks)
        results = []

        for i, task in enumerate(tasks):
            logger.info(f"Running task {i + 1}/{len(tasks)}: {task.task_id}")

            result = await self._run_task(agent, task, timeout)
            results.append(result)

            if result.passed:
                logger.info(f"  ✓ Passed")
            else:
                logger.info(f"  ✗ Failed: {result.error}")

        passed = sum(1 for r in results if r.passed)
        total = len(results)

        return {
            "pass_rate": passed / total if total > 0 else 0,
            "passed": passed,
            "total": total,
            "results": [r.__dict__ for r in results],
            "avg_duration": sum(r.duration for r in results) / total if total > 0 else 0,
        }

    async def _run_task(
        self,
        agent: DoraemonAgent,
        task: HumanEvalTask,
        timeout: float,
    ) -> HumanEvalResult:
        """Run a single task."""
        start_time = time.time()

        prompt = f"""Complete the following Python function. Only output the function code, no explanation.

```python
{task.prompt}
```

The function should be named `{task.entry_point}`."""

        try:
            agent.reset()
            result = await asyncio.wait_for(
                agent.run(prompt),
                timeout=timeout,
            )

            code = self._extract_code(result.response or "")

            if not code:
                return HumanEvalResult(
                    task_id=task.task_id,
                    passed=False,
                    generated_code="",
                    error="No code generated",
                    duration=time.time() - start_time,
                )

            passed, error = await self._verify_solution(code, task)

            return HumanEvalResult(
                task_id=task.task_id,
                passed=passed,
                generated_code=code,
                error=error,
                duration=time.time() - start_time,
                tokens_used=result.tokens_used,
            )

        except asyncio.TimeoutError:
            return HumanEvalResult(
                task_id=task.task_id,
                passed=False,
                generated_code="",
                error="Timeout",
                duration=time.time() - start_time,
            )
        except Exception as e:
            return HumanEvalResult(
                task_id=task.task_id,
                passed=False,
                generated_code="",
                error=str(e),
                duration=time.time() - start_time,
            )

    def _extract_code(self, response: str) -> str:
        """Extract code from response."""
        code_block = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        if code_block:
            return code_block.group(1).strip()

        code_block = re.search(r"```\n(.*?)```", response, re.DOTALL)
        if code_block:
            return code_block.group(1).strip()

        if "def " in response:
            lines = response.split("\n")
            code_lines = []
            in_function = False
            for line in lines:
                if line.startswith("def "):
                    in_function = True
                if in_function:
                    code_lines.append(line)
            return "\n".join(code_lines)

        return ""

    async def _verify_solution(
        self,
        code: str,
        task: HumanEvalTask,
    ) -> tuple[bool, str | None]:
        """Verify solution against tests."""
        full_code = f"{code}\n\n{task.test}"

        try:
            import tempfile
            import subprocess

            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
            ) as f:
                f.write(full_code)
                temp_path = f.name

            proc = await asyncio.create_subprocess_exec(
                "python3",
                temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=30.0,
            )

            Path(temp_path).unlink()

            if proc.returncode == 0:
                return True, None
            else:
                error = stderr.decode() or stdout.decode()
                return False, error[:500]

        except Exception as e:
            return False, str(e)


async def run_humaneval_quick(
    agent: DoraemonAgent,
    n_tasks: int = 10,
) -> dict[str, Any]:
    """Quick HumanEval+ run for CI."""
    benchmark = HumanEvalBenchmark()
    return await benchmark.run(agent, n_tasks=n_tasks, timeout=60.0)
