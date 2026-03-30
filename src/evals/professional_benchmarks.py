"""Professional benchmark runners for coding agents."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkTask:
    id: str
    suite: str
    prompt: str
    files: dict[str, str]
    verify: dict[str, Any]
    metadata: dict[str, Any]


@dataclass
class BenchmarkResult:
    task_id: str
    suite: str
    passed: bool
    duration: float
    output: str
    tool_calls: list[str]
    trace_path: str | None
    error: str | None = None


class ProfessionalBenchmarkRunner:
    """Minimal benchmark runner aligned with the production agent path."""

    def __init__(self, agent) -> None:
        self.agent = agent

    def load_tasks(self, suite: str, dataset_path: str | None = None, limit: int = 0) -> list[BenchmarkTask]:
        path = Path(dataset_path) if dataset_path else self._default_dataset_path(suite)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tasks = [self._normalize_task(suite, item) for item in data]
        return tasks[:limit] if limit > 0 else tasks

    def run(self, suite: str, dataset_path: str | None = None, limit: int = 0) -> dict[str, Any]:
        tasks = self.load_tasks(suite, dataset_path=dataset_path, limit=limit)
        results = [self._run_task(task) for task in tasks]
        passed = sum(1 for result in results if result.passed)
        return {
            "suite": suite,
            "total": len(results),
            "passed": passed,
            "pass_rate": passed / len(results) if results else 0.0,
            "results": [result.__dict__ for result in results],
        }

    def _default_dataset_path(self, suite: str) -> Path:
        base = Path("benchmarks/samples")
        mapping = {
            "humaneval_plus": base / "humaneval_plus_sample.json",
            "repo_patch": base / "repo_patch_sample.json",
            "terminal_bench": base / "terminal_bench_sample.json",
            "real_repo": Path("benchmarks/real_repo_tasks.json"),
        }
        if suite not in mapping:
            raise ValueError(f"Unsupported suite: {suite}")
        return mapping[suite]

    def _normalize_task(self, suite: str, item: dict[str, Any]) -> BenchmarkTask:
        if suite == "humaneval_plus":
            prompt = (
                f"Create a file named solution.py and implement this Python function exactly.\n\n"
                f"{item['prompt']}\n\n"
                f"Return only a short completion message after writing the file."
            )
            verify = {
                "type": "python_test",
                "code": f"{item['test']}\ncheck({item['entry_point']})",
            }
            return BenchmarkTask(
                id=item["task_id"],
                suite=suite,
                prompt=prompt,
                files={},
                verify=verify,
                metadata={"entry_point": item["entry_point"]},
            )

        if suite == "repo_patch":
            files = item.get("files", {})
            verify = item.get("verify", {})
            prompt = item["problem_statement"]
            return BenchmarkTask(
                id=item["instance_id"],
                suite=suite,
                prompt=prompt,
                files=files,
                verify=verify,
                metadata={"repo": item.get("repo"), "base_commit": item.get("base_commit")},
            )

        if suite == "real_repo":
            files = item.get("files", {})
            verify = item.get("verify", {})
            prompt = item["problem_statement"]
            return BenchmarkTask(
                id=item["instance_id"],
                suite=suite,
                prompt=prompt,
                files=files,
                verify=verify,
                metadata={
                    "source_commit": item.get("source_commit"),
                    "theme": item.get("theme"),
                },
            )

        if suite == "terminal_bench":
            return BenchmarkTask(
                id=item["id"],
                suite=suite,
                prompt=item["instruction"],
                files=item.get("files", {}),
                verify=item["verify"],
                metadata={"category": item.get("category", "terminal")},
            )

        raise ValueError(f"Unsupported suite: {suite}")

    def _run_task(self, task: BenchmarkTask) -> BenchmarkResult:
        start = time.time()
        original_cwd = Path.cwd()

        with tempfile.TemporaryDirectory(prefix=f"{task.suite}_{task.id}_") as sandbox:
            sandbox_dir = Path(sandbox)
            try:
                for relative_path, content in task.files.items():
                    path = sandbox_dir / relative_path
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content, encoding="utf-8")

                os.chdir(sandbox_dir)
                response = self.agent.execute(task.prompt)
                trace_path = response.metadata.get("trace_path")
                tool_calls = [tc.name for tc in response.tool_calls]

                if not response.success:
                    error = "; ".join(response.errors) or "agent execution failed"
                    return BenchmarkResult(
                        task_id=task.id,
                        suite=task.suite,
                        passed=False,
                        duration=time.time() - start,
                        output=response.output,
                        tool_calls=tool_calls,
                        trace_path=trace_path,
                        error=error,
                    )

                passed, error = self._verify_task(task, sandbox_dir, response.output)
                return BenchmarkResult(
                    task_id=task.id,
                    suite=task.suite,
                    passed=passed,
                    duration=time.time() - start,
                    output=response.output,
                    tool_calls=tool_calls,
                    trace_path=trace_path,
                    error=error,
                )
            except Exception as e:
                return BenchmarkResult(
                    task_id=task.id,
                    suite=task.suite,
                    passed=False,
                    duration=time.time() - start,
                    output="",
                    tool_calls=[],
                    trace_path=None,
                    error=str(e),
                )
            finally:
                os.chdir(original_cwd)

    def _verify_task(self, task: BenchmarkTask, sandbox_dir: Path, output: str) -> tuple[bool, str | None]:
        verify_type = task.verify.get("type")

        if verify_type == "python_test":
            solution_path = sandbox_dir / "solution.py"
            if not solution_path.exists():
                code = self._extract_python_code(output)
                if not code:
                    return False, "solution.py was not created"
                solution_path.write_text(code, encoding="utf-8")

            test_path = sandbox_dir / "verify_solution.py"
            test_path.write_text(
                solution_path.read_text(encoding="utf-8") + "\n\n" + task.verify["code"],
                encoding="utf-8",
            )
            proc = subprocess.run(
                ["python3", str(test_path)],
                cwd=sandbox_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode == 0:
                return True, None
            return False, (proc.stderr or proc.stdout or "python test failed")[:1000]

        if verify_type == "command":
            proc = subprocess.run(
                task.verify["command"],
                cwd=sandbox_dir,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if proc.returncode == 0:
                return True, None
            return False, (proc.stderr or proc.stdout or "command failed")[:1000]

        return False, f"Unsupported verify type: {verify_type}"

    def _extract_python_code(self, output: str) -> str:
        python_block = re.search(r"```python\n(.*?)```", output, re.DOTALL)
        if python_block:
            return python_block.group(1).strip()
        generic_block = re.search(r"```\n(.*?)```", output, re.DOTALL)
        if generic_block:
            return generic_block.group(1).strip()
        if "def " in output:
            return output.strip()
        return ""
