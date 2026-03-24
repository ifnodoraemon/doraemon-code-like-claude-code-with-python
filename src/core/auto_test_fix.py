"""
Auto Test-Fix Loop

Implements an automated test-fix cycle similar to Aider.
Runs tests, analyzes failures, and attempts automatic fixes.
"""

import asyncio
import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Status of a test."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class TestResult:
    """Result of a single test."""

    name: str
    status: TestStatus
    message: str | None = None
    file_path: str | None = None
    line_number: int | None = None
    error_output: str | None = None


@dataclass
class TestRunResult:
    """Result of running a test suite."""

    command: str
    exit_code: int
    stdout: str
    stderr: str
    tests: list[TestResult] = field(default_factory=list)
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    duration: float = 0.0

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and self.failed == 0 and self.errors == 0

    @property
    def summary(self) -> str:
        return (
            f"Passed: {self.passed}, Failed: {self.failed}, "
            f"Errors: {self.errors}, Skipped: {self.skipped}"
        )


@dataclass
class FixAttempt:
    """Record of a fix attempt."""

    iteration: int
    test_name: str
    error_message: str
    fix_description: str
    success: bool = False


class TestRunner:
    """Runs tests and parses results."""

    # Test framework detection patterns
    FRAMEWORK_PATTERNS = {
        "pytest": ["pytest.ini", "pyproject.toml", "setup.cfg"],
        "unittest": ["unittest", "test_*.py"],
        "jest": ["jest.config.js", "jest.config.ts", "package.json"],
        "vitest": ["vitest.config.ts", "vite.config.ts"],
        "mocha": [".mocharc.js", ".mocharc.json"],
        "cargo": ["Cargo.toml"],
        "go": ["go.mod"],
    }

    # Test commands by framework
    TEST_COMMANDS = {
        "pytest": ["pytest", "-v", "--tb=short"],
        "unittest": ["python", "-m", "unittest", "discover", "-v"],
        "jest": ["npm", "test", "--", "--verbose"],
        "vitest": ["npm", "test", "--", "--run"],
        "mocha": ["npm", "test"],
        "cargo": ["cargo", "test", "--", "--nocapture"],
        "go": ["go", "test", "-v", "./..."],
        "ruff": ["ruff", "check", "."],
        "mypy": ["mypy", "."],
    }

    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path).resolve()
        self._detected_framework: str | None = None

    def detect_framework(self) -> str | None:
        """Detect the test framework used in the project."""
        if self._detected_framework:
            return self._detected_framework

        for framework, patterns in self.FRAMEWORK_PATTERNS.items():
            for pattern in patterns:
                if (self.project_path / pattern).exists():
                    self._detected_framework = framework
                    return framework

        # Check package.json for npm projects
        pkg_json = self.project_path / "package.json"
        if pkg_json.exists():
            try:
                data = json.loads(pkg_json.read_text())
                scripts = data.get("scripts", {})
                if "test" in scripts:
                    test_script = scripts["test"].lower()
                    if "jest" in test_script:
                        self._detected_framework = "jest"
                    elif "vitest" in test_script:
                        self._detected_framework = "vitest"
                    elif "mocha" in test_script:
                        self._detected_framework = "mocha"
                    else:
                        self._detected_framework = "jest"  # Default
                    return self._detected_framework
            except Exception:
                pass

        # Check pyproject.toml for pytest
        pyproject = self.project_path / "pyproject.toml"
        if pyproject.exists():
            content = pyproject.read_text()
            if "pytest" in content:
                self._detected_framework = "pytest"
                return "pytest"

        return None

    def get_test_command(self, framework: str | None = None) -> list[str]:
        """Get the test command for the detected framework."""
        fw = framework or self.detect_framework() or "pytest"
        return self.TEST_COMMANDS.get(fw, ["pytest", "-v"])

    async def run_tests(
        self,
        command: list[str] | None = None,
        timeout: int = 300,
        env: dict[str, str] | None = None,
    ) -> TestRunResult:
        """
        Run tests and return results.

        Args:
            command: Test command to run (auto-detected if None)
            timeout: Timeout in seconds
            env: Additional environment variables

        Returns:
            TestRunResult with parsed test results
        """
        import time

        cmd = command or self.get_test_command()
        cmd_str = " ".join(cmd)

        logger.info(f"Running tests: {cmd_str}")

        start_time = time.time()

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.project_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**dict(subprocess.os.environ), **(env or {})},
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            exit_code = process.returncode or 0

        except asyncio.TimeoutError:
            return TestRunResult(
                command=cmd_str,
                exit_code=-1,
                stdout="",
                stderr=f"Test execution timed out after {timeout} seconds",
            )
        except Exception as e:
            return TestRunResult(
                command=cmd_str,
                exit_code=-1,
                stdout="",
                stderr=str(e),
            )

        duration = time.time() - start_time

        result = TestRunResult(
            command=cmd_str,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration=duration,
        )

        # Parse test results
        framework = self.detect_framework()
        if framework == "pytest":
            self._parse_pytest_output(result)
        elif framework in {"jest", "vitest"}:
            self._parse_jest_output(result)
        elif framework == "cargo":
            self._parse_cargo_output(result)

        return result

    def _parse_pytest_output(self, result: TestRunResult) -> None:
        """Parse pytest output to extract test results."""
        # Pattern for passed/failed tests
        passed_pattern = r"(\d+) passed"
        failed_pattern = r"(\d+) failed"
        error_pattern = r"(\d+) error"
        skipped_pattern = r"(\d+) skipped"

        summary_line = ""
        for line in result.stdout.split("\n"):
            if "passed" in line or "failed" in line:
                summary_line = line

        if match := re.search(passed_pattern, summary_line):
            result.passed = int(match.group(1))
        if match := re.search(failed_pattern, summary_line):
            result.failed = int(match.group(1))
        if match := re.search(error_pattern, summary_line):
            result.errors = int(match.group(1))
        if match := re.search(skipped_pattern, summary_line):
            result.skipped = int(match.group(1))

        # Extract individual failures
        failure_pattern = r"FAILED (.*?) - (.*?)(?:\n|$)"
        for match in re.finditer(failure_pattern, result.stdout):
            result.tests.append(
                TestResult(
                    name=match.group(1).strip(),
                    status=TestStatus.FAILED,
                    message=match.group(2).strip(),
                )
            )

        # Extract error details
        error_block_pattern = r"={3,} ERRORS ={3,}(.*?)(?=={3,}|$)"
        for match in re.finditer(error_block_pattern, result.stdout, re.DOTALL):
            error_block = match.group(1)
            result.tests.append(
                TestResult(
                    name="Test Error",
                    status=TestStatus.ERROR,
                    error_output=error_block[:1000],
                )
            )

    def _parse_jest_output(self, result: TestRunResult) -> None:
        """Parse Jest/Vitest output."""
        # Pattern for summary
        summary_pattern = r"Tests:\s+(\d+) passed,?\s*(\d+) failed,?\s*(\d+) total"
        if match := re.search(summary_pattern, result.stdout):
            result.passed = int(match.group(1))
            result.failed = int(match.group(2))

        # Extract failures
        fail_pattern = r"FAIL\s+(.*?)(?:\n|$)"
        for match in re.finditer(fail_pattern, result.stdout):
            result.tests.append(
                TestResult(
                    name=match.group(1).strip(),
                    status=TestStatus.FAILED,
                )
            )

    def _parse_cargo_output(self, result: TestRunResult) -> None:
        """Parse cargo test output."""
        # Pattern for summary
        summary_pattern = r"(\d+) passed; (\d+) failed"
        if match := re.search(summary_pattern, result.stdout):
            result.passed = int(match.group(1))
            result.failed = int(match.group(2))

    def get_failed_tests(self, result: TestRunResult) -> list[TestResult]:
        """Get list of failed tests from result."""
        return [t for t in result.tests if t.status in {TestStatus.FAILED, TestStatus.ERROR}]


class AutoTestFix:
    """
    Automatic test-fix loop.

    Runs tests, analyzes failures, and provides guidance for fixes.
    """

    def __init__(
        self,
        project_path: str = ".",
        max_iterations: int = 3,
        auto_apply: bool = False,
    ):
        self.project_path = Path(project_path).resolve()
        self.max_iterations = max_iterations
        self.auto_apply = auto_apply
        self.runner = TestRunner(project_path)
        self.fix_history: list[FixAttempt] = []

    async def run_test_fix_loop(
        self,
        test_command: list[str] | None = None,
        on_fix_attempt: Any | None = None,
    ) -> tuple[bool, list[FixAttempt]]:
        """
        Run the automatic test-fix loop.

        Args:
            test_command: Custom test command
            on_fix_attempt: Callback for each fix attempt

        Returns:
            Tuple of (success, fix_history)
        """
        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"Test-fix iteration {iteration}/{self.max_iterations}")

            # Run tests
            result = await self.runner.run_tests(test_command)

            if result.success:
                logger.info("All tests passed!")
                return True, self.fix_history

            # Analyze failures
            failed_tests = self.runner.get_failed_tests(result)
            logger.info(f"Found {len(failed_tests)} failed tests")

            if not failed_tests:
                # No specific failures found, but tests failed
                logger.warning(
                    f"Tests failed but no specific failures detected. Summary: {result.summary}"
                )
                return False, self.fix_history

            # Process each failure
            for test in failed_tests[:5]:  # Limit to first 5 failures
                fix_attempt = FixAttempt(
                    iteration=iteration,
                    test_name=test.name,
                    error_message=test.error_output or test.message or "",
                    fix_description="",
                )

                # Generate fix guidance
                fix_guidance = self._analyze_failure(test, result)
                fix_attempt.fix_description = fix_guidance

                if on_fix_attempt:
                    await self._call_callback(on_fix_attempt, fix_attempt, result)

                self.fix_history.append(fix_attempt)

            # If not auto-apply, stop after first iteration
            if not self.auto_apply:
                break

        return False, self.fix_history

    def _analyze_failure(self, test: TestResult, result: TestRunResult) -> str:
        """Analyze a test failure and generate fix guidance."""
        guidance_parts = []

        # Identify error type
        error_output = test.error_output or ""
        message = test.message or ""

        # Python traceback analysis
        if "AssertionError" in error_output:
            guidance_parts.append("Assertion failed. Check expected vs actual values.")
            if test.file_path:
                guidance_parts.append(f"Check file: {test.file_path}:{test.line_number}")

        elif "TypeError" in error_output:
            guidance_parts.append("Type error detected. Check function signatures and types.")
            type_match = re.search(r"expected\s+(\w+).*got\s+(\w+)", error_output)
            if type_match:
                guidance_parts.append(f"Expected {type_match.group(1)}, got {type_match.group(2)}")

        elif "ImportError" in error_output or "ModuleNotFoundError" in error_output:
            guidance_parts.append("Import error. Check module availability and paths.")
            module_match = re.search(r"No module named '([^']+)'", error_output)
            if module_match:
                guidance_parts.append(f"Missing module: {module_match.group(1)}")

        elif "NameError" in error_output:
            guidance_parts.append("Name error. Undefined variable or function.")
            name_match = re.search(r"name '([^']+)' is not defined", error_output)
            if name_match:
                guidance_parts.append(f"Undefined: {name_match.group(1)}")

        elif "AttributeError" in error_output:
            guidance_parts.append("Attribute error. Object missing expected attribute.")
            attr_match = re.search(r"'(\w+)' object has no attribute '(\w+)'", error_output)
            if attr_match:
                guidance_parts.append(
                    f"Object '{attr_match.group(1)}' missing attribute '{attr_match.group(2)}'"
                )

        # Generic guidance
        if not guidance_parts:
            guidance_parts.append("Review the test output for specific error details.")
            if test.file_path:
                guidance_parts.append(f"Test file: {test.file_path}")

        return "\n".join(f"- {g}" for g in guidance_parts)

    async def _call_callback(
        self, callback: Any, fix_attempt: FixAttempt, result: TestRunResult
    ) -> None:
        """Call a callback function."""
        import asyncio
        import inspect

        if inspect.iscoroutinefunction(callback):
            await callback(fix_attempt, result)
        else:
            callback(fix_attempt, result)

    def get_fix_summary(self) -> str:
        """Get a summary of all fix attempts."""
        if not self.fix_history:
            return "No fix attempts made."

        lines = ["# Fix Attempt Summary\n"]
        for attempt in self.fix_history:
            status = "✅" if attempt.success else "❌"
            lines.append(f"\n## Iteration {attempt.iteration}: {status}\n")
            lines.append(f"**Test:** {attempt.test_name}\n")
            lines.append(f"**Error:**\n```\n{attempt.error_message[:500]}\n```\n")
            lines.append(f"**Guidance:**\n{attempt.fix_description}\n")

        return "".join(lines)


async def run_tests_with_fix_loop(
    project_path: str = ".",
    max_iterations: int = 3,
    test_command: list[str] | None = None,
    on_fix_attempt: Any | None = None,
) -> tuple[bool, str]:
    """
    Convenience function to run tests with auto-fix loop.

    Args:
        project_path: Path to the project
        max_iterations: Maximum fix iterations
        test_command: Custom test command
        on_fix_attempt: Callback for each fix attempt

    Returns:
        Tuple of (success, summary)
    """
    auto_fix = AutoTestFix(
        project_path=project_path,
        max_iterations=max_iterations,
    )

    success, history = await auto_fix.run_test_fix_loop(
        test_command=test_command,
        on_fix_attempt=on_fix_attempt,
    )

    return success, auto_fix.get_fix_summary()
