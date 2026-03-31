"""
Unified Code Quality and Linting Tools

Provides code analysis, linting, and type checking capabilities.
Similar to Claude Code's ReadLints tool but more comprehensive.

Design Philosophy:
- Occam's Razor: Single unified `lint` tool replaces 8 scattered tools
- Single Responsibility: Each operation has one clear purpose
- Functional Cohesion: Related operations grouped via `operation` parameter
- Parameterized Design: Use parameters instead of multiple tools

Unified Tool:
    lint(path, operation, language, fix)

    Operations:
    - check: Run linter (Ruff for Python, ESLint for JS)
    - format: Format code (Ruff format for Python)
    - typecheck: Type checking (MyPy for Python)
    - security: Security-focused linting
    - complexity: Code complexity analysis
    - summary: Get lint summary by category
    - all: Run all checks

Legacy tools (lint_python_ruff, format_python_ruff, etc.) are kept
for backward compatibility but delegate to the unified lint function.
"""

import json
import logging
import os
import subprocess
from dataclasses import dataclass

from src.core.logger import configure_root_logger
from src.core.security.security import validate_path

# Setup logging
configure_root_logger()
logger = logging.getLogger(__name__)


# ========================================
# Data Structures
# ========================================


@dataclass
class LintIssue:
    """Represents a single lint issue."""

    file: str
    line: int
    column: int
    severity: str  # error, warning, info
    code: str
    message: str
    source: str  # ruff, mypy, eslint, etc.

    def to_string(self) -> str:
        return (
            f"{self.file}:{self.line}:{self.column} [{self.severity}] {self.code}: {self.message}"
        )


# ========================================
# Helper Functions
# ========================================


def _run_command(
    args: list[str],
    cwd: str = ".",
    timeout: int = 60,
) -> tuple[int, str, str]:
    """
    Run a command and return (exit_code, stdout, stderr).
    """
    try:
        resolved_cwd = validate_path(cwd)
    except (PermissionError, ValueError) as e:
        return -1, "", f"Invalid path: {e}"

    try:
        result = subprocess.run(
            args,
            cwd=resolved_cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr

    except FileNotFoundError:
        return -1, "", f"Command not found: {args[0]}"
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout}s"
    except Exception as e:
        return -1, "", f"Error: {str(e)}"


def _check_tool_installed(tool: str) -> bool:
    """Check if a tool is installed and available."""
    try:
        subprocess.run([tool, "--version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _detect_language(path: str) -> str | None:
    """
    Auto-detect language based on file extension or directory contents.

    Returns:
        "python", "javascript", or None if cannot detect
    """
    try:
        resolved_path = validate_path(path)
    except (PermissionError, ValueError):
        return None

    if os.path.isfile(resolved_path):
        ext = os.path.splitext(resolved_path)[1].lower()
        if ext == ".py":
            return "python"
        elif ext in (".js", ".jsx", ".ts", ".tsx"):
            return "javascript"
        return None

    # Check directory contents
    has_python = False
    has_js = False

    for _, _, files in os.walk(resolved_path):
        for f in files:
            if f.endswith(".py"):
                has_python = True
            elif f.endswith((".js", ".jsx", ".ts", ".tsx")):
                has_js = True
        if has_python and has_js:
            break

    # Prefer Python if both exist
    if has_python:
        return "python"
    if has_js:
        return "javascript"
    return None


# ========================================
# Internal Lint Operations
# ========================================


def _lint_check_python(
    path: str,
    fix: bool = False,
    select: list[str] | None = None,
    ignore: list[str] | None = None,
) -> str:
    """Run Ruff linter on Python code."""
    if not _check_tool_installed("ruff"):
        return "Error: Ruff is not installed. Install with: pip install ruff"

    try:
        resolved_path = validate_path(path)
    except (PermissionError, ValueError) as e:
        return f"Error: {e}"

    args = ["ruff", "check", resolved_path]

    if fix:
        args.append("--fix")
    if select:
        args.extend(["--select", ",".join(select)])
    if ignore:
        args.extend(["--ignore", ",".join(ignore)])

    # Use JSON output for structured parsing
    args.extend(["--output-format", "json"])

    exit_code, stdout, stderr = _run_command(args)

    if stderr and "error" in stderr.lower():
        return f"Error running Ruff: {stderr}"

    try:
        if stdout.strip():
            issues = json.loads(stdout)
            if not issues:
                return "No issues found!"

            # Format output
            lines = [f"Found {len(issues)} issue(s):\n"]
            for issue in issues:
                code = issue.get("code", "")
                # E/F codes are errors, W codes are warnings
                severity = "error" if code and code[0] in ("E", "F") else "warning"
                lines.append(
                    f"  {issue['filename']}:{issue['location']['row']}:{issue['location']['column']} "
                    f"[{severity}] {issue['code']}: {issue['message']}"
                )
                if issue.get("fix"):
                    lines.append("    -> Auto-fixable")

            if fix:
                lines.append("\nApplied automatic fixes where possible.")

            return "\n".join(lines)
        else:
            return "No issues found!"

    except json.JSONDecodeError:
        # Fallback to raw output
        return stdout if stdout else "No issues found!"


def _lint_check_javascript(
    path: str,
    fix: bool = False,
    ext: list[str] | None = None,
) -> str:
    """Run ESLint on JavaScript/TypeScript code."""
    # Check for npx (comes with npm)
    if not _check_tool_installed("npx"):
        return "Error: npx is not available. Make sure Node.js/npm is installed."

    try:
        resolved_path = validate_path(path)
    except (PermissionError, ValueError) as e:
        return f"Error: {e}"

    args = ["npx", "eslint", resolved_path]

    if fix:
        args.append("--fix")

    if ext:
        args.extend(["--ext", ",".join(ext)])
    else:
        args.extend(["--ext", ".js,.jsx,.ts,.tsx"])

    args.append("--format=stylish")

    exit_code, stdout, stderr = _run_command(args, timeout=120)

    if "eslint: command not found" in stderr or "Cannot find module" in stderr:
        return "Error: ESLint is not installed. Run: npm install eslint"

    if exit_code == 0:
        return "No issues found!"

    return stdout if stdout else stderr


def _lint_format_python(path: str, check_only: bool = False) -> str:
    """Format Python code using Ruff formatter."""
    if not _check_tool_installed("ruff"):
        return "Error: Ruff is not installed. Install with: pip install ruff"

    try:
        resolved_path = validate_path(path)
    except (PermissionError, ValueError) as e:
        return f"Error: {e}"

    args = ["ruff", "format", resolved_path]

    if check_only:
        args.append("--check")

    exit_code, stdout, stderr = _run_command(args)

    if exit_code == 0:
        if check_only:
            return "All files are properly formatted."
        return stdout if stdout else "Formatting complete."
    else:
        if check_only:
            return f"Formatting issues found:\n{stdout}"
        return f"Error: {stderr}"


def _lint_typecheck_python(
    path: str,
    strict: bool = False,
    ignore_missing_imports: bool = True,
) -> str:
    """Run MyPy type checker on Python code."""
    if not _check_tool_installed("mypy"):
        return "Error: MyPy is not installed. Install with: pip install mypy"

    try:
        resolved_path = validate_path(path)
    except (PermissionError, ValueError) as e:
        return f"Error: {e}"

    args = ["mypy", resolved_path]

    if strict:
        args.append("--strict")
    if ignore_missing_imports:
        args.append("--ignore-missing-imports")

    exit_code, stdout, stderr = _run_command(args, timeout=120)

    if exit_code == 0:
        return "No type errors found!"

    # Parse output
    if stdout:
        lines = stdout.strip().split("\n")
        error_count = len([line for line in lines if ": error:" in line])
        warning_count = len([line for line in lines if ": warning:" in line or ": note:" in line])

        result = f"Found {error_count} error(s), {warning_count} warning(s)/note(s):\n\n{stdout}"
        return result

    return stderr if stderr else "Type checking complete."


def _lint_security(path: str) -> str:
    """Run security-focused linting using Ruff's security rules (S)."""
    if not _check_tool_installed("ruff"):
        return "Error: Ruff is not installed. Install with: pip install ruff"

    try:
        resolved_path = validate_path(path)
    except (PermissionError, ValueError) as e:
        return f"Error: {e}"

    # Use S (bandit) rules for security
    args = [
        "ruff",
        "check",
        resolved_path,
        "--select",
        "S",
        "--output-format",
        "json",
    ]

    exit_code, stdout, stderr = _run_command(args)

    if stderr and "error" in stderr.lower():
        return f"Error: {stderr}"

    try:
        if stdout.strip():
            issues = json.loads(stdout)
            if not issues:
                return "No security issues found!"

            lines = [f"Found {len(issues)} potential security issue(s):\n"]
            for issue in issues:
                lines.append(
                    f"  {issue['filename']}:{issue['location']['row']} "
                    f"[{issue['code']}]: {issue['message']}"
                )

            return "\n".join(lines)
        else:
            return "No security issues found!"

    except json.JSONDecodeError:
        return stdout if stdout else "No security issues found!"


def _lint_complexity(path: str, max_complexity: int = 10) -> str:
    """Analyze code complexity using Ruff's mccabe checker."""
    if not _check_tool_installed("ruff"):
        return "Error: Ruff is not installed. Install with: pip install ruff"

    try:
        resolved_path = validate_path(path)
    except (PermissionError, ValueError) as e:
        return f"Error: {e}"

    # Use C901 (McCabe complexity) rule
    # Note: max_complexity is configured in pyproject.toml/ruff.toml, not CLI
    args = [
        "ruff",
        "check",
        resolved_path,
        "--select",
        "C901",
        "--output-format",
        "json",
    ]

    exit_code, stdout, stderr = _run_command(args)

    if stderr and "error" in stderr.lower():
        return f"Error: {stderr}"

    try:
        if stdout.strip():
            issues = json.loads(stdout)
            if not issues:
                return f"All functions have complexity <= {max_complexity} (default threshold)"

            # Filter issues based on max_complexity parameter
            # Parse complexity from message like "`func` is too complex (15 > 10)"
            filtered_issues = []
            for issue in issues:
                msg = issue.get("message", "")
                # Extract complexity value from message
                import re

                match = re.search(r"\((\d+)\s*>", msg)
                if match:
                    complexity = int(match.group(1))
                    if complexity > max_complexity:
                        filtered_issues.append(issue)
                else:
                    filtered_issues.append(issue)

            if not filtered_issues:
                return f"All functions have complexity <= {max_complexity}"

            lines = [f"Functions with complexity > {max_complexity}:\n"]
            for issue in filtered_issues:
                lines.append(
                    f"  {issue['filename']}:{issue['location']['row']} - {issue['message']}"
                )

            return "\n".join(lines)
        else:
            return f"All functions have complexity <= {max_complexity}"

    except json.JSONDecodeError:
        return stdout if stdout else f"All functions have complexity <= {max_complexity}"


def _lint_summary(path: str) -> str:
    """Get a summary of all lint issues in a project."""
    if not _check_tool_installed("ruff"):
        return "Error: Ruff is not installed. Install with: pip install ruff"

    try:
        resolved_path = validate_path(path)
    except (PermissionError, ValueError) as e:
        return f"Error: {e}"

    args = [
        "ruff",
        "check",
        resolved_path,
        "--output-format",
        "json",
    ]

    exit_code, stdout, stderr = _run_command(args)

    if stderr and "error" in stderr.lower():
        return f"Error: {stderr}"

    try:
        if stdout.strip():
            issues = json.loads(stdout)

            # Group by code prefix
            categories: dict[str, int] = {}
            for issue in issues:
                code = issue.get("code", "Unknown")
                prefix = code[0] if code else "?"

                category_names = {
                    "E": "Style (pycodestyle)",
                    "W": "Warnings",
                    "F": "Errors (Pyflakes)",
                    "I": "Imports (isort)",
                    "N": "Naming",
                    "D": "Docstrings",
                    "S": "Security",
                    "B": "Bugbear",
                    "C": "Complexity",
                }

                cat_name = category_names.get(prefix, f"Other ({prefix})")
                categories[cat_name] = categories.get(cat_name, 0) + 1

            if not categories:
                return "No issues found!"

            lines = [f"Lint Summary for {path}:\n"]
            total = sum(categories.values())

            for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
                bar_len = int(count / total * 20) if total > 0 else 0
                bar = "#" * bar_len + "-" * (20 - bar_len)
                lines.append(f"  {cat:<25} {bar} {count:>4}")

            lines.append(f"\n  {'Total':<25} {'=' * 20} {total:>4}")

            return "\n".join(lines)
        else:
            return "No issues found!"

    except json.JSONDecodeError:
        return stdout if stdout else "No issues found!"


# ========================================
# Unified Lint Tool (Primary Interface)
# ========================================


def lint(
    path: str = ".",
    operation: str = "check",
    language: str | None = None,
    fix: bool = False,
) -> str:
    """
    Unified code checking tool.

    Consolidates 8 lint tools into one with operation parameter:
    - check: Run linter (Ruff for Python, ESLint for JS)
    - format: Format code (Ruff format for Python)
    - typecheck: Type checking (MyPy for Python)
    - security: Security-focused linting (Ruff S rules)
    - complexity: Code complexity analysis (McCabe)
    - summary: Get lint summary by category
    - all: Run all checks

    Args:
        path: File or directory to check
        operation: check | format | typecheck | security | complexity | summary | all
        language: python | javascript | auto (default: auto-detect)
        fix: Automatically fix fixable issues (for check/format operations)

    Returns:
        Lint results

    Examples:
        lint("src/")                             # Check src directory
        lint("src/main.py", "format")            # Format a file
        lint("src/", "typecheck")                # Type check
        lint("src/", "security")                 # Security scan
        lint("src/", "all")                      # Run all checks
        lint("src/", "check", fix=True)          # Check and auto-fix
    """
    # Validate operation
    valid_operations = ("check", "format", "typecheck", "security", "complexity", "summary", "all")
    if operation not in valid_operations:
        return f"Error: Invalid operation '{operation}'. Valid: {', '.join(valid_operations)}"

    # Auto-detect language if not specified
    detected_lang = language if language else _detect_language(path)

    # Handle 'all' operation
    if operation == "all":
        results = []

        # Run check
        results.append("=== Lint Check ===")
        results.append(lint(path, "check", language, fix))
        results.append("")

        # Run format check (not fix)
        if detected_lang == "python":
            results.append("=== Format Check ===")
            results.append(_lint_format_python(path, check_only=True))
            results.append("")

            # Run typecheck
            results.append("=== Type Check ===")
            results.append(_lint_typecheck_python(path))
            results.append("")

            # Run security
            results.append("=== Security Check ===")
            results.append(_lint_security(path))
            results.append("")

            # Run complexity
            results.append("=== Complexity Check ===")
            results.append(_lint_complexity(path))
            results.append("")

        return "\n".join(results)

    # Handle individual operations
    if operation == "check":
        if detected_lang == "python":
            return _lint_check_python(path, fix=fix)
        elif detected_lang == "javascript":
            return _lint_check_javascript(path, fix=fix)
        else:
            # Try both if language not detected
            results = []
            if _check_tool_installed("ruff"):
                results.append("=== Python (Ruff) ===")
                results.append(_lint_check_python(path, fix=fix))
            if _check_tool_installed("npx"):
                results.append("=== JavaScript (ESLint) ===")
                results.append(_lint_check_javascript(path, fix=fix))
            if not results:
                return "No supported files found or no linters available."
            return "\n".join(results)

    elif operation == "format":
        if detected_lang == "javascript":
            return "Error: JavaScript formatting not yet supported. Use Prettier directly."
        # Default to Python formatting
        return _lint_format_python(path, check_only=not fix)

    elif operation == "typecheck":
        if detected_lang == "javascript":
            return "Error: TypeScript type checking not yet supported. Use tsc directly."
        return _lint_typecheck_python(path)

    elif operation == "security":
        return _lint_security(path)

    elif operation == "complexity":
        return _lint_complexity(path)

    elif operation == "summary":
        return _lint_summary(path)

    return f"Error: Unknown operation '{operation}'"

