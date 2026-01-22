"""
Code Quality and Linting MCP Server

Provides code analysis, linting, and type checking capabilities.
Similar to Claude Code's ReadLints tool but more comprehensive.

Features:
- Ruff linting (Python)
- MyPy type checking (Python)
- ESLint (JavaScript/TypeScript)
- General code quality metrics
"""

import json
import logging
import os
import subprocess
from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP

from src.core.security import validate_path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("PolymathLint")


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


# ========================================
# Python Linting (Ruff)
# ========================================


@mcp.tool()
def lint_python_ruff(
    path: str = ".",
    fix: bool = False,
    select: list[str] | None = None,
    ignore: list[str] | None = None,
) -> str:
    """
    Run Ruff linter on Python code.

    Ruff is an extremely fast Python linter that covers:
    - pycodestyle (E, W)
    - Pyflakes (F)
    - isort (I)
    - pydocstyle (D)
    - and many more...

    Args:
        path: File or directory to lint
        fix: Automatically fix fixable issues
        select: Rule codes to enable (e.g., ["E", "F", "I"])
        ignore: Rule codes to ignore (e.g., ["E501"])

    Returns:
        Lint results with issues found

    Example:
        lint_python_ruff("src/")
        lint_python_ruff("src/main.py", fix=True)
    """
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
                return "✅ No issues found!"

            # Format output
            lines = [f"Found {len(issues)} issue(s):\n"]
            for issue in issues:
                severity = "error" if issue.get("fix") is None else "warning"
                lines.append(
                    f"  {issue['filename']}:{issue['location']['row']}:{issue['location']['column']} "
                    f"[{severity}] {issue['code']}: {issue['message']}"
                )
                if issue.get("fix"):
                    lines.append("    ↳ Auto-fixable")

            if fix:
                lines.append("\n✨ Applied automatic fixes where possible.")

            return "\n".join(lines)
        else:
            return "✅ No issues found!"

    except json.JSONDecodeError:
        # Fallback to raw output
        return stdout if stdout else "✅ No issues found!"


@mcp.tool()
def format_python_ruff(
    path: str = ".",
    check_only: bool = False,
) -> str:
    """
    Format Python code using Ruff formatter.

    Args:
        path: File or directory to format
        check_only: Only check formatting without making changes

    Returns:
        Formatting result
    """
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
            return "✅ All files are properly formatted."
        return stdout if stdout else "✅ Formatting complete."
    else:
        if check_only:
            return f"❌ Formatting issues found:\n{stdout}"
        return f"Error: {stderr}"


# ========================================
# Python Type Checking (MyPy)
# ========================================


@mcp.tool()
def typecheck_python_mypy(
    path: str = ".",
    strict: bool = False,
    ignore_missing_imports: bool = True,
) -> str:
    """
    Run MyPy type checker on Python code.

    Args:
        path: File or directory to check
        strict: Enable strict mode (more rigorous checking)
        ignore_missing_imports: Ignore imports that can't be resolved

    Returns:
        Type checking results

    Example:
        typecheck_python_mypy("src/")
    """
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
        return "✅ No type errors found!"

    # Parse output
    if stdout:
        lines = stdout.strip().split("\n")
        error_count = len([l for l in lines if ": error:" in l])
        warning_count = len([l for l in lines if ": warning:" in l or ": note:" in l])

        result = f"Found {error_count} error(s), {warning_count} warning(s)/note(s):\n\n{stdout}"
        return result

    return stderr if stderr else "Type checking complete."


# ========================================
# JavaScript/TypeScript Linting (ESLint)
# ========================================


@mcp.tool()
def lint_javascript_eslint(
    path: str = ".",
    fix: bool = False,
    ext: list[str] | None = None,
) -> str:
    """
    Run ESLint on JavaScript/TypeScript code.

    Args:
        path: File or directory to lint
        fix: Automatically fix fixable issues
        ext: File extensions to lint (default: [".js", ".jsx", ".ts", ".tsx"])

    Returns:
        Lint results

    Requires:
        ESLint must be installed in the project (npm install eslint)
    """
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
        return "✅ No issues found!"

    return stdout if stdout else stderr


# ========================================
# Multi-Language Lint
# ========================================


@mcp.tool()
def lint_all(
    path: str = ".",
    fix: bool = False,
) -> str:
    """
    Run all available linters on a path.

    Automatically detects file types and runs appropriate linters:
    - Python: Ruff
    - JavaScript/TypeScript: ESLint (if available)

    Args:
        path: File or directory to lint
        fix: Automatically fix fixable issues

    Returns:
        Combined lint results from all linters
    """
    results = []

    # Check for Python files
    try:
        resolved_path = validate_path(path)
    except (PermissionError, ValueError) as e:
        return f"Error: {e}"

    has_python = False
    has_js = False

    if os.path.isfile(resolved_path):
        if resolved_path.endswith(".py"):
            has_python = True
        elif resolved_path.endswith((".js", ".jsx", ".ts", ".tsx")):
            has_js = True
    else:
        # Check directory contents
        for root, _, files in os.walk(resolved_path):
            for f in files:
                if f.endswith(".py"):
                    has_python = True
                elif f.endswith((".js", ".jsx", ".ts", ".tsx")):
                    has_js = True
            if has_python and has_js:
                break

    # Run Python linter
    if has_python and _check_tool_installed("ruff"):
        results.append("=== Python (Ruff) ===")
        results.append(lint_python_ruff(path, fix=fix))
        results.append("")

    # Run JS/TS linter
    if has_js and _check_tool_installed("npx"):
        results.append("=== JavaScript/TypeScript (ESLint) ===")
        results.append(lint_javascript_eslint(path, fix=fix))
        results.append("")

    if not results:
        return "No supported files found or no linters available."

    return "\n".join(results)


# ========================================
# Code Quality Metrics
# ========================================


@mcp.tool()
def code_complexity(
    path: str = ".",
    max_complexity: int = 10,
) -> str:
    """
    Analyze code complexity using Ruff's mccabe checker.

    Args:
        path: File or directory to analyze
        max_complexity: Threshold for flagging complex functions (default: 10)

    Returns:
        Functions that exceed the complexity threshold
    """
    if not _check_tool_installed("ruff"):
        return "Error: Ruff is not installed. Install with: pip install ruff"

    try:
        resolved_path = validate_path(path)
    except (PermissionError, ValueError) as e:
        return f"Error: {e}"

    # Use C901 (McCabe complexity) rule
    args = [
        "ruff",
        "check",
        resolved_path,
        "--select",
        "C901",
        f"--max-complexity={max_complexity}",
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
                return f"✅ All functions have complexity ≤ {max_complexity}"

            lines = [f"Functions with complexity > {max_complexity}:\n"]
            for issue in issues:
                lines.append(
                    f"  {issue['filename']}:{issue['location']['row']} - {issue['message']}"
                )

            return "\n".join(lines)
        else:
            return f"✅ All functions have complexity ≤ {max_complexity}"

    except json.JSONDecodeError:
        return stdout if stdout else f"✅ All functions have complexity ≤ {max_complexity}"


@mcp.tool()
def check_security(path: str = ".") -> str:
    """
    Run security-focused linting using Ruff's security rules (S).

    Checks for common security issues:
    - Hardcoded passwords
    - SQL injection vulnerabilities
    - Use of eval/exec
    - Insecure random number generation
    - And more...

    Args:
        path: File or directory to check

    Returns:
        Security issues found
    """
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
                return "✅ No security issues found!"

            lines = [f"⚠️ Found {len(issues)} potential security issue(s):\n"]
            for issue in issues:
                lines.append(
                    f"  {issue['filename']}:{issue['location']['row']} "
                    f"[{issue['code']}]: {issue['message']}"
                )

            return "\n".join(lines)
        else:
            return "✅ No security issues found!"

    except json.JSONDecodeError:
        return stdout if stdout else "✅ No security issues found!"


@mcp.tool()
def get_lint_summary(path: str = ".") -> str:
    """
    Get a summary of all lint issues in a project.

    Args:
        path: Project root path

    Returns:
        Summary with counts by category
    """
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
        "--statistics",
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
                return "✅ No issues found!"

            lines = [f"Lint Summary for {path}:\n"]
            total = sum(categories.values())

            for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
                bar_len = int(count / total * 20) if total > 0 else 0
                bar = "█" * bar_len + "░" * (20 - bar_len)
                lines.append(f"  {cat:<25} {bar} {count:>4}")

            lines.append(f"\n  {'Total':<25} {'=' * 20} {total:>4}")

            return "\n".join(lines)
        else:
            return "✅ No issues found!"

    except json.JSONDecodeError:
        return stdout if stdout else "✅ No issues found!"


if __name__ == "__main__":
    mcp.run()
