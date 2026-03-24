"""
Health Check System (Doctor)

Diagnoses installation and configuration issues.

Features:
- Environment verification
- Dependency checking
- Configuration validation
- Network connectivity tests
- API key verification
"""

import importlib
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from src.core.config.config import load_config

logger = logging.getLogger(__name__)


class CheckStatus(Enum):
    """Health check status."""

    OK = "ok"
    WARNING = "warning"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class CheckResult:
    """Result of a health check."""

    name: str
    status: CheckStatus
    message: str
    details: str = ""
    fix_suggestion: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "fix_suggestion": self.fix_suggestion,
        }


class Doctor:
    """
    Health check and diagnostic system.

    Usage:
        doctor = Doctor()

        # Run all checks
        results = doctor.run_all_checks()

        # Run specific check
        result = doctor.check_api_key()

        # Get summary
        summary = doctor.get_summary(results)
    """

    def __init__(self, project_dir: Path | None = None):
        """
        Initialize doctor.

        Args:
            project_dir: Project directory to check
        """
        self.project_dir = project_dir or Path.cwd()

    def run_all_checks(self) -> list[CheckResult]:
        """Run all health checks."""
        checks = [
            self.check_python_version,
            self.check_api_key,
            self.check_dependencies,
            self.check_optional_dependencies,
            self.check_git,
            self.check_project_config,
            self.check_permissions,
            self.check_disk_space,
            self.check_network,
        ]

        results = []
        for check in checks:
            try:
                result = check()
                results.append(result)
            except Exception as e:
                results.append(
                    CheckResult(
                        name=check.__name__,
                        status=CheckStatus.ERROR,
                        message=f"Check failed: {e}",
                    )
                )

        return results

    def check_python_version(self) -> CheckResult:
        """Check Python version."""
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"

        if version.major < 3 or (version.major == 3 and version.minor < 10):
            return CheckResult(
                name="Python Version",
                status=CheckStatus.ERROR,
                message=f"Python {version_str} is too old",
                details="The agent requires Python 3.10 or higher",
                fix_suggestion="Upgrade Python to 3.10+",
            )

        return CheckResult(
            name="Python Version",
            status=CheckStatus.OK,
            message=f"Python {version_str}",
        )

    def check_api_key(self) -> CheckResult:
        """Check model credentials from project config."""
        config = load_config(validate=False)
        gateway_url = config.get("gateway_url")
        gateway_key = config.get("gateway_key")
        provider_keys = {
            "google_api_key": config.get("google_api_key"),
            "openai_api_key": config.get("openai_api_key"),
            "anthropic_api_key": config.get("anthropic_api_key"),
        }

        if gateway_url:
            if not gateway_key:
                return CheckResult(
                    name="Model Credentials",
                    status=CheckStatus.ERROR,
                    message="gateway_url is set but gateway_key is missing",
                    details="Gateway mode requires both gateway_url and gateway_key",
                    fix_suggestion="Set gateway_key in .agent/config.json",
                )

            return CheckResult(
                name="Model Credentials",
                status=CheckStatus.OK,
                message="Gateway credentials configured",
                details=f"Gateway: {gateway_url}",
            )

        configured = [name for name, value in provider_keys.items() if value]
        if not configured:
            return CheckResult(
                name="Model Credentials",
                status=CheckStatus.ERROR,
                message="No model credentials configured",
                details="No provider keys found in .agent/config.json",
                fix_suggestion="Set google_api_key, openai_api_key, or anthropic_api_key in .agent/config.json",
            )

        return CheckResult(
            name="Model Credentials",
            status=CheckStatus.OK,
            message=f"Configured: {', '.join(configured)}",
        )

    def check_dependencies(self) -> CheckResult:
        """Check required dependencies."""
        required = [
            ("google.genai", "google-genai"),
            ("typer", "typer"),
            ("rich", "rich"),
            ("pydantic", "pydantic"),
            ("dotenv", "python-dotenv"),
        ]

        missing = []
        for module, package in required:
            try:
                importlib.import_module(module)
            except ImportError:
                missing.append(package)

        if missing:
            return CheckResult(
                name="Dependencies",
                status=CheckStatus.ERROR,
                message=f"Missing packages: {', '.join(missing)}",
                fix_suggestion=f"pip install {' '.join(missing)}",
            )

        return CheckResult(
            name="Dependencies",
            status=CheckStatus.OK,
            message="All required packages installed",
        )

    def check_optional_dependencies(self) -> CheckResult:
        """Check optional dependencies."""
        optional = [
            ("chromadb", "chromadb", "Vector memory"),
            ("playwright", "playwright", "Browser automation"),
            ("pdfplumber", "pdfplumber", "PDF parsing"),
        ]

        missing = []
        installed = []

        for module, package, feature in optional:
            try:
                importlib.import_module(module)
                installed.append(feature)
            except ImportError:
                missing.append(f"{package} ({feature})")

        if missing:
            return CheckResult(
                name="Optional Dependencies",
                status=CheckStatus.WARNING,
                message="Some features unavailable",
                details=f"Missing: {', '.join(missing)}",
                fix_suggestion="Install with: pip install "
                + " ".join(m.split(" ")[0] for m in missing),
            )

        return CheckResult(
            name="Optional Dependencies",
            status=CheckStatus.OK,
            message="All optional packages installed",
            details=f"Features: {', '.join(installed)}",
        )

    def check_git(self) -> CheckResult:
        """Check Git installation and repository."""
        # Check git is installed
        if not shutil.which("git"):
            return CheckResult(
                name="Git",
                status=CheckStatus.WARNING,
                message="Git not found",
                fix_suggestion="Install Git for version control features",
            )

        # Check if in git repo
        git_dir = self.project_dir / ".git"
        if not git_dir.exists():
            return CheckResult(
                name="Git",
                status=CheckStatus.WARNING,
                message="Not a Git repository",
                details=str(self.project_dir),
                fix_suggestion="Run 'git init' to initialize",
            )

        # Get git version
        try:
            result = subprocess.run(["git", "--version"], capture_output=True, text=True)
            version = result.stdout.strip()
        except Exception:
            version = "unknown"

        return CheckResult(
            name="Git",
            status=CheckStatus.OK,
            message=f"{version}",
            details=f"Repository: {self.project_dir}",
        )

    def check_project_config(self) -> CheckResult:
        """Check project configuration."""
        config_paths = [
            self.project_dir / ".agent" / "config.json",
            self.project_dir / "AGENTS.md",
        ]

        found = []
        missing = []

        for path in config_paths:
            if path.exists():
                found.append(path.name)
            else:
                missing.append(path.name)

        if not found:
            return CheckResult(
                name="Project Config",
                status=CheckStatus.WARNING,
                message="No configuration found",
                details="Project may need initialization",
                fix_suggestion="Run /init to create AGENTS.md",
            )

        status = CheckStatus.OK if not missing else CheckStatus.WARNING
        return CheckResult(
            name="Project Config",
            status=status,
            message=f"Found: {', '.join(found)}",
            details=f"Missing: {', '.join(missing)}" if missing else "",
        )

    def check_permissions(self) -> CheckResult:
        """Check file system permissions."""
        test_paths = [
            self.project_dir,
            self.project_dir / ".agent",
        ]

        issues = []

        for path in test_paths:
            if path.exists():
                if not os.access(path, os.R_OK):
                    issues.append(f"Cannot read: {path}")
                if not os.access(path, os.W_OK):
                    issues.append(f"Cannot write: {path}")

        if issues:
            return CheckResult(
                name="Permissions",
                status=CheckStatus.ERROR,
                message="Permission issues detected",
                details="\n".join(issues),
                fix_suggestion="Check directory permissions",
            )

        return CheckResult(
            name="Permissions",
            status=CheckStatus.OK,
            message="File system permissions OK",
        )

    def check_disk_space(self) -> CheckResult:
        """Check available disk space."""
        try:
            import shutil

            usage = shutil.disk_usage(self.project_dir)
            free_gb = usage.free / (1024**3)

            if free_gb < 1:
                return CheckResult(
                    name="Disk Space",
                    status=CheckStatus.ERROR,
                    message=f"Low disk space: {free_gb:.1f} GB",
                    fix_suggestion="Free up disk space",
                )

            if free_gb < 5:
                return CheckResult(
                    name="Disk Space",
                    status=CheckStatus.WARNING,
                    message=f"Disk space low: {free_gb:.1f} GB free",
                )

            return CheckResult(
                name="Disk Space",
                status=CheckStatus.OK,
                message=f"{free_gb:.1f} GB free",
            )

        except Exception as e:
            return CheckResult(
                name="Disk Space",
                status=CheckStatus.SKIPPED,
                message=f"Could not check: {e}",
            )

    def check_network(self) -> CheckResult:
        """Check network connectivity."""
        try:
            import socket

            # Try to connect to Google API
            socket.setdefaulttimeout(5)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(
                ("generativelanguage.googleapis.com", 443)
            )

            return CheckResult(
                name="Network",
                status=CheckStatus.OK,
                message="Connected to Google API",
            )

        except TimeoutError:
            return CheckResult(
                name="Network",
                status=CheckStatus.ERROR,
                message="Connection timeout",
                fix_suggestion="Check your internet connection",
            )

        except OSError as e:
            return CheckResult(
                name="Network",
                status=CheckStatus.ERROR,
                message=f"Connection failed: {e}",
                fix_suggestion="Check firewall and proxy settings",
            )

    def get_summary(self, results: list[CheckResult]) -> dict[str, Any]:
        """Get summary of check results."""
        ok_count = sum(1 for r in results if r.status == CheckStatus.OK)
        warning_count = sum(1 for r in results if r.status == CheckStatus.WARNING)
        error_count = sum(1 for r in results if r.status == CheckStatus.ERROR)

        overall = CheckStatus.OK
        if warning_count > 0:
            overall = CheckStatus.WARNING
        if error_count > 0:
            overall = CheckStatus.ERROR

        return {
            "overall": overall.value,
            "total": len(results),
            "ok": ok_count,
            "warnings": warning_count,
            "errors": error_count,
            "results": [r.to_dict() for r in results],
        }

    def format_results(self, results: list[CheckResult]) -> str:
        """Format results for display."""
        lines = ["", "=== Health Check Results ===", ""]

        status_icons = {
            CheckStatus.OK: "✅",
            CheckStatus.WARNING: "⚠️",
            CheckStatus.ERROR: "❌",
            CheckStatus.SKIPPED: "⏭️",
        }

        for result in results:
            icon = status_icons.get(result.status, "?")
            lines.append(f"{icon} {result.name}: {result.message}")

            if result.details:
                lines.append(f"   {result.details}")

            if result.fix_suggestion and result.status in (
                CheckStatus.ERROR,
                CheckStatus.WARNING,
            ):
                lines.append(f"   💡 {result.fix_suggestion}")

        # Summary
        summary = self.get_summary(results)
        lines.append("")
        lines.append(
            f"Summary: {summary['ok']} OK, "
            f"{summary['warnings']} warnings, "
            f"{summary['errors']} errors"
        )

        return "\n".join(lines)
