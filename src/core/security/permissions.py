"""
Permission System

Fine-grained permission control for tools and operations.

Features:
- Rule-based permissions
- Path-based access control
- Tool allow/deny lists
- HITL approval integration
- Audit logging
"""

import fnmatch
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PermissionLevel(Enum):
    """Permission levels."""

    ALLOW = "allow"  # Always allowed
    DENY = "deny"  # Always denied
    ASK = "ask"  # Require user approval (HITL)
    WARN = "warn"  # Warn but allow


class OperationType(Enum):
    """Types of operations."""

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    NETWORK = "network"


@dataclass
class PermissionRule:
    """A permission rule."""

    name: str
    description: str
    level: PermissionLevel
    tools: list[str] = field(default_factory=list)  # Tool names or patterns
    paths: list[str] = field(default_factory=list)  # Path patterns
    operations: list[OperationType] = field(default_factory=list)
    conditions: dict[str, Any] = field(default_factory=dict)  # Additional conditions
    priority: int = 0  # Higher = evaluated first

    def matches_tool(self, tool: str) -> bool:
        """Check if rule matches a tool."""
        if not self.tools:
            return True  # No tool restriction
        return any(fnmatch.fnmatch(tool, pattern) for pattern in self.tools)

    def matches_path(self, path: str) -> bool:
        """Check if rule matches a path."""
        if not self.paths:
            return True  # No path restriction
        return any(fnmatch.fnmatch(path, pattern) for pattern in self.paths)

    def matches_operation(self, operation: OperationType) -> bool:
        """Check if rule matches an operation type."""
        if not self.operations:
            return True  # No operation restriction
        return operation in self.operations

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "level": self.level.value,
            "tools": self.tools,
            "paths": self.paths,
            "operations": [o.value for o in self.operations],
            "priority": self.priority,
        }


@dataclass
class PermissionRequest:
    """A request for permission check."""

    tool: str
    operation: OperationType | None = None
    path: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class PermissionResult:
    """Result of a permission check."""

    level: PermissionLevel
    rule: PermissionRule | None = None
    message: str = ""
    requires_approval: bool = False

    @property
    def is_allowed(self) -> bool:
        return self.level in (PermissionLevel.ALLOW, PermissionLevel.WARN)


@dataclass
class AuditEntry:
    """Audit log entry."""

    timestamp: float
    tool: str
    operation: OperationType
    path: str | None
    result: PermissionLevel
    approved_by: str | None = None
    rule_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "tool": self.tool,
            "operation": self.operation.value,
            "path": self.path,
            "result": self.result.value,
            "approved_by": self.approved_by,
            "rule_name": self.rule_name,
        }


# Default permission rules
DEFAULT_RULES: list[PermissionRule] = [
    # Read operations are generally safe
    PermissionRule(
        name="allow_read",
        description="Allow read operations",
        level=PermissionLevel.ALLOW,
        operations=[OperationType.READ],
        priority=0,
    ),
    # Deny sensitive paths
    PermissionRule(
        name="deny_sensitive",
        description="Deny access to sensitive files",
        level=PermissionLevel.DENY,
        paths=[
            # Secrets and credentials
            "**/.env",
            "**/.env.*",
            "**/secrets.*",
            "**/credentials.*",
            "**/*.pem",
            "**/*.key",
            "**/*.p12",
            "**/*.pfx",
            "**/*.jks",
            "**/id_rsa*",
            "**/id_ed25519*",
            "**/.ssh/*",
            # Cloud provider credentials
            "**/.aws/*",
            "**/.azure/*",
            "**/.gcloud/*",
            "**/.config/gcloud/*",
            # Container/orchestration credentials
            "**/.docker/config.json",
            "**/.kube/config",
            "**/.kube/*.yaml",
            # Token and password files
            "**/token",
            "**/token.*",
            "**/*password*",
            "**/*secret*",
            # Package manager tokens
            "**/.npmrc",
            "**/.pypirc",
            "**/.gem/credentials",
            # Database credentials
            "**/.pgpass",
            "**/.my.cnf",
            "**/.mongocli.yaml",
        ],
        priority=100,
    ),
    # Warn on system paths
    PermissionRule(
        name="warn_system",
        description="Warn when accessing system paths",
        level=PermissionLevel.WARN,
        paths=[
            "/etc/*",
            "/usr/*",
            "/var/*",
            "/root/*",
            "C:\\Windows\\*",
            "C:\\Program Files*",
        ],
        operations=[OperationType.WRITE, OperationType.DELETE],
        priority=90,
    ),
    # Ask for destructive operations
    PermissionRule(
        name="ask_destructive",
        description="Require approval for destructive operations",
        level=PermissionLevel.ASK,
        operations=[OperationType.DELETE],
        priority=80,
    ),
    # Ask for network operations
    PermissionRule(
        name="ask_network",
        description="Require approval for network operations",
        level=PermissionLevel.ASK,
        tools=["run"],
        operations=[OperationType.NETWORK],
        priority=70,
    ),
    # Default allow for write in plan mode is actually deny
    PermissionRule(
        name="default_allow",
        description="Default permission",
        level=PermissionLevel.ALLOW,
        priority=-100,
    ),
]


class PermissionManager:
    """
    Manages permissions for tools and operations.

    Usage:
        pm = PermissionManager()

        # Check permission
        request = PermissionRequest(
            tool="file_write",
            operation=OperationType.WRITE,
            path="/path/to/file"
        )
        result = pm.check(request)

        if not result.is_allowed:
            if result.requires_approval:
                # Ask user
                approved = await ask_user()
                if approved:
                    pm.approve(request)
            else:
                # Denied
                raise PermissionError(result.message)

        # Add custom rule
        pm.add_rule(PermissionRule(...))
    """

    # Tool to operation mapping
    TOOL_OPERATIONS = {
        "read": OperationType.READ,
        "search": OperationType.READ,
        "write": OperationType.WRITE,
        "run": OperationType.EXECUTE,
        "web_search": OperationType.NETWORK,
        "web_fetch": OperationType.NETWORK,
        "memory_get": OperationType.READ,
        "memory_search": OperationType.READ,
        "memory_list": OperationType.READ,
        "memory_put": OperationType.WRITE,
        "memory_delete": OperationType.WRITE,
    }

    def __init__(
        self,
        rules: list[PermissionRule] | None = None,
        mode: str = "build",
        approval_callback: Callable[[PermissionRequest], bool] | None = None,
    ):
        """
        Initialize permission manager.

        Args:
            rules: Custom rules (uses defaults if None)
            mode: Current mode (plan/build)
            approval_callback: Function to call for HITL approval
        """
        self._rules = rules or DEFAULT_RULES.copy()
        self._mode = mode
        self._approval_callback = approval_callback
        self._approved_requests: set[str] = set()  # Session approvals
        self._approval_timestamps: dict[str, float] = {}  # Expiry tracking
        self._approval_ttl: float = 300.0  # Approvals expire after 5 minutes
        self._audit_log: list[AuditEntry] = []
        self._max_audit_entries = 1000

    def add_rule(self, rule: PermissionRule):
        """Add a permission rule.

        Rules with level=DENY and high priority cannot be overridden
        by rules with level=ALLOW loaded from external files.
        """
        existing = next((r for r in self._rules if r.name == rule.name), None)
        if existing and existing.level == PermissionLevel.DENY and rule.level == PermissionLevel.ALLOW:
            logger.warning(
                "Refusing to override DENY rule '%s' with ALLOW rule from external source",
                rule.name,
            )
            return
        self._rules.append(rule)
        # Re-sort by priority
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def remove_rule(self, name: str):
        """Remove a rule by name."""
        self._rules = [r for r in self._rules if r.name != name]

    def set_mode(self, mode: str):
        """Set the current mode."""
        self._mode = mode

    def check(self, request: PermissionRequest) -> PermissionResult:
        """
        Check permission for a request.

        Args:
            request: Permission request

        Returns:
            PermissionResult
        """
        # Get operation type if not specified
        if request.operation is None:
            request.operation = self.TOOL_OPERATIONS.get(request.tool, OperationType.READ)

        # Mode-based restrictions
        if self._mode == "plan" and request.operation in (
            OperationType.WRITE,
            OperationType.DELETE,
            OperationType.EXECUTE,
        ):
            return PermissionResult(
                level=PermissionLevel.DENY,
                message="Write operations not allowed in plan mode",
            )

        # Check if already approved in this session (with expiry)
        request_key = self._make_request_key(request)
        if request_key in self._approved_requests:
            approved_at = self._approval_timestamps.get(request_key, 0)
            if time.time() - approved_at > self._approval_ttl:
                self._approved_requests.discard(request_key)
                self._approval_timestamps.pop(request_key, None)
            else:
                return PermissionResult(
                    level=PermissionLevel.ALLOW,
                    message="Previously approved",
                )

        # Find matching rule (highest priority first)
        for rule in self._rules:
            if not rule.matches_tool(request.tool):
                continue
            if request.path and not rule.matches_path(request.path):
                continue
            if not rule.matches_operation(request.operation):
                continue

            # Found matching rule
            result = PermissionResult(
                level=rule.level,
                rule=rule,
                message=rule.description,
                requires_approval=rule.level == PermissionLevel.ASK,
            )

            # Log the check
            self._audit(request, result)

            return result

        # Default allow
        result = PermissionResult(
            level=PermissionLevel.ALLOW,
            message="No matching rules, allowing",
        )
        self._audit(request, result)
        return result

    def approve(self, request: PermissionRequest, approver: str = "user"):
        """
        Approve a request for this session.

        Args:
            request: The request to approve
            approver: Who approved it
        """
        request_key = self._make_request_key(request)
        self._approved_requests.add(request_key)
        self._approval_timestamps[request_key] = time.time()

        # Update audit log
        entry = AuditEntry(
            timestamp=time.time(),
            tool=request.tool,
            operation=request.operation,
            path=request.path,
            result=PermissionLevel.ALLOW,
            approved_by=approver,
        )
        self._add_audit_entry(entry)

    def _make_request_key(self, request: PermissionRequest) -> str:
        """Create a unique key for a request."""
        return f"{request.tool}:{request.operation.value}:{request.path}"

    def _audit(self, request: PermissionRequest, result: PermissionResult):
        """Add audit entry."""
        entry = AuditEntry(
            timestamp=time.time(),
            tool=request.tool,
            operation=request.operation,
            path=request.path,
            result=result.level,
            rule_name=result.rule.name if result.rule else None,
        )
        self._add_audit_entry(entry)

    def _add_audit_entry(self, entry: AuditEntry):
        """Add entry to audit log."""
        self._audit_log.append(entry)
        # Trim if too large
        if len(self._audit_log) > self._max_audit_entries:
            self._audit_log = self._audit_log[-self._max_audit_entries // 2 :]

    def get_audit_log(self, limit: int = 50) -> list[dict]:
        """Get recent audit entries."""
        return [e.to_dict() for e in self._audit_log[-limit:]]

    def get_rules(self) -> list[dict]:
        """Get all rules."""
        return [r.to_dict() for r in self._rules]

    def clear_approvals(self):
        """Clear all session approvals."""
        self._approved_requests.clear()

    def load_rules_from_file(self, path: Path) -> bool:
        """
        Load rules from a JSON file.

        Args:
            path: Path to rules file

        Returns:
            True if loaded successfully
        """
        try:
            data = json.loads(path.read_text(encoding="utf-8"))

            for rule_data in data.get("rules", []):
                rule = PermissionRule(
                    name=rule_data["name"],
                    description=rule_data.get("description", ""),
                    level=PermissionLevel(rule_data["level"]),
                    tools=rule_data.get("tools", []),
                    paths=rule_data.get("paths", []),
                    operations=[OperationType(o) for o in rule_data.get("operations", [])],
                    priority=rule_data.get("priority", 0),
                )
                self.add_rule(rule)

            logger.info("Loaded %s permission rules", len(data.get('rules', [])))
            return True

        except Exception as e:
            logger.error("Failed to load permission rules: %s", e)
            return False

    def get_summary(self) -> dict[str, Any]:
        """Get permission system summary."""
        return {
            "mode": self._mode,
            "total_rules": len(self._rules),
            "approved_count": len(self._approved_requests),
            "audit_entries": len(self._audit_log),
        }
