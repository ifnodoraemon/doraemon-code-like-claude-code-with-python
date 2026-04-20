import json
import time
from pathlib import Path

import pytest

from src.core.security.permissions import (
    AuditEntry,
    OperationType,
    PermissionLevel,
    PermissionManager,
    PermissionRequest,
    PermissionResult,
    PermissionRule,
)


class TestPermissionLevel:
    def test_values(self):
        assert PermissionLevel.ALLOW.value == "allow"
        assert PermissionLevel.DENY.value == "deny"
        assert PermissionLevel.ASK.value == "ask"
        assert PermissionLevel.WARN.value == "warn"


class TestOperationType:
    def test_values(self):
        assert OperationType.READ.value == "read"
        assert OperationType.WRITE.value == "write"
        assert OperationType.EXECUTE.value == "execute"
        assert OperationType.DELETE.value == "delete"
        assert OperationType.NETWORK.value == "network"


class TestPermissionRule:
    def test_matches_tool_no_restriction(self):
        rule = PermissionRule(name="test", description="t", level=PermissionLevel.ALLOW)
        assert rule.matches_tool("anything") is True

    def test_matches_tool_with_pattern(self):
        rule = PermissionRule(
            name="test", description="t", level=PermissionLevel.ALLOW, tools=["file_*"]
        )
        assert rule.matches_tool("file_write") is True
        assert rule.matches_tool("run") is False

    def test_matches_path_no_restriction(self):
        rule = PermissionRule(name="test", description="t", level=PermissionLevel.ALLOW)
        assert rule.matches_path("/any/path") is True

    def test_matches_path_with_pattern(self):
        rule = PermissionRule(
            name="test", description="t", level=PermissionLevel.ALLOW, paths=["/etc/*"]
        )
        assert rule.matches_path("/etc/passwd") is True
        assert rule.matches_path("/home/user/file") is False

    def test_matches_operation_no_restriction(self):
        rule = PermissionRule(name="test", description="t", level=PermissionLevel.ALLOW)
        assert rule.matches_operation(OperationType.DELETE) is True

    def test_matches_operation_with_list(self):
        rule = PermissionRule(
            name="test",
            description="t",
            level=PermissionLevel.ALLOW,
            operations=[OperationType.READ],
        )
        assert rule.matches_operation(OperationType.READ) is True
        assert rule.matches_operation(OperationType.WRITE) is False

    def test_to_dict(self):
        rule = PermissionRule(
            name="test",
            description="desc",
            level=PermissionLevel.DENY,
            tools=["run"],
            paths=["/secret"],
            operations=[OperationType.EXECUTE],
            priority=10,
        )
        d = rule.to_dict()
        assert d["name"] == "test"
        assert d["level"] == "deny"
        assert d["operations"] == ["execute"]
        assert d["priority"] == 10


class TestPermissionResult:
    def test_is_allowed_allow(self):
        result = PermissionResult(level=PermissionLevel.ALLOW)
        assert result.is_allowed is True

    def test_is_allowed_warn(self):
        result = PermissionResult(level=PermissionLevel.WARN)
        assert result.is_allowed is True

    def test_is_allowed_deny(self):
        result = PermissionResult(level=PermissionLevel.DENY)
        assert result.is_allowed is False

    def test_is_allowed_ask(self):
        result = PermissionResult(level=PermissionLevel.ASK)
        assert result.is_allowed is False


class TestAuditEntry:
    def test_to_dict(self):
        entry = AuditEntry(
            timestamp=1000.0,
            tool="write",
            operation=OperationType.WRITE,
            path="/tmp/file",
            result=PermissionLevel.ALLOW,
            approved_by="user",
            rule_name="rule1",
        )
        d = entry.to_dict()
        assert d["timestamp"] == 1000.0
        assert d["operation"] == "write"
        assert d["result"] == "allow"
        assert d["approved_by"] == "user"


class TestPermissionManager:
    def test_check_default_allow(self):
        pm = PermissionManager(mode="build")
        result = pm.check(PermissionRequest(tool="search", operation=OperationType.READ))
        assert result.is_allowed is True

    def test_check_plan_mode_blocks_write(self):
        pm = PermissionManager(mode="plan")
        result = pm.check(PermissionRequest(tool="write", operation=OperationType.WRITE))
        assert result.is_allowed is False
        assert "plan mode" in result.message.lower()

    def test_check_plan_mode_blocks_execute(self):
        pm = PermissionManager(mode="plan")
        result = pm.check(PermissionRequest(tool="run", operation=OperationType.EXECUTE))
        assert result.is_allowed is False

    def test_check_plan_mode_allows_read(self):
        pm = PermissionManager(mode="plan")
        result = pm.check(PermissionRequest(tool="read", operation=OperationType.READ))
        assert result.is_allowed is True

    def test_check_sensitive_path_denied(self):
        pm = PermissionManager(mode="build")
        result = pm.check(
            PermissionRequest(
                tool="write",
                operation=OperationType.WRITE,
                path="/home/user/.env",
            )
        )
        assert result.is_allowed is False
        assert result.level == PermissionLevel.DENY

    def test_check_delete_requires_approval(self):
        pm = PermissionManager(mode="build")
        result = pm.check(
            PermissionRequest(tool="rm", operation=OperationType.DELETE, path="/tmp/scratch/file")
        )
        assert result.requires_approval is True
        assert result.level == PermissionLevel.ASK

    def test_add_rule(self):
        pm = PermissionManager(mode="build")
        custom_rule = PermissionRule(
            name="custom_deny",
            description="Deny test tool",
            level=PermissionLevel.DENY,
            tools=["test_tool"],
            priority=200,
        )
        pm.add_rule(custom_rule)
        result = pm.check(PermissionRequest(tool="test_tool", operation=OperationType.READ))
        assert result.is_allowed is False

    def test_remove_rule(self):
        pm = PermissionManager(mode="build")
        pm.remove_rule("deny_sensitive")
        result = pm.check(
            PermissionRequest(
                tool="read",
                operation=OperationType.READ,
                path="/home/user/.env",
            )
        )
        assert result.is_allowed is True

    def test_set_mode(self):
        pm = PermissionManager(mode="build")
        pm.set_mode("plan")
        result = pm.check(PermissionRequest(tool="write", operation=OperationType.WRITE))
        assert result.is_allowed is False

    def test_approve_and_recheck(self):
        pm = PermissionManager(mode="build")
        req = PermissionRequest(tool="write", operation=OperationType.WRITE, path="/tmp/f")
        result1 = pm.check(req)
        assert result1.is_allowed is True

        req_ask = PermissionRequest(tool="rm", operation=OperationType.DELETE, path="/tmp/f")
        result_ask = pm.check(req_ask)
        assert result_ask.requires_approval is True

        pm.approve(req_ask)
        result_after = pm.check(req_ask)
        assert result_after.is_allowed is True
        assert "Previously approved" in result_after.message

    def test_approval_expiry(self):
        pm = PermissionManager(mode="build", approval_callback=None)
        pm._approval_ttl = 0.01
        req = PermissionRequest(tool="rm", operation=OperationType.DELETE, path="/tmp/f")
        pm.approve(req)
        time.sleep(0.02)
        result = pm.check(req)
        assert result.requires_approval is True

    def test_clear_approvals(self):
        pm = PermissionManager(mode="build")
        req = PermissionRequest(tool="rm", operation=OperationType.DELETE, path="/tmp/f")
        pm.approve(req)
        pm.clear_approvals()
        result = pm.check(req)
        assert result.requires_approval is True

    def test_get_audit_log(self):
        pm = PermissionManager(mode="build")
        pm.check(PermissionRequest(tool="read", operation=OperationType.READ))
        log = pm.get_audit_log()
        assert len(log) >= 1
        assert "tool" in log[0]

    def test_get_audit_log_limit(self):
        pm = PermissionManager(mode="build")
        for i in range(20):
            pm.check(PermissionRequest(tool=f"tool_{i}", operation=OperationType.READ))
        log = pm.get_audit_log(limit=5)
        assert len(log) <= 5

    def test_get_rules(self):
        pm = PermissionManager(mode="build")
        rules = pm.get_rules()
        assert len(rules) > 0
        assert all("name" in r for r in rules)

    def test_get_summary(self):
        pm = PermissionManager(mode="build")
        summary = pm.get_summary()
        assert summary["mode"] == "build"
        assert "total_rules" in summary

    def test_load_rules_from_file(self, tmp_path):
        rules_file = tmp_path / "rules.json"
        rules_file.write_text(
            json.dumps(
                {
                    "rules": [
                        {
                            "name": "deny_all",
                            "description": "Deny all",
                            "level": "deny",
                            "tools": ["*"],
                            "paths": [],
                            "operations": [],
                            "priority": 500,
                        }
                    ]
                }
            )
        )
        pm = PermissionManager(mode="build")
        result = pm.load_rules_from_file(rules_file)
        assert result is True
        result_check = pm.check(PermissionRequest(tool="anything", operation=OperationType.READ))
        assert result_check.is_allowed is False

    def test_load_rules_from_invalid_file(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json {{{")
        pm = PermissionManager(mode="build")
        assert pm.load_rules_from_file(bad_file) is False

    def test_operation_inferred_from_tool(self):
        pm = PermissionManager(mode="build")
        req = PermissionRequest(tool="web_search")
        assert req.operation is None
        pm.check(req)
        assert req.operation == OperationType.NETWORK

    def test_audit_log_trimming(self):
        pm = PermissionManager(mode="build")
        pm._max_audit_entries = 10
        for i in range(20):
            pm.check(PermissionRequest(tool=f"tool_{i}", operation=OperationType.READ))
        assert len(pm._audit_log) <= 10
