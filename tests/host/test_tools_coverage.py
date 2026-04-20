"""Targeted coverage tests for host.tools - LazyToolFunction error paths, audit, registry operations."""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from src.host.tools import LazyToolFunction, ToolAuditEntry, ToolRegistry


class TestLazyToolFunctionErrorPaths:
    def test_load_error_raises_on_subsequent_call(self):
        ltf = LazyToolFunction("bad.module", "func")
        ltf._load_error = "import failed"
        with pytest.raises(ImportError, match="import failed"):
            ltf()

    def test_doc_on_import_error(self):
        ltf = LazyToolFunction("bad.module", "func")
        ltf._load_error = "import failed"
        doc = ltf.__doc__
        assert "bad.module" in doc
        assert "func" in doc

    def test_signature_raises_on_import_error(self):
        ltf = LazyToolFunction("bad.module", "func")
        ltf._load_error = "import failed"
        with pytest.raises(ImportError):
            _ = ltf.__signature__

    @pytest.mark.asyncio
    async def test_acall_with_loaded_func(self):
        async def my_async(x):
            return x * 2

        ltf = LazyToolFunction("os.path", "exists")
        ltf._loaded_func = my_async
        result = await ltf.__acall__(7)
        assert result == 14


class TestToolAuditLogOperations:
    def test_audit_log_trims_when_exceeds_max(self):
        registry = ToolRegistry()
        registry._max_audit_entries = 10
        for i in range(20):
            registry._record_audit(ToolAuditEntry(
                timestamp=time.time(),
                tool_name=f"tool_{i}",
                action="test",
                allowed=True,
                mode="build",
                source="built_in",
            ))
        assert len(registry._audit_log) <= 10

    def test_clear_audit_log(self):
        registry = ToolRegistry()
        registry._record_audit(ToolAuditEntry(
            timestamp=time.time(), tool_name="t", action="x", allowed=True, mode="build", source="built_in",
        ))
        assert len(registry._audit_log) == 1
        registry.clear_audit_log()
        assert len(registry._audit_log) == 0

    def test_get_audit_log_limit(self):
        registry = ToolRegistry()
        for i in range(10):
            registry._record_audit(ToolAuditEntry(
                timestamp=time.time(), tool_name=f"t{i}", action="x", allowed=True, mode="build", source="built_in",
            ))
        result = registry.get_audit_log(limit=3)
        assert len(result) == 3


class TestCheckToolExecutionVisibility:
    def test_invisible_tool_blocked(self):
        registry = ToolRegistry()

        def secret_tool(cmd: str) -> str:
            return cmd

        registry.register(secret_tool, name="secret_tool", metadata={"capability_group": "edit"})
        with patch("src.host.tools.get_default_tool_policy_engine") as mock_engine:
            mock_policy = MagicMock()
            mock_policy.to_dict.return_value = {
                "visible": False,
                "requires_approval": True,
                "background_safe": False,
                "sandbox": "none",
                "audit_level": "full",
                "source": "built_in",
            }
            mock_engine.return_value.describe_tool.return_value = mock_policy
            allowed, reason, policy = registry.check_tool_execution("secret_tool", mode="build")
            assert allowed is False
            assert reason is not None

    def test_unknown_tool_returns_false(self):
        registry = ToolRegistry()
        allowed, reason, policy = registry.check_tool_execution("no_such_tool", mode="build")
        assert allowed is False
        assert "not found" in reason
        assert policy is None


class TestRecordToolExecutionWithMissingPolicy:
    def test_records_with_unknown_source(self):
        registry = ToolRegistry()
        registry.record_tool_execution(
            "missing_tool",
            action="executed",
            mode="build",
            allowed=True,
        )
        log = registry.get_audit_log()
        assert log[0]["source"] == "unknown"


class TestToolRegistryRegisterLazy:
    def test_register_lazy_unloaded_uses_generic_schema(self):
        registry = ToolRegistry()
        ltf = LazyToolFunction("nonexistent_module_xyz", "func")
        ltf._loaded_func = None
        ltf._load_error = None
        registry.register(ltf, name="lazy_test")
        assert "lazy_test" in registry.get_tool_names()
        tool = registry._tools["lazy_test"]
        assert tool.parameters.get("additionalProperties") is True


class TestToolRegistryCallToolTimeout:
    @pytest.mark.asyncio
    async def test_call_tool_timeout_returns_error(self):
        registry = ToolRegistry()

        async def slow_tool():
            await asyncio.sleep(10)
            return "done"

        registry.register(slow_tool, name="slow", timeout=0.01)
        result = await registry.call_tool("slow", {})
        assert "timed out" in result


class TestToolRegistryCallToolSync:
    @pytest.mark.asyncio
    async def test_sync_tool_with_none_result(self):
        registry = ToolRegistry()

        def none_tool():
            return None

        registry.register(none_tool, name="none_t")
        result = await registry.call_tool("none_t", {})
        assert result == "Success"


class TestToolPolicies:
    def test_get_tool_policies(self):
        registry = ToolRegistry()

        def read_t(path: str) -> str:
            return path

        registry.register(read_t, metadata={"capability_group": "read"})
        policies = registry.get_tool_policies(mode="build")
        assert "read_t" in policies
        assert policies["read_t"]["visible"] is True


class TestIsSensitive:
    def test_not_registered(self):
        registry = ToolRegistry()
        assert registry.is_sensitive("no_such_tool") is False
