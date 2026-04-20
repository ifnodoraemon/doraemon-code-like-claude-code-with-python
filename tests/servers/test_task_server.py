"""Tests for servers.task — unified task tool interface."""

import json
from contextvars import ContextVar
from unittest.mock import MagicMock, patch

import pytest

from src.core.tasks import TaskClaimError, TaskManager, TaskStatus
from src.servers.task import (
    _dump,
    _normalize_dependencies,
    _normalize_status,
    get_task_manager,
    reset_task_manager,
    set_task_manager,
    task,
)


class TestNormalizeStatus:
    def test_valid_status(self):
        assert _normalize_status("pending") == TaskStatus.PENDING
        assert _normalize_status("in_progress") == TaskStatus.IN_PROGRESS
        assert _normalize_status("completed") == TaskStatus.COMPLETED

    def test_aliases(self):
        assert _normalize_status("todo") == TaskStatus.PENDING
        assert _normalize_status("done") == TaskStatus.COMPLETED

    def test_case_insensitive(self):
        assert _normalize_status("PENDING") == TaskStatus.PENDING
        assert _normalize_status("TODO") == TaskStatus.PENDING

    def test_invalid(self):
        assert _normalize_status("invalid_status") is None

    def test_none(self):
        assert _normalize_status(None) is None


class TestDump:
    def test_dumps_json(self):
        result = _dump({"a": 1, "b": 2})
        parsed = json.loads(result)
        assert parsed == {"a": 1, "b": 2}

    def test_sorted_keys(self):
        result = _dump({"z": 1, "a": 2})
        keys = list(json.loads(result).keys())
        assert keys == ["a", "z"]


class TestNormalizeDependencies:
    def test_comma_separated(self):
        result = _normalize_dependencies("abc,def,ghi")
        assert result == ["abc", "def", "ghi"]

    def test_trims_whitespace(self):
        result = _normalize_dependencies("  abc , def  ")
        assert result == ["abc", "def"]

    def test_filters_empty(self):
        result = _normalize_dependencies("abc,,def,")
        assert result == ["abc", "def"]

    def test_none(self):
        assert _normalize_dependencies(None) is None

    def test_empty_string(self):
        result = _normalize_dependencies("")
        assert result == []


class TestTaskContextManager:
    def test_get_default(self):
        mgr = get_task_manager()
        assert isinstance(mgr, TaskManager)

    def test_set_and_get(self):
        custom = MagicMock(spec=TaskManager)
        token = set_task_manager(custom)
        assert get_task_manager() is custom
        reset_task_manager(token)
        assert get_task_manager() is not custom


class TestTaskCreate:
    def test_create_success(self, tmp_path):
        from src.servers.task import _active_task_manager

        mgr = TaskManager(storage_path=tmp_path / "tasks.json")
        token = set_task_manager(mgr)
        try:
            result = json.loads(task(operation="create", title="My Task", description="desc"))
            assert result["ok"] is True
            assert result["task"]["title"] == "My Task"
            assert result["task"]["description"] == "desc"
            assert result["task"]["status"] == "pending"
        finally:
            reset_task_manager(token)

    def test_create_no_title(self, tmp_path):
        mgr = TaskManager(storage_path=tmp_path / "tasks.json")
        token = set_task_manager(mgr)
        try:
            result = json.loads(task(operation="create"))
            assert result["ok"] is False
            assert "title is required" in result["error"]
        finally:
            reset_task_manager(token)

    def test_create_with_dependencies(self, tmp_path):
        mgr = TaskManager(storage_path=tmp_path / "tasks.json")
        token = set_task_manager(mgr)
        try:
            parent = mgr.create_task(title="Parent")
            result = json.loads(task(operation="create", title="Child", dependencies=parent.id))
            assert result["ok"] is True
            assert parent.id in result["task"]["dependencies"]
        finally:
            reset_task_manager(token)


class TestTaskList:
    def test_list_all(self, tmp_path):
        mgr = TaskManager(storage_path=tmp_path / "tasks.json")
        token = set_task_manager(mgr)
        try:
            mgr.create_task(title="Task 1")
            mgr.create_task(title="Task 2")
            result = json.loads(task(operation="list"))
            assert result["ok"] is True
            assert len(result["tasks"]) == 2
        finally:
            reset_task_manager(token)

    def test_list_with_status_filter(self, tmp_path):
        mgr = TaskManager(storage_path=tmp_path / "tasks.json")
        token = set_task_manager(mgr)
        try:
            mgr.create_task(title="T1")
            t2 = mgr.create_task(title="T2")
            mgr.update_task(t2.id, status=TaskStatus.COMPLETED)
            result = json.loads(task(operation="list", status="completed"))
            assert result["ok"] is True
            assert len(result["tasks"]) == 1
        finally:
            reset_task_manager(token)

    def test_list_invalid_status(self, tmp_path):
        mgr = TaskManager(storage_path=tmp_path / "tasks.json")
        token = set_task_manager(mgr)
        try:
            result = json.loads(task(operation="list", status="invalid_status"))
            assert result["ok"] is False
            assert "invalid status" in result["error"]
        finally:
            reset_task_manager(token)


class TestTaskGet:
    def test_get_existing(self, tmp_path):
        mgr = TaskManager(storage_path=tmp_path / "tasks.json")
        token = set_task_manager(mgr)
        try:
            created = mgr.create_task(title="Find me")
            result = json.loads(task(operation="get", task_id=created.id))
            assert result["ok"] is True
            assert result["task"]["title"] == "Find me"
        finally:
            reset_task_manager(token)

    def test_get_missing(self, tmp_path):
        mgr = TaskManager(storage_path=tmp_path / "tasks.json")
        token = set_task_manager(mgr)
        try:
            result = json.loads(task(operation="get", task_id="nonexistent"))
            assert result["ok"] is False
        finally:
            reset_task_manager(token)

    def test_get_no_task_id(self, tmp_path):
        mgr = TaskManager(storage_path=tmp_path / "tasks.json")
        token = set_task_manager(mgr)
        try:
            result = json.loads(task(operation="get"))
            assert result["ok"] is False
        finally:
            reset_task_manager(token)


class TestTaskUpdate:
    def test_update_status(self, tmp_path):
        mgr = TaskManager(storage_path=tmp_path / "tasks.json")
        token = set_task_manager(mgr)
        try:
            created = mgr.create_task(title="Update me")
            result = json.loads(task(operation="update", task_id=created.id, status="completed"))
            assert result["ok"] is True
            assert result["task"]["status"] == "completed"
        finally:
            reset_task_manager(token)

    def test_update_no_task_id(self, tmp_path):
        mgr = TaskManager(storage_path=tmp_path / "tasks.json")
        token = set_task_manager(mgr)
        try:
            result = json.loads(task(operation="update"))
            assert result["ok"] is False
        finally:
            reset_task_manager(token)

    def test_update_not_found(self, tmp_path):
        mgr = TaskManager(storage_path=tmp_path / "tasks.json")
        token = set_task_manager(mgr)
        try:
            result = json.loads(task(operation="update", task_id="nonexistent", status="completed"))
            assert result["ok"] is False
        finally:
            reset_task_manager(token)

    def test_update_invalid_status(self, tmp_path):
        mgr = TaskManager(storage_path=tmp_path / "tasks.json")
        token = set_task_manager(mgr)
        try:
            created = mgr.create_task(title="T")
            result = json.loads(task(operation="update", task_id=created.id, status="invalid"))
            assert result["ok"] is False
        finally:
            reset_task_manager(token)

    def test_update_with_agent(self, tmp_path):
        mgr = TaskManager(storage_path=tmp_path / "tasks.json")
        token = set_task_manager(mgr)
        try:
            created = mgr.create_task(title="T")
            result = json.loads(task(operation="update", task_id=created.id, agent_id="agent-1"))
            assert result["ok"] is True
            assert result["task"]["assigned_agent"] == "agent-1"
        finally:
            reset_task_manager(token)


class TestTaskReady:
    def test_ready_excludes_blocked(self, tmp_path):
        mgr = TaskManager(storage_path=tmp_path / "tasks.json")
        token = set_task_manager(mgr)
        try:
            dep = mgr.create_task(title="Dep")
            blocked = mgr.create_task(title="Blocked", dependencies=[dep.id])
            result = json.loads(task(operation="ready"))
            ready_ids = [t["id"] for t in result["tasks"]]
            assert dep.id in ready_ids
            assert blocked.id not in ready_ids
        finally:
            reset_task_manager(token)

    def test_ready_includes_unblocked(self, tmp_path):
        mgr = TaskManager(storage_path=tmp_path / "tasks.json")
        token = set_task_manager(mgr)
        try:
            mgr.create_task(title="Free")
            result = json.loads(task(operation="ready"))
            assert result["ok"] is True
            assert len(result["tasks"]) == 1
        finally:
            reset_task_manager(token)


class TestTaskClaim:
    def test_claim_success(self, tmp_path):
        mgr = TaskManager(storage_path=tmp_path / "tasks.json")
        token = set_task_manager(mgr)
        try:
            created = mgr.create_task(title="Claim me")
            result = json.loads(task(operation="claim", task_id=created.id, agent_id="agent-1"))
            assert result["ok"] is True
            assert result["task"]["assigned_agent"] == "agent-1"
        finally:
            reset_task_manager(token)

    def test_claim_no_task_id(self, tmp_path):
        mgr = TaskManager(storage_path=tmp_path / "tasks.json")
        token = set_task_manager(mgr)
        try:
            result = json.loads(task(operation="claim"))
            assert result["ok"] is False
        finally:
            reset_task_manager(token)

    def test_claim_no_agent_id(self, tmp_path):
        mgr = TaskManager(storage_path=tmp_path / "tasks.json")
        token = set_task_manager(mgr)
        try:
            created = mgr.create_task(title="T")
            result = json.loads(task(operation="claim", task_id=created.id))
            assert result["ok"] is False
        finally:
            reset_task_manager(token)

    def test_claim_already_claimed(self, tmp_path):
        mgr = TaskManager(storage_path=tmp_path / "tasks.json")
        token = set_task_manager(mgr)
        try:
            created = mgr.create_task(title="T")
            mgr.claim_task(created.id, "agent-1")
            result = json.loads(task(operation="claim", task_id=created.id, agent_id="agent-2"))
            assert result["ok"] is False
        finally:
            reset_task_manager(token)


class TestTaskRelease:
    def test_release_success(self, tmp_path):
        mgr = TaskManager(storage_path=tmp_path / "tasks.json")
        token = set_task_manager(mgr)
        try:
            created = mgr.create_task(title="T")
            mgr.claim_task(created.id, "agent-1")
            result = json.loads(task(operation="release", task_id=created.id))
            assert result["ok"] is True
            assert result["task"]["assigned_agent"] is None
        finally:
            reset_task_manager(token)

    def test_release_no_task_id(self, tmp_path):
        mgr = TaskManager(storage_path=tmp_path / "tasks.json")
        token = set_task_manager(mgr)
        try:
            result = json.loads(task(operation="release"))
            assert result["ok"] is False
        finally:
            reset_task_manager(token)


class TestTaskDelete:
    def test_delete_success(self, tmp_path):
        mgr = TaskManager(storage_path=tmp_path / "tasks.json")
        token = set_task_manager(mgr)
        try:
            created = mgr.create_task(title="Delete me")
            result = json.loads(task(operation="delete", task_id=created.id))
            assert result["ok"] is True
            assert result["deleted"] == created.id
        finally:
            reset_task_manager(token)

    def test_delete_not_found(self, tmp_path):
        mgr = TaskManager(storage_path=tmp_path / "tasks.json")
        token = set_task_manager(mgr)
        try:
            result = json.loads(task(operation="delete", task_id="nonexistent"))
            assert result["ok"] is False
        finally:
            reset_task_manager(token)

    def test_delete_no_task_id(self, tmp_path):
        mgr = TaskManager(storage_path=tmp_path / "tasks.json")
        token = set_task_manager(mgr)
        try:
            result = json.loads(task(operation="delete"))
            assert result["ok"] is False
        finally:
            reset_task_manager(token)

    def test_delete_with_subtasks(self, tmp_path):
        mgr = TaskManager(storage_path=tmp_path / "tasks.json")
        token = set_task_manager(mgr)
        try:
            parent = mgr.create_task(title="Parent")
            child = mgr.create_task(title="Child", parent_id=parent.id)
            result = json.loads(task(operation="delete", task_id=parent.id, delete_subtasks=True))
            assert result["ok"] is True
            assert mgr.get_task(child.id) is None
        finally:
            reset_task_manager(token)


class TestTaskClear:
    def test_clear(self, tmp_path):
        mgr = TaskManager(storage_path=tmp_path / "tasks.json")
        token = set_task_manager(mgr)
        try:
            mgr.create_task(title="T1")
            mgr.create_task(title="T2")
            result = json.loads(task(operation="clear"))
            assert result["ok"] is True
            assert result["cleared"] is True
            assert len(mgr.list_tasks()) == 0
        finally:
            reset_task_manager(token)


class TestTaskUnknownOperation:
    def test_unknown_op(self, tmp_path):
        mgr = TaskManager(storage_path=tmp_path / "tasks.json")
        token = set_task_manager(mgr)
        try:
            result = json.loads(task(operation="fly_to_moon"))
            assert result["ok"] is False
            assert "unknown operation" in result["error"]
        finally:
            reset_task_manager(token)
