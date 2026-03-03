"""Tests for spec server tools."""

import pytest

from src.core.spec_manager import SpecManager
from src.servers.spec import (
    set_spec_manager,
    spec_check_item,
    spec_progress,
    spec_update_task,
)


@pytest.fixture
def spec_mgr(tmp_path):
    mgr = SpecManager(root=tmp_path / "specs")
    set_spec_manager(mgr)
    return mgr


class TestSpecUpdateTask:
    def test_no_active_session(self, spec_mgr):
        result = spec_update_task("T1", "done")
        assert "Error" in result

    def test_update_valid_task(self, spec_mgr):
        spec_mgr.create_spec("Test")
        spec_mgr.write_spec_doc("tasks.md", "- [ ] T1: Do thing\n- [ ] T2: Other")
        result = spec_update_task("T1", "done")
        assert "Updated T1" in result
        assert "done" in result

    def test_update_invalid_task(self, spec_mgr):
        spec_mgr.create_spec("Test")
        spec_mgr.write_spec_doc("tasks.md", "- [ ] T1: Do thing")
        result = spec_update_task("T99", "done")
        assert "Error" in result

    def test_update_invalid_status(self, spec_mgr):
        spec_mgr.create_spec("Test")
        spec_mgr.write_spec_doc("tasks.md", "- [ ] T1: Do thing")
        result = spec_update_task("T1", "invalid")
        assert "Error" in result


class TestSpecCheckItem:
    def test_no_active_session(self, spec_mgr):
        result = spec_check_item("C1")
        assert "Error" in result

    def test_check_valid_item(self, spec_mgr):
        spec_mgr.create_spec("Test")
        spec_mgr.write_spec_doc("checklist.md", "- [ ] C1: Verify tests")
        result = spec_check_item("C1", True)
        assert "checked" in result
        assert "C1" in result

    def test_uncheck_item(self, spec_mgr):
        spec_mgr.create_spec("Test")
        spec_mgr.write_spec_doc("checklist.md", "- [x] C1: Verify tests")
        result = spec_check_item("C1", False)
        assert "unchecked" in result

    def test_check_nonexistent_item(self, spec_mgr):
        spec_mgr.create_spec("Test")
        spec_mgr.write_spec_doc("checklist.md", "- [ ] C1: Verify tests")
        result = spec_check_item("C99")
        assert "Error" in result


class TestSpecProgress:
    def test_no_active_session(self, spec_mgr):
        result = spec_progress()
        assert "No active" in result

    def test_progress_report(self, spec_mgr):
        spec_mgr.create_spec("Build auth")
        spec_mgr.write_spec_doc("tasks.md", "- [ ] T1: A\n- [ ] T2: B")
        spec_mgr.write_spec_doc("checklist.md", "- [ ] C1: X")

        result = spec_progress()
        assert "build-auth" in result
        assert "Tasks" in result
        assert "Checklist" in result
        assert "0%" in result

    def test_progress_with_completed(self, spec_mgr):
        spec_mgr.create_spec("Test")
        spec_mgr.write_spec_doc("tasks.md", "- [ ] T1: A\n- [ ] T2: B")
        spec_mgr.write_spec_doc("checklist.md", "- [ ] C1: X")
        spec_mgr.update_task_status("T1", "done")
        spec_mgr.check_item("C1", True)

        result = spec_progress()
        assert "66%" in result  # 2 of 3
