"""Tests for spec_manager.py - Spec-driven development workflow."""

import json

import pytest

from src.core.spec_manager import (
    SPEC_DOCS,
    SpecCheckItem,
    SpecManager,
    SpecPhase,
    SpecSession,
    SpecTask,
)


class TestSpecPhase:
    def test_phase_values(self):
        assert SpecPhase.DRAFT.value == "draft"
        assert SpecPhase.REVIEW.value == "review"
        assert SpecPhase.EXECUTE.value == "execute"
        assert SpecPhase.COMPLETE.value == "complete"


class TestSpecManager:
    @pytest.fixture
    def mgr(self, tmp_path):
        return SpecManager(root=tmp_path / "specs")

    def test_create_spec(self, mgr):
        session = mgr.create_spec("Build auth system")
        assert session is not None
        assert session.name == "build-auth-system"
        assert session.phase == SpecPhase.DRAFT
        assert session.description == "Build auth system"
        assert len(session.id) == 8
        assert mgr.is_active

    def test_create_spec_creates_directory(self, mgr):
        session = mgr.create_spec("Test feature")
        assert session.spec_path.exists()
        assert (session.spec_path / "meta.json").exists()

    def test_advance_phase(self, mgr):
        mgr.create_spec("Test")
        mgr.advance_phase(SpecPhase.REVIEW)
        assert mgr.phase == SpecPhase.REVIEW

    def test_advance_phase_no_session_raises(self, mgr):
        with pytest.raises(RuntimeError):
            mgr.advance_phase(SpecPhase.REVIEW)

    def test_abort(self, mgr):
        mgr.create_spec("Test")
        assert mgr.is_active
        mgr.abort()
        assert not mgr.is_active
        assert mgr.session is None

    def test_write_and_read_spec_doc(self, mgr):
        mgr.create_spec("Test")
        path = mgr.write_spec_doc("spec.md", "# Spec\nTest content")
        assert path.exists()
        content = mgr.read_spec_doc("spec.md")
        assert content == "# Spec\nTest content"

    def test_write_invalid_doc_raises(self, mgr):
        mgr.create_spec("Test")
        with pytest.raises(ValueError):
            mgr.write_spec_doc("invalid.md", "content")

    def test_read_nonexistent_doc(self, mgr):
        mgr.create_spec("Test")
        assert mgr.read_spec_doc("spec.md") is None

    def test_get_all_spec_content(self, mgr):
        mgr.create_spec("Test")
        mgr.write_spec_doc("spec.md", "Spec content")
        mgr.write_spec_doc("tasks.md", "- [ ] T1: Task one")
        mgr.write_spec_doc("checklist.md", "- [ ] C1: Check one")

        content = mgr.get_all_spec_content()
        assert "Spec content" in content
        assert "T1" in content
        assert "C1" in content

    def test_parse_tasks(self, mgr):
        mgr.create_spec("Test")
        tasks_md = """# Tasks

- [ ] T1: Create database schema
- [ ] T2: Implement API endpoints
- [x] T3: Write unit tests
"""
        mgr.write_spec_doc("tasks.md", tasks_md)
        tasks = mgr.session.tasks
        assert len(tasks) == 3
        assert tasks[0].id == "T1"
        assert tasks[0].title == "Create database schema"
        assert tasks[0].status == "pending"
        assert tasks[2].id == "T3"
        assert tasks[2].status == "done"

    def test_parse_checklist(self, mgr):
        mgr.create_spec("Test")
        checklist_md = """# Checklist

- [ ] C1: Unit tests pass
- [x] C2: Integration tests pass
- [ ] C3: No regressions
"""
        mgr.write_spec_doc("checklist.md", checklist_md)
        items = mgr.session.checklist
        assert len(items) == 3
        assert items[0].id == "C1"
        assert items[0].checked is False
        assert items[1].id == "C2"
        assert items[1].checked is True

    def test_update_task_status(self, mgr):
        mgr.create_spec("Test")
        mgr.write_spec_doc("tasks.md", "- [ ] T1: Do thing\n- [ ] T2: Do other")
        assert mgr.update_task_status("T1", "done")
        assert mgr.session.tasks[0].status == "done"

        # Verify markdown was synced
        content = mgr.read_spec_doc("tasks.md")
        assert "- [x] T1:" in content
        assert "- [ ] T2:" in content

    def test_update_task_invalid_status(self, mgr):
        mgr.create_spec("Test")
        mgr.write_spec_doc("tasks.md", "- [ ] T1: Do thing")
        assert not mgr.update_task_status("T1", "invalid_status")

    def test_update_task_nonexistent(self, mgr):
        mgr.create_spec("Test")
        mgr.write_spec_doc("tasks.md", "- [ ] T1: Do thing")
        assert not mgr.update_task_status("T99", "done")

    def test_check_item(self, mgr):
        mgr.create_spec("Test")
        mgr.write_spec_doc("checklist.md", "- [ ] C1: Check one\n- [ ] C2: Check two")
        assert mgr.check_item("C1", True)
        assert mgr.session.checklist[0].checked is True

        # Verify markdown was synced
        content = mgr.read_spec_doc("checklist.md")
        assert "- [x] C1:" in content
        assert "- [ ] C2:" in content

    def test_check_item_uncheck(self, mgr):
        mgr.create_spec("Test")
        mgr.write_spec_doc("checklist.md", "- [x] C1: Check one")
        assert mgr.check_item("C1", False)
        assert mgr.session.checklist[0].checked is False

    def test_get_progress(self, mgr):
        mgr.create_spec("Test")
        mgr.write_spec_doc("tasks.md", "- [ ] T1: A\n- [ ] T2: B\n- [ ] T3: C\n- [ ] T4: D")
        mgr.write_spec_doc("checklist.md", "- [ ] C1: X\n- [ ] C2: Y")

        p = mgr.get_progress()
        assert p["tasks_total"] == 4
        assert p["tasks_done"] == 0
        assert p["checks_total"] == 2
        assert p["checks_done"] == 0
        assert p["percent"] == 0

        mgr.update_task_status("T1", "done")
        mgr.update_task_status("T2", "done")
        mgr.check_item("C1", True)
        p = mgr.get_progress()
        assert p["tasks_done"] == 2
        assert p["checks_done"] == 1
        assert p["percent"] == 50  # 3 of 6

    def test_check_draft_complete(self, mgr):
        mgr.create_spec("Test")
        assert not mgr.check_draft_complete()

        mgr.write_spec_doc("spec.md", "# Spec")
        assert not mgr.check_draft_complete()

        mgr.write_spec_doc("tasks.md", "- [ ] T1: Task")
        assert not mgr.check_draft_complete()

        mgr.write_spec_doc("checklist.md", "- [ ] C1: Check")
        assert mgr.check_draft_complete()
        assert mgr.phase == SpecPhase.REVIEW

    def test_check_draft_complete_wrong_phase(self, mgr):
        mgr.create_spec("Test")
        mgr.advance_phase(SpecPhase.REVIEW)
        assert not mgr.check_draft_complete()

    def test_list_specs(self, mgr):
        mgr.create_spec("First spec")
        mgr.create_spec("Second spec")
        specs = mgr.list_specs()
        assert len(specs) == 2
        names = {s["name"] for s in specs}
        assert "first-spec" in names
        assert "second-spec" in names

    def test_list_specs_empty(self, mgr):
        assert mgr.list_specs() == []

    def test_resume_spec_by_name(self, mgr):
        session1 = mgr.create_spec("My feature")
        original_id = session1.id
        mgr._session = None  # Clear active session

        resumed = mgr.resume_spec("my-feature")
        assert resumed is not None
        assert resumed.id == original_id
        assert resumed.name == "my-feature"

    def test_resume_spec_by_id(self, mgr):
        session1 = mgr.create_spec("My feature")
        original_id = session1.id
        mgr._session = None

        resumed = mgr.resume_spec(original_id)
        assert resumed is not None
        assert resumed.id == original_id

    def test_resume_spec_not_found(self, mgr):
        assert mgr.resume_spec("nonexistent") is None

    def test_persistence(self, mgr, tmp_path):
        """Test that state survives creating a new SpecManager instance."""
        session = mgr.create_spec("Persist test")
        mgr.write_spec_doc("tasks.md", "- [ ] T1: Task one\n- [ ] T2: Task two")
        mgr.update_task_status("T1", "done")
        original_id = session.id

        # Create new manager pointing to same root
        mgr2 = SpecManager(root=tmp_path / "specs")
        resumed = mgr2.resume_spec(original_id)
        assert resumed is not None
        assert resumed.tasks[0].status == "done"
        assert resumed.tasks[1].status == "pending"

    def test_slugify(self):
        assert SpecManager._slugify("Build Auth System") == "build-auth-system"
        assert SpecManager._slugify("Add user login!!!") == "add-user-login"
        assert SpecManager._slugify("  spaces  ") == "spaces"
        assert SpecManager._slugify("a" * 100) == "a" * 60
        assert SpecManager._slugify("!!!") == "unnamed-spec"

    def test_is_active_after_complete(self, mgr):
        mgr.create_spec("Test")
        assert mgr.is_active
        mgr.advance_phase(SpecPhase.COMPLETE)
        assert not mgr.is_active

    def test_get_progress_no_session(self, mgr):
        p = mgr.get_progress()
        assert p["percent"] == 0
        assert p["tasks_total"] == 0

    def test_is_draft_write_allowed_in_spec_dir(self, mgr):
        session = mgr.create_spec("Test")
        spec_dir = session.spec_path
        assert mgr.is_draft_write_allowed(str(spec_dir / "spec.md"))
        assert mgr.is_draft_write_allowed(str(spec_dir / "tasks.md"))
        assert mgr.is_draft_write_allowed(str(spec_dir / "subdir" / "file.txt"))

    def test_is_draft_write_blocked_outside_spec_dir(self, mgr):
        mgr.create_spec("Test")
        assert not mgr.is_draft_write_allowed("/tmp/evil.py")
        assert not mgr.is_draft_write_allowed("src/main.py")

    def test_is_draft_write_allowed_non_draft_phase(self, mgr):
        mgr.create_spec("Test")
        mgr.advance_phase(SpecPhase.EXECUTE)
        # In EXECUTE phase, writes to any path are allowed
        assert mgr.is_draft_write_allowed("/tmp/anything.py")
        assert mgr.is_draft_write_allowed("src/main.py")

    def test_is_draft_write_allowed_no_session(self, mgr):
        # No active session → allow (no restriction applies)
        assert mgr.is_draft_write_allowed("src/main.py")
