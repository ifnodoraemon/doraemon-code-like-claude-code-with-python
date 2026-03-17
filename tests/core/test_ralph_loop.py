from src.core.ralph_loop import RalphLoopManager


def test_ralph_manager_add_choose_and_complete_task(tmp_path):
    mgr = RalphLoopManager(project_dir=tmp_path)
    mgr.init_storage()
    task = mgr.add_task("Optimize startup latency", "Startup should be faster.")

    assert task.title == "Optimize startup latency"

    next_task = mgr.choose_next_task()
    assert next_task is not None
    assert next_task.id == task.id
    assert next_task.status == "in_progress"
    assert "[Ralph Task" in next_task.last_prompt
    assert mgr.get_active_task() is not None
    assert mgr.get_active_task().id == task.id

    assert mgr.mark_done(task.id, "validated") is True
    assert mgr.get_task(task.id).status == "completed"
    assert mgr.get_active_task() is None


def test_ralph_manager_marks_task_blocked(tmp_path):
    mgr = RalphLoopManager(project_dir=tmp_path)
    task = mgr.add_task("Investigate flaky test")

    assert mgr.mark_blocked(task.id, "need repro steps") is True
    blocked = mgr.get_task(task.id)
    assert blocked is not None
    assert blocked.status == "blocked"
    assert blocked.notes[-1] == "blocked: need repro steps"


def test_ralph_manager_records_progress_notes(tmp_path):
    mgr = RalphLoopManager(project_dir=tmp_path)
    task = mgr.add_task("Improve test stability")
    mgr.choose_next_task()

    assert mgr.record_progress(task.id, "inspected failing tests") is True

    updated = mgr.get_task(task.id)
    assert updated is not None
    assert updated.notes[-1] == "progress: inspected failing tests"


def test_ralph_manager_can_resume_active_prompt(tmp_path):
    mgr = RalphLoopManager(project_dir=tmp_path)
    task = mgr.add_task("Continue refactor", "Keep behavior unchanged.")
    mgr.choose_next_task()

    prompt = mgr.resume_active_prompt()

    assert prompt is not None
    assert task.id in prompt
    assert "Choose the next best action dynamically" in prompt
