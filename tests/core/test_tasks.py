import json

import pytest

from src.core.tasks import Task, TaskClaimError, TaskManager, TaskStatus
from src.servers.task import reset_task_manager, set_task_manager, task


class TestTask:
    def test_roundtrip(self):
        original = Task(
            id="task_1",
            title="Ship task tool",
            description="Keep only the useful fields",
            status=TaskStatus.IN_PROGRESS,
            parent_id="root",
            created_at=1.0,
            updated_at=2.0,
        )

        loaded = Task.from_dict(original.to_dict())

        assert loaded == original

    def test_legacy_status_mapping(self):
        loaded = Task.from_dict({"id": "task_1", "title": "Legacy", "status": "todo"})
        assert loaded.status == TaskStatus.PENDING

    def test_from_dict_done_status(self):
        loaded = Task.from_dict({"id": "t1", "title": "Done", "status": "done"})
        assert loaded.status == TaskStatus.COMPLETED

    def test_from_dict_invalid_status_raises(self):
        with pytest.raises(ValueError):
            Task.from_dict({"id": "t1", "title": "Bad", "status": "invalid"})

    def test_from_dict_dependencies_dedup(self):
        loaded = Task.from_dict({"id": "t1", "title": "Dedup", "status": "pending", "dependencies": ["a", "a", "b"]})
        assert loaded.dependencies == ["a", "b"]


class TestTaskManager:
    def test_create_get_and_persist(self, tmp_path):
        storage = tmp_path / "tasks.json"
        manager = TaskManager(storage_path=storage)
        created = manager.create_task("Implement feature", description="Minimal state store")

        reloaded = TaskManager(storage_path=storage)
        persisted = reloaded.get_task(created.id)

        assert persisted is not None
        assert persisted.title == "Implement feature"
        assert persisted.description == "Minimal state store"
        assert persisted.workspace_id == f"task-{created.id}"
        assert persisted.workspace_path is not None

    def test_list_filters_by_status_and_parent(self, tmp_path):
        manager = TaskManager(storage_path=tmp_path / "tasks.json")
        parent = manager.create_task("Parent")
        child = manager.create_task("Child", parent_id=parent.id)
        manager.create_task("Other")
        manager.update_task_status(child.id, TaskStatus.COMPLETED)

        completed_children = manager.list_tasks(
            status=TaskStatus.COMPLETED,
            parent_id=parent.id,
        )

        assert [task.id for task in completed_children] == [child.id]

    def test_list_tasks_preserves_creation_order(self, tmp_path):
        manager = TaskManager(storage_path=tmp_path / "tasks.json")
        first = manager.create_task("First", priority=100)
        second = manager.create_task("Second", priority=0)

        assert [task.id for task in manager.list_tasks()] == [first.id, second.id]

    def test_get_task_tree_is_computed_from_parent_links(self, tmp_path):
        manager = TaskManager(storage_path=tmp_path / "tasks.json")
        root = manager.create_task("Root")
        child = manager.create_task("Child", parent_id=root.id)
        grandchild = manager.create_task("Grandchild", parent_id=child.id)

        tree = manager.get_task_tree()

        assert tree[0]["id"] == root.id
        assert tree[0]["children"][0]["id"] == child.id
        assert tree[0]["children"][0]["children"][0]["id"] == grandchild.id

    def test_get_task_tree_can_scope_to_single_root(self, tmp_path):
        manager = TaskManager(storage_path=tmp_path / "tasks.json")
        root_one = manager.create_task("Root One")
        child_one = manager.create_task("Child One", parent_id=root_one.id)
        root_two = manager.create_task("Root Two")

        tree = manager.get_task_tree(root_one.id)

        assert len(tree) == 1
        assert tree[0]["id"] == root_one.id
        assert tree[0]["children"][0]["id"] == child_one.id
        assert tree[0]["title"] == "Root One"
        assert tree[0]["id"] != root_two.id

    def test_get_task_tree_nonexistent_root(self, tmp_path):
        manager = TaskManager(storage_path=tmp_path / "tasks.json")
        assert manager.get_task_tree("nonexistent") == []

    def test_delete_orphans_children_by_default(self, tmp_path):
        manager = TaskManager(storage_path=tmp_path / "tasks.json")
        parent = manager.create_task("Parent")
        child = manager.create_task("Child", parent_id=parent.id)

        manager.delete_task(parent.id)

        assert manager.get_task(parent.id) is None
        assert manager.get_task(child.id).parent_id is None

    def test_delete_can_cascade(self, tmp_path):
        manager = TaskManager(storage_path=tmp_path / "tasks.json")
        parent = manager.create_task("Parent")
        child = manager.create_task("Child", parent_id=parent.id)

        manager.delete_task(parent.id, delete_subtasks=True)

        assert manager.get_task(parent.id) is None
        assert manager.get_task(child.id) is None

    def test_delete_removes_dependency_refs(self, tmp_path):
        manager = TaskManager(storage_path=tmp_path / "tasks.json")
        dep = manager.create_task("Dep")
        dependent = manager.create_task("Dependent", dependencies=[dep.id])
        manager.delete_task(dep.id)
        reloaded = manager.get_task(dependent.id)
        assert dep.id not in reloaded.dependencies

    def test_delete_nonexistent_task(self, tmp_path):
        manager = TaskManager(storage_path=tmp_path / "tasks.json")
        assert manager.delete_task("nonexistent") is False

    def test_clear_all_tasks(self, tmp_path):
        manager = TaskManager(storage_path=tmp_path / "tasks.json")
        manager.create_task("A")
        manager.create_task("B")

        manager.clear_all_tasks()

        assert manager.list_tasks() == []

    def test_ready_tasks_require_completed_dependencies(self, tmp_path):
        manager = TaskManager(storage_path=tmp_path / "tasks.json")
        dep = manager.create_task("Dependency")
        blocked = manager.create_task("Blocked", dependencies=[dep.id], priority=10)
        ready = manager.create_task("Ready", priority=1)

        ready_ids = [task.id for task in manager.list_ready_tasks()]
        assert ready.id in ready_ids
        assert blocked.id not in ready_ids

        manager.update_task_status(dep.id, TaskStatus.COMPLETED)
        ready_ids = [task.id for task in manager.list_ready_tasks()]
        assert ready_ids[0] == blocked.id

    def test_claim_non_pending_task(self, tmp_path):
        manager = TaskManager(storage_path=tmp_path / "tasks.json")
        t = manager.create_task("Done")
        manager.update_task_status(t.id, TaskStatus.COMPLETED)
        with pytest.raises(TaskClaimError, match="not pending"):
            manager.claim_task(t.id, "agent-a")

    def test_release_nonexistent_task(self, tmp_path):
        manager = TaskManager(storage_path=tmp_path / "tasks.json")
        with pytest.raises(TaskClaimError, match="task not found"):
            manager.release_task("nonexistent")

    def test_release_unclaimed_task(self, tmp_path):
        manager = TaskManager(storage_path=tmp_path / "tasks.json")
        t = manager.create_task("Unclaimed")
        with pytest.raises(TaskClaimError, match="not claimed"):
            manager.release_task(t.id)

    def test_release_wrong_agent(self, tmp_path):
        manager = TaskManager(storage_path=tmp_path / "tasks.json")
        t = manager.create_task("Claimed")
        manager.claim_task(t.id, "agent-a")
        with pytest.raises(TaskClaimError, match="claimed by"):
            manager.release_task(t.id, agent_id="agent-b")

    def test_update_rejects_dependency_cycles(self, tmp_path):
        manager = TaskManager(storage_path=tmp_path / "tasks.json")
        first = manager.create_task("First")
        second = manager.create_task("Second", dependencies=[first.id])

        try:
            manager.update_task(first.id, dependencies=[second.id])
        except ValueError as exc:
            assert "dependency cycle detected" in str(exc)
        else:
            raise AssertionError("Expected update_task to reject dependency cycles")

    def test_update_task_nonexistent(self, tmp_path):
        manager = TaskManager(storage_path=tmp_path / "tasks.json")
        assert manager.update_task("nonexistent") is None

    def test_update_task_with_workspace_id(self, tmp_path):
        manager = TaskManager(storage_path=tmp_path / "tasks.json")
        t = manager.create_task("Task")
        updated = manager.update_task(t.id, workspace_id="custom-ws")
        assert updated.workspace_id == "custom-ws"

    def test_update_task_completed_clears_agent(self, tmp_path):
        manager = TaskManager(storage_path=tmp_path / "tasks.json")
        t = manager.create_task("Task")
        manager.claim_task(t.id, "agent-a")
        updated = manager.update_task(t.id, status=TaskStatus.COMPLETED)
        assert updated.assigned_agent is None
        assert updated.claimed_at is None

    def test_update_task_cancelled_clears_agent(self, tmp_path):
        manager = TaskManager(storage_path=tmp_path / "tasks.json")
        t = manager.create_task("Task")
        manager.claim_task(t.id, "agent-a")
        updated = manager.update_task(t.id, status=TaskStatus.CANCELLED)
        assert updated.assigned_agent is None

    def test_update_task_with_assigned_agent(self, tmp_path):
        manager = TaskManager(storage_path=tmp_path / "tasks.json")
        t = manager.create_task("Task")
        updated = manager.update_task(t.id, assigned_agent="agent-b")
        assert updated.assigned_agent == "agent-b"
        assert updated.claimed_at is not None

    def test_update_task_clear_agent(self, tmp_path):
        manager = TaskManager(storage_path=tmp_path / "tasks.json")
        t = manager.create_task("Task")
        manager.claim_task(t.id, "agent-a")
        updated = manager.update_task(t.id, assigned_agent=None)
        assert updated.assigned_agent is None
        assert updated.claimed_at is None

    def test_create_task_with_invalid_dependency(self, tmp_path):
        manager = TaskManager(storage_path=tmp_path / "tasks.json")
        with pytest.raises(ValueError, match="unknown dependency"):
            manager.create_task("Bad", dependencies=["nonexistent"])

    def test_create_task_with_self_dependency(self, tmp_path):
        manager = TaskManager(storage_path=tmp_path / "tasks.json")
        with pytest.raises(ValueError):
            manager.create_task("Self", dependencies=["self-id"])

    def test_create_task_with_empty_dependency(self, tmp_path):
        manager = TaskManager(storage_path=tmp_path / "tasks.json")
        t = manager.create_task("Empty dep", dependencies=[""])
        assert t.dependencies == []

    def test_create_task_with_invalid_parent(self, tmp_path):
        manager = TaskManager(storage_path=tmp_path / "tasks.json")
        t = manager.create_task("Orphan parent", parent_id="nonexistent")
        assert t.parent_id is None

    def test_load_corrupt_storage(self, tmp_path):
        storage = tmp_path / "tasks.json"
        storage.write_text("not json")
        manager = TaskManager(storage_path=storage)
        assert manager.list_tasks() == []

    def test_load_invalid_format_storage(self, tmp_path):
        storage = tmp_path / "tasks.json"
        storage.write_text("42")
        manager = TaskManager(storage_path=storage)
        assert manager.list_tasks() == []

    def test_load_invalid_task_entry(self, tmp_path):
        storage = tmp_path / "tasks.json"
        storage.write_text(json.dumps({"bad": {"id": "x", "title": "T", "status": "invalid_status"}}))
        manager = TaskManager(storage_path=storage)
        assert manager.list_tasks() == []

    def test_are_dependencies_satisfied_nonexistent(self, tmp_path):
        manager = TaskManager(storage_path=tmp_path / "tasks.json")
        assert manager.are_dependencies_satisfied("nonexistent") is False

    def test_normalize_dependencies_dedup(self, tmp_path):
        manager = TaskManager(storage_path=tmp_path / "tasks.json")
        a = manager.create_task("A")
        result = manager._normalize_dependencies([a.id, a.id], task_id="new")
        assert result == [a.id]

    def test_save_failure_handled(self, tmp_path):
        storage = tmp_path / "readonly" / "tasks.json"
        storage.parent.mkdir(parents=True)
        manager = TaskManager(storage_path=storage)
        manager.create_task("Task")

    def test_update_task_status_nonexistent(self, tmp_path):
        manager = TaskManager(storage_path=tmp_path / "tasks.json")
        assert manager.update_task_status("nonexistent", TaskStatus.COMPLETED) is False


class TestTaskTool:
    def test_task_tool_returns_structured_json(self, tmp_path, monkeypatch):
        manager = TaskManager(storage_path=tmp_path / "tasks.json")
        monkeypatch.setattr("src.servers.task.manager", manager)

        created = json.loads(task(operation="create", title="Plan work"))
        listing = json.loads(task(operation="list"))

        assert created["ok"] is True
        assert created["task"]["title"] == "Plan work"
        assert listing["ok"] is True
        assert listing["tasks"][0]["title"] == "Plan work"

    def test_task_tool_supports_get_update_delete_and_clear(self, tmp_path, monkeypatch):
        manager = TaskManager(storage_path=tmp_path / "tasks.json")
        monkeypatch.setattr("src.servers.task.manager", manager)

        created = json.loads(task(operation="create", title="Fix bug"))
        task_id = created["task"]["id"]

        fetched = json.loads(task(operation="get", task_id=task_id))
        updated = json.loads(task(operation="update", task_id=task_id, status="completed"))
        deleted = json.loads(task(operation="delete", task_id=task_id))
        cleared = json.loads(task(operation="clear"))

        assert fetched["task"]["id"] == task_id
        assert updated["task"]["status"] == "completed"
        assert deleted == {"deleted": task_id, "ok": True}
        assert cleared == {"cleared": True, "ok": True}

    def test_task_tool_supports_dependencies_claims_and_ready_listing(
        self, tmp_path, monkeypatch
    ):
        manager = TaskManager(storage_path=tmp_path / "tasks.json")
        monkeypatch.setattr("src.servers.task.manager", manager)

        dep = json.loads(task(operation="create", title="Dep"))
        blocked = json.loads(
            task(
                operation="create",
                title="Blocked",
                dependencies=dep["task"]["id"],
                priority=9,
            )
        )
        ready_before = json.loads(task(operation="ready"))
        assert [item["id"] for item in ready_before["tasks"]] == [dep["task"]["id"]]

        json.loads(task(operation="update", task_id=dep["task"]["id"], status="completed"))
        claimed = json.loads(
            task(operation="claim", task_id=blocked["task"]["id"], agent_id="agent-1")
        )
        released = json.loads(
            task(operation="release", task_id=blocked["task"]["id"], agent_id="agent-1")
        )
        reset_priority = json.loads(
            task(operation="update", task_id=blocked["task"]["id"], priority=0)
        )

        assert claimed["task"]["assigned_agent"] == "agent-1"
        assert claimed["task"]["status"] == "in_progress"
        assert released["task"]["status"] == "pending"
        assert reset_priority["task"]["priority"] == 0

    def test_task_tool_validates_inputs(self, tmp_path, monkeypatch):
        manager = TaskManager(storage_path=tmp_path / "tasks.json")
        monkeypatch.setattr("src.servers.task.manager", manager)

        missing_title = json.loads(task(operation="create"))
        invalid_status = json.loads(task(operation="update", task_id="x", status="bad"))
        unknown = json.loads(task(operation="boom"))

        assert missing_title["ok"] is False
        assert invalid_status["ok"] is False
        assert unknown["ok"] is False

    def test_task_tool_uses_runtime_task_manager_context(self, tmp_path, monkeypatch):
        default_manager = TaskManager(storage_path=tmp_path / "default.json")
        runtime_manager = TaskManager(storage_path=tmp_path / "runtime.json")

        import src.servers.task as task_module

        monkeypatch.setattr(task_module, "manager", default_manager)
        token = set_task_manager(runtime_manager)
        try:
            created = json.loads(task(operation="create", title="Runtime task"))
        finally:
            reset_task_manager(token)

        assert created["ok"] is True
        assert runtime_manager.get_task(created["task"]["id"]) is not None
        assert default_manager.get_task(created["task"]["id"]) is None
