import json

from src.core.tasks import Task, TaskManager, TaskStatus
from src.servers.task import task


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

    def test_claim_and_release_task(self, tmp_path):
        manager = TaskManager(storage_path=tmp_path / "tasks.json")
        created = manager.create_task("Claim me")

        claimed = manager.claim_task(created.id, "agent-alpha")
        assert claimed.assigned_agent == "agent-alpha"
        assert claimed.status == TaskStatus.IN_PROGRESS

        released = manager.release_task(created.id, agent_id="agent-alpha")
        assert released.assigned_agent is None
        assert released.status == TaskStatus.PENDING

    def test_claim_is_idempotent_for_same_agent(self, tmp_path):
        manager = TaskManager(storage_path=tmp_path / "tasks.json")
        created = manager.create_task("Claim me")

        first = manager.claim_task(created.id, "agent-alpha")
        second = manager.claim_task(created.id, "agent-alpha")

        assert first.id == second.id
        assert second.assigned_agent == "agent-alpha"
        assert second.status == TaskStatus.IN_PROGRESS

    def test_claim_requires_dependencies(self, tmp_path):
        manager = TaskManager(storage_path=tmp_path / "tasks.json")
        dep = manager.create_task("Dependency")
        created = manager.create_task("Blocked", dependencies=[dep.id])

        try:
            manager.claim_task(created.id, "agent-alpha")
        except ValueError as exc:
            assert "unresolved dependencies" in str(exc)
        else:
            raise AssertionError("Expected claim_task to reject unresolved dependencies")

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
