import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.webui.routes.chat import ChatRequest, chat_endpoint
from src.webui.routes.projects import get_project_context
from src.webui.routes.sessions import get_session, get_session_run
from src.webui.routes.tasks import list_tasks
from src.webui.server import app


class StubTaskManager:
    def get_task_tree(self, root_task_id=None):
        scoped = [{"id": "root", "title": "Root", "status": "completed", "ready": True}]
        if root_task_id:
            return scoped
        return scoped + [{"id": "other-root", "title": "Other", "status": "completed", "ready": True}]

    def list_tasks(self):
        return [
            SimpleNamespace(id="root", parent_id=None, to_dict=lambda: {"id": "root", "title": "Root"}),
            SimpleNamespace(
                id="child",
                parent_id="root",
                to_dict=lambda: {"id": "child", "title": "Child", "parent_id": "root"},
            ),
        ]

    def list_ready_tasks(self):
        return [
            SimpleNamespace(
                id="ready",
                parent_id="root",
                to_dict=lambda: {"id": "ready", "title": "Ready", "parent_id": "root"},
            )
        ]


@pytest.mark.asyncio
async def test_chat_endpoint_streams_orchestration_result(monkeypatch):
    captured = {}

    class StubSession:
        def __init__(self, *args, **kwargs):
            captured["kwargs"] = kwargs
            self._task_manager = StubTaskManager()
            self.session_id = kwargs.get("session_id") or "generated-session"

        async def orchestrate(self, message: str, *, context=None, max_workers=None, resume_run_id=None):
            captured["resume_run_id"] = resume_run_id
            return SimpleNamespace(
                summary=f"planned: {message}",
                to_dict=lambda: {
                    "run_id": "run-1",
                    "goal": message,
                    "root_task_id": "root",
                    "plan_id": "plan-1",
                    "executed_task_ids": ["task-1"],
                    "completed_task_ids": ["task-1"],
                    "failed_task_ids": [],
                    "blocked_task_id": None,
                    "success": True,
                    "summary": f"planned: {message}",
                    "task_summaries": {},
                    "worker_assignments": {},
                    "max_workers": max_workers,
                },
            )

        def get_orchestration_state(self):
            return {
                "run_id": "run-1",
                "goal": "Implement authentication",
                "root_task_id": "root",
                "plan_id": "plan-1",
                "executed_task_ids": ["task-1"],
                "completed_task_ids": ["task-1"],
                "failed_task_ids": [],
                "blocked_task_id": None,
                "success": True,
                "summary": "planned: Implement authentication",
                "task_summaries": {},
                "worker_assignments": {},
                "task_graph": self._task_manager.get_task_tree("root"),
            }

        def get_task_manager(self):
            return self._task_manager

        async def aclose(self):
            return None

    monkeypatch.setattr("src.webui.routes.chat.AgentSession", StubSession)

    response = await chat_endpoint(
        ChatRequest(
            message="Implement authentication",
            execution_mode="orchestrate",
            max_workers=3,
            session_id="resume-me",
            model="test-model",
        )
    )

    payloads = []
    async for chunk in response.body_iterator:
        payloads.append(chunk)

    assert payloads[-1] == "data: [DONE]\n\n"
    first_payload = json.loads(payloads[0].removeprefix("data: ").strip())
    assert first_payload["type"] == "orchestration"
    assert first_payload["session_id"] == "resume-me"
    assert first_payload["content"] == "planned: Implement authentication"
    assert first_payload["result"]["root_task_id"] == "root"
    assert first_payload["task_graph"][0]["id"] == "root"
    assert len(first_payload["task_graph"]) == 1
    assert captured["kwargs"]["session_id"] == "resume-me"
    assert captured["kwargs"]["model_name"] == "test-model"
    assert captured["resume_run_id"] is None


@pytest.mark.asyncio
async def test_list_tasks_returns_tree_and_ready_views(monkeypatch):
    runtime = SimpleNamespace(
        task_manager=StubTaskManager(),
        aclose=lambda: _noop_async(),
    )

    async def fake_bootstrap_runtime(**kwargs):
        return runtime

    monkeypatch.setattr("src.webui.routes.tasks.bootstrap_runtime", fake_bootstrap_runtime)

    listing = await list_tasks(project="demo", mode="build", ready_only=False)
    ready_listing = await list_tasks(project="demo", mode="build", ready_only=True)
    scoped_listing = await list_tasks(project="demo", mode="build", ready_only=False, root_task_id="root")

    assert listing["tasks"][0]["id"] == "root"
    assert listing["tree"][0]["id"] == "root"
    assert ready_listing["tasks"][0]["id"] == "ready"
    assert all(task["id"] != "other-root" for task in scoped_listing["tree"])


@pytest.mark.asyncio
async def test_chat_endpoint_rejects_invalid_session_id():
    with pytest.raises(HTTPException) as exc_info:
        await chat_endpoint(ChatRequest(message="hi", session_id="../escape"))

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_chat_endpoint_rejects_invalid_run_id():
    with pytest.raises(HTTPException) as exc_info:
        await chat_endpoint(ChatRequest(message="hi", resume_run_id="../escape"))

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_chat_endpoint_forwards_resume_run_id(monkeypatch):
    captured = {}

    class StubSession:
        def __init__(self, *args, **kwargs):
            self.session_id = "resume-session"

        async def orchestrate(self, message: str, *, context=None, max_workers=None, resume_run_id=None):
            captured["resume_run_id"] = resume_run_id
            return SimpleNamespace(
                summary="resumed",
                to_dict=lambda: {
                    "run_id": "run-2",
                    "goal": "resume goal",
                    "root_task_id": "root",
                    "plan_id": "plan-1",
                    "executed_task_ids": [],
                    "completed_task_ids": [],
                    "failed_task_ids": [],
                    "blocked_task_id": None,
                    "success": True,
                    "summary": "resumed",
                    "task_summaries": {},
                    "worker_assignments": {},
                },
            )

        def get_orchestration_state(self):
            return {
                "run_id": "run-2",
                "goal": "resume goal",
                "root_task_id": "root",
                "plan_id": "plan-1",
                "executed_task_ids": [],
                "completed_task_ids": [],
                "failed_task_ids": [],
                "blocked_task_id": None,
                "success": True,
                "summary": "resumed",
                "task_summaries": {},
                "worker_assignments": {},
                "task_graph": [{"id": "root", "title": "Root", "status": "completed", "ready": True}],
            }

        async def aclose(self):
            return None

    monkeypatch.setattr("src.webui.routes.chat.AgentSession", StubSession)

    response = await chat_endpoint(
        ChatRequest(message="", execution_mode="orchestrate", resume_run_id="run-1")
    )

    payloads = []
    async for chunk in response.body_iterator:
        payloads.append(chunk)

    assert payloads[-1] == "data: [DONE]\n\n"
    assert captured["resume_run_id"] == "run-1"


@pytest.mark.asyncio
async def test_chat_endpoint_accepts_large_message_without_length_cap(monkeypatch):
    captured = {}

    class StubSession:
        def __init__(self, *args, **kwargs):
            self.session_id = "large-session"

        async def turn_stream(self, message: str):
            captured["message"] = message
            yield {"type": "response", "content": "ok"}
            yield {"type": "done"}

        async def aclose(self):
            return None

    monkeypatch.setattr("src.webui.routes.chat.AgentSession", StubSession)

    large_message = "x" * 200_001
    response = await chat_endpoint(ChatRequest(message=large_message))

    payloads = []
    async for chunk in response.body_iterator:
        payloads.append(chunk)

    assert payloads[-1] == "data: [DONE]\n\n"
    assert captured["message"] == large_message


@pytest.mark.asyncio
async def test_chat_endpoint_streams_orchestration_failure_payload(monkeypatch):
    class StubSession:
        def __init__(self, *args, **kwargs):
            self._task_manager = StubTaskManager()
            self.session_id = "failed-session"

        async def orchestrate(self, message: str, *, context=None, max_workers=None, resume_run_id=None):
            raise RuntimeError(f"planner boom for {message}")

        def get_orchestration_state(self):
            return {
                "run_id": "run-2",
                "goal": "Broken auth flow",
                "root_task_id": "root",
                "plan_id": None,
                "executed_task_ids": [],
                "completed_task_ids": [],
                "failed_task_ids": [],
                "blocked_task_id": "root",
                "success": False,
                "summary": "Orchestration failed: planner boom for Broken auth flow",
                "task_summaries": {},
                "worker_assignments": {},
                "task_graph": self._task_manager.get_task_tree("root"),
            }

        def get_task_manager(self):
            return self._task_manager

        async def aclose(self):
            return None

    monkeypatch.setattr("src.webui.routes.chat.AgentSession", StubSession)

    response = await chat_endpoint(
        ChatRequest(
            message="Broken auth flow",
            execution_mode="orchestrate",
        )
    )

    payloads = []
    async for chunk in response.body_iterator:
        payloads.append(chunk)

    assert payloads[-1] == "data: [DONE]\n\n"
    first_payload = json.loads(payloads[0].removeprefix("data: ").strip())
    assert first_payload["type"] == "orchestration"
    assert first_payload["session_id"] == "failed-session"
    assert first_payload["result"]["success"] is False
    assert first_payload["result"]["root_task_id"] is None
    assert "planner boom" in first_payload["content"]
    assert first_payload["task_graph"] == []


@pytest.mark.asyncio
async def test_get_session_returns_orchestration_state(monkeypatch):
    class StubSessionManager:
        def __init__(self, *args, **kwargs):
            return None

        def resume_session(self, session_id: str):
            assert session_id == "session-123"
            return SimpleNamespace(
                metadata=SimpleNamespace(id="session-123", name="Demo"),
                messages=[{"role": "user", "content": "hello"}],
                orchestration_state={
                    "run_id": "run-1",
                    "success": True,
                    "summary": "planned: hello",
                    "executed_task_ids": ["task-1", "task-2"],
                    "completed_task_ids": ["task-1"],
                    "failed_task_ids": ["task-2"],
                    "task_graph": [{"id": "root", "title": "Root", "status": "completed", "ready": True}],
                    "worker_assignments": {},
                },
                orchestration_runs=[
                    {
                        "run_id": "run-1",
                        "success": True,
                        "summary": "planned: hello",
                        "executed_task_ids": ["task-1", "task-2"],
                        "completed_task_ids": ["task-1"],
                        "failed_task_ids": ["task-2"],
                        "task_graph": [{"id": "root", "title": "Root", "status": "completed", "ready": True}],
                        "worker_assignments": {},
                    }
                ],
                active_orchestration_run_id="run-1",
            )

    monkeypatch.setattr("src.webui.routes.sessions.SessionManager", StubSessionManager)

    payload = await get_session("session-123")

    assert payload["id"] == "session-123"
    assert payload["messages"][0]["content"] == "hello"
    assert payload["orchestration_state"]["summary"] == "planned: hello"
    assert payload["orchestration_state"]["executed_task_count"] == 2
    assert payload["orchestration_state"]["completed_task_count"] == 1
    assert payload["orchestration_state"]["failed_task_count"] == 1
    assert "executed_task_ids" not in payload["orchestration_state"]
    assert payload["orchestration_runs"][0]["run_id"] == "run-1"
    assert payload["orchestration_runs"][0]["executed_task_count"] == 2
    assert payload["orchestration_runs"][0]["completed_task_count"] == 1
    assert payload["orchestration_runs"][0]["failed_task_count"] == 1
    assert "executed_task_ids" not in payload["orchestration_runs"][0]
    assert "task_graph" not in payload["orchestration_runs"][0]
    assert payload["active_orchestration_run_id"] == "run-1"


@pytest.mark.asyncio
async def test_get_session_supports_message_window(monkeypatch):
    class StubSessionManager:
        def __init__(self, *args, **kwargs):
            return None

        def resume_session(self, session_id: str):
            assert session_id == "session-123"
            return SimpleNamespace(
                metadata=SimpleNamespace(id="session-123", name="Demo"),
                messages=[
                    {"role": "user", "content": "m0"},
                    {"role": "assistant", "content": "m1"},
                    {"role": "user", "content": "m2"},
                ],
                orchestration_state={},
                orchestration_runs=[],
                active_orchestration_run_id=None,
            )

    monkeypatch.setattr("src.webui.routes.sessions.SessionManager", StubSessionManager)

    payload = await get_session("session-123", message_limit=2)

    assert payload["message_count"] == 3
    assert payload["message_offset"] == 1
    assert payload["has_more_messages"] is True
    assert [message["content"] for message in payload["messages"]] == ["m1", "m2"]


@pytest.mark.asyncio
async def test_get_session_run_returns_full_run_details(monkeypatch):
    class StubSessionManager:
        def __init__(self, *args, **kwargs):
            return None

        def resume_session(self, session_id: str):
            assert session_id == "session-123"
            return SimpleNamespace(
                metadata=SimpleNamespace(id="session-123", name="Demo"),
                messages=[],
                orchestration_state={},
                orchestration_runs=[
                    {
                        "run_id": "run-1",
                        "success": True,
                        "summary": "planned: hello",
                        "task_graph": [{"id": "root", "title": "Root", "status": "completed", "ready": True}],
                        "worker_assignments": {"task-1": {"role": "inspect"}},
                    }
                ],
                active_orchestration_run_id="run-1",
            )

    monkeypatch.setattr("src.webui.routes.sessions.SessionManager", StubSessionManager)

    payload = await get_session_run("session-123", "run-1")

    assert payload["run"]["run_id"] == "run-1"
    assert payload["run"]["task_graph"][0]["id"] == "root"
    assert payload["run"]["worker_assignments"]["task-1"]["role"] == "inspect"


def test_root_api_routes_accept_paths_without_trailing_slash(monkeypatch):
    route_paths = {route.path for route in app.routes}

    assert "/api/chat" in route_paths
    assert "/api/projects" in route_paths
    assert "/api/sessions" in route_paths
    assert "/api/tasks" in route_paths
    assert "/api/tools" in route_paths


@pytest.mark.asyncio
async def test_get_project_context_returns_startup_directory():
    payload = await get_project_context()

    assert payload["project_dir"]
    assert payload["project_name"]


async def _noop_async():
    return None
