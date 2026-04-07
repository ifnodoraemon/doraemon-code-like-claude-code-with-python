import json
from types import SimpleNamespace

import pytest

from src.webui.routes.chat import ChatRequest, chat_endpoint
from src.webui.routes.tasks import list_tasks


class StubTaskManager:
    def get_task_tree(self):
        return [{"id": "root", "title": "Root", "status": "completed", "ready": True}]

    def list_tasks(self):
        return [SimpleNamespace(to_dict=lambda: {"id": "root", "title": "Root"})]

    def list_ready_tasks(self):
        return [SimpleNamespace(to_dict=lambda: {"id": "ready", "title": "Ready"})]


@pytest.mark.asyncio
async def test_chat_endpoint_streams_orchestration_result(monkeypatch):
    class StubSession:
        def __init__(self, *args, **kwargs):
            self._task_manager = StubTaskManager()

        async def orchestrate(self, message: str, *, context=None, max_workers=None):
            return SimpleNamespace(
                summary=f"planned: {message}",
                to_dict=lambda: {
                    "success": True,
                    "summary": f"planned: {message}",
                    "max_workers": max_workers,
                },
            )

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
        )
    )

    payloads = []
    async for chunk in response.body_iterator:
        payloads.append(chunk)

    assert payloads[-1] == "data: [DONE]\n\n"
    first_payload = json.loads(payloads[0].removeprefix("data: ").strip())
    assert first_payload["type"] == "orchestration"
    assert first_payload["content"] == "planned: Implement authentication"
    assert first_payload["result"]["max_workers"] == 3
    assert first_payload["task_graph"][0]["id"] == "root"


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

    assert listing["tasks"][0]["id"] == "root"
    assert listing["tree"][0]["id"] == "root"
    assert ready_listing["tasks"][0]["id"] == "ready"


async def _noop_async():
    return None
