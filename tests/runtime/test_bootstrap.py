from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agent.adapter import AgentSession
from src.host.tools import ToolRegistry
from src.runtime.bootstrap import ProjectContext, RuntimeBootstrap, bootstrap_runtime
from src.core.tasks import TaskManager


class DummyRegistry:
    def __init__(self):
        self._tools = {}
        self._active_mcp_extensions = ["docs"]
        self._mcp_clients = []

    def get_tool_names(self) -> list[str]:
        return ["read", "write"]


@pytest.mark.asyncio
async def test_bootstrap_runtime_creates_missing_dependencies(monkeypatch, tmp_path):
    registry = DummyRegistry()
    created = {}

    class DummyCheckpointManager:
        def __init__(self, project: str):
            created["checkpoint_project"] = project

    class DummySkillManager:
        def __init__(self, project_dir: Path):
            created["skills_project_dir"] = project_dir

    class DummyHookManager:
        def __init__(self, project_dir: Path):
            created["hooks_project_dir"] = project_dir

    class DummyTaskManager:
        def __init__(self, storage_path=None, *, project_dir: Path | None = None):
            created["task_manager_project_dir"] = project_dir
            self.project_dir = project_dir

    async def fake_model_create():
        created["model_client"] = object()
        return created["model_client"]

    async def fake_create_tool_registry(config_path=None, *, mode=None, extension_tools=None):
        created["registry_args"] = {
            "config_path": config_path,
            "mode": mode,
            "extension_tools": extension_tools,
        }
        return registry

    import src.core.checkpoint as checkpoint_mod
    import src.core.hooks as hooks_mod
    import src.core.llm.model_client as model_client_mod
    import src.core.skills as skills_mod
    import src.core.tasks as tasks_mod
    import src.host.mcp_registry as mcp_registry_mod

    monkeypatch.setattr(checkpoint_mod, "CheckpointManager", DummyCheckpointManager)
    monkeypatch.setattr(skills_mod, "SkillManager", DummySkillManager)
    monkeypatch.setattr(hooks_mod, "HookManager", DummyHookManager)
    monkeypatch.setattr(tasks_mod, "TaskManager", DummyTaskManager)
    monkeypatch.setattr(model_client_mod.ModelClient, "create", staticmethod(fake_model_create))
    monkeypatch.setattr(mcp_registry_mod, "create_tool_registry", fake_create_tool_registry)

    runtime = await bootstrap_runtime(
        mode="plan",
        project="demo",
        project_dir=tmp_path,
        config_path=tmp_path / "config.json",
        extension_tools=["browser"],
    )

    assert runtime.model_client is created["model_client"]
    assert runtime.registry is registry
    assert runtime.owns_model_client is True
    assert runtime.owns_registry is True
    assert runtime.context == ProjectContext(
        project="demo",
        mode="plan",
        project_dir=tmp_path.resolve(),
        config_path=tmp_path / "config.json",
        capability_groups=["read", "memory", "research", "task"],
        tool_names=["read", "write"],
        active_mcp_extensions=["docs"],
    )
    assert created["checkpoint_project"] == "demo"
    assert created["skills_project_dir"] == tmp_path.resolve()
    assert created["hooks_project_dir"] == tmp_path.resolve()
    assert created["task_manager_project_dir"] == tmp_path.resolve()
    assert created["registry_args"]["mode"] == "plan"
    assert created["registry_args"]["extension_tools"] == ["browser"]


@pytest.mark.asyncio
async def test_bootstrap_runtime_reuses_provided_dependencies(monkeypatch, tmp_path):
    model_client = object()
    registry = DummyRegistry()
    hooks = object()
    checkpoints = object()
    skills = object()
    task_manager = object()

    async def fail_model_create():
        raise AssertionError("ModelClient.create should not be called")

    import src.core.llm.model_client as model_client_mod

    monkeypatch.setattr(model_client_mod.ModelClient, "create", staticmethod(fail_model_create))

    runtime = await bootstrap_runtime(
        mode="build",
        project="demo",
        project_dir=tmp_path,
        model_client=model_client,
        registry=registry,
        hooks=hooks,
        checkpoints=checkpoints,
        skills=skills,
        task_manager=task_manager,
    )

    assert runtime.model_client is model_client
    assert runtime.registry is registry
    assert runtime.hooks is hooks
    assert runtime.checkpoints is checkpoints
    assert runtime.skills is skills
    assert runtime.task_manager is task_manager
    assert runtime.owns_model_client is False
    assert runtime.owns_registry is False


@pytest.mark.asyncio
async def test_bootstrap_runtime_can_skip_model_client_creation(monkeypatch, tmp_path):
    async def fail_model_create():
        raise AssertionError("ModelClient.create should not be called")

    import src.core.llm.model_client as model_client_mod

    monkeypatch.setattr(model_client_mod.ModelClient, "create", staticmethod(fail_model_create))

    runtime = await bootstrap_runtime(
        mode="build",
        project="demo",
        project_dir=tmp_path,
        registry=DummyRegistry(),
        create_model_client=False,
    )

    assert runtime.model_client is None
    assert runtime.task_manager is not None
    assert runtime.owns_model_client is False
    assert runtime.owns_registry is False


@pytest.mark.asyncio
async def test_agent_session_initializes_from_shared_bootstrap(monkeypatch, tmp_path):
    registry = ToolRegistry()

    def read(path: str) -> str:
        return path

    registry.register(read, name="read")

    runtime = RuntimeBootstrap(
        context=ProjectContext(
            project="demo",
            mode="build",
            project_dir=tmp_path.resolve(),
            config_path=None,
            capability_groups=["read", "edit", "memory", "research", "task"],
            tool_names=["read"],
            active_mcp_extensions=["docs"],
        ),
        model_client=object(),
        registry=registry,
        hooks=None,
        checkpoints=None,
        skills=None,
        task_manager=object(),
        owns_model_client=False,
        owns_registry=False,
    )

    async def fake_bootstrap_runtime(**kwargs):
        return runtime

    monkeypatch.setattr("src.agent.adapter.bootstrap_runtime", fake_bootstrap_runtime)

    session = AgentSession(
        model_client=None,
        registry=None,
        project="demo",
        mode="build",
        project_dir=tmp_path,
        enable_trace=False,
    )

    await session.initialize()

    assert session.registry is registry
    assert session.model_client is runtime.model_client
    assert session._runtime is runtime
    assert session._mcp_extensions == ["docs"]
    assert session.get_task_manager() is runtime.task_manager
    assert [tool.name for tool in session._agent.tools] == ["read"]


@pytest.mark.asyncio
async def test_agent_session_orchestrate_uses_lead_runtime(monkeypatch, tmp_path):
    registry = ToolRegistry()

    def read(path: str) -> str:
        return path

    registry.register(read, name="read")

    runtime = RuntimeBootstrap(
        context=ProjectContext(
            project="demo",
            mode="build",
            project_dir=tmp_path.resolve(),
            config_path=None,
            capability_groups=["read", "edit", "memory", "research", "task"],
            tool_names=["read"],
            active_mcp_extensions=[],
        ),
        model_client=object(),
        registry=registry,
        hooks=None,
        checkpoints=None,
        skills=None,
        task_manager=TaskManager(storage_path=tmp_path / "tasks.json"),
        owns_model_client=False,
        owns_registry=False,
    )

    async def fake_bootstrap_runtime(**kwargs):
        return runtime

    async def fake_execute(self, goal: str, *, context=None):
        return SimpleNamespace(success=True, summary=f"planned: {goal}", goal=goal, context=context)

    monkeypatch.setattr("src.agent.adapter.bootstrap_runtime", fake_bootstrap_runtime)
    monkeypatch.setattr("src.runtime.lead.LeadAgentRuntime.execute", fake_execute)

    session = AgentSession(
        model_client=None,
        registry=None,
        project="demo",
        mode="build",
        project_dir=tmp_path,
        enable_trace=False,
    )

    result = await session.orchestrate("Implement authentication", context={"files": ["auth.py"]})

    assert result.success is True
    assert result.summary == "planned: Implement authentication"


@pytest.mark.asyncio
async def test_runtime_bootstrap_only_closes_owned_registry_clients():
    closed = []

    class DummyClient:
        def __init__(self, name: str):
            self.name = name

        async def close(self):
            closed.append(self.name)

    shared_runtime = RuntimeBootstrap(
        context=ProjectContext(
            project="demo",
            mode="build",
            project_dir=Path.cwd(),
            config_path=None,
            capability_groups=["read"],
            tool_names=["read"],
            active_mcp_extensions=[],
        ),
        model_client=None,
        registry=SimpleNamespace(_mcp_clients=[DummyClient("shared")]),
        hooks=None,
        checkpoints=None,
        skills=None,
        task_manager=None,
        owns_model_client=False,
        owns_registry=False,
    )

    owned_runtime = RuntimeBootstrap(
        context=shared_runtime.context,
        model_client=None,
        registry=SimpleNamespace(_mcp_clients=[DummyClient("owned")]),
        hooks=None,
        checkpoints=None,
        skills=None,
        task_manager=None,
        owns_model_client=False,
        owns_registry=True,
    )

    await shared_runtime.aclose()
    await owned_runtime.aclose()

    assert closed == ["owned"]
