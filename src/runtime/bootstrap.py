"""Shared runtime bootstrap for CLI, web, and automation entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.home import set_project_dir
from src.core.tool_selector import get_capability_groups_for_mode


@dataclass(slots=True)
class ProjectContext:
    """Stable project/runtime context shared across entry points."""

    project: str
    mode: str
    project_dir: Path
    config_path: Path | None
    capability_groups: list[str]
    tool_names: list[str]
    active_mcp_extensions: list[str]


@dataclass(slots=True)
class RuntimeBootstrap:
    """Bootstrapped runtime dependencies for an agent session."""

    context: ProjectContext
    model_client: Any | None
    registry: Any
    hooks: Any
    checkpoints: Any
    skills: Any
    task_manager: Any
    owns_model_client: bool = False
    owns_registry: bool = False

    async def aclose(self) -> None:
        """Close owned runtime resources."""
        if self.owns_registry:
            for client in getattr(self.registry, "_mcp_clients", []):
                await client.close()

        if self.owns_model_client and hasattr(self.model_client, "close"):
            await self.model_client.close()


async def bootstrap_runtime(
    *,
    mode: str = "build",
    project: str = "default",
    project_dir: Path | None = None,
    config_path: Path | None = None,
    extension_tools: list[str] | None = None,
    create_model_client: bool = True,
    model_client: Any | None = None,
    registry: Any | None = None,
    hooks: Any | None = None,
    checkpoints: Any | None = None,
    skills: Any | None = None,
    task_manager: Any | None = None,
) -> RuntimeBootstrap:
    """Create the shared runtime objects used by all entry points."""
    project_dir = (project_dir or Path.cwd()).resolve()
    set_project_dir(project_dir)

    owns_model_client = False
    if model_client is None and create_model_client:
        from src.core.llm.model_client import ModelClient

        model_client = await ModelClient.create()
        owns_model_client = True

    if checkpoints is None:
        from src.core.checkpoint import CheckpointManager

        checkpoints = CheckpointManager(project=project)

    if skills is None:
        from src.core.skills import SkillManager

        skills = SkillManager(project_dir=project_dir)

    if hooks is None:
        from src.core.hooks import HookManager

        hooks = HookManager(project_dir=project_dir)

    if task_manager is None:
        from src.core.tasks import TaskManager

        task_manager = TaskManager(project_dir=project_dir)

    owns_registry = False
    if registry is None:
        from src.host.mcp_registry import create_tool_registry

        registry = await create_tool_registry(
            config_path=config_path,
            mode=mode,
            extension_tools=extension_tools,
        )
        owns_registry = True

    context = ProjectContext(
        project=project,
        mode=mode,
        project_dir=project_dir,
        config_path=config_path,
        capability_groups=get_capability_groups_for_mode(mode),
        tool_names=registry.get_tool_names(),
        active_mcp_extensions=getattr(registry, "_active_mcp_extensions", []).copy(),
    )

    return RuntimeBootstrap(
        context=context,
        model_client=model_client,
        registry=registry,
        hooks=hooks,
        checkpoints=checkpoints,
        skills=skills,
        task_manager=task_manager,
        owns_model_client=owns_model_client,
        owns_registry=owns_registry,
    )


async def get_tool_catalog(
    *,
    mode: str = "build",
    config_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return the visible tool catalog for a runtime mode."""
    from src.host.mcp_registry import create_tool_registry

    registry = await create_tool_registry(config_path=config_path, mode=mode)
    try:
        return [
            {
                "name": definition.name,
                "description": definition.description,
                "parameters": definition.parameters,
                "source": definition.source,
                "sensitive": definition.sensitive,
                "metadata": definition.metadata,
                "policy": registry.get_tool_policy(
                    definition.name,
                    mode=mode,
                    active_mcp_extensions=getattr(registry, "_active_mcp_extensions", []),
                ),
            }
            for definition in registry._tools.values()
        ]
    finally:
        for client in getattr(registry, "_mcp_clients", []):
            await client.close()
