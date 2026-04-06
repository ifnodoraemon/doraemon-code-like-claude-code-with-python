"""
Doraemon Agent - Production Agent for Doraemon Code

Integrates the standard ReActAgent with Doraemon's existing infrastructure:
- ToolRegistry for built-in tool execution
- HookManager for lifecycle events
- Skills system
- Checkpoint system
- HITL approval workflow
- Trace recording
"""

import logging
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.agent.react import ReActAgent
from src.agent.state import AgentState
from src.agent.types import (
    Action,
    AgentResult,
    Observation,
    Thought,
    ToolDefinition,
)
from src.core.home import Trace, set_project_dir
from src.core.hooks import HookEvent, HookManager
from src.core.tool_selector import get_capability_groups_for_mode

logger = logging.getLogger(__name__)


class PermissionMatrix:
    """Hard enforcement of tool permissions based on agent mode."""

    MODE_PERMISSIONS = {
        "plan": {
            "read",
            "search",
            "ask_user",
            "ls",
            "grep",
            "find",
            "cat",
            "get_codebase_map",
            "get_file_outline",
        },
        "build": {"all"},
    }

    @classmethod
    def check(cls, mode: str, tool_name: str) -> bool:
        allowed = cls.MODE_PERMISSIONS.get(mode, set())
        if "all" in allowed:
            return True
        return tool_name in allowed


class DoraemonAgent(ReActAgent):
    """
    Production agent for Doraemon Code.

    Extends ReActAgent with:
    - Integration with existing ToolRegistry
    - Hook lifecycle events
    - Checkpoint management
    - Skills integration
    - Rich display output
    - Trace recording with session persistence
    """

    def __init__(
        self,
        llm_client: Any,
        tool_registry: Any,
        state: AgentState | None = None,
        *,
        hooks: HookManager | None = None,
        checkpoints: Any = None,
        skills: Any = None,
        permission_callback: Callable | None = None,
        display_callback: Callable | None = None,
        project_dir: Path | None = None,
        enable_trace: bool = True,
        trace: Trace | None = None,
        session_id: str | None = None,
        active_mcp_extensions: list[str] | None = None,
        **kwargs,
    ):
        mode = state.mode if state else "build"
        tools = self._convert_registry_to_tools(
            tool_registry,
            mode=mode,
            active_mcp_extensions=active_mcp_extensions or [],
        )
        super().__init__(
            llm_client=llm_client,
            state=state,
            tools=tools,
            permission_callback=permission_callback,
            **kwargs,
        )

        self.tool_registry = tool_registry
        self.hooks = hooks
        self.checkpoints = checkpoints
        self.skills = skills
        self.display_callback = display_callback
        self.project_dir = project_dir
        self.enable_trace = enable_trace
        self.session_id = session_id or self._generate_session_id()
        self.active_mcp_extensions = (active_mcp_extensions or []).copy()

        if trace:
            self._trace = trace
        elif enable_trace:
            self._trace = Trace(self.session_id, project_dir)
        else:
            self._trace = None

        if self._trace:
            self.set_trace(self._trace)

        if project_dir:
            set_project_dir(project_dir)

    @staticmethod
    def _generate_session_id() -> str:
        """Generate a unique session ID."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique = str(uuid.uuid4())[:8]
        return f"{timestamp}_{unique}"

    def get_trace(self) -> Trace | None:
        """Get current trace object."""
        return self._trace

    def save_trace(self) -> Path | None:
        """Save trace to file, returns path if saved."""
        if self._trace:
            return self._trace.save()
        return None

    def _convert_registry_to_tools(
        self,
        registry: Any,
        *,
        mode: str = "build",
        active_mcp_extensions: list[str] | None = None,
    ) -> list[ToolDefinition]:
        """Convert registry tool definitions into agent ToolDefinitions."""
        from src.core.tool_selector import get_tools_for_mode
        from src.host.mcp_registry import resolve_extension_tools
        from src.host.tools import LazyToolFunction

        tools = []
        tool_name_set: set[str] = set()
        allowed_tool_names = set(get_tools_for_mode(mode))
        allowed_tool_names.update(resolve_extension_tools(active_mcp_extensions or []))

        if hasattr(registry, "_tool_schemas"):
            for name, schema in registry._tool_schemas.items():
                tools.append(
                    ToolDefinition(
                        name=schema["name"],
                        description=schema.get("description", ""),
                        parameters=schema.get("parameters", {}),
                        sensitive=name in registry._sensitive_tools
                        if hasattr(registry, "_sensitive_tools")
                        else False,
                    )
                )
                tool_name_set.add(name)

        if hasattr(registry, "_tools"):
            for name in registry.get_tool_names():
                tool_def = registry._tools.get(name)
                if not tool_def:
                    continue

                is_remote_mcp_tool = getattr(
                    tool_def, "source", "built_in"
                ) == "mcp_remote" and tool_def.metadata.get("mcp_server") in (
                    active_mcp_extensions or []
                )
                if name not in allowed_tool_names and not is_remote_mcp_tool:
                    continue
                if name not in tool_name_set:
                    func = getattr(tool_def, "function", None)
                    if isinstance(func, LazyToolFunction) and getattr(func, "_load_error", None):
                        logger.warning(
                            "Skipping unavailable tool '%s' from agent-visible tool list: %s",
                            name,
                            func._load_error,
                        )
                        continue
                    tools.append(
                        ToolDefinition(
                            name=tool_def.name,
                            description=tool_def.description,
                            parameters=tool_def.parameters,
                            sensitive=tool_def.sensitive,
                        )
                    )
                    tool_name_set.add(name)

        return tools

    async def execute_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> tuple[str, str | None]:
        """
        Execute a tool from the registry with hooks, trace recording,
        and hard permission enforcement.
        """
        if not PermissionMatrix.check(self.state.mode, name):
            logger.warning(
                f"Permission denied: Tool '{name}' is not allowed in '{self.state.mode}' mode"
            )
            return (
                "",
                f"Permission Error: Tool '{name}' is not available in {self.state.mode} mode.",
            )

        start_time = time.time()

        if self.hooks:
            hook_result = await self.hooks.trigger(
                HookEvent.PRE_TOOL_USE,
                tool_name=name,
                tool_input=arguments,
            )

            if hook_result.decision.value == "deny":
                return "", hook_result.reason or "Blocked by hook"

            if hook_result.modified_input:
                arguments = hook_result.modified_input

        try:
            result = await self.tool_registry.call_tool(name, arguments)
            error = None
        except Exception as e:
            result = ""
            error = str(e)

        duration = time.time() - start_time

        if self._trace and self.enable_trace:
            tool_definition = None
            if hasattr(self.tool_registry, "_tools"):
                tool_definition = self.tool_registry._tools.get(name)
            source = (
                getattr(tool_definition, "source", "built_in") if tool_definition else "built_in"
            )
            metadata = getattr(tool_definition, "metadata", {}) if tool_definition else {}
            self._trace.tool_call(
                name,
                arguments,
                result or f"Error: {error}",
                duration,
                error=error,
                source=source,
                metadata=metadata,
            )

        if self.hooks:
            await self.hooks.trigger(
                HookEvent.POST_TOOL_USE,
                tool_name=name,
                tool_input=arguments,
                tool_output=result,
            )

        if self.checkpoints and self._is_modifying_tool(name):
            await self._create_checkpoint(name, arguments)

        return result, error

    def _is_modifying_tool(self, name: str) -> bool:
        """Check if a tool modifies files."""
        return name == "write"

    @staticmethod
    def _get_checkpoint_paths(args: dict[str, Any]) -> list[str]:
        """Resolve paths that should be snapshotted for a write operation."""
        paths: list[str] = []

        path = args.get("path")
        if path:
            paths.append(path)

        operation = args.get("operation", "create")
        if operation in {"move", "copy"}:
            destination = args.get("destination")
            if destination:
                paths.append(destination)

        return paths

    async def _create_checkpoint(
        self,
        tool_name: str,
        args: dict[str, Any],
    ) -> None:
        """Create a checkpoint before file modifications."""
        if not self.checkpoints:
            return

        for path in self._get_checkpoint_paths(args):
            try:
                self.checkpoints.snapshot(
                    path,
                    reason=f"Before {tool_name}",
                )
            except Exception as e:
                logger.warning("Failed to create checkpoint for %s: %s", path, e)

    async def run(
        self,
        input: str,
        **kwargs,
    ) -> AgentResult:
        """Run the agent with lifecycle hooks and trace recording."""
        if self.enable_trace and self._trace:
            self._trace.start_turn(
                input,
                metadata={
                    "mode": self.state.mode,
                    "capability_groups": get_capability_groups_for_mode(self.state.mode),
                    "active_tools": [tool.name for tool in self.tools],
                    "active_skills": [],
                    "active_mcp_extensions": self.active_mcp_extensions.copy(),
                },
            )

        if self.hooks:
            await self.hooks.trigger(
                HookEvent.USER_PROMPT_SUBMIT,
                user_prompt=input,
            )

        try:
            result = await super().run(input, **kwargs)

            if self._trace and self.enable_trace:
                self._trace.end_turn(success=result.success, error=result.error)

            if self.hooks:
                await self.hooks.trigger(
                    HookEvent.STOP,
                    stop_reason="completed" if result.success else "error",
                )

            return result

        except Exception as e:
            if self._trace and self.enable_trace:
                self._trace.error(
                    message=str(e),
                    details={"exception_type": type(e).__name__},
                )
                self._trace.end_turn(success=False, error=str(e))
            raise

    async def observe(self) -> Observation:
        """Observe with skills context."""
        observation = await super().observe()

        if self.skills:
            skills_content = self.skills.get_skills_for_context(
                observation.user_input or "",
                mode=self.state.mode,
            )
            if skills_content:
                observation.context["skills"] = skills_content
                if self._trace and self.enable_trace:
                    self._trace.event(
                        "context",
                        "skills_activated",
                        {
                            "active_skills": self.skills.get_active_skills(),
                        },
                    )

        return observation

    async def think(self, observation: Observation) -> Thought:
        """Think with display output."""
        if self.display_callback:
            await self.display_callback("thinking", {"observation": observation})

        thought = await super().think(observation)

        if self.display_callback and thought.reasoning:
            await self.display_callback("thought", {"reasoning": thought.reasoning})

        return thought

    async def act(self, thought: Thought) -> Action:
        """Act with display output."""
        action = await super().act(thought)

        if self.display_callback:
            await self.display_callback("action", {"action": action.to_dict()})

        return action

    def _build_messages(self, observation: Observation) -> list[dict]:
        """Build messages with skills and context."""
        messages = super()._build_messages(observation)

        if "skills" in observation.context:
            for i, msg in enumerate(messages):
                if msg.get("role") == "system":
                    messages[i]["content"] += (
                        f"\n\n## Active Skills\n{observation.context['skills']}"
                    )
                    break

        return messages

    def _get_system_prompt(self) -> str:
        """Get mode-specific system prompt."""
        if self.state.mode == "plan":
            return """You are a planning agent. Analyze the task and create a step-by-step plan.

IMPORTANT:
- Use read-only tools to explore the codebase
- Do NOT modify any files
- Create a detailed plan with specific steps
- End your response with "## Plan" followed by numbered steps

Available tools are for reading and searching only."""

        return """You are a coding agent. Complete the given task using available tools.

WORKFLOW:
1. Understand the task and explore relevant files
2. Plan your approach
3. Make necessary changes
4. Verify your changes work correctly

TOOLS:
- read: Read files, directories, or code outlines
- write: Create, edit, or delete files
- search: Find files or search content
- run: Execute commands (tests, builds, etc.)

Always verify your changes by running relevant tests or checks."""


def create_doraemon_agent(
    llm_client: Any,
    tool_registry: Any,
    mode: str = "build",
    hooks: HookManager | None = None,
    checkpoints: Any = None,
    skills: Any = None,
    permission_callback: Callable | None = None,
    display_callback: Callable | None = None,
    max_turns: int = 100,
    project_dir: Path | None = None,
    enable_trace: bool = True,
    trace: Trace | None = None,
    session_id: str | None = None,
    active_mcp_extensions: list[str] | None = None,
) -> DoraemonAgent:
    """Factory function to create a DoraemonAgent."""
    state = AgentState(mode=mode, max_turns=max_turns)

    return DoraemonAgent(
        llm_client=llm_client,
        tool_registry=tool_registry,
        state=state,
        hooks=hooks,
        checkpoints=checkpoints,
        skills=skills,
        permission_callback=permission_callback,
        display_callback=display_callback,
        max_turns=max_turns,
        project_dir=project_dir,
        enable_trace=enable_trace,
        trace=trace,
        session_id=session_id,
        active_mcp_extensions=active_mcp_extensions,
    )


async def create_doraemon_agent_with_tools(
    llm_client: Any,
    mode: str = "build",
    config_path: Path | None = None,
    hooks: HookManager | None = None,
    checkpoints: Any = None,
    skills: Any = None,
    permission_callback: Callable | None = None,
    display_callback: Callable | None = None,
    max_turns: int = 100,
) -> DoraemonAgent:
    """Create a DoraemonAgent with the built-in tool registry."""
    from src.host.mcp_registry import create_tool_registry

    registry = await create_tool_registry(config_path, mode=mode)
    active_mcp_extensions = getattr(registry, "_active_mcp_extensions", []).copy()

    return create_doraemon_agent(
        llm_client=llm_client,
        tool_registry=registry,
        mode=mode,
        hooks=hooks,
        checkpoints=checkpoints,
        skills=skills,
        permission_callback=permission_callback,
        display_callback=display_callback,
        max_turns=max_turns,
        active_mcp_extensions=active_mcp_extensions,
    )


async def create_doraemon_agent_with_mcp(
    llm_client: Any,
    mode: str = "build",
    config_path: Path | None = None,
    hooks: HookManager | None = None,
    checkpoints: Any = None,
    skills: Any = None,
    permission_callback: Callable | None = None,
    display_callback: Callable | None = None,
    max_turns: int = 100,
) -> DoraemonAgent:
    """Backward-compatible alias for older callers."""
    return await create_doraemon_agent_with_tools(
        llm_client=llm_client,
        mode=mode,
        config_path=config_path,
        hooks=hooks,
        checkpoints=checkpoints,
        skills=skills,
        permission_callback=permission_callback,
        display_callback=display_callback,
        max_turns=max_turns,
    )
