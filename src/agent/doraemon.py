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

from src.core.checkpoint import CheckpointManager
from src.core.hooks import HookManager
from src.core.llm.model_client import ModelClient
from src.core.skills import SkillManager
from src.core.tasks import TaskManager
from src.host.tools import ToolRegistry

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
from src.core.tasks import TaskStatus
from src.core.tool_selector import get_capability_groups_for_mode
from src.host.tools import LazyToolFunction

logger = logging.getLogger(__name__)


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
        llm_client: ModelClient,
        tool_registry: ToolRegistry,
        state: AgentState | None = None,
        *,
        hooks: HookManager | None = None,
        checkpoints: CheckpointManager | None = None,
        skills: SkillManager | None = None,
        task_manager: TaskManager | None = None,
        permission_callback: Callable | None = None,
        display_callback: Callable | None = None,
        project_dir: Path | None = None,
        enable_trace: bool = True,
        trace: Trace | None = None,
        session_id: str | None = None,
        active_mcp_extensions: list[str] | None = None,
        worker_role: str | None = None,
        allowed_tool_names: list[str] | None = None,
        **kwargs,
    ):
        mode = state.mode if state else "build"
        tools = self._convert_registry_to_tools(
            tool_registry,
            mode=mode,
            active_mcp_extensions=active_mcp_extensions or [],
            allowed_tool_names=allowed_tool_names,
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
        self.task_manager = task_manager
        self.display_callback = display_callback
        self.project_dir = project_dir
        self.enable_trace = enable_trace
        self.session_id = session_id or self._generate_session_id()
        self.active_mcp_extensions = (active_mcp_extensions or []).copy()
        self.worker_role = worker_role
        self.allowed_tool_names = (
            set(allowed_tool_names) if allowed_tool_names is not None else None
        )
        self._active_trace_run_id: str | None = None

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

    def _begin_runtime_task(self, user_input: str) -> str | None:
        """Create and claim a runtime task for the current top-level agent turn."""
        if self.task_manager is None:
            return None

        title = " ".join(user_input.strip().split())
        if not title:
            title = "Agent task"
        if len(title) > 80:
            title = f"{title[:77]}..."

        task = self.task_manager.create_task(
            title=title,
            description=user_input,
        )
        try:
            self.task_manager.claim_task(task.id, self.session_id)
        except Exception:
            self.task_manager.update_task(
                task.id,
                status=TaskStatus.IN_PROGRESS,
                assigned_agent=self.session_id,
            )
        return task.id

    def _finish_runtime_task(
        self, task_id: str | None, *, success: bool, error: str | None = None
    ) -> None:
        """Finalize the current runtime task after an agent turn completes."""
        if self.task_manager is None or task_id is None:
            return

        final_status = TaskStatus.COMPLETED if success else TaskStatus.BLOCKED
        self.task_manager.update_task(
            task_id,
            status=final_status,
            assigned_agent=None,
        )

    def _get_runtime_tool_policy(self, tool_name: str) -> dict[str, Any] | None:
        """Safely fetch runtime tool policy from a registry that may be mocked."""
        get_tool_policy = getattr(self.tool_registry, "get_tool_policy", None)
        if not callable(get_tool_policy):
            return None

        policy = get_tool_policy(
            tool_name,
            mode=self.state.mode,
            active_mcp_extensions=self.active_mcp_extensions,
        )
        return policy if isinstance(policy, dict) else None

    def _convert_registry_to_tools(
        self,
        registry: Any,
        *,
        mode: str = "build",
        active_mcp_extensions: list[str] | None = None,
        allowed_tool_names: list[str] | None = None,
    ) -> list[ToolDefinition]:
        """Convert registry tool definitions into agent ToolDefinitions."""
        tools: list[ToolDefinition] = []
        tool_name_set: set[str] = set()
        allowed_names = set(allowed_tool_names) if allowed_tool_names is not None else None

        if hasattr(registry, "_tool_schemas"):
            for name, schema in registry._tool_schemas.items():
                if allowed_names is not None and name not in allowed_names:
                    continue
                if hasattr(registry, "get_tool_policy"):
                    policy = registry.get_tool_policy(
                        name,
                        mode=mode,
                        active_mcp_extensions=active_mcp_extensions,
                    )
                    if not isinstance(policy, dict):
                        policy = None
                    if policy is not None and not policy["visible"]:
                        continue
                    sensitive = (
                        policy["requires_approval"]
                        if policy is not None
                        else (
                            name in registry._sensitive_tools
                            if hasattr(registry, "_sensitive_tools")
                            else False
                        )
                    )
                else:
                    sensitive = (
                        name in registry._sensitive_tools
                        if hasattr(registry, "_sensitive_tools")
                        else False
                    )
                tools.append(
                    ToolDefinition(
                        name=schema["name"],
                        description=schema.get("description", ""),
                        parameters=schema.get("parameters", {}),
                        sensitive=sensitive,
                    )
                )
                tool_name_set.add(name)

        if hasattr(registry, "_tools"):
            for name in registry.get_tool_names():
                if allowed_names is not None and name not in allowed_names:
                    continue
                tool_def = registry._tools.get(name)
                if not tool_def:
                    continue

                policy = None
                if hasattr(registry, "get_tool_policy"):
                    policy = registry.get_tool_policy(
                        name,
                        mode=mode,
                        active_mcp_extensions=active_mcp_extensions,
                    )
                    if not isinstance(policy, dict):
                        policy = None
                if policy is not None and not policy["visible"]:
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
                            sensitive=policy["requires_approval"]
                            if policy is not None
                            else tool_def.sensitive,
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
        policy = None
        if self.allowed_tool_names is not None and name not in self.allowed_tool_names:
            logger.warning("Worker tool scope denied access to '%s'", name)
            return (
                "",
                f"Permission Error: Tool '{name}' is not available for worker role '{self.worker_role or 'default'}'.",
            )

        check_tool_execution = getattr(self.tool_registry, "check_tool_execution", None)
        if callable(check_tool_execution):
            decision = check_tool_execution(
                name,
                mode=self.state.mode,
                active_mcp_extensions=self.active_mcp_extensions,
            )
            if isinstance(decision, tuple) and len(decision) == 3:
                allowed, reason, policy = decision
            else:
                policy = self._get_runtime_tool_policy(name)
                if policy is not None and not policy["visible"]:
                    allowed = False
                    reason = f"Tool '{name}' is not available in {self.state.mode} mode."
                else:
                    allowed, reason = True, None
            if not allowed:
                logger.warning(
                    f"Permission denied: Tool '{name}' is not allowed in '{self.state.mode}' mode"
                )
                return "", f"Permission Error: {reason}"

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
            raw_metadata = getattr(tool_definition, "metadata", None) if tool_definition else None
            metadata = dict(raw_metadata or {})
            if self._active_trace_run_id:
                metadata["run_id"] = self._active_trace_run_id
            self._trace.tool_call(
                name,
                arguments,
                result or f"Error: {error}",
                duration,
                error=error,
                source=source,
                metadata=metadata,
            )

        if hasattr(self.tool_registry, "record_tool_execution"):
            self.tool_registry.record_tool_execution(
                name,
                action="executed" if error is None else "failed",
                mode=self.state.mode,
                active_mcp_extensions=self.active_mcp_extensions,
                allowed=error is None,
                error=error,
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

    async def _execute_tool_with_permission(
        self,
        name: str,
        args: dict[str, Any],
    ) -> tuple[str, str | None]:
        """Execute tool with policy-backed approval tracking."""
        policy = self._get_runtime_tool_policy(name)

        needs_approval = (policy["requires_approval"] if policy is not None else False) or (
            self.is_sensitive_tool(name)
        )
        if needs_approval:
            allowed = await self.check_permission(name, args)
            if hasattr(self.tool_registry, "record_tool_execution"):
                self.tool_registry.record_tool_execution(
                    name,
                    action="approval_granted" if allowed else "approval_denied",
                    mode=self.state.mode,
                    active_mcp_extensions=self.active_mcp_extensions,
                    allowed=allowed,
                    error=None if allowed else "Permission denied by user",
                )
            if not allowed:
                return "", "Permission denied by user"

        return await self.execute_tool(name, args)

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
        create_runtime_task: bool = False,
        **kwargs,
    ) -> AgentResult:
        """Run the agent with lifecycle hooks and trace recording."""
        trace_run_id = kwargs.pop("trace_run_id", None)
        runtime_task_id = self._begin_runtime_task(input) if create_runtime_task else None
        previous_trace_run_id = self._active_trace_run_id
        self._active_trace_run_id = trace_run_id

        task_manager_token = None
        if self.task_manager is not None:
            from src.servers.task import set_task_manager

            task_manager_token = set_task_manager(self.task_manager)

        if self.enable_trace and self._trace:
            self._trace.start_turn(
                input,
                metadata={
                    "mode": self.state.mode,
                    "capability_groups": get_capability_groups_for_mode(self.state.mode),
                    "active_tools": [tool.name for tool in self.tools],
                    "active_skills": [],
                    "active_mcp_extensions": self.active_mcp_extensions.copy(),
                    "run_id": trace_run_id,
                    "runtime_task_id": runtime_task_id,
                    "worker_role": self.worker_role,
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

            self._finish_runtime_task(
                runtime_task_id,
                success=result.success,
                error=result.error,
            )

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
            self._finish_runtime_task(runtime_task_id, success=False, error=str(e))
            raise
        finally:
            self._active_trace_run_id = previous_trace_run_id
            if task_manager_token is not None:
                from src.servers.task import reset_task_manager

                reset_task_manager(task_manager_token)

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
            base_prompt = """You are a planning agent. Analyze the task and produce a concrete implementation strategy.

IMPORTANT:
- Use read-only tools to explore the codebase
- Do NOT modify any files
- Base your strategy on actual repository state, not assumptions
- End your response with "## Plan" followed by numbered steps

Available tools are for reading and searching only."""
            if self.worker_role:
                return (
                    f"{base_prompt}\n\nEXECUTION PROFILE:\n"
                    f"- You are operating under the '{self.worker_role}' profile for a coordinating agent.\n"
                    "- Stay within your visible tool scope and return concrete findings."
                )
            return base_prompt

        base_prompt = """You are a coding agent. Complete the given task using available tools.

OPERATING PRINCIPLES:
- Inspect the relevant context before making changes
- Choose the next action based on the current repository state
- Make only the changes needed to complete the task
- Validate the result before concluding when validation is possible

TOOLS:
- read: Read files, directories, or code outlines
- write: Create, edit, or delete files
- search: Find files or search content
- run: Execute commands (tests, builds, etc.)

Always verify your changes by running relevant tests or checks."""
        if self.worker_role:
            return (
                f"{base_prompt}\n\nEXECUTION PROFILE:\n"
                f"- You are operating under the '{self.worker_role}' profile for a coordinating agent.\n"
                "- Use only the tools visible to your role.\n"
                "- Keep your output focused on the subtask and hand results back to the coordinator."
            )
        return base_prompt


def create_doraemon_agent(
    llm_client: ModelClient,
    tool_registry: ToolRegistry,
    mode: str = "build",
    hooks: HookManager | None = None,
    checkpoints: CheckpointManager | None = None,
    skills: SkillManager | None = None,
    task_manager: TaskManager | None = None,
    permission_callback: Callable | None = None,
    display_callback: Callable | None = None,
    max_turns: int = 100,
    project_dir: Path | None = None,
    enable_trace: bool = True,
    trace: Trace | None = None,
    session_id: str | None = None,
    active_mcp_extensions: list[str] | None = None,
    worker_role: str | None = None,
    allowed_tool_names: list[str] | None = None,
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
        task_manager=task_manager,
        permission_callback=permission_callback,
        display_callback=display_callback,
        max_turns=max_turns,
        project_dir=project_dir,
        enable_trace=enable_trace,
        trace=trace,
        session_id=session_id,
        active_mcp_extensions=active_mcp_extensions,
        worker_role=worker_role,
        allowed_tool_names=allowed_tool_names,
    )


async def create_doraemon_agent_with_tools(
    llm_client: ModelClient,
    mode: str = "build",
    config_path: Path | None = None,
    hooks: HookManager | None = None,
    checkpoints: CheckpointManager | None = None,
    skills: SkillManager | None = None,
    task_manager: TaskManager | None = None,
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
        task_manager=task_manager,
        permission_callback=permission_callback,
        display_callback=display_callback,
        max_turns=max_turns,
        active_mcp_extensions=active_mcp_extensions,
    )


async def create_doraemon_agent_with_mcp(
    llm_client: ModelClient,
    mode: str = "build",
    config_path: Path | None = None,
    hooks: HookManager | None = None,
    checkpoints: CheckpointManager | None = None,
    skills: SkillManager | None = None,
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
