"""
Agent Adapter - Bridge between old chat_loop and new Agent architecture

This module provides a compatibility layer that allows the existing
chat_loop.py to use the new Agent architecture without major changes.

Usage:
    from src.agent.adapter import run_agent_turn, AgentSession

    result = await run_agent_turn(
        user_input="Read the README",
        model_client=model_client,
        registry=registry,
        ...
    )

    # Or use AgentSession for multi-turn conversations:
    session = AgentSession(model_client, mode="build")
    await session.initialize()
    result = await session.turn("First message")
    result = await session.turn("Second message")
    session.close()
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.agent.doraemon import DoraemonAgent, create_doraemon_agent
from src.agent.state import AgentState
from src.core.home import Trace
from src.runtime import LeadExecutionResult, LeadAgentRuntime
from src.runtime.bootstrap import RuntimeBootstrap, bootstrap_runtime

logger = logging.getLogger(__name__)
console = Console()


def _collect_modified_paths(tool_calls: list[Any]) -> list[str]:
    """Collect modified paths from unified write tool calls."""
    modified_paths: list[str] = []

    for tc in tool_calls:
        if tc.name != "write":
            continue

        arguments = tc.arguments or {}
        operation = arguments.get("operation", "create")

        path = arguments.get("path")
        if path:
            modified_paths.append(path)

        if operation in {"move", "copy"}:
            destination = arguments.get("destination")
            if destination:
                modified_paths.append(destination)

    return modified_paths


@dataclass
class AgentTurnResult:
    """Result from running an agent turn, compatible with chat_loop."""

    response: str
    tool_calls: list[dict[str, Any]]
    tokens_used: int
    duration: float
    success: bool
    error: str | None = None
    files_modified: list[str] = None

    def __post_init__(self):
        if self.files_modified is None:
            self.files_modified = []


async def run_agent_turn(
    user_input: str,
    model_client: Any,
    registry: Any,
    context: Any,
    *,
    mode: str = "build",
    hooks: Any = None,
    checkpoints: Any = None,
    skills: Any = None,
    sensitive_tools: set[str] | None = None,
    headless: bool = False,
    permission_callback: Callable | None = None,
    max_turns: int = 15,
    display_output: bool = True,
) -> AgentTurnResult:
    """
    Run a single agent turn.

    This function bridges the old chat_loop interface with the new Agent.

    Args:
        user_input: User's input message
        model_client: LLM client
        registry: Tool registry
        context: Context manager (for history)
        mode: Agent mode ("plan" or "build")
        hooks: Hook manager
        checkpoints: Checkpoint manager
        skills: Skill manager
        sensitive_tools: Set of sensitive tool names
        headless: Whether running in headless mode
        permission_callback: Callback for HITL approval
        max_turns: Maximum turns per session
        display_output: Whether to display output to console

    Returns:
        AgentTurnResult with response and metadata
    """
    start_time = time.time()

    async def display_callback(event_type: str, data: dict):
        if not display_output:
            return

        if event_type == "thinking":
            pass
        elif event_type == "thought":
            if data.get("reasoning"):
                console.print(
                    Panel(
                        Markdown(data["reasoning"]),
                        title="[bold dim]Thinking[/bold dim]",
                        border_style="dim white",
                        expand=False,
                    )
                )
        elif event_type == "action":
            action = data.get("action")
            if isinstance(action, dict) and action.get("tool_name"):
                console.print(f"[dim]→ Tool: {action['tool_name']}[/dim]")

    async def default_permission_callback(
        tool_name: str,
        args: dict[str, Any],
    ) -> bool:
        if headless:
            return False

        if permission_callback:
            return await permission_callback(tool_name, args)

        from rich.prompt import Confirm

        return Confirm.ask(
            f"[yellow]Allow tool call:[/yellow] {tool_name}({list(args.keys())})?",
            default=True,
        )

    agent = create_doraemon_agent(
        llm_client=model_client,
        tool_registry=registry,
        mode=mode,
        hooks=hooks,
        checkpoints=checkpoints,
        skills=skills,
        permission_callback=default_permission_callback if sensitive_tools else None,
        display_callback=display_callback,
        max_turns=max_turns,
    )

    if context and hasattr(context, "messages"):
        for msg in context.messages[-10:]:
            if hasattr(msg, "role") and hasattr(msg, "content"):
                agent.state.add_message(
                    type(
                        "M",
                        (),
                        {
                            "role": msg.role,
                            "content": getattr(msg, "content", None),
                            "tool_calls": getattr(msg, "tool_calls", None),
                            "to_api_format": lambda m=msg: {
                                "role": m.role,
                                "content": getattr(m, "content", ""),
                            },
                        },
                    )()
                )

    try:
        result = await agent.run(user_input)

        return AgentTurnResult(
            response=result.response or "",
            tool_calls=[tc.to_dict() for tc in result.tool_calls],
            tokens_used=result.tokens_used,
            duration=time.time() - start_time,
            success=result.success,
            error=result.error,
            files_modified=_collect_modified_paths(result.tool_calls),
        )

    except asyncio.TimeoutError:
        return AgentTurnResult(
            response="",
            tool_calls=[],
            tokens_used=0,
            duration=time.time() - start_time,
            success=False,
            error="Agent timed out",
            files_modified=[],
        )
    except Exception as e:
        logger.error(f"Agent turn failed: {e}")
        return AgentTurnResult(
            response="",
            tool_calls=[],
            tokens_used=0,
            duration=time.time() - start_time,
            success=False,
            error=str(e),
            files_modified=[],
        )


class AgentSession:
    """
    Manages a persistent agent session.

    Use this for multi-turn conversations where context should persist.

    Session Lifecycle:
        session = AgentSession(...)
        await session.initialize()
        result1 = await session.turn("First message")
        result2 = await session.turn("Second message")
        session.close()  # Saves trace
    """

    def __init__(
        self,
        model_client: Any | None,
        registry: Any | None = None,
        *,
        mode: str = "build",
        project: str = "default",
        hooks: Any = None,
        checkpoints: Any = None,
        skills: Any = None,
        task_manager: Any = None,
        max_turns: int = 100,
        config_path: Path | None = None,
        project_dir: Path | None = None,
        enable_trace: bool = True,
        worker_role: str | None = None,
        allowed_tool_names: list[str] | None = None,
    ):
        self.model_client = model_client
        self.registry = registry
        self.mode = mode
        self.project = project
        self.hooks = hooks
        self.checkpoints = checkpoints
        self.skills = skills
        self.task_manager = task_manager
        self.max_turns = max_turns
        self.config_path = config_path
        self.project_dir = project_dir or Path.cwd()
        self.enable_trace = enable_trace
        self.worker_role = worker_role
        self.allowed_tool_names = (
            allowed_tool_names.copy() if allowed_tool_names is not None else None
        )

        self._agent: DoraemonAgent | None = None
        self._state: AgentState | None = None
        self._tool_registry: Any = None
        self._trace: Trace | None = None
        self._session_id: str | None = None
        self._mcp_extensions: list[str] = []
        self._runtime: RuntimeBootstrap | None = None

    @property
    def session_id(self) -> str:
        """Get session ID."""
        if not self._session_id:
            self._session_id = DoraemonAgent._generate_session_id()
        return self._session_id

    async def initialize(self) -> None:
        """Initialize the agent session through the shared runtime bootstrap."""
        self._state = AgentState(mode=self.mode, max_turns=self.max_turns)

        self._runtime = await bootstrap_runtime(
            mode=self.mode,
            project=self.project,
            project_dir=self.project_dir,
            config_path=self.config_path,
            model_client=self.model_client,
            registry=self.registry,
            hooks=self.hooks,
            checkpoints=self.checkpoints,
            skills=self.skills,
            task_manager=self.task_manager,
        )
        self.model_client = self._runtime.model_client
        self.registry = self._runtime.registry
        self._tool_registry = self._runtime.registry
        self.hooks = self._runtime.hooks
        self.checkpoints = self._runtime.checkpoints
        self.skills = self._runtime.skills
        self.task_manager = self._runtime.task_manager
        self.project_dir = self._runtime.context.project_dir
        self._mcp_extensions = self._runtime.context.active_mcp_extensions.copy()

        if self.enable_trace:
            from src.core.home import set_project_dir

            set_project_dir(self.project_dir)
            self._trace = Trace(self.session_id, self.project_dir)

        self._agent = create_doraemon_agent(
            llm_client=self.model_client,
            tool_registry=self.registry,
            mode=self.mode,
            hooks=self.hooks,
            checkpoints=self.checkpoints,
            skills=self.skills,
            task_manager=self.task_manager,
            max_turns=self.max_turns,
            project_dir=self.project_dir,
            enable_trace=self.enable_trace,
            trace=self._trace,
            session_id=self.session_id,
            active_mcp_extensions=self._mcp_extensions,
            worker_role=self.worker_role,
            allowed_tool_names=self.allowed_tool_names,
        )
        self._agent.state = self._state

    async def turn(
        self,
        user_input: str,
        **kwargs,
    ) -> AgentTurnResult:
        """Run a single turn in the session."""
        if not self._agent:
            await self.initialize()

        start_time = time.time()

        try:
            result = await self._agent.run(user_input, **kwargs)

            return AgentTurnResult(
                response=result.response or "",
                tool_calls=[tc.to_dict() for tc in result.tool_calls],
                tokens_used=result.tokens_used,
                duration=time.time() - start_time,
                success=result.success,
                error=result.error,
                files_modified=_collect_modified_paths(result.tool_calls),
            )
        except Exception as e:
            logger.error(f"Agent turn failed: {e}")
            if self._trace:
                self._trace.error(str(e), {"exception_type": type(e).__name__})
            return AgentTurnResult(
                response="",
                tool_calls=[],
                tokens_used=0,
                duration=time.time() - start_time,
                success=False,
                error=str(e),
                files_modified=[],
            )

    async def orchestrate(
        self,
        user_input: str,
        *,
        context: dict[str, Any] | None = None,
        max_workers: int | None = None,
    ) -> LeadExecutionResult:
        """Execute a goal through the thin lead-runtime orchestration path."""
        if not self._agent:
            await self.initialize()

        runtime = LeadAgentRuntime(self, max_workers=max_workers or 2)
        return await runtime.execute(user_input, context=context)

    async def turn_stream(
        self,
        user_input: str,
        **kwargs,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream a single turn using the underlying agent event protocol."""
        if not self._agent:
            await self.initialize()

        try:
            async for event in self._agent.run_stream(user_input, **kwargs):
                yield event
        except Exception as e:
            logger.error(f"Agent streaming turn failed: {e}")
            if self._trace:
                self._trace.error(str(e), {"exception_type": type(e).__name__})
            yield {"type": "error", "error": str(e)}

    def get_state(self) -> AgentState | None:
        """Get current agent state."""
        return self._state

    def get_trace(self) -> Trace | None:
        """Get current trace object."""
        return self._trace

    def get_task_manager(self):
        """Get the shared runtime task manager."""
        return self.task_manager

    async def spawn_worker_session(
        self,
        *,
        enable_trace: bool | None = None,
        worker_role: str | None = None,
        allowed_tool_names: list[str] | None = None,
    ) -> "AgentSession":
        """Create an isolated worker session that reuses shared runtime resources."""
        if not self._agent:
            await self.initialize()

        worker = AgentSession(
            model_client=self.model_client,
            registry=self.registry,
            mode=self.mode,
            project=self.project,
            hooks=self.hooks,
            checkpoints=self.checkpoints,
            skills=self.skills,
            task_manager=self.task_manager,
            max_turns=self.max_turns,
            config_path=self.config_path,
            project_dir=self.project_dir,
            enable_trace=self.enable_trace if enable_trace is None else enable_trace,
            worker_role=worker_role,
            allowed_tool_names=allowed_tool_names,
        )
        await worker.initialize()
        return worker

    def save_trace(self) -> Path | None:
        """Save trace to file, returns path if saved."""
        if self._trace:
            return self._trace.save()
        return None

    async def aclose(self) -> Path | None:
        """Close session resources and save trace."""
        if self._runtime:
            await self._runtime.aclose()
        elif self.registry is not None:
            for client in getattr(self.registry, "_mcp_clients", []):
                await client.close()
        return self.save_trace()

    def close(self) -> Path | None:
        """Close session and save trace."""
        return self.save_trace()

    def reset(self) -> None:
        """Reset the session (preserves trace)."""
        if self._agent:
            self._agent.reset()
        self._state = None

    def set_mode(self, mode: str) -> None:
        """Change agent mode."""
        self.mode = mode
        if self._state:
            self._state.mode = mode
