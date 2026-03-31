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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.agent.doraemon import DoraemonAgent, create_doraemon_agent
from src.agent.state import AgentState
from src.core.home import Trace

logger = logging.getLogger(__name__)
console = Console()


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

        files_modified = []
        for tc in result.tool_calls:
            if tc.name in {"write", "edit", "write_file", "edit_file"}:
                if tc.arguments.get("path"):
                    files_modified.append(tc.arguments["path"])

        return AgentTurnResult(
            response=result.response or "",
            tool_calls=[tc.to_dict() for tc in result.tool_calls],
            tokens_used=result.tokens_used,
            duration=time.time() - start_time,
            success=result.success,
            error=result.error,
            files_modified=files_modified,
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
        model_client: Any,
        registry: Any | None = None,
        *,
        mode: str = "build",
        hooks: Any = None,
        checkpoints: Any = None,
        skills: Any = None,
        max_turns: int = 100,
        config_path: Path | None = None,
        project_dir: Path | None = None,
        enable_trace: bool = True,
    ):
        self.model_client = model_client
        self.registry = registry
        self.mode = mode
        self.hooks = hooks
        self.checkpoints = checkpoints
        self.skills = skills
        self.max_turns = max_turns
        self.config_path = config_path
        self.project_dir = project_dir or Path.cwd()
        self.enable_trace = enable_trace

        self._agent: DoraemonAgent | None = None
        self._state: AgentState | None = None
        self._tool_registry: Any = None
        self._trace: Trace | None = None
        self._session_id: str | None = None
        self._mcp_extensions: list[str] = []

    @property
    def session_id(self) -> str:
        """Get session ID."""
        if not self._session_id:
            self._session_id = DoraemonAgent._generate_session_id()
        return self._session_id

    async def initialize(self) -> None:
        """Initialize the agent session with the built-in tool registry."""
        self._state = AgentState(mode=self.mode, max_turns=self.max_turns)

        if self.registry is None:
            from src.host.mcp_registry import create_tool_registry

            self._tool_registry = await create_tool_registry(self.config_path, mode=self.mode)
            self.registry = self._tool_registry
            self._mcp_extensions = getattr(self._tool_registry, "_active_mcp_extensions", []).copy()
        elif not self._mcp_extensions:
            self._tool_registry = self.registry
            self._mcp_extensions = getattr(self.registry, "_active_mcp_extensions", []).copy()

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
            max_turns=self.max_turns,
            project_dir=self.project_dir,
            enable_trace=self.enable_trace,
            trace=self._trace,
            session_id=self.session_id,
            active_mcp_extensions=self._mcp_extensions,
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
            result = await self._agent.run(user_input)

            files_modified = []
            for tc in result.tool_calls:
                if tc.name in {"write", "edit", "write_file", "edit_file"}:
                    if tc.arguments.get("path"):
                        files_modified.append(tc.arguments["path"])

            return AgentTurnResult(
                response=result.response or "",
                tool_calls=[tc.to_dict() for tc in result.tool_calls],
                tokens_used=result.tokens_used,
                duration=time.time() - start_time,
                success=result.success,
                error=result.error,
                files_modified=files_modified,
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

    def get_state(self) -> AgentState | None:
        """Get current agent state."""
        return self._state

    def get_trace(self) -> Trace | None:
        """Get current trace object."""
        return self._trace

    def save_trace(self) -> Path | None:
        """Save trace to file, returns path if saved."""
        if self._trace:
            return self._trace.save()
        return None

    async def aclose(self) -> Path | None:
        """Close session resources and save trace."""
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
