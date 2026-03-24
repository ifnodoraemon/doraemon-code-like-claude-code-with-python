"""
Agent Adapter - Bridge between old chat_loop and new Agent architecture

This module provides a compatibility layer that allows the existing
chat_loop.py to use the new Agent architecture without major changes.

Usage:
    from src.agent.adapter import run_agent_turn

    result = await run_agent_turn(
        user_input="Read the README",
        model_client=model_client,
        registry=registry,
        ...
    )
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
from src.agent.types import AgentResult

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
    ctx: Any,
    *,
    mode: str = "build",
    hook_mgr: Any = None,
    checkpoint_mgr: Any = None,
    skill_mgr: Any = None,
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
        ctx: Context manager (for history)
        mode: Agent mode ("plan" or "build")
        hook_mgr: Hook manager
        checkpoint_mgr: Checkpoint manager
        skill_mgr: Skill manager
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
        hook_mgr=hook_mgr,
        checkpoint_mgr=checkpoint_mgr,
        skill_mgr=skill_mgr,
        permission_callback=default_permission_callback if sensitive_tools else None,
        display_callback=display_callback,
        max_turns=max_turns,
    )

    if ctx and hasattr(ctx, "messages"):
        for msg in ctx.messages[-10:]:
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
    """

    def __init__(
        self,
        model_client: Any,
        registry: Any | None = None,
        *,
        mode: str = "build",
        hook_mgr: Any = None,
        checkpoint_mgr: Any = None,
        skill_mgr: Any = None,
        max_turns: int = 100,
        config_path: Path | None = None,
    ):
        self.model_client = model_client
        self.registry = registry
        self.mode = mode
        self.hook_mgr = hook_mgr
        self.checkpoint_mgr = checkpoint_mgr
        self.skill_mgr = skill_mgr
        self.max_turns = max_turns
        self.config_path = config_path

        self._agent: DoraemonAgent | None = None
        self._state: AgentState | None = None
        self._mcp_registry: Any = None

    async def initialize(self) -> None:
        """Initialize the agent session with MCP support."""
        self._state = AgentState(mode=self.mode, max_turns=self.max_turns)

        if self.registry is None:
            from src.host.mcp_registry import create_unified_registry

            self._mcp_registry = await create_unified_registry(self.config_path)
            self.registry = self._mcp_registry

        self._agent = create_doraemon_agent(
            llm_client=self.model_client,
            tool_registry=self.registry,
            mode=self.mode,
            hook_mgr=self.hook_mgr,
            checkpoint_mgr=self.checkpoint_mgr,
            skill_mgr=self.skill_mgr,
            max_turns=self.max_turns,
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

    def reset(self) -> None:
        """Reset the session."""
        if self._agent:
            self._agent.reset()
        self._state = None
        self._agent = None

    def set_mode(self, mode: str) -> None:
        """Change agent mode."""
        self.mode = mode
        if self._state:
            self._state.mode = mode
