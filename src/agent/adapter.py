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
    await session.aclose()
"""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.agent.doraemon import DoraemonAgent, create_doraemon_agent
from src.agent.state import AgentState
from src.agent.types import Message
from src.core.home import Trace
from src.core.session import SessionData, SessionManager
from src.runtime import LeadAgentRuntime, LeadExecutionResult
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


def _message_from_session_data(payload: dict[str, Any]) -> Message:
    """Convert persisted session payloads into agent messages."""
    return Message(
        role=payload.get("role", "assistant"),
        content=payload.get("content"),
        provider_items=payload.get("provider_items"),
        tool_calls=payload.get("tool_calls"),
        tool_call_id=payload.get("tool_call_id"),
        name=payload.get("name"),
        thought=payload.get("thought"),
    )


def _message_to_session_data(message: Message) -> dict[str, Any]:
    """Convert agent messages into persisted session payloads."""
    payload: dict[str, Any] = {"role": message.role}
    if message.content is not None:
        payload["content"] = message.content
    if message.provider_items:
        payload["provider_items"] = message.provider_items
    if message.tool_calls:
        payload["tool_calls"] = message.tool_calls
    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    if message.name:
        payload["name"] = message.name
    if message.thought:
        payload["thought"] = message.thought
    return payload


@dataclass
class AgentTurnResult:
    """Result from running an agent turn, compatible with chat_loop."""

    response: str
    tool_calls: list[dict[str, Any]]
    tokens_used: int
    duration: float
    success: bool
    error: str | None = None
    files_modified: list[str] = field(default_factory=list)


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
    except (RuntimeError, ValueError, OSError, TypeError, KeyError) as e:
        logger.error("Agent turn failed: %s", e)
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
        await session.aclose()  # Saves trace
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
        session_id: str | None = None,
        model_name: str | None = None,
        worker_role: str | None = None,
        allowed_tool_names: list[str] | None = None,
        persist_session: bool = True,
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
        self.model_name = model_name
        self.worker_role = worker_role
        self.persist_session = persist_session
        self.allowed_tool_names = (
            allowed_tool_names.copy() if allowed_tool_names is not None else None
        )

        self._agent: DoraemonAgent | None = None
        self._state: AgentState | None = None
        self._tool_registry: Any = None
        self._trace: Trace | None = None
        self._session_id: str | None = session_id
        self._mcp_extensions: list[str] = []
        self._runtime: RuntimeBootstrap | None = None
        self._session_manager: SessionManager | None = None
        self._session_record: SessionData | None = None

    @property
    def session_id(self) -> str:
        """Get session ID."""
        if not self._session_id:
            self._session_id = DoraemonAgent._generate_session_id()
        return self._session_id

    async def initialize(self) -> None:
        """Initialize the agent session through the shared runtime bootstrap."""
        if self._agent is not None:
            return

        self._prepare_session_record()
        self._initialize_state()
        await self._ensure_runtime()
        self._ensure_session_record()
        self._initialize_trace()
        self._rebuild_agent()

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
            self._save_session_state()

            return AgentTurnResult(
                response=result.response or "",
                tool_calls=[tc.to_dict() for tc in result.tool_calls],
                tokens_used=result.tokens_used,
                duration=time.time() - start_time,
                success=result.success,
                error=result.error,
                files_modified=_collect_modified_paths(result.tool_calls),
            )
        except (RuntimeError, ValueError, OSError, TypeError, KeyError) as e:
            logger.error("Agent turn failed: %s", e)
            if self._trace:
                self._trace.error(str(e), {"exception_type": type(e).__name__})
            self._save_session_state()
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
        resume_run_id: str | None = None,
    ) -> LeadExecutionResult:
        """Execute a goal through the thin lead-runtime orchestration path."""
        if not self._agent:
            await self.initialize()

        prior_run = self._get_orchestration_run(resume_run_id) if resume_run_id else None
        if resume_run_id and prior_run is None:
            raise ValueError(f"unknown orchestration run: {resume_run_id}")
        run_id = uuid.uuid4().hex[:12]
        goal = user_input
        if prior_run is not None:
            goal = prior_run.get("goal") or prior_run.get("summary") or "Resume orchestration"
            user_input = f"Resume orchestration: {goal}"

        message_start_index = len(self._state.messages) if self._state is not None else 0
        if self._state is not None:
            self._state.add_user_message(user_input)

        if self._trace and self.enable_trace:
            self._trace.start_turn(
                user_input,
                metadata={
                    "mode": self.mode,
                    "execution_mode": "orchestrate",
                    "run_id": run_id,
                    "goal": goal,
                    "resume_run_id": resume_run_id,
                    "max_workers": max_workers or 2,
                },
            )

        runtime = LeadAgentRuntime(self, max_workers=max_workers or 2)
        try:
            if prior_run is None:
                result = await runtime.execute(goal, context=context, trace_run_id=run_id)
            else:
                result = await runtime.resume(
                    prior_run["root_task_id"],
                    prior_state=prior_run,
                    trace_run_id=run_id,
                )
        except (RuntimeError, ValueError, OSError, TypeError, KeyError) as e:
            self._rollback_orchestration_messages(message_start_index)
            if self._trace and self.enable_trace:
                self._trace.error(str(e), {"exception_type": type(e).__name__, "run_id": run_id})
                self._trace.end_turn(success=False, error=str(e))
            raise

        if self._state is not None:
            self._state.add_assistant_message(result.summary)
        message_end_index = (
            len(self._state.messages) - 1
            if self._state is not None and self._state.messages
            else message_start_index
        )
        self._record_orchestration_run(
            self._build_orchestration_state(
                result=result,
                run_id=run_id,
                goal=goal,
                message_start_index=message_start_index,
                message_end_index=message_end_index,
                resumed_from_run_id=resume_run_id,
            )
        )
        if self._trace and self.enable_trace:
            self._trace.end_turn(
                success=result.success,
                error=None if result.success else result.summary,
            )
        self._save_session_state()
        return result

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
        except (RuntimeError, ValueError, OSError, TypeError, KeyError) as e:
            logger.error("Agent streaming turn failed: %s", e)
            if self._trace:
                self._trace.error(str(e), {"exception_type": type(e).__name__})
            yield {"type": "error", "error": str(e)}
        finally:
            self._save_session_state()

    def get_state(self) -> AgentState | None:
        """Get current agent state."""
        return self._state

    def get_trace(self) -> Trace | None:
        """Get current trace object."""
        return self._trace

    def get_orchestration_state(self) -> dict[str, Any]:
        """Return the persisted orchestration snapshot for the current session."""
        if self._session_record is None:
            return {}
        return deepcopy(self._session_record.orchestration_state)

    def get_orchestration_runs(self) -> list[dict[str, Any]]:
        """Return persisted orchestration runs for the current session."""
        if self._session_record is None:
            return []
        return deepcopy(self._session_record.orchestration_runs)

    def get_active_orchestration_run_id(self) -> str | None:
        """Return the currently active orchestration run identifier."""
        if self._session_record is None:
            return None
        return self._session_record.active_orchestration_run_id

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
            model_name=self.model_name,
            worker_role=worker_role,
            allowed_tool_names=allowed_tool_names,
            persist_session=False,
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
        self._save_session_state()
        owns_model_client = self._runtime.owns_model_client if self._runtime else False
        owns_registry = self._runtime.owns_registry if self._runtime else False
        if self._runtime:
            await self._runtime.aclose()
        elif self.registry is not None:
            for client in getattr(self.registry, "_mcp_clients", []):
                await client.close()
        if owns_model_client:
            self.model_client = None
        if owns_registry:
            self.registry = None
        self._runtime = None
        self._agent = None
        self._tool_registry = None
        self._mcp_extensions = []
        return self.save_trace()

    def close(self) -> Path | None:
        """Close session resources from synchronous callers."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aclose())

        raise RuntimeError("close() cannot run inside an active event loop; use await aclose().")

    def reset(self) -> None:
        """Reset the session (preserves trace)."""
        if self._agent:
            self._agent.reset()
        if self._session_record is not None:
            self._session_record.messages = []
            self._session_record.orchestration_state = {}
            self._session_record.orchestration_runs = []
            self._session_record.active_orchestration_run_id = None
            self._session_record.metadata.message_count = 0
            self._session_record.metadata.total_tokens = 0
            self._session_record.metadata.name = None
            if self._session_manager is not None:
                self._session_manager.save_session(self._session_record)
        self._agent = None
        self._state = None

    async def set_mode(self, mode: str) -> None:
        """Change agent mode and rebuild the agent-visible tool surface."""
        if mode == self.mode:
            return

        self._update_mode(mode)
        await self._rebootstrap_runtime_for_mode()
        self._rebuild_agent()
        self._save_session_state()

    def _create_session_manager(self) -> SessionManager:
        return SessionManager(base_dir=self.project_dir / ".agent" / "sessions")

    def _prepare_session_record(self) -> None:
        if not self.persist_session:
            self._session_manager = None
            self._session_record = None
            _ = self.session_id
            return

        self._session_manager = self._create_session_manager()
        self._session_record = self._resume_existing_session()

    def _ensure_session_record(self) -> None:
        if (
            not self.persist_session
            or self._session_record is not None
            or self._session_manager is None
        ):
            return

        self._session_record = self._session_manager.create_session(
            project=self.project,
            mode=self.mode,
        )
        self._session_id = self._session_record.metadata.id

    def _resume_existing_session(self) -> SessionData | None:
        if self._session_manager is None or not self._session_id:
            return None

        session_record = self._session_manager.resume_session(self._session_id)
        if session_record is None:
            return None

        self._session_id = session_record.metadata.id
        self.mode = session_record.metadata.mode or self.mode
        return session_record

    def _initialize_state(self) -> None:
        self._state = AgentState(mode=self.mode, max_turns=self.max_turns)
        self._restore_session_state()

    async def _bootstrap_runtime(self, *, registry: Any | None) -> RuntimeBootstrap:
        return await bootstrap_runtime(
            mode=self.mode,
            project=self.project,
            project_dir=self.project_dir,
            config_path=self.config_path,
            model_client=self.model_client,
            model_name=self.model_name,
            registry=registry,
            hooks=self.hooks,
            checkpoints=self.checkpoints,
            skills=self.skills,
            task_manager=self.task_manager,
        )

    async def _ensure_runtime(self) -> None:
        if self._runtime is not None:
            self._apply_runtime(self._runtime)
            return

        runtime = await self._bootstrap_runtime(registry=self.registry)
        self._apply_runtime(runtime)

    def _apply_runtime(self, runtime: RuntimeBootstrap) -> None:
        self._runtime = runtime
        self.model_client = runtime.model_client
        self.registry = runtime.registry
        self._tool_registry = runtime.registry
        self.hooks = runtime.hooks
        self.checkpoints = runtime.checkpoints
        self.skills = runtime.skills
        self.task_manager = runtime.task_manager
        self.project_dir = runtime.context.project_dir
        self._mcp_extensions = runtime.context.active_mcp_extensions.copy()

    def _initialize_trace(self) -> None:
        if not self.enable_trace:
            self._trace = None
            return

        from src.core.home import set_project_dir

        set_project_dir(self.project_dir)
        self._trace = Trace(self.session_id, self.project_dir)

    def _rebuild_agent(self) -> None:
        if self.registry is None or self._state is None:
            return

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

    def _update_mode(self, mode: str) -> None:
        self.mode = mode
        if self._state is not None:
            self._state.mode = mode
        if self._session_record is not None:
            self._session_record.metadata.mode = mode

    async def _rebootstrap_runtime_for_mode(self) -> None:
        if self._runtime is None or not self._runtime.owns_registry:
            return

        previous_runtime = self._runtime
        transfer_model_ownership = previous_runtime.owns_model_client
        previous_runtime.owns_model_client = False
        runtime = await self._bootstrap_runtime(registry=None)
        runtime.owns_model_client = runtime.owns_model_client or transfer_model_ownership
        self._apply_runtime(runtime)
        await previous_runtime.aclose()

    def _restore_session_state(self) -> None:
        if self._state is None or self._session_record is None:
            return

        self._state.mode = self.mode
        for payload in self._session_record.messages:
            self._state.add_message(_message_from_session_data(payload))

        for message in reversed(self._state.messages):
            if (
                self._state.last_response is None
                and message.role == "assistant"
                and message.content
            ):
                self._state.last_response = message.content
            if self._state.user_input is None and message.role == "user" and message.content:
                self._state.user_input = message.content
            if self._state.last_response is not None and self._state.user_input is not None:
                break

    def _save_session_state(self) -> None:
        if self._session_manager is None or self._session_record is None or self._state is None:
            return

        self._session_record.metadata.project = self.project
        self._session_record.metadata.mode = self.mode
        self._session_record.metadata.message_count = len(self._state.messages)
        self._session_record.metadata.total_tokens = self._state.estimated_tokens
        if not self._session_record.metadata.name:
            self._session_record.metadata.name = self._derive_session_name()
        self._session_record.messages = [
            _message_to_session_data(message) for message in self._state.messages
        ]
        self._session_manager.save_session(self._session_record)

    def _get_task_graph_snapshot(self, root_task_id: str | None = None) -> list[dict[str, Any]]:
        if self.task_manager is None:
            return []
        get_task_tree = getattr(self.task_manager, "get_task_tree", None)
        if not callable(get_task_tree):
            return []
        return get_task_tree(root_task_id)

    def _build_orchestration_state(
        self,
        *,
        result: LeadExecutionResult | None = None,
        run_id: str | None = None,
        goal: str | None = None,
        message_start_index: int | None = None,
        message_end_index: int | None = None,
        resumed_from_run_id: str | None = None,
        success: bool | None = None,
        summary: str | None = None,
    ) -> dict[str, Any]:
        if result is None:
            return {
                "run_id": run_id,
                "goal": goal or "",
                "resumed_from_run_id": resumed_from_run_id,
                "message_start_index": message_start_index,
                "message_end_index": message_end_index,
                "root_task_id": None,
                "plan_id": None,
                "executed_task_ids": [],
                "completed_task_ids": [],
                "failed_task_ids": [],
                "blocked_task_id": None,
                "success": bool(success),
                "summary": summary or "",
                "task_summaries": {},
                "worker_assignments": {},
                "task_graph": [],
            }

        state = result.to_dict()
        state["run_id"] = run_id
        state["goal"] = goal or ""
        state["resumed_from_run_id"] = resumed_from_run_id
        state["message_start_index"] = message_start_index
        state["message_end_index"] = message_end_index
        state["task_graph"] = self._get_task_graph_snapshot(result.root_task_id)
        return state

    def _record_orchestration_run(self, payload: dict[str, Any]) -> None:
        if self._session_record is None:
            return
        self._session_record.orchestration_runs.append(payload)
        self._session_record.orchestration_state = payload
        self._session_record.active_orchestration_run_id = payload.get("run_id")

    def _rollback_orchestration_messages(self, message_start_index: int) -> None:
        if self._state is None:
            return
        self._state.messages = self._state.messages[:message_start_index]
        self._state._update_token_estimate()
        self._state.user_input = next(
            (
                message.content
                for message in reversed(self._state.messages)
                if message.role == "user"
            ),
            None,
        )
        self._state.last_response = next(
            (
                message.content
                for message in reversed(self._state.messages)
                if message.role == "assistant"
            ),
            None,
        )

    def _get_orchestration_run(self, run_id: str | None) -> dict[str, Any] | None:
        if self._session_record is None or not run_id:
            return None
        for run in reversed(self._session_record.orchestration_runs):
            if run.get("run_id") == run_id:
                return deepcopy(run)
        if self._session_record.orchestration_state.get("run_id") == run_id:
            return deepcopy(self._session_record.orchestration_state)
        return None

    def _derive_session_name(self) -> str | None:
        if self._state is None:
            return None
        for message in self._state.messages:
            if message.role == "user" and message.content:
                compact = " ".join(message.content.strip().split())
                if not compact:
                    continue
                return compact[:60] + ("..." if len(compact) > 60 else "")
        return None
