"""
Subagent System

Enables creating specialized subagents for complex tasks.

Features:
- Dynamic subagent creation with custom prompts
- Tool restrictions per subagent
- Model selection per subagent
- Parallel execution support
- Built-in agent types (code-reviewer, debugger, etc.)
- Agent communication protocol (message passing)
- Agent state management (idle/running/completed/failed)
- Parallel execution with asyncio
- Result aggregation and monitoring
- Timeout and error handling with metrics
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from src.core.config import get_required_config_value
from src.core.paths import mailboxes_dir

logger = logging.getLogger(__name__)


class SubagentModel(Enum):
    """Available models for subagents."""

    INHERIT = "inherit"  # Use parent's model
    PRO = "pro"  # Most capable model
    FLASH = "flash"  # Fast and capable (default)


class AgentState(Enum):
    """Agent execution state."""

    IDLE = "idle"  # Not running
    RUNNING = "running"  # Currently executing
    COMPLETED = "completed"  # Finished successfully
    FAILED = "failed"  # Execution failed
    TIMEOUT = "timeout"  # Execution timed out
    CANCELLED = "cancelled"  # Execution was cancelled


@dataclass
class AgentMessage:
    """Message for inter-agent communication."""

    sender_id: str
    recipient_id: str | None  # None for broadcast
    message_type: str  # "request", "response", "status", "error"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "message_type": self.message_type,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentMessage":
        return cls(
            sender_id=data["sender_id"],
            recipient_id=data.get("recipient_id"),
            message_type=data["message_type"],
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AgentMetrics:
    """Metrics for agent execution monitoring."""

    agent_id: str
    agent_name: str
    state: AgentState
    start_time: float
    end_time: float | None = None
    turns_used: int = 0
    tokens_used: int = 0
    tool_calls: int = 0
    errors: int = 0
    messages_sent: int = 0
    messages_received: int = 0

    @property
    def duration(self) -> float:
        """Calculate execution duration."""
        end = self.end_time or time.time()
        return end - self.start_time

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "state": self.state.value,
            "duration": round(self.duration, 2),
            "turns_used": self.turns_used,
            "tokens_used": self.tokens_used,
            "tool_calls": self.tool_calls,
            "errors": self.errors,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
        }


@dataclass
class SubagentConfig:
    """Configuration for a subagent."""

    name: str
    description: str
    prompt: str  # System prompt for the subagent
    tools: list[str] | None = None  # Tool whitelist (None = all tools)
    model: SubagentModel = SubagentModel.INHERIT
    max_turns: int = 10  # Maximum conversation turns
    timeout: float = 300  # Timeout in seconds (5 min default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "prompt": self.prompt,
            "tools": self.tools,
            "model": self.model.value,
            "max_turns": self.max_turns,
            "timeout": self.timeout,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubagentConfig":
        return cls(
            name=data["name"],
            description=data["description"],
            prompt=data["prompt"],
            tools=data.get("tools"),
            model=SubagentModel(data.get("model", "inherit")),
            max_turns=data.get("max_turns", 10),
            timeout=data.get("timeout", 300),
        )


@dataclass
class SubagentResult:
    """Result from a subagent execution."""

    agent_id: str
    agent_name: str
    success: bool
    output: str
    turns_used: int
    tokens_used: int
    duration: float
    error: str | None = None
    state: AgentState = AgentState.COMPLETED
    metrics: AgentMetrics | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "success": self.success,
            "output": self.output,
            "turns_used": self.turns_used,
            "tokens_used": self.tokens_used,
            "duration": round(self.duration, 2),
            "error": self.error,
            "state": self.state.value,
            "metrics": self.metrics.to_dict() if self.metrics else None,
        }


# ========================================
# Agent Communication & State Management
# ========================================


class AgentMessageQueue:
    """Recipient-aware queue with durable mailbox append logs."""

    def __init__(self, storage_dir: Path | None = None):
        """Initialize queues and mailbox storage."""
        self._global_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self._recipient_queues: dict[str, asyncio.Queue[AgentMessage]] = defaultdict(asyncio.Queue)
        self._subscribers: dict[str, list[Callable[[AgentMessage], None]]] = {}
        self._storage_dir = storage_dir or mailboxes_dir()
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    async def send(self, message: AgentMessage) -> None:
        """Send a message to the queue."""
        await self._global_queue.put(message)
        if message.recipient_id:
            await self._recipient_queues[message.recipient_id].put(message)
            self._append_to_mailbox(message.recipient_id, message)
        else:
            self._append_to_mailbox("broadcast", message)
        await self._notify_subscribers(message)

    async def receive(
        self,
        agent_id: str | None = None,
        timeout: float | None = None,
    ) -> AgentMessage | None:
        """Receive the next message globally or for a specific recipient."""
        queue = self._recipient_queues[agent_id] if agent_id else self._global_queue
        try:
            if timeout:
                return await asyncio.wait_for(queue.get(), timeout=timeout)
            return await queue.get()
        except asyncio.TimeoutError:
            return None

    def subscribe(self, agent_id: str, callback: Callable[[AgentMessage], None]) -> None:
        """Subscribe to messages for a specific agent."""
        if agent_id not in self._subscribers:
            self._subscribers[agent_id] = []
        self._subscribers[agent_id].append(callback)

    async def _notify_subscribers(self, message: AgentMessage) -> None:
        """Notify subscribers of a new message."""
        if message.recipient_id and message.recipient_id in self._subscribers:
            for callback in self._subscribers[message.recipient_id]:
                try:
                    callback(message)
                except Exception as e:
                    logger.error(f"Error in message subscriber: {e}")

    def get_mailbox_path(self, agent_id: str) -> Path:
        """Return the append-only mailbox path for an agent."""
        return self._storage_dir / f"{agent_id}.jsonl"

    def get_mailbox_messages(self, agent_id: str, limit: int = 100) -> list[AgentMessage]:
        """Read recent durable messages for an agent without loading all history."""
        path = self.get_mailbox_path(agent_id)
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as handle:
            lines = handle.readlines()
        recent = lines[-limit:]
        messages: list[AgentMessage] = []
        for line in recent:
            try:
                payload = json.loads(line)
                messages.append(AgentMessage.from_dict(payload))
            except Exception as exc:
                logger.warning(f"Skipping invalid mailbox entry in {path}: {exc}")
        return messages

    def _append_to_mailbox(self, mailbox_id: str, message: AgentMessage) -> None:
        path = self.get_mailbox_path(mailbox_id)
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(message.to_dict(), ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.error(f"Failed to append mailbox message for {mailbox_id}: {exc}")


class AgentStateManager:
    """Manages agent state and transitions."""

    def __init__(self):
        """Initialize state manager."""
        self._states: dict[str, AgentState] = {}
        self._metrics: dict[str, AgentMetrics] = {}
        self._lock = asyncio.Lock()

    async def set_state(self, agent_id: str, state: AgentState) -> None:
        """Set agent state."""
        async with self._lock:
            self._states[agent_id] = state
            logger.debug(f"Agent {agent_id} state changed to {state.value}")

    async def get_state(self, agent_id: str) -> AgentState | None:
        """Get agent state."""
        async with self._lock:
            return self._states.get(agent_id)

    async def create_metrics(self, agent_id: str, agent_name: str) -> AgentMetrics:
        """Create metrics for an agent."""
        async with self._lock:
            metrics = AgentMetrics(
                agent_id=agent_id,
                agent_name=agent_name,
                state=AgentState.IDLE,
                start_time=time.time(),
            )
            self._metrics[agent_id] = metrics
            return metrics

    async def get_metrics(self, agent_id: str) -> AgentMetrics | None:
        """Get agent metrics."""
        async with self._lock:
            return self._metrics.get(agent_id)

    async def update_metrics(self, agent_id: str, **kwargs: Any) -> None:
        """Update agent metrics."""
        async with self._lock:
            if agent_id in self._metrics:
                metrics = self._metrics[agent_id]
                for key, value in kwargs.items():
                    if hasattr(metrics, key):
                        setattr(metrics, key, value)

    async def get_all_metrics(self) -> dict[str, AgentMetrics]:
        """Get all agent metrics."""
        async with self._lock:
            return dict(self._metrics)


def _create_agent_id() -> str:
    """Create a unique agent ID."""
    return str(uuid.uuid4())[:8]


def _get_model_name(model: SubagentModel, parent_model: str) -> str:
    """Get model name. Currently inherits parent model for all types."""
    # TODO: Implement actual model selection when multi-model routing is ready
    return parent_model


async def _execute_agent_task(
    agent_id: str,
    config: "SubagentConfig",
    task: str,
    context: str,
    model_client: Any,
    tool_registry: Any,
    parent_model: str,
    on_output: Callable[[str], None] | None = None,
    state_manager: AgentStateManager | None = None,
) -> SubagentResult:
    """
    Execute a single agent task using the unified ModelClient.

    Args:
        agent_id: Unique agent ID
        config: Agent configuration
        task: Task description
        context: Additional context
        model_client: Unified ModelClient (provider-agnostic)
        tool_registry: Tool registry
        parent_model: Parent model name
        on_output: Output callback
        state_manager: State manager for tracking

    Returns:
        SubagentResult with execution details
    """
    from src.core.model_utils import Message, ToolDefinition

    start_time = time.time()
    metrics = None

    try:
        # Initialize metrics
        if state_manager:
            metrics = await state_manager.create_metrics(agent_id, config.name)
            await state_manager.set_state(agent_id, AgentState.RUNNING)

        logger.info(f"Spawning subagent {agent_id} ({config.name}): {task[:100]}")

        # Determine model
        model_name = _get_model_name(config.model, parent_model)

        # Get tool definitions
        tool_defs = None
        if config.tools:
            genai_tools = tool_registry.get_genai_tools(config.tools)
            tool_defs = [
                ToolDefinition(
                    name=getattr(t, "name", ""),
                    description=getattr(t, "description", "") or "",
                    parameters=getattr(t, "parameters", {}) or {},
                )
                for t in genai_tools
            ]

        # Build system prompt
        system_prompt = config.prompt
        if context:
            system_prompt += f"\n\n[Context]\n{context}"

        # Build conversation using unified Message format
        conversation: list[Message] = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=task),
        ]

        # Run agent loop
        output_parts = []
        turns_used = 0
        total_tokens = 0
        tool_calls_count = 0

        while turns_used < config.max_turns:
            # Call model using unified interface
            response = await asyncio.wait_for(
                model_client.chat(conversation, tools=tool_defs, model=model_name),
                timeout=config.timeout,
            )
            turns_used += 1

            # Track tokens
            if response.usage:
                total_tokens += response.usage.get("total_tokens", 0) or (
                    response.usage.get("prompt_tokens", 0)
                    + response.usage.get("completion_tokens", 0)
                )

            # Collect text output
            if response.content:
                output_parts.append(response.content)
                if on_output:
                    on_output(response.content)

            # Add assistant message to conversation
            conversation.append(
                Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                    thought=response.thought,
                )
            )

            # No tool calls - done
            if not response.has_tool_calls:
                break

            # Execute tool calls
            for tc in response.tool_calls:
                func = tc.get("function", {})
                tool_name = func.get("name", "")
                args = func.get("arguments", {})
                tool_call_id = tc.get("id", "")

                # Parse string arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}

                tool_calls_count += 1

                # Execute tool
                try:
                    result = await tool_registry.call_tool(tool_name, args)
                except Exception as e:
                    result = f"Error: {e}"

                # Add tool result to conversation
                conversation.append(
                    Message(
                        role="tool",
                        content=str(result),
                        tool_call_id=tool_call_id,
                        name=tool_name,
                    )
                )

        duration = time.time() - start_time
        output = "\n".join(output_parts)

        # Update metrics
        if metrics and state_manager:
            await state_manager.update_metrics(
                agent_id,
                state=AgentState.COMPLETED,
                turns_used=turns_used,
                tokens_used=total_tokens,
                tool_calls=tool_calls_count,
                end_time=time.time(),
            )

        logger.info(
            f"Subagent {agent_id} completed in {duration:.1f}s "
            f"({turns_used} turns, {total_tokens} tokens)"
        )

        return SubagentResult(
            agent_id=agent_id,
            agent_name=config.name,
            success=True,
            output=output,
            turns_used=turns_used,
            tokens_used=total_tokens,
            duration=duration,
            state=AgentState.COMPLETED,
            metrics=metrics,
        )

    except asyncio.TimeoutError:
        duration = time.time() - start_time
        if metrics:
            await state_manager.update_metrics(
                agent_id,
                state=AgentState.TIMEOUT,
                end_time=time.time(),
            )
        return SubagentResult(
            agent_id=agent_id,
            agent_name=config.name,
            success=False,
            output="",
            turns_used=0,
            tokens_used=0,
            duration=duration,
            error=f"Timeout after {config.timeout}s",
            state=AgentState.TIMEOUT,
            metrics=metrics,
        )

    except Exception as e:
        duration = time.time() - start_time
        if metrics:
            await state_manager.update_metrics(
                agent_id,
                state=AgentState.FAILED,
                errors=1,
                end_time=time.time(),
            )
        logger.error(f"Subagent {agent_id} failed: {e}")
        return SubagentResult(
            agent_id=agent_id,
            agent_name=config.name,
            success=False,
            output="",
            turns_used=0,
            tokens_used=0,
            duration=duration,
            error=str(e),
            state=AgentState.FAILED,
            metrics=metrics,
        )


def _aggregate_results(results: list[SubagentResult]) -> dict[str, Any]:
    """
    Aggregate results from multiple agents.

    Args:
        results: List of SubagentResult objects

    Returns:
        Aggregated metrics and summary
    """
    total_duration = sum(r.duration for r in results)
    total_tokens = sum(r.tokens_used for r in results)
    successful = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)

    return {
        "total_agents": len(results),
        "successful": successful,
        "failed": failed,
        "total_duration": round(total_duration, 2),
        "total_tokens": total_tokens,
        "average_duration": round(total_duration / len(results), 2) if results else 0,
        "results": [r.to_dict() for r in results],
    }


BUILTIN_AGENTS: dict[str, SubagentConfig] = {
    "code-reviewer": SubagentConfig(
        name="code-reviewer",
        description="Expert code reviewer. Use proactively after code changes.",
        prompt="""You are a senior code reviewer focused on quality, security, and best practices.

Review the provided code and provide:
1. Security issues (critical)
2. Bug risks (high priority)
3. Performance concerns
4. Code style and readability suggestions
5. Test coverage recommendations

Be specific and actionable. Reference line numbers when applicable.""",
        tools=["read", "search"],
        model=SubagentModel.FLASH,
        max_turns=5,
    ),
    "debugger": SubagentConfig(
        name="debugger",
        description="Debugging specialist for errors and test failures.",
        prompt="""You are an expert debugger. Your task is to:

1. Analyze error messages and stack traces
2. Identify root causes
3. Suggest specific fixes
4. Explain why the error occurred

Focus on finding the actual cause, not just symptoms.
Test your hypotheses by examining relevant code.""",
        tools=["read", "search", "run"],
        model=SubagentModel.FLASH,
        max_turns=10,
    ),
    "test-writer": SubagentConfig(
        name="test-writer",
        description="Test writing specialist for creating comprehensive tests.",
        prompt="""You are a test writing expert. Create comprehensive tests that:

1. Cover edge cases and error conditions
2. Test both positive and negative scenarios
3. Use appropriate mocking and fixtures
4. Follow the project's testing conventions
5. Are maintainable and well-documented

Focus on meaningful test coverage, not just line coverage.""",
        tools=["read", "write", "search", "run"],
        model=SubagentModel.FLASH,
        max_turns=10,
    ),
    "documenter": SubagentConfig(
        name="documenter",
        description="Documentation specialist for creating clear docs.",
        prompt="""You are a documentation expert. Create clear, helpful documentation:

1. API documentation with examples
2. Usage guides and tutorials
3. Architecture explanations
4. Code comments for complex logic
5. README updates

Write for the target audience. Include practical examples.""",
        tools=["read", "write"],
        model=SubagentModel.FLASH,
        max_turns=8,
    ),
    "security-auditor": SubagentConfig(
        name="security-auditor",
        description="Security specialist for auditing code vulnerabilities.",
        prompt="""You are a security auditor. Analyze code for:

1. Injection vulnerabilities (SQL, command, XSS)
2. Authentication and authorization issues
3. Data exposure risks
4. Insecure dependencies
5. Configuration weaknesses

Provide severity ratings and specific remediation steps.
Reference OWASP guidelines when applicable.""",
        tools=["read", "search", "run"],
        model=SubagentModel.FLASH,
        max_turns=8,
    ),
    "refactorer": SubagentConfig(
        name="refactorer",
        description="Refactoring specialist for improving code structure.",
        prompt="""You are a refactoring expert. Improve code by:

1. Reducing complexity and duplication
2. Improving naming and organization
3. Applying design patterns appropriately
4. Breaking down large functions/classes
5. Improving testability

Make incremental, safe changes. Preserve behavior.""",
        tools=["read", "write", "search"],
        model=SubagentModel.FLASH,
        max_turns=10,
    ),
    "explorer": SubagentConfig(
        name="explorer",
        description="Codebase exploration specialist for understanding code.",
        prompt="""You are a codebase exploration expert. Your task is to:

1. Map the project structure and architecture
2. Find relevant files and functions
3. Trace code paths and dependencies
4. Identify patterns and conventions used
5. Answer questions about how the code works

Be thorough but focused. Summarize your findings clearly.""",
        tools=["read", "search"],
        model=SubagentModel.FLASH,
        max_turns=10,
    ),
}


class SubagentManager:
    """
    Manages subagent creation and execution.

    Features:
    - Agent communication protocol (message passing)
    - Agent state management (idle/running/completed/failed)
    - Parallel execution with asyncio
    - Result aggregation and monitoring
    - Timeout and error handling with metrics

    Usage:
        mgr = SubagentManager(client, tool_registry)

        # Use a built-in agent
        result = await mgr.spawn("code-reviewer", task="Review auth.py")

        # Create a custom agent
        config = SubagentConfig(
            name="my-agent",
            description="Custom agent",
            prompt="You are a helpful assistant.",
        )
        result = await mgr.spawn_custom(config, task="Do something")

        # Run multiple agents in parallel
        results = await mgr.spawn_parallel([
            ("code-reviewer", "Review api.py"),
            ("security-auditor", "Audit api.py"),
        ])

        # Get agent metrics
        metrics = await mgr.get_agent_metrics(agent_id)

        # Get all metrics
        all_metrics = await mgr.get_all_metrics()
    """

    def __init__(
        self,
        model_client: Any,
        tool_registry: Any,  # ToolRegistry
        parent_model: str | None = None,
    ):
        """
        Initialize subagent manager.

        Args:
            model_client: Unified ModelClient (provider-agnostic)
            tool_registry: Tool registry for getting tools
            parent_model: Parent model name (used for INHERIT)
        """
        self.model_client = model_client
        self.tool_registry = tool_registry
        self.parent_model = parent_model or get_required_config_value("model")
        self._custom_agents: dict[str, SubagentConfig] = {}
        self._running_agents: dict[str, asyncio.Task] = {}
        self._message_queue = AgentMessageQueue()
        self._state_manager = AgentStateManager()

    def register_agent(self, config: SubagentConfig) -> None:
        """Register a custom agent configuration."""
        self._custom_agents[config.name] = config
        logger.info(f"Registered custom agent: {config.name}")

    def get_agent_config(self, name: str) -> SubagentConfig | None:
        """Get agent configuration by name."""
        if name in self._custom_agents:
            return self._custom_agents[name]
        return BUILTIN_AGENTS.get(name)

    def list_agents(self) -> list[dict[str, str]]:
        """List all available agents."""
        agents = []

        # Built-in agents
        for name, config in BUILTIN_AGENTS.items():
            agents.append(
                {
                    "name": name,
                    "description": config.description,
                    "type": "builtin",
                }
            )

        # Custom agents
        for name, config in self._custom_agents.items():
            agents.append(
                {
                    "name": name,
                    "description": config.description,
                    "type": "custom",
                }
            )

        return agents

    async def get_agent_state(self, agent_id: str) -> AgentState | None:
        """Get current state of an agent."""
        return await self._state_manager.get_state(agent_id)

    async def get_agent_metrics(self, agent_id: str) -> AgentMetrics | None:
        """Get metrics for a specific agent."""
        return await self._state_manager.get_metrics(agent_id)

    async def get_all_metrics(self) -> dict[str, AgentMetrics]:
        """Get metrics for all agents."""
        return await self._state_manager.get_all_metrics()

    async def send_message(self, message: AgentMessage) -> None:
        """Send a message between agents."""
        await self._message_queue.send(message)
        logger.debug(
            f"Message from {message.sender_id} to {message.recipient_id}: {message.message_type}"
        )

    async def receive_message(
        self, agent_id: str, timeout: float | None = None
    ) -> AgentMessage | None:
        """Receive a message for an agent."""
        return await self._message_queue.receive(agent_id=agent_id, timeout=timeout)

    def subscribe_to_messages(
        self, agent_id: str, callback: Callable[[AgentMessage], None]
    ) -> None:
        """Subscribe to messages for an agent."""
        self._message_queue.subscribe(agent_id, callback)

    def get_mailbox_messages(self, agent_id: str, limit: int = 100) -> list[AgentMessage]:
        """Read recent durable mailbox messages for an agent."""
        return self._message_queue.get_mailbox_messages(agent_id, limit=limit)

    async def spawn(
        self,
        agent_name: str,
        task: str,
        context: str = "",
        on_output: Callable[[str], None] | None = None,
    ) -> SubagentResult:
        """
        Spawn a named agent to perform a task.

        Args:
            agent_name: Name of agent (builtin or custom)
            task: Task description for the agent
            context: Additional context to provide
            on_output: Callback for streaming output

        Returns:
            SubagentResult with execution details
        """
        config = self.get_agent_config(agent_name)
        if not config:
            available = list(BUILTIN_AGENTS.keys()) + list(self._custom_agents.keys())
            return SubagentResult(
                agent_id="",
                agent_name=agent_name,
                success=False,
                output="",
                turns_used=0,
                tokens_used=0,
                duration=0,
                error=f"Unknown agent: {agent_name}. Available: {', '.join(available)}",
                state=AgentState.FAILED,
            )

        return await self.spawn_custom(config, task, context, on_output)

    async def spawn_custom(
        self,
        config: SubagentConfig,
        task: str,
        context: str = "",
        on_output: Callable[[str], None] | None = None,
    ) -> SubagentResult:
        """
        Spawn a custom agent to perform a task.

        Args:
            config: Agent configuration
            task: Task description
            context: Additional context
            on_output: Callback for streaming output

        Returns:
            SubagentResult with execution details
        """
        agent_id = _create_agent_id()

        return await _execute_agent_task(
            agent_id=agent_id,
            config=config,
            task=task,
            context=context,
            model_client=self.model_client,
            tool_registry=self.tool_registry,
            parent_model=self.parent_model,
            on_output=on_output,
            state_manager=self._state_manager,
        )

    async def spawn_parallel(
        self,
        tasks: list[tuple[str, str]],  # (agent_name, task)
        context: str = "",
    ) -> dict[str, Any]:
        """
        Run multiple agents in parallel.

        Args:
            tasks: List of (agent_name, task) tuples
            context: Shared context for all agents

        Returns:
            Aggregated results with metrics
        """
        coroutines = [self.spawn(name, task, context) for name, task in tasks]

        results = await asyncio.gather(*coroutines, return_exceptions=True)

        # Convert exceptions to failed results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                agent_name = str(tasks[i][0])
                final_results.append(
                    SubagentResult(
                        agent_id="",
                        agent_name=agent_name,
                        success=False,
                        output="",
                        turns_used=0,
                        tokens_used=0,
                        duration=0,
                        error=str(result),
                        state=AgentState.FAILED,
                    )
                )
            elif isinstance(result, SubagentResult):
                final_results.append(result)
            else:
                # Should not happen but for type safety
                logger.error(f"Unexpected result type from spawn: {type(result)}")

        return _aggregate_results(final_results)

    async def spawn_with_timeout(
        self,
        agent_name: str,
        task: str,
        timeout: float,
        context: str = "",
        on_output: Callable[[str], None] | None = None,
    ) -> SubagentResult:
        """
        Spawn an agent with a custom timeout.

        Args:
            agent_name: Name of agent
            task: Task description
            timeout: Timeout in seconds
            context: Additional context
            on_output: Output callback

        Returns:
            SubagentResult with execution details
        """
        config = self.get_agent_config(agent_name)
        if not config:
            available = list(BUILTIN_AGENTS.keys()) + list(self._custom_agents.keys())
            return SubagentResult(
                agent_id="",
                agent_name=agent_name,
                success=False,
                output="",
                turns_used=0,
                tokens_used=0,
                duration=0,
                error=f"Unknown agent: {agent_name}. Available: {', '.join(available)}",
                state=AgentState.FAILED,
            )

        # Create a modified config with custom timeout
        modified_config = SubagentConfig(
            name=config.name,
            description=config.description,
            prompt=config.prompt,
            tools=config.tools,
            model=config.model,
            max_turns=config.max_turns,
            timeout=timeout,
        )

        return await self.spawn_custom(modified_config, task, context, on_output)

    async def cancel_agent(self, agent_id: str) -> bool:
        """
        Cancel a running agent.

        Args:
            agent_id: ID of agent to cancel

        Returns:
            True if cancelled, False if not running
        """
        if agent_id in self._running_agents:
            task = self._running_agents[agent_id]
            task.cancel()
            await self._state_manager.set_state(agent_id, AgentState.CANCELLED)
            logger.info(f"Cancelled agent {agent_id}")
            return True
        return False

    async def get_running_agents(self) -> list[str]:
        """Get list of currently running agent IDs."""
        return list(self._running_agents.keys())

    async def wait_for_agent(self, agent_id: str, timeout: float | None = None) -> bool:
        """
        Wait for an agent to complete.

        Args:
            agent_id: ID of agent to wait for
            timeout: Timeout in seconds

        Returns:
            True if completed, False if timeout
        """
        if agent_id not in self._running_agents:
            return False

        try:
            if timeout:
                await asyncio.wait_for(self._running_agents[agent_id], timeout=timeout)
            else:
                await self._running_agents[agent_id]
            return True
        except asyncio.TimeoutError:
            return False
