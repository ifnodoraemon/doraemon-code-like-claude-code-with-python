"""
Base Agent - Abstract Agent Interface

Defines the standard interface for all agents, following agentic principles:
- Observe-Think-Act loop
- Pull-based state access
- Tool orchestration
- Clear lifecycle management
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from .state import AgentState
from .types import Action, AgentResult, Observation, Thought, ToolDefinition


class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    This follows the standard agentic pattern:
    - observe() -> Gather information from environment
    - think() -> Reason about what to do next
    - act() -> Execute the decided action
    - run() -> Main loop combining the above

    Key principles:
    1. Judge belongs to the model, not the runtime
    2. Runtime provides state, state exposes current world
    3. Tools return facts, not next actions
    4. Thin guardrails, hard constraints
    """

    def __init__(
        self,
        state: AgentState | None = None,
        tools: list[ToolDefinition] | None = None,
        **kwargs,
    ):
        self.state = state or AgentState()
        self._tools: dict[str, ToolDefinition] = {}
        self._sensitive_tools: set[str] = set()

        if tools:
            for tool in tools:
                self.register_tool(tool)

    @property
    def tools(self) -> list[ToolDefinition]:
        """Get all registered tools."""
        return list(self._tools.values())

    def register_tool(self, tool: ToolDefinition) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        if tool.sensitive:
            self._sensitive_tools.add(tool.name)

    def is_sensitive_tool(self, name: str) -> bool:
        """Check if a tool is sensitive (requires approval)."""
        return name in self._sensitive_tools

    def get_tool_definitions_for_api(self) -> list[dict]:
        """Get tools in API format."""
        return [t.to_api_format() for t in self.tools]

    @abstractmethod
    async def observe(self) -> Observation:
        """
        Observe the environment and gather information.

        Returns:
            Observation containing current state, tool results, errors
        """
        raise NotImplementedError

    @abstractmethod
    async def think(self, observation: Observation) -> Thought:
        """
        Reason about what to do next.

        Args:
            observation: Current observation of the environment

        Returns:
            Thought containing reasoning and planned actions
        """
        raise NotImplementedError

    @abstractmethod
    async def act(self, thought: Thought) -> Action:
        """
        Execute the decided action.

        Args:
            thought: Agent's reasoning about what to do

        Returns:
            Action to take (tool call, respond, ask user, etc.)
        """
        raise NotImplementedError

    @abstractmethod
    async def execute_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> tuple[str, str | None]:
        """
        Execute a tool and return the result.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tuple of (result, error)
        """
        raise NotImplementedError

    async def run(
        self,
        input: str,
        **kwargs,
    ) -> AgentResult:
        """
        Main entry point - execute the agent loop.

        This implements the Observe-Think-Act loop:
        1. Set goal from input
        2. Loop until finished or max turns:
           a. Observe environment
           b. Think about next action
           c. Act on the decision
           d. Update state
        3. Return result

        Args:
            input: User input/goal
            **kwargs: Additional options

        Returns:
            AgentResult with response and tool history
        """
        raise NotImplementedError

    async def run_stream(
        self,
        input: str,
        **kwargs,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Streaming version of run.

        Yields events:
        - {"type": "thinking", "content": "..."}
        - {"type": "tool_call", "name": "...", "args": {...}}
        - {"type": "tool_result", "result": "..."}
        - {"type": "response", "content": "..."}
        - {"type": "error", "error": "..."}
        """
        raise NotImplementedError

    async def ask_user(
        self,
        question: str,
        options: list[str] | None = None,
    ) -> str:
        """
        Ask user for input/clarification.

        Args:
            question: Question to ask
            options: Optional list of choices

        Returns:
            User's response
        """
        raise NotImplementedError

    async def check_permission(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> bool:
        """
        Check if a sensitive tool call should be allowed.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            True if allowed, False otherwise
        """
        if not self.is_sensitive_tool(tool_name):
            return True

        description = self._tools.get(tool_name)
        if not description:
            return False

        return (
            await self.ask_user(
                f"Allow tool call: {tool_name}({arguments})?",
                options=["yes", "no", "always"],
            )
            == "yes"
        )

    def reset(self) -> None:
        """Reset agent state for a new task."""
        self.state.clear_history()
        self.state.is_finished = False
        self.state.status = "idle"
        self.state.goal = None
        self.state.last_error = None

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the agent's current state."""
        return {
            **self.state.to_dict(),
            "tools_registered": len(self._tools),
            "sensitive_tools": list(self._sensitive_tools),
        }


class AgentError(Exception):
    """Base exception for agent errors."""

    pass


class ToolNotFoundError(AgentError):
    """Raised when a tool is not found."""

    pass


class ToolExecutionError(AgentError):
    """Raised when tool execution fails."""

    pass


class MaxTurnsExceededError(AgentError):
    """Raised when max turns is exceeded."""

    pass


class ContextOverflowError(AgentError):
    """Raised when context exceeds limits."""

    pass
