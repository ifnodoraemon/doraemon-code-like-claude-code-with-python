"""
Agent Adapter for Evaluation System

Provides adapters to connect the evaluation system with real and mock agents.
Enables testing of Doraemon Code Agent in automated evaluation scenarios.
"""

import asyncio
import logging
import random
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


# ========================================
# Data Classes
# ========================================


@dataclass
class TokenUsage:
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class ToolCall:
    """Represents a single tool call made by the agent."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str | None = None
    success: bool = True
    execution_time: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "result": self.result,
            "success": self.success,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCall":
        return cls(
            name=data["name"],
            arguments=data.get("arguments", {}),
            result=data.get("result"),
            success=data.get("success", True),
            execution_time=data.get("execution_time", 0.0),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class AgentResponse:
    """Response from an agent execution."""

    success: bool
    output: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    execution_time: float = 0.0
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "execution_time": self.execution_time,
            "token_usage": self.token_usage.to_dict(),
            "errors": self.errors,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentResponse":
        return cls(
            success=data["success"],
            output=data["output"],
            tool_calls=[ToolCall.from_dict(tc) for tc in data.get("tool_calls", [])],
            execution_time=data.get("execution_time", 0.0),
            token_usage=TokenUsage(**data.get("token_usage", {})),
            errors=data.get("errors", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""

    role: str  # "user" or "assistant"
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


# ========================================
# Agent Adapter Base Class
# ========================================


class AgentAdapter(ABC):
    """
    Abstract base class for agent adapters.

    Provides a unified interface for different agent implementations
    to be used with the evaluation system.
    """

    def __init__(self, name: str = "BaseAgent"):
        self.name = name
        self._tool_calls: list[ToolCall] = []
        self._conversation_history: list[ConversationTurn] = []
        self._total_token_usage = TokenUsage()

    @abstractmethod
    def execute(self, prompt: str) -> AgentResponse:
        """
        Execute a task given a prompt.

        Args:
            prompt: The task prompt to execute

        Returns:
            AgentResponse containing the result
        """
        pass

    @abstractmethod
    def get_tool_calls(self) -> list[ToolCall]:
        """
        Get the history of tool calls made by the agent.

        Returns:
            List of ToolCall objects
        """
        pass

    def get_conversation_history(self) -> list[ConversationTurn]:
        """Get the conversation history."""
        return self._conversation_history.copy()

    def get_total_token_usage(self) -> TokenUsage:
        """Get total token usage across all executions."""
        return self._total_token_usage

    def reset(self) -> None:
        """Reset the agent state for a new evaluation."""
        self._tool_calls.clear()
        self._conversation_history.clear()
        self._total_token_usage = TokenUsage()

    def _record_tool_call(self, tool_call: ToolCall) -> None:
        """Record a tool call."""
        self._tool_calls.append(tool_call)

    def _record_conversation_turn(self, turn: ConversationTurn) -> None:
        """Record a conversation turn."""
        self._conversation_history.append(turn)


# ========================================
# Doraemon Agent Adapter
# ========================================


class DoraemonAgentAdapter(AgentAdapter):
    """
    Adapter for the real Doraemon Code Agent.

    Connects to the Doraemon Code ModelClient and ToolRegistry
    to execute tasks and track tool usage.
    """

    def __init__(
        self,
        model: str | None = None,
        timeout: float = 300.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        auto_approve_tools: bool = True,
    ):
        """
        Initialize the Doraemon Agent adapter.

        Args:
            model: Model name to use (defaults to env DORAEMON_MODEL)
            timeout: Maximum execution time in seconds
            max_retries: Number of retries on failure
            retry_delay: Delay between retries in seconds
            auto_approve_tools: Auto-approve sensitive tool calls (for eval)
        """
        super().__init__(name="DoraemonAgent")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.auto_approve_tools = auto_approve_tools

        # Lazy initialization
        self._model_client = None
        self._tool_registry = None
        self._context_manager = None
        self._initialized = False

    def _initialize(self) -> None:
        """Lazy initialization of Doraemon components."""
        if self._initialized:
            return

        try:
            from src.core.context_manager import ContextManager
            from src.core.model_client import ModelClient
            from src.host.tools import get_tool_registry

            self._model_client = ModelClient(model=self.model)
            self._context_manager = ContextManager(project="eval")
            self._tool_registry = get_tool_registry()
            self._initialized = True
            logger.info("DoraemonAgentAdapter initialized successfully")

        except ImportError as e:
            logger.error(f"Failed to import Doraemon components: {e}")
            raise RuntimeError(
                "Doraemon Code components not available. "
                "Ensure the project is properly installed."
            ) from e

    def execute(self, prompt: str) -> AgentResponse:
        """
        Execute a task using the Doraemon Agent.

        Args:
            prompt: The task prompt

        Returns:
            AgentResponse with results
        """
        self._initialize()

        start_time = time.time()
        errors: list[str] = []
        tool_calls: list[ToolCall] = []
        output = ""
        success = False

        # Record user turn
        self._record_conversation_turn(
            ConversationTurn(role="user", content=prompt)
        )

        for attempt in range(self.max_retries):
            try:
                # Execute with timeout
                response = self._execute_with_timeout(prompt)

                output = response.get("output", "")
                tool_calls = response.get("tool_calls", [])
                token_usage = response.get("token_usage", TokenUsage())

                # Update totals
                self._total_token_usage = self._total_token_usage + token_usage
                for tc in tool_calls:
                    self._record_tool_call(tc)

                # Record assistant turn
                self._record_conversation_turn(
                    ConversationTurn(
                        role="assistant",
                        content=output,
                        tool_calls=tool_calls,
                    )
                )

                success = True
                break

            except asyncio.TimeoutError:
                error_msg = f"Execution timed out after {self.timeout}s (attempt {attempt + 1})"
                logger.warning(error_msg)
                errors.append(error_msg)

            except Exception as e:
                error_msg = f"Execution failed: {str(e)} (attempt {attempt + 1})"
                logger.warning(error_msg)
                errors.append(error_msg)

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        execution_time = time.time() - start_time

        return AgentResponse(
            success=success,
            output=output,
            tool_calls=tool_calls,
            execution_time=execution_time,
            token_usage=self._total_token_usage,
            errors=errors,
            metadata={
                "model": self.model,
                "attempts": attempt + 1,
            },
        )

    def _execute_with_timeout(self, prompt: str) -> dict[str, Any]:
        """Execute the prompt with timeout control."""
        # Run async execution in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                asyncio.wait_for(
                    self._async_execute(prompt),
                    timeout=self.timeout,
                )
            )
            return result
        finally:
            loop.close()

    async def _async_execute(self, prompt: str) -> dict[str, Any]:
        """Async execution of the agent task."""
        tool_calls: list[ToolCall] = []
        total_output = ""
        token_usage = TokenUsage()

        # Add message to context
        self._context_manager.add_message("user", prompt)

        # Get conversation history for API
        messages = self._context_manager.get_messages_for_api()

        # Get available tools
        tools = self._tool_registry.get_tool_definitions() if self._tool_registry else []

        # Call model
        response = await self._model_client.chat_async(
            messages=messages,
            tools=tools,
        )

        # Process response
        if hasattr(response, "content"):
            total_output = response.content or ""

        # Track token usage
        if hasattr(response, "usage"):
            usage = response.usage
            token_usage = TokenUsage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                completion_tokens=getattr(usage, "completion_tokens", 0),
                total_tokens=getattr(usage, "total_tokens", 0),
            )

        # Process tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                tool_start = time.time()

                # Execute tool
                try:
                    if self._tool_registry:
                        result = self._tool_registry.execute(
                            tc.name,
                            tc.arguments,
                            auto_approve=self.auto_approve_tools,
                        )
                        tool_success = True
                    else:
                        result = "Tool registry not available"
                        tool_success = False
                except Exception as e:
                    result = f"Error: {str(e)}"
                    tool_success = False

                tool_call = ToolCall(
                    name=tc.name,
                    arguments=tc.arguments if hasattr(tc, "arguments") else {},
                    result=str(result),
                    success=tool_success,
                    execution_time=time.time() - tool_start,
                )
                tool_calls.append(tool_call)

        # Add assistant response to context
        self._context_manager.add_message("assistant", total_output)

        return {
            "output": total_output,
            "tool_calls": tool_calls,
            "token_usage": token_usage,
        }

    def get_tool_calls(self) -> list[ToolCall]:
        """Get all tool calls made during execution."""
        return self._tool_calls.copy()

    def reset(self) -> None:
        """Reset agent state."""
        super().reset()
        if self._context_manager:
            self._context_manager.clear()


# ========================================
# Mock Agent for Testing
# ========================================


class MockAgentAdapter(AgentAdapter):
    """
    Mock agent for testing the evaluation system.

    Simulates agent behavior with configurable success rate,
    delays, and tool usage patterns.
    """

    def __init__(
        self,
        success_rate: float = 0.8,
        min_delay: float = 0.1,
        max_delay: float = 1.0,
        tool_patterns: dict[str, list[str]] | None = None,
        response_generator: Callable[[str], str] | None = None,
    ):
        """
        Initialize the mock agent.

        Args:
            success_rate: Probability of successful execution (0.0 to 1.0)
            min_delay: Minimum simulated execution delay
            max_delay: Maximum simulated execution delay
            tool_patterns: Mapping of keywords to tool names to simulate
            response_generator: Custom function to generate responses
        """
        super().__init__(name="MockAgent")
        self.success_rate = success_rate
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.tool_patterns = tool_patterns or self._default_tool_patterns()
        self.response_generator = response_generator or self._default_response_generator

    def _default_tool_patterns(self) -> dict[str, list[str]]:
        """Default tool patterns based on prompt keywords."""
        return {
            "read": ["read"],
            "write": ["write"],
            "search": ["search"],
            "file": ["read", "write"],
            "code": ["read", "write", "search"],
            "git": ["shell_execute"],
            "test": ["shell_execute", "read"],
            "debug": ["read", "search"],
            "refactor": ["read", "write", "search"],
        }

    def _default_response_generator(self, prompt: str) -> str:
        """Generate a default mock response."""
        return f"Mock response for: {prompt[:100]}..."

    def execute(self, prompt: str) -> AgentResponse:
        """
        Execute a mock task.

        Args:
            prompt: The task prompt

        Returns:
            AgentResponse with simulated results
        """
        start_time = time.time()

        # Simulate processing delay
        delay = random.uniform(self.min_delay, self.max_delay)
        time.sleep(delay)

        # Determine success based on configured rate
        success = random.random() < self.success_rate

        # Generate tool calls based on prompt keywords
        tool_calls = self._generate_tool_calls(prompt)

        # Record tool calls
        for tc in tool_calls:
            self._record_tool_call(tc)

        # Generate response
        if success:
            output = self.response_generator(prompt)
            errors = []
        else:
            output = ""
            errors = ["Simulated failure for testing"]

        # Simulate token usage
        token_usage = TokenUsage(
            prompt_tokens=len(prompt.split()) * 2,
            completion_tokens=len(output.split()) * 2 if output else 0,
            total_tokens=len(prompt.split()) * 2 + (len(output.split()) * 2 if output else 0),
        )
        self._total_token_usage = self._total_token_usage + token_usage

        # Record conversation turns
        self._record_conversation_turn(
            ConversationTurn(role="user", content=prompt)
        )
        self._record_conversation_turn(
            ConversationTurn(role="assistant", content=output, tool_calls=tool_calls)
        )

        execution_time = time.time() - start_time

        return AgentResponse(
            success=success,
            output=output,
            tool_calls=tool_calls,
            execution_time=execution_time,
            token_usage=token_usage,
            errors=errors,
            metadata={
                "mock": True,
                "success_rate": self.success_rate,
            },
        )

    def _generate_tool_calls(self, prompt: str) -> list[ToolCall]:
        """Generate simulated tool calls based on prompt."""
        tool_calls = []
        prompt_lower = prompt.lower()

        # Find matching tool patterns
        tools_to_call = set()
        for keyword, tools in self.tool_patterns.items():
            if keyword in prompt_lower:
                tools_to_call.update(tools)

        # Create tool call objects
        for tool_name in tools_to_call:
            tc = ToolCall(
                name=tool_name,
                arguments={"mock": True, "prompt_snippet": prompt[:50]},
                result="Mock tool result",
                success=True,
                execution_time=random.uniform(0.01, 0.1),
            )
            tool_calls.append(tc)

        return tool_calls

    def get_tool_calls(self) -> list[ToolCall]:
        """Get all simulated tool calls."""
        return self._tool_calls.copy()


# ========================================
# Agent Factory
# ========================================


class AgentFactory:
    """Factory for creating agent adapters."""

    _registry: dict[str, type] = {
        "doraemon": DoraemonAgentAdapter,
        "mock": MockAgentAdapter,
    }

    @classmethod
    def register(cls, name: str, adapter_class: type) -> None:
        """Register a new agent adapter type."""
        cls._registry[name] = adapter_class

    @classmethod
    def create(cls, agent_type: str, **kwargs) -> AgentAdapter:
        """
        Create an agent adapter.

        Args:
            agent_type: Type of agent ("doraemon", "mock", etc.)
            **kwargs: Arguments to pass to the adapter constructor

        Returns:
            AgentAdapter instance
        """
        if agent_type not in cls._registry:
            raise ValueError(
                f"Unknown agent type: {agent_type}. "
                f"Available types: {list(cls._registry.keys())}"
            )

        return cls._registry[agent_type](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """Get list of available agent types."""
        return list(cls._registry.keys())


# ========================================
# Utility Functions
# ========================================


def create_doraemon_agent(
    model: str | None = None,
    timeout: float = 300.0,
    **kwargs,
) -> DoraemonAgentAdapter:
    """
    Convenience function to create a Doraemon agent adapter.

    Args:
        model: Model name to use
        timeout: Execution timeout in seconds
        **kwargs: Additional arguments

    Returns:
        DoraemonAgentAdapter instance
    """
    return DoraemonAgentAdapter(model=model, timeout=timeout, **kwargs)


def create_mock_agent(
    success_rate: float = 0.8,
    **kwargs,
) -> MockAgentAdapter:
    """
    Convenience function to create a mock agent adapter.

    Args:
        success_rate: Probability of success
        **kwargs: Additional arguments

    Returns:
        MockAgentAdapter instance
    """
    return MockAgentAdapter(success_rate=success_rate, **kwargs)


# ========================================
# Main (Demo)
# ========================================


def main():
    """Demonstrate agent adapter usage."""
    print("=" * 60)
    print("Agent Adapter Demo")
    print("=" * 60)

    # Demo with mock agent
    print("\n1. Mock Agent Demo")
    print("-" * 40)

    mock_agent = create_mock_agent(success_rate=0.9)

    prompts = [
        "Read the file src/main.py and explain its structure",
        "Search for all TODO comments in the codebase",
        "Write a new test file for the utils module",
    ]

    for prompt in prompts:
        print(f"\nPrompt: {prompt[:50]}...")
        response = mock_agent.execute(prompt)
        print(f"  Success: {response.success}")
        print(f"  Execution time: {response.execution_time:.3f}s")
        print(f"  Tool calls: {[tc.name for tc in response.tool_calls]}")
        print(f"  Token usage: {response.token_usage.total_tokens}")

    print(f"\nTotal tool calls: {len(mock_agent.get_tool_calls())}")
    print(f"Total tokens used: {mock_agent.get_total_token_usage().total_tokens}")

    # Demo agent factory
    print("\n2. Agent Factory Demo")
    print("-" * 40)
    print(f"Available agent types: {AgentFactory.available_types()}")

    factory_agent = AgentFactory.create("mock", success_rate=1.0)
    response = factory_agent.execute("Test prompt")
    print(f"Factory-created agent response: {response.success}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
