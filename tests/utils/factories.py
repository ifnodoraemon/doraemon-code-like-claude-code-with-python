"""
Test Utilities and Mock Factories

Provides factory functions for creating test objects and mocks.
"""

from typing import Any
from unittest.mock import AsyncMock

from src.agent import AgentState
from src.core.llm.model_utils import (
    ChatResponse,
    ClientConfig,
    ClientMode,
    Message,
    ToolDefinition,
)
from src.host.tools import ToolRegistry


def create_mock_model_client(responses: list[str | ChatResponse]) -> AsyncMock:
    """
    Create a mock model client with predefined responses.

    Args:
        responses: List of response strings or ChatResponse objects

    Returns:
        AsyncMock configured to return the responses in sequence

    Example:
        client = create_mock_model_client(["Hello", "World"])
        response1 = await client.chat([])  # Returns ChatResponse(content="Hello")
        response2 = await client.chat([])  # Returns ChatResponse(content="World")
    """
    client = AsyncMock()

    # Convert strings to ChatResponse objects
    chat_responses = []
    for resp in responses:
        if isinstance(resp, str):
            chat_responses.append(ChatResponse(content=resp, finish_reason="stop"))
        else:
            chat_responses.append(resp)

    # Configure chat method to return responses in sequence
    client.chat.side_effect = chat_responses

    # Configure other methods
    client.connect = AsyncMock()
    client.close = AsyncMock()
    client.list_models = AsyncMock(return_value=[])

    return client


def create_test_agent_state(
    mode: str = "build",
    max_turns: int = 100,
) -> AgentState:
    """
    Create an AgentState for testing.

    Args:
        mode: Agent mode (plan/build)
        max_turns: Maximum turns

    Returns:
        Configured AgentState instance
    """
    return AgentState(mode=mode, max_turns=max_turns)


def create_test_tool_registry() -> ToolRegistry:
    """
    Create a ToolRegistry with test tools.

    Returns:
        ToolRegistry with simple test tools registered
    """
    registry = ToolRegistry()

    # Register simple test tools
    def test_read(path: str) -> str:
        """Test tool: read a file."""
        return f"Content of {path}"

    def test_write(path: str, content: str) -> str:
        """Test tool: write a file."""
        return f"Wrote {len(content)} bytes to {path}"

    def test_execute_command(command: str) -> str:
        """Test tool: execute a command."""
        return f"Executed: {command}"

    registry.register(test_read, sensitive=False)
    registry.register(test_write, sensitive=True)
    registry.register(test_execute_command, sensitive=True)

    return registry


def create_test_messages(count: int = 3) -> list[Message]:
    """
    Create a list of test messages.

    Args:
        count: Number of messages to create

    Returns:
        List of Message objects
    """
    messages = []
    for i in range(count):
        if i % 2 == 0:
            messages.append(Message(role="user", content=f"User message {i}"))
        else:
            messages.append(Message(role="assistant", content=f"Assistant message {i}"))
    return messages


def create_test_tool_definition(
    name: str = "test_tool",
    description: str = "A test tool",
    parameters: dict[str, Any] | None = None,
) -> ToolDefinition:
    """
    Create a test tool definition.

    Args:
        name: Tool name
        description: Tool description
        parameters: Tool parameters schema

    Returns:
        ToolDefinition instance
    """
    if parameters is None:
        parameters = {
            "type": "object",
            "properties": {
                "arg1": {"type": "string", "description": "First argument"},
                "arg2": {"type": "integer", "description": "Second argument"},
            },
            "required": ["arg1"],
        }

    return ToolDefinition(name=name, description=description, parameters=parameters)


def create_test_client_config(mode: ClientMode = ClientMode.DIRECT) -> ClientConfig:
    """
    Create a test ClientConfig.

    Args:
        mode: Client mode (GATEWAY or DIRECT)

    Returns:
        ClientConfig instance
    """
    if mode == ClientMode.GATEWAY:
        return ClientConfig(
            mode=ClientMode.GATEWAY,
            gateway_url="http://localhost:8000",
            gateway_key="test-key",
            model="test-model",
        )
    else:
        return ClientConfig(
            mode=ClientMode.DIRECT,
            google_api_key="test-google-key",
            model="gemini-test",
        )


def create_chat_response_with_tools(tool_calls: list[dict[str, Any]]) -> ChatResponse:
    """
    Create a ChatResponse with tool calls.

    Args:
        tool_calls: List of tool call dictionaries

    Returns:
        ChatResponse with tool calls
    """
    return ChatResponse(
        content=None,
        tool_calls=tool_calls,
        finish_reason="tool_calls",
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    )


def create_simple_tool_call(
    name: str = "test_tool",
    arguments: dict[str, Any] | None = None,
    call_id: str = "call_123",
) -> dict[str, Any]:
    """
    Create a simple tool call dictionary.

    Args:
        name: Tool name
        arguments: Tool arguments
        call_id: Tool call ID

    Returns:
        Tool call dictionary
    """
    if arguments is None:
        arguments = {"arg1": "value1"}

    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": str(arguments),
        },
    }
