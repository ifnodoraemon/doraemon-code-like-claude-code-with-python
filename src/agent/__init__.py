"""
Doraemon Agent - Standard Agentic Interface

This module provides a standard agent abstraction following agentic principles:
- Observe-Think-Act loop (ReAct pattern)
- Pull-based state access
- Tool orchestration
- Clear lifecycle management

Usage:
    from src.agent import ReActAgent, AgentState, ToolDefinition

    # Define tools
    tools = [
        ToolDefinition(
            name="read_file",
            description="Read a file",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        ),
    ]

    # Create agent
    agent = ReActAgent(
        llm_client=model_client,
        state=AgentState(mode="build"),
        tools=tools,
    )

    # Run agent
    result = await agent.run("Read the README file")
    print(result.response)
"""

from .adapter import AgentSession, AgentTurnResult, run_agent_turn
from .base import (
    AgentError,
    BaseAgent,
    ContextOverflowError,
    MaxTurnsExceededError,
    ToolExecutionError,
    ToolNotFoundError,
)
from .doraemon import (
    DoraemonAgent,
    create_doraemon_agent,
    create_doraemon_agent_with_mcp,
)
from .react import ReActAgent
from .state import AgentState
from .types import (
    Action,
    ActionType,
    AgentResult,
    AgentStatus,
    Message,
    Observation,
    Thought,
    ToolCall,
    ToolDefinition,
)

__all__ = [
    "BaseAgent",
    "ReActAgent",
    "DoraemonAgent",
    "create_doraemon_agent",
    "AgentState",
    "AgentResult",
    "AgentStatus",
    "AgentError",
    "ToolNotFoundError",
    "ToolExecutionError",
    "MaxTurnsExceededError",
    "ContextOverflowError",
    "Message",
    "ToolCall",
    "ToolDefinition",
    "Observation",
    "Thought",
    "Action",
    "ActionType",
    "run_agent_turn",
    "AgentSession",
    "AgentTurnResult",
]
