"""Targeted coverage for agent/react.py uncovered lines: 135,141-142,144,186,274,307-308,323-325,353-355,385-388,401-402,426-429,473-474,601,604,763."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.react import ReActAgent, TraceInterface
from src.agent.state import AgentState
from src.agent.types import (
    Action,
    ActionType,
    AgentResult,
    Message,
    Observation,
    Thought,
    ToolCall,
    ToolDefinition,
)


class TestObserveGoalRelevanceEdge:
    @pytest.mark.asyncio
    async def test_observe_seen_ids_dedup(self):
        llm = AsyncMock()
        agent = ReActAgent(llm_client=llm, tools=[])
        agent.state.goal = "analyze database schema"
        for i in range(10):
            tc = ToolCall(id=str(i), name="read", arguments={}, result=f"result_{i}")
            agent.state.add_tool_call(tc)
        obs = await agent.observe()
        ids = [id(tc) for tc in obs.tool_results]
        assert len(ids) == len(set(ids))

    @pytest.mark.asyncio
    async def test_observe_limits_relevant_results(self):
        llm = AsyncMock()
        agent = ReActAgent(llm_client=llm, tools=[])
        agent.state.goal = "find database"
        for i in range(20):
            tc = ToolCall(id=str(i), name="read", arguments={}, result="database info")
            agent.state.add_tool_call(tc)
        obs = await agent.observe()
        assert len(obs.tool_results) <= 16

    @pytest.mark.asyncio
    async def test_observe_write_is_relevant(self):
        llm = AsyncMock()
        agent = ReActAgent(llm_client=llm, tools=[])
        agent.state.goal = "fix bug"
        tc = ToolCall(id="old", name="write", arguments={}, result="wrote file")
        agent.state.add_tool_call(tc)
        tc2 = ToolCall(id="recent", name="read", arguments={}, result="read file")
        agent.state.add_tool_call(tc2)
        obs = await agent.observe()
        names = [tc.name for tc in obs.tool_results]
        assert "write" in names


class TestThinkTimeoutTrace:
    @pytest.mark.asyncio
    async def test_think_timeout_records_trace(self):
        trace = TraceInterface()
        trace.event = MagicMock()
        llm = AsyncMock()

        async def slow_chat(*a, **kw):
            await asyncio.sleep(10)

        llm.chat = slow_chat
        agent = ReActAgent(llm_client=llm, tools=[], timeout=0.1)
        agent.set_trace(trace)
        agent.state.add_user_message("test")
        with pytest.raises(TimeoutError):
            await agent.think(Observation(user_input="test"))
        trace.event.assert_called()
        assert trace.event.call_args[0][0] == "error"


class TestRunParallelToolArgsParsing:
    @pytest.mark.asyncio
    async def test_parallel_string_args_invalid_json(self):
        llm = AsyncMock()
        llm.chat = AsyncMock(return_value=MagicMock(content="Done!", tool_calls=None))
        tools = [ToolDefinition(name="read", description="Read", parameters={})]
        agent = ReActAgent(llm_client=llm, tools=tools)
        agent.permission_callback = AsyncMock(return_value="yes")
        executor = AsyncMock(return_value=("result", None))
        agent.set_tool_executor(executor)

        tool_calls = [
            {"name": "read", "arguments": "{bad json}"},
            {"name": "read", "arguments": '{"path": "a"}'},
        ]
        results = await agent._execute_tools_parallel(tool_calls)
        assert len(results) == 2


class TestRunSingleToolActionError:
    @pytest.mark.asyncio
    async def test_run_with_error_action_type(self):
        llm = AsyncMock()
        agent = ReActAgent(llm_client=llm, tools=[])
        thought = Thought(
            reasoning="error",
            tool_calls=[{"name": "bad", "arguments": {}}],
        )
        action = Action(type=ActionType.ERROR, error="something broke")
        with patch.object(agent, "observe", new_callable=AsyncMock) as mock_obs:
            with patch.object(agent, "think", new_callable=AsyncMock) as mock_think:
                with patch.object(agent, "act", new_callable=AsyncMock) as mock_act:
                    mock_obs.return_value = Observation(user_input="test")
                    mock_think.return_value = thought
                    mock_act.return_value = action
                    result = await agent.run("test error")
        assert result.success is False


class TestRunStreamToolCallArgParsing:
    @pytest.mark.asyncio
    async def test_stream_tool_call_with_string_args_invalid_json(self):
        llm = AsyncMock()
        tools = [ToolDefinition(name="read", description="Read", parameters={})]
        agent = ReActAgent(llm_client=llm, tools=tools)
        agent.set_tool_executor(lambda n, a: ("result", None))
        agent.permission_callback = AsyncMock(return_value="yes")

        tc = MagicMock()
        tc.id = "1"
        tc.function = MagicMock(name="read", arguments='{"path": "a"}')
        first_response = MagicMock(content=None, tool_calls=[tc])
        second_response = MagicMock(content="Done!", tool_calls=None)
        llm.chat = AsyncMock(side_effect=[first_response, second_response])

        events = []
        async for event in agent.run_stream("test"):
            events.append(event)
        tool_events = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_events) >= 1


class TestCallLlmCallable:
    @pytest.mark.asyncio
    async def test_call_llm_with_callable_client(self):
        async def my_callable(messages, tools=None):
            return MagicMock(content="callable result", tool_calls=None)

        agent = ReActAgent(llm_client=my_callable, tools=[])
        result = await agent._call_llm([{"role": "user", "content": "hi"}], [])
        assert result["content"] == "callable result"


class TestEstimateTokensEdge:
    def test_with_none_content(self):
        llm = AsyncMock()
        agent = ReActAgent(llm_client=llm, tools=[])
        msgs = [{"content": None}]
        assert agent._estimate_tokens(msgs) == 0


class TestCompressContextSemantic:
    @pytest.mark.asyncio
    async def test_compress_updates_token_estimate(self):
        llm = AsyncMock()
        llm.chat = AsyncMock(return_value=MagicMock(content="compressed summary", tool_calls=None))
        agent = ReActAgent(llm_client=llm, tools=[])
        for i in range(20):
            agent.state.add_user_message(f"Message {i}" * 5)
        await agent._compress_context()
        assert len(agent.state.messages) < 20


class TestSummarizeMessages:
    @pytest.mark.asyncio
    async def test_summarize_messages_removed(self):
        llm = AsyncMock()
        agent = ReActAgent(llm_client=llm, tools=[])
        assert not hasattr(agent, "_summarize_messages")


class TestBuildResultNoStartTime:
    def test_build_result_no_start(self):
        llm = AsyncMock()
        agent = ReActAgent(llm_client=llm, tools=[])
        agent._start_time = None
        result = agent._build_result()
        assert result.duration == 0
