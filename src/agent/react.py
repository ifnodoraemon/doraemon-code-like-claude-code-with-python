"""
ReAct Agent - Reasoning and Acting Agent

Implements the ReAct pattern:
- Thought: Agent reasons about the current state
- Action: Agent decides what to do (tool call or respond)
- Observation: Agent sees the result of the action

This is the core agentic loop that makes decisions, not the runtime.
"""

import asyncio
import json
import time
import uuid
from typing import Any, AsyncIterator, Callable

from .base import (
    BaseAgent,
    ContextOverflowError,
    MaxTurnsExceededError,
    ToolExecutionError,
    ToolNotFoundError,
)
from .state import AgentState
from .types import (
    Action,
    ActionType,
    AgentResult,
    Message,
    Observation,
    Thought,
    ToolCall,
    ToolDefinition,
)


class ReActAgent(BaseAgent):
    """
    ReAct (Reasoning + Acting) Agent.

    This is the standard agentic pattern where the agent:
    1. Observes the environment
    2. Thinks about what to do
    3. Acts on the decision
    4. Repeats until done

    The key difference from non-agentic systems:
    - The MODEL decides the path, not the runtime
    - The runtime provides state and tools, but doesn't dictate workflow
    - Tools return facts, not suggestions for next actions
    """

    def __init__(
        self,
        llm_client: Any,
        state: AgentState | None = None,
        tools: list[ToolDefinition] | None = None,
        *,
        max_turns: int = 100,
        timeout: float = 300.0,
        permission_callback: Callable | None = None,
        **kwargs,
    ):
        super().__init__(state=state, tools=tools, **kwargs)
        self.llm = llm_client
        self.max_turns = max_turns
        self.timeout = timeout
        self.permission_callback = permission_callback
        self._start_time: float | None = None
        self._tool_executor: Callable | None = None

    def set_tool_executor(self, executor: Callable) -> None:
        """Set the tool executor function."""
        self._tool_executor = executor

    async def observe(self) -> Observation:
        """
        Observe the current environment state.

        Pull-based: Agent gets what it needs, not everything.
        """
        tool_results = []
        errors = []

        for tc in self.state.tool_history[-5:]:
            if tc.result or tc.error:
                tool_results.append(tc)
            if tc.error:
                errors.append(tc.error)

        return Observation(
            user_input=self.state.user_input,
            tool_results=tool_results,
            errors=errors,
            context={
                "turn_count": self.state.turn_count,
                "goal": self.state.goal,
                "is_finished": self.state.is_finished,
            },
        )

    async def think(self, observation: Observation) -> Thought:
        """
        Reason about the next action.

        This is where the LLM makes decisions.
        """
        messages = self._build_messages(observation)
        tools = self.get_tool_definitions_for_api()

        try:
            response = await asyncio.wait_for(
                self._call_llm(messages, tools),
                timeout=60.0,
            )

            return Thought(
                reasoning=response.get("reasoning", ""),
                tool_calls=response.get("tool_calls", []),
                response=response.get("content"),
                is_finished=response.get("is_finished", False),
            )

        except asyncio.TimeoutError:
            return Thought(
                reasoning="LLM call timed out",
                is_finished=True,
            )
        except Exception as e:
            return Thought(
                reasoning=f"Error in thinking: {e}",
                is_finished=True,
            )

    async def act(self, thought: Thought) -> Action:
        """
        Execute the decided action.

        Note: Only processes the first tool call. Remaining tool calls
        will be handled in subsequent iterations of the ReAct loop.
        This ensures each action has proper observation before proceeding.
        """
        if thought.is_finished or not thought.tool_calls:
            return Action.respond(thought.response or "Task completed.")

        tool_call = thought.tool_calls[0]
        name = tool_call.get("name") or tool_call.get("function", {}).get("name")
        args = tool_call.get("arguments") or tool_call.get("function", {}).get("arguments", {})

        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}

        call_id = tool_call.get("id") or str(uuid.uuid4())

        return Action.tool_call(name, args, call_id)

    async def execute_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> tuple[str, str | None]:
        """
        Execute a tool and return the result.

        Tools return facts, not next actions.
        """
        if name not in self._tools:
            return "", f"Tool not found: {name}"

        if self._tool_executor:
            return await self._tool_executor(name, arguments)

        return "", "No tool executor configured"

    async def run(
        self,
        input: str,
        **kwargs,
    ) -> AgentResult:
        """
        Main ReAct loop.

        The runtime does NOT decide the path - the model does.
        The runtime only:
        1. Provides state
        2. Executes tools (in parallel when independent)
        3. Enforces hard constraints (max turns, timeout)
        """
        self._start_time = time.time()
        self.state.set_goal(input)
        self.state.add_user_message(input)

        try:
            while not self.state.is_finished:
                if not self.state.increment_turn():
                    raise MaxTurnsExceededError(f"Exceeded max turns ({self.max_turns})")

                if time.time() - self._start_time > self.timeout:
                    raise TimeoutError(f"Agent timeout ({self.timeout}s)")

                if self.state.needs_compression():
                    await self._compress_context()

                observation = await self.observe()
                thought = await self.think(observation)

                if thought.is_finished or not thought.tool_calls:
                    self.state.add_assistant_message(thought.response)
                    self.state.mark_finished()
                    break

                if len(thought.tool_calls) > 1:
                    results = await self._execute_tools_parallel(thought.tool_calls)
                    for tc_data, (result, error) in zip(thought.tool_calls, results):
                        call_id = tc_data.get("id") or str(uuid.uuid4())
                        name = tc_data.get("name") or tc_data.get("function", {}).get("name", "")
                        args = tc_data.get("arguments") or tc_data.get("function", {}).get(
                            "arguments", {}
                        )
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except:
                                args = {}

                        self.state.add_tool_call(
                            ToolCall(
                                id=call_id,
                                name=name,
                                arguments=args,
                                result=result,
                                error=error,
                            )
                        )
                        self.state.add_tool_result(call_id, name, result or f"Error: {error}")
                else:
                    action = await self.act(thought)

                    if action.type == ActionType.RESPOND:
                        self.state.add_assistant_message(action.response)
                        self.state.mark_finished()
                        break

                    elif action.type == ActionType.TOOL_CALL:
                        result, error = await self._execute_tool_with_permission(
                            action.tool_name,
                            action.tool_args,
                        )

                        tool_call = ToolCall(
                            id=action.tool_call_id or str(uuid.uuid4()),
                            name=action.tool_name,
                            arguments=action.tool_args or {},
                            result=result,
                            error=error,
                        )
                        self.state.add_tool_call(tool_call)
                        self.state.add_tool_result(
                            tool_call.id,
                            tool_call.name,
                            result or f"Error: {error}",
                        )

                    elif action.type == ActionType.ERROR:
                        self.state.mark_error(action.error or "Unknown error")
                        break

            return self._build_result()

        except Exception as e:
            self.state.mark_error(str(e))
            return self._build_result(error=str(e))

    async def _execute_tools_parallel(
        self,
        tool_calls: list[dict],
    ) -> list[tuple[str, str | None]]:
        """Execute multiple tool calls in parallel."""
        tasks = []
        for tc in tool_calls:
            name = tc.get("name") or tc.get("function", {}).get("name", "")
            args = tc.get("arguments") or tc.get("function", {}).get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except:
                    args = {}
            tasks.append(self._execute_tool_with_permission(name, args))

        return await asyncio.gather(*tasks)

    async def run_stream(
        self,
        input: str,
        **kwargs,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Streaming version of run.

        Yields events for UI updates.
        """
        self._start_time = time.time()
        self.state.set_goal(input)
        self.state.add_user_message(input)

        yield {"type": "start", "input": input}

        try:
            while not self.state.is_finished:
                if not self.state.increment_turn():
                    yield {"type": "error", "error": "Max turns exceeded"}
                    break

                observation = await self.observe()
                thought = await self.think(observation)

                if thought.reasoning:
                    yield {"type": "thinking", "content": thought.reasoning}

                if not thought.tool_calls:
                    self.state.mark_finished()
                    yield {"type": "response", "content": thought.response}
                    break

                for tool_call_data in thought.tool_calls:
                    name = tool_call_data.get("name") or tool_call_data.get("function", {}).get(
                        "name", ""
                    )
                    args = tool_call_data.get("arguments") or tool_call_data.get(
                        "function", {}
                    ).get("arguments", {})

                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except:
                            args = {}

                    yield {"type": "tool_call", "name": name, "args": args}

                    result, error = await self._execute_tool_with_permission(name, args)

                    yield {"type": "tool_result", "result": result, "error": error}

                    call_id = tool_call_data.get("id") or str(uuid.uuid4())
                    self.state.add_tool_call(
                        ToolCall(
                            id=call_id,
                            name=name,
                            arguments=args,
                            result=result,
                            error=error,
                        )
                    )
                    self.state.add_tool_result(call_id, name, result or f"Error: {error}")

            yield {"type": "done", "result": self._build_result().to_dict()}

        except Exception as e:
            yield {"type": "error", "error": str(e)}

    async def ask_user(
        self,
        question: str,
        options: list[str] | None = None,
    ) -> str:
        """Ask user for input."""
        if self.permission_callback:
            return await self.permission_callback(question, options)
        return "yes"

    async def _call_llm(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> dict[str, Any]:
        """Call the LLM with messages and tools."""
        if hasattr(self.llm, "chat"):
            response = await self.llm.chat(messages, tools=tools)
        elif hasattr(self.llm, "ainvoke"):
            response = await self.llm.ainvoke(messages, tools=tools)
        elif callable(self.llm):
            response = await self.llm(messages, tools=tools)
        else:
            raise ValueError("LLM client must have chat, ainvoke, or be callable")

        return self._parse_llm_response(response)

    def _parse_llm_response(self, response: Any) -> dict[str, Any]:
        """Parse LLM response into standard format."""
        if hasattr(response, "content"):
            content = response.content
            tool_calls = getattr(response, "tool_calls", None)

            parsed_calls = []
            if tool_calls:
                for tc in tool_calls:
                    if hasattr(tc, "function"):
                        parsed_calls.append(
                            {
                                "id": getattr(tc, "id", str(uuid.uuid4())),
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        )
                    else:
                        parsed_calls.append(
                            {
                                "id": getattr(tc, "id", str(uuid.uuid4())),
                                "name": tc.get("name"),
                                "arguments": tc.get("arguments"),
                            }
                        )

            return {
                "content": content,
                "tool_calls": parsed_calls,
                "is_finished": not tool_calls,
            }

        if isinstance(response, dict):
            return {
                "content": response.get("content"),
                "tool_calls": response.get("tool_calls", []),
                "is_finished": not response.get("tool_calls"),
            }

        return {"content": str(response), "tool_calls": [], "is_finished": True}

    def _build_messages(self, observation: Observation) -> list[dict]:
        """Build messages for LLM call."""
        messages = []

        messages.append(
            {
                "role": "system",
                "content": self._get_system_prompt(),
            }
        )

        for msg in self.state.get_recent_messages(20):
            messages.append(msg.to_api_format())

        return messages

    def _get_system_prompt(self) -> str:
        """Get system prompt based on mode."""
        if self.state.mode == "plan":
            return """You are a planning agent. Analyze the task and create a step-by-step plan.
Do not execute anything, only create a plan. Use available tools to explore the codebase.
Once you have a plan, respond with "PLAN:" followed by the steps."""

        return """You are a coding agent. Complete the given task using available tools.
Think step by step. Use tools to read files, write code, and run commands.
When the task is complete, provide a summary of what was done."""

    async def _execute_tool_with_permission(
        self,
        name: str,
        args: dict[str, Any],
    ) -> tuple[str, str | None]:
        """Execute tool with permission check for sensitive tools."""
        if self.is_sensitive_tool(name):
            allowed = await self.check_permission(name, args)
            if not allowed:
                return "", "Permission denied by user"

        return await self.execute_tool(name, args)

    async def _compress_context(self) -> None:
        """Compress context when approaching limits."""
        recent = self.state.get_recent_messages(12)

        older = self.state.messages[:-12]
        if older:
            summary = await self._summarize_messages(older)
            self.state.messages = [
                Message(role="system", content=f"[Previous context summary]\n{summary}"),
                *recent,
            ]
            self.state._update_token_estimate()

    async def _summarize_messages(self, messages: list[Message]) -> str:
        """Summarize older messages."""
        content = "\n".join(f"{m.role}: {m.content[:200]}" for m in messages if m.content)

        if len(content) < 500:
            return content

        try:
            response = await self._call_llm(
                [{"role": "user", "content": f"Summarize this conversation:\n{content}"}],
                [],
            )
            return response.get("content", content[:500])
        except:
            return content[:500]

    def _build_result(self, error: str | None = None) -> AgentResult:
        """Build the final result."""
        duration = time.time() - self._start_time if self._start_time else 0

        return AgentResult(
            success=error is None and self.state.status != "error",
            response=self.state.last_response,
            tool_calls=self.state.tool_history,
            messages=self.state.messages,
            duration=duration,
            error=error or self.state.last_error,
            metadata={
                "turn_count": self.state.turn_count,
                "tool_call_count": self.state.get_tool_call_count(),
                "status": self.state.status,
            },
        )
