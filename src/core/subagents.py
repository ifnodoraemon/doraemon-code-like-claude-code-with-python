"""
Subagent System

Enables creating specialized subagents for complex tasks.

Features:
- Dynamic subagent creation with custom prompts
- Tool restrictions per subagent
- Model selection per subagent
- Parallel execution support
- Built-in agent types (code-reviewer, debugger, etc.)
"""

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class SubagentModel(Enum):
    """Available models for subagents (Gemini 3 Series - Latest)."""

    INHERIT = "inherit"  # Use parent's model
    PRO = "pro"  # gemini-3-pro - Most capable, multimodal
    FLASH = "flash"  # gemini-3-flash - Fast and capable (default)


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
        }


# ========================================
# Built-in Agent Configurations
# ========================================

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
        tools=["read_file", "read_file_outline", "list_directory", "glob_files", "grep_search"],
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
        tools=[
            "read_file",
            "grep_search",
            "glob_files",
            "execute_python",
            "shell_execute",
        ],
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
        tools=["read_file", "write_file", "edit_file", "grep_search", "execute_python"],
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
        tools=["read_file", "write_file", "edit_file", "list_directory"],
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
        tools=["read_file", "grep_search", "glob_files", "shell_execute"],
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
        tools=["read_file", "write_file", "edit_file", "grep_search"],
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
        tools=[
            "read_file",
            "read_file_outline",
            "list_directory",
            "glob_files",
            "grep_search",
            "find_symbol",
        ],
        model=SubagentModel.FLASH,
        max_turns=10,
    ),
}


class SubagentManager:
    """
    Manages subagent creation and execution.

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
    """

    def __init__(
        self,
        client: genai.Client,
        tool_registry: Any,  # ToolRegistry
        parent_model: str = "gemini-2.0-flash",
    ):
        """
        Initialize subagent manager.

        Args:
            client: GenAI client
            tool_registry: Tool registry for getting tools
            parent_model: Parent model name (used for INHERIT)
        """
        self.client = client
        self.tool_registry = tool_registry
        self.parent_model = parent_model
        self._custom_agents: dict[str, SubagentConfig] = {}
        self._running_agents: dict[str, asyncio.Task] = {}

    def register_agent(self, config: SubagentConfig):
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
        agent_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        logger.info(f"Spawning subagent {agent_id} ({config.name}): {task[:100]}")

        try:
            # Determine model
            # Model selection with latest models
            # Model mapping - Gemini 3 Series (Latest)
            model_mapping = {
                SubagentModel.INHERIT: self.parent_model,
                SubagentModel.PRO: "gemini-3-pro-preview",
                SubagentModel.FLASH: "gemini-3-flash-preview",
            }
            model_name = model_mapping.get(config.model, self.parent_model)

            # Get tools
            if config.tools:
                tools = self.tool_registry.get_genai_tools(config.tools)
            else:
                tools = self.tool_registry.get_genai_tools()

            # Build system prompt
            system_prompt = config.prompt
            if context:
                system_prompt += f"\n\n[Context]\n{context}"

            # Create chat config
            gen_config = types.GenerateContentConfig(
                tools=[types.Tool(function_declarations=tools)] if tools else None,
                system_instruction=system_prompt,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            )

            # Create chat session
            chat = self.client.chats.create(model=model_name, config=gen_config)

            # Run agent loop
            output_parts = []
            turns_used = 0
            total_tokens = 0

            # Initial message
            response = await asyncio.wait_for(
                asyncio.to_thread(chat.send_message, task),
                timeout=config.timeout,
            )
            turns_used += 1

            while turns_used < config.max_turns:
                if not response.candidates:
                    break

                parts = response.candidates[0].content.parts
                has_tool_call = False
                tool_results = []

                for part in parts:
                    # Collect text output
                    if part.text:
                        output_parts.append(part.text)
                        if on_output:
                            on_output(part.text)

                    # Handle tool calls
                    if part.function_call:
                        has_tool_call = True
                        fc = part.function_call
                        tool_name = fc.name
                        args = dict(fc.args.items()) if hasattr(fc.args, "items") else {}

                        # Execute tool
                        try:
                            result = await self.tool_registry.call_tool(tool_name, args)
                        except Exception as e:
                            result = f"Error: {e}"

                        tool_results.append(
                            {"name": tool_name, "result": {"result": result}}
                        )

                # Track tokens
                if response.usage_metadata:
                    total_tokens += response.usage_metadata.prompt_token_count or 0
                    total_tokens += response.usage_metadata.candidates_token_count or 0

                # No more tool calls - done
                if not has_tool_call:
                    break

                # Send tool results
                response_parts = [
                    types.Part.from_function_response(
                        name=tr["name"], response=tr["result"]
                    )
                    for tr in tool_results
                ]

                response = await asyncio.wait_for(
                    asyncio.to_thread(chat.send_message, response_parts),
                    timeout=config.timeout,
                )
                turns_used += 1

            duration = time.time() - start_time
            output = "\n".join(output_parts)

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
            )

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            return SubagentResult(
                agent_id=agent_id,
                agent_name=config.name,
                success=False,
                output="",
                turns_used=0,
                tokens_used=0,
                duration=duration,
                error=f"Timeout after {config.timeout}s",
            )

        except Exception as e:
            duration = time.time() - start_time
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
            )

    async def spawn_parallel(
        self,
        tasks: list[tuple[str, str]],  # (agent_name, task)
        context: str = "",
    ) -> list[SubagentResult]:
        """
        Run multiple agents in parallel.

        Args:
            tasks: List of (agent_name, task) tuples
            context: Shared context for all agents

        Returns:
            List of results in same order as tasks
        """
        coroutines = [self.spawn(name, task, context) for name, task in tasks]

        results = await asyncio.gather(*coroutines, return_exceptions=True)

        # Convert exceptions to failed results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                agent_name = tasks[i][0]
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
                    )
                )
            else:
                final_results.append(result)

        return final_results
