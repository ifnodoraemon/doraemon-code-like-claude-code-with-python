"""
Polymath CLI - Main Command Line Interface

Provides the primary interactive interface for Polymath AI agent.
Supports multiple modes, slash commands, and tool execution with HITL approval.
"""

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import typer
from dotenv import load_dotenv

# 强制设置标准流编码，防止 UnicodeDecodeError
if sys.stdin and hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Google GenAI (New SDK)
from google import genai
from google.genai import types
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

# Core Imports
from src.core.config import load_config
from src.core.diff import print_diff

# DI Container and Services
from src.core.events import MessageEvents, SessionEvents, publish
from src.core.prompts import get_system_prompt
from src.core.rules import (
    format_instructions_for_prompt,
    load_all_instructions,
)
from src.core.services import initialize_services, resolve
from src.core.telemetry import StructuredLogger, Tracer

# 导入通用 MCP 客户端
from src.host.client import MultiServerMCPClient

app = typer.Typer()

# 加载 .env 环境变量
load_dotenv()

# Initialize services at module load (lazy initialization)
_services_initialized = False


def _ensure_services():
    """Ensure services are initialized."""
    global _services_initialized
    if not _services_initialized:
        initialize_services()
        _services_initialized = True


def get_console() -> Console:
    """Get the console instance from DI container."""
    _ensure_services()
    console = resolve(Console)
    return console if console else Console()


def get_logger() -> StructuredLogger | None:
    """Get the structured logger from DI container."""
    _ensure_services()
    return resolve(StructuredLogger)


def get_tracer() -> Tracer | None:
    """Get the tracer from DI container."""
    _ensure_services()
    return resolve(Tracer)


# For backward compatibility
console = Console()  # Will be replaced by DI when services are initialized


@dataclass
class SessionState:
    """
    Encapsulates the session state for the chat loop.

    This replaces global variables with a proper state container,
    making the code more testable and maintainable.
    """

    mode: str = "default"
    project: str = "default"
    turn_count: int = 0
    chat_session: Any = None
    tools: list = field(default_factory=list)
    config: dict = field(default_factory=dict)

    # Mode colors for display
    MODE_COLORS: dict = field(
        default_factory=lambda: {
            "default": "green",
            "plan": "blue",
            "build": "green",
            "coder": "cyan",
            "architect": "magenta",
        }
    )

    def get_mode_color(self) -> str:
        """Get the display color for current mode."""
        return self.MODE_COLORS.get(self.mode, "yellow")

    def increment_turn(self) -> int:
        """Increment and return turn count."""
        self.turn_count += 1
        return self.turn_count


# Global session state (for backward compatibility with cli_commands)
_session_state: SessionState | None = None


def get_session_state() -> SessionState:
    """Get the current session state."""
    global _session_state
    if _session_state is None:
        _session_state = SessionState()
    return _session_state


def init_chat_model(client: genai.Client, mode: str, tools: list, history: list | None = None):
    """Initialize the Gemini chat with a specific mode (system prompt)."""
    config = load_config()
    persona = config.get("persona", {})

    # Get prompt for mode
    sys_instruction = get_system_prompt(mode, persona)

    # Load Project Rules (AGENTS.md)
    instructions = load_all_instructions(config)
    if instructions:
        sys_instruction += format_instructions_for_prompt(instructions)

    # Load Project Memory if exists (legacy support for POLYMATH.md)
    if os.path.exists("POLYMATH.md"):
        try:
            with open("POLYMATH.md") as f:
                memory_content = f.read()
            sys_instruction += f"\n\n=== PROJECT MEMORY (POLYMATH.md - DEPRECATED) ===\n{memory_content}\n=== Use AGENTS.md instead ===\n"
        except Exception:
            pass

    # Create Chat with New SDK
    # tools is a list of types.FunctionDeclaration
    # We need to wrap them in types.Tool
    tool_obj = types.Tool(function_declarations=tools)

    # GenerateContentConfig
    gen_config = types.GenerateContentConfig(
        tools=[tool_obj],
        system_instruction=sys_instruction,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    # Check if history needs processing (New SDK expects list of Content objects or dicts)
    # We assume 'history' passed here is compatible or empty
    # Model can be configured via environment variable
    if history is None:
        history = []
    model_name = os.getenv("POLYMATH_MODEL", os.getenv("MODEL_NAME", "gemini-2.0-flash"))
    chat = client.chats.create(model=model_name, config=gen_config, history=history)
    return chat


async def handle_slash_command(
    command: str,
    chat_session_ref: dict,
    mcp_client: MultiServerMCPClient,
    active_tools: list,
    client: genai.Client,
    session: SessionState,
) -> bool:
    """
    Handle slash commands using the new command dispatch system.

    Args:
        command: The slash command string
        chat_session_ref: Dict containing the chat session (for modification)
        mcp_client: MCP client instance
        active_tools: List of available tools
        client: GenAI client
        session: Session state object

    Returns:
        True to continue, "EXIT" to quit, False on error
    """
    # Import the command dispatcher
    from src.host.cli_commands import dispatch_command

    # Build context for command execution
    context = {
        "chat": chat_session_ref,
        "client": client,
        "mode": session.mode,
        "tools": active_tools,
        "mcp_client": mcp_client,
        "mcp_servers": list(mcp_client.sessions.keys()) if mcp_client else [],
    }

    # Dispatch to command handler
    result = await dispatch_command(command, context)

    # Update session mode if it changed
    if context["mode"] != session.mode:
        session.mode = context["mode"]

    return result


async def chat_loop(project: str = "default"):
    """
    Main chat loop for interactive session.

    Args:
        project: Project name for memory isolation
    """
    global _session_state

    # Initialize services (DI container)
    _ensure_services()
    output = get_console()
    logger = get_logger()
    _tracer = get_tracer()  # Reserved for future distributed tracing

    # Log session start
    if logger:
        logger.info("Session starting", project=project)

    # Publish session started event
    publish(SessionEvents.STARTED, project=project)

    # Initialize session state
    session = SessionState(project=project)
    _session_state = session

    # 1. 环境检查
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        if logger:
            logger.error("Missing API key", key="GOOGLE_API_KEY")
        output.print("[red]错误: 未设置 GOOGLE_API_KEY[/red]")
        return

    # Initialize New Client
    try:
        client = genai.Client(api_key=api_key)
        if logger:
            logger.debug("GenAI client initialized")
    except Exception as e:
        if logger:
            logger.error("Failed to initialize GenAI client", exception=e)
        output.print(f"[red]Failed to initialize Google GenAI Client: {e}[/red]")
        return

    output.print(f"[bold yellow]Project: {project}[/bold yellow]")

    # 2. 初始化 MCP 客户端
    config = load_config()
    session.config = config
    mcp_client = MultiServerMCPClient()

    if logger:
        logger.debug("Connecting to MCP servers")

    await mcp_client.connect_to_config(config)

    # Show connection errors if any
    if mcp_client.has_errors():
        for err in mcp_client.get_connection_errors():
            if logger:
                logger.warning("MCP connection error", error=str(err))
            output.print(f"[yellow]Warning: {err}[/yellow]")

    if logger:
        logger.info(
            "MCP servers connected",
            servers=list(mcp_client.sessions.keys()),
            tool_count=len(mcp_client.tool_map),
        )

    # 敏感工具列表
    sensitive_tools = config.get(
        "sensitive_tools", ["execute_python", "write_file", "save_note", "move_file", "delete_file"]
    )

    try:
        # 3. 准备工具
        genai_tools = await mcp_client.get_genai_tools()
        active_tools = genai_tools
        session.tools = active_tools

        # 4. 初始化模型 (Initial Chat)
        chat_ref = {"session": init_chat_model(client, session.mode, active_tools, [])}
        session.chat_session = chat_ref["session"]

        output.print(
            Panel.fit(
                f"[bold blue]Polymath v0.4.0 (Multi-Mode + DI)[/bold blue]\n"
                f"[dim]Servers: {', '.join(mcp_client.sessions.keys())}[/dim]\n"
                f"[dim]Type /help for commands. Current Mode: {session.mode}[/dim]",
                border_style="blue",
            )
        )

        # 5. 聊天主循环
        while True:
            # Show current mode in prompt
            mode_color = session.get_mode_color()
            user_input = Prompt.ask(
                f"\n[bold {mode_color}]You ({session.mode})[/bold {mode_color}]"
            )

            if user_input.lower() in ["exit", "quit"]:
                break

            # 处理 Slash Commands
            if user_input.startswith("/"):
                result = await handle_slash_command(
                    user_input, chat_ref, mcp_client, active_tools, client, session
                )
                if result == "EXIT":
                    break
                if result:
                    continue

            chat = chat_ref["session"]

            # 发送消息
            response = None
            try:
                with output.status(
                    f"[bold {mode_color}]Thinking ({session.mode})...[/bold {mode_color}]",
                    spinner="dots",
                ):
                    response = chat.send_message(user_input)
            except Exception as e:
                output.print(f"[red]API Error: {e}[/red]")
                continue

            # 处理多轮对话 (Tool Loop)
            while True:
                if not response.candidates:
                    output.print("[red]Error: Empty response from model.[/red]")
                    break

                # New SDK: response.candidates[0].content.parts
                parts = response.candidates[0].content.parts
                has_tool_call = False
                tool_results = []

                for part in parts:
                    # 1. 处理思考文本 (Text)
                    if part.text:
                        output.print(
                            Panel(
                                Markdown(part.text),
                                title="[bold purple]Thought[/bold purple]",
                                border_style="purple",
                                expand=False,
                            )
                        )

                    # 2. 处理工具调用 (Function Call)
                    if part.function_call:
                        has_tool_call = True
                        fc = part.function_call
                        tool_name = fc.name
                        args = fc.args

                        # Convert args to dict safely
                        if hasattr(args, "items"):
                            args_dict = dict(args.items())
                        else:
                            try:
                                args_dict = dict(args)
                            except (TypeError, ValueError):
                                args_dict = {}

                        # 注入当前项目上下文
                        if tool_name in ["save_note", "search_notes"]:
                            args_dict["collection_name"] = session.project

                        # --- Transparency: Diff View ---
                        if (
                            tool_name == "write_file"
                            and "content" in args_dict
                            and "path" in args_dict
                        ):
                            output.print(
                                f"\n[bold yellow]📝 Proposing changes to:[/bold yellow] {args_dict['path']}"
                            )
                            print_diff(args_dict["path"], args_dict["content"])

                        # --- Security: Approval ---
                        tool_result = None
                        if tool_name in sensitive_tools:
                            output.print(f"\n[bold red]⚠️  Sensitive Action:[/bold red] {tool_name}")
                            if tool_name != "write_file":
                                output.print(
                                    f"[dim]Args: {json.dumps(args_dict, indent=2, ensure_ascii=False)}[/dim]"
                                )

                            confirm = Prompt.ask("Execute?", choices=["y", "n"], default="n")
                            if confirm.lower() != "y":
                                tool_result = "User denied the operation."
                                if logger:
                                    logger.info("Tool call denied by user", tool=tool_name)
                                publish(
                                    MessageEvents.TOOL_CALL,
                                    tool_name=tool_name,
                                    status="denied",
                                )
                                output.print("[red]Cancelled.[/red]")
                            else:
                                if logger:
                                    logger.info("Executing sensitive tool", tool=tool_name)
                                publish(
                                    MessageEvents.TOOL_CALL,
                                    tool_name=tool_name,
                                    status="executing",
                                    sensitive=True,
                                )
                                output.print(f"[cyan]Running {tool_name}...[/cyan]")
                                tool_result = await mcp_client.call_tool(tool_name, args_dict)
                                publish(
                                    MessageEvents.TOOL_RESULT,
                                    tool_name=tool_name,
                                    success=True,
                                )
                        else:
                            if logger:
                                logger.debug("Executing tool", tool=tool_name)
                            publish(
                                MessageEvents.TOOL_CALL,
                                tool_name=tool_name,
                                status="executing",
                            )
                            output.print(f"[cyan]Running {tool_name}...[/cyan]")
                            tool_result = await mcp_client.call_tool(tool_name, args_dict)
                            publish(
                                MessageEvents.TOOL_RESULT,
                                tool_name=tool_name,
                                success=True,
                            )

                        tool_results.append({"name": tool_name, "result": {"result": tool_result}})

                if not has_tool_call:
                    turn = session.increment_turn()
                    usage = response.usage_metadata
                    if usage:
                        output.print(
                            f"\n[dim italic]Turn {turn} | Input: {usage.prompt_token_count} | Output: {usage.candidates_token_count}[/dim italic]"
                        )
                    break

                # 发回结果 (New SDK Style)
                response_parts = []
                for tr in tool_results:
                    response_parts.append(
                        types.Part.from_function_response(name=tr["name"], response=tr["result"])
                    )

                with output.status("[bold cyan]Processing Results...[/bold cyan]", spinner="dots"):
                    response = chat.send_message(response_parts)

    finally:
        if logger:
            logger.info(
                "Session ending",
                project=project,
                turns=session.turn_count,
                mode=session.mode,
            )
        # Publish session ended event
        publish(
            SessionEvents.ENDED,
            project=project,
            turns=session.turn_count,
            mode=session.mode,
        )
        await mcp_client.cleanup()


@app.command()
def setup():
    """初始化环境并安装依赖"""
    import subprocess

    console.print("[bold cyan]正在启动自动设置流程...[/bold cyan]")
    try:
        # 运行 shell 脚本
        subprocess.run(["bash", "scripts/setup.sh"], check=True)
    except Exception as e:
        console.print(f"[bold red]设置失败: {e}[/bold red]")


@app.command()
def tui(project: str = "default"):
    """启动 Polymath TUI (Textual界面)"""
    try:
        from .tui import run_tui

        console.print("[cyan]Launching Polymath TUI...[/cyan]")
        run_tui()
    except ImportError as e:
        console.print("[red]Error: Textual not installed. Run: pip install textual[/red]")
        console.print(f"[dim]{e}[/dim]")
    except Exception as e:
        console.print(f"[red]TUI Error: {e}[/red]")


@app.command()
def start(project: str = "default"):
    """启动 Polymath CLI（命令行界面）"""
    # 依赖检查 - use find_spec to avoid unused import warnings
    import importlib.util

    missing_deps = []
    if importlib.util.find_spec("google.genai") is None:
        missing_deps.append("google-genai")
    if importlib.util.find_spec("mcp") is None:
        missing_deps.append("mcp")

    if missing_deps:
        console.print(
            f"[yellow]警告: 核心依赖未安装: {', '.join(missing_deps)}。"
            f"请运行 'pip install {' '.join(missing_deps)}' 或 'pl setup'。[/yellow]"
        )
        if not Prompt.ask("是否继续启动？", choices=["y", "n"], default="n") == "y":
            return

    asyncio.run(chat_loop(project=project))


def entry_point():
    app()


if __name__ == "__main__":
    app()
