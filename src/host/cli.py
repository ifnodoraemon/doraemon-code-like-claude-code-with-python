import os
import sys
import asyncio
import json
import typer
from dotenv import load_dotenv

# 强制设置标准流编码，防止 UnicodeDecodeError
if sys.stdin and hasattr(sys.stdin, 'reconfigure'):
    sys.stdin.reconfigure(encoding='utf-8', errors='replace')
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.panel import Panel
from rich.table import Table

# Google GenAI (New SDK)
from google import genai
from google.genai import types

# 导入通用 MCP 客户端
from src.host.client import MultiServerMCPClient
# Core Imports
from src.core.config import load_config
from src.core.diff import print_diff
from src.core.prompts import get_system_prompt, PROMPTS
from src.core.rules import load_all_instructions, format_instructions_for_prompt, create_default_agents_md

app = typer.Typer()
console = Console()

# 加载 .env 环境变量
load_dotenv()

# Global State for Mode
CURRENT_MODE = "default"

def init_chat_model(client: genai.Client, mode: str, tools: list, history: list = []):
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
            with open("POLYMATH.md", "r") as f:
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
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
    )
    
    # Check if history needs processing (New SDK expects list of Content objects or dicts)
    # We assume 'history' passed here is compatible or empty
    chat = client.chats.create(model="gemini-3-pro-preview", config=gen_config, history=history)
    return chat

async def handle_slash_command(command: str, chat_session_ref: dict, mcp_client, active_tools, client) -> bool:
    """
    Handle slash commands using the new command dispatch system.
    chat_session_ref is a dict {"session": chat} to allow modification.
    """
    global CURRENT_MODE
    
    # Import the command dispatcher
    from src.host.cli_commands import dispatch_command
    
    # Build context for command execution
    context = {
        'chat': chat_session_ref,
        'client': client,
        'mode': CURRENT_MODE,
        'tools': active_tools,
        'mcp_client': mcp_client,
        'mcp_servers': list(mcp_client.sessions.keys()) if mcp_client else []
    }
    
    # Dispatch to command handler
    result = await dispatch_command(command, context)
    
    # Update global mode if it changed
    if context['mode'] != CURRENT_MODE:
        CURRENT_MODE = context['mode']
    
    return result


async def chat_loop(project: str = "default"):
    global CURRENT_MODE
    # 1. 环境检查
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        console.print("[red]错误: 未设置 GOOGLE_API_KEY[/red]")
        return
    
    # Initialize New Client
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        console.print(f"[red]Failed to initialize Google GenAI Client: {e}[/red]")
        return

    console.print(f"[bold yellow]Project: {project}[/bold yellow]")

    # 2. 初始化 MCP 客户端
    config = load_config()
    mcp_client = MultiServerMCPClient()
    await mcp_client.connect_to_config(config)

    # 敏感工具列表
    SENSITIVE_TOOLS = config.get("sensitive_tools", ["execute_python", "write_file", "save_note", "move_file", "delete_file"])

    try:
        # 3. 准备工具
        genai_tools = await mcp_client.get_genai_tools() # Returns list[types.FunctionDeclaration]
        active_tools = genai_tools

        # 4. 初始化模型 (Initial Chat)
        chat_ref = {
            "session": init_chat_model(client, CURRENT_MODE, active_tools, [])
        }

        console.print(Panel.fit(
            f"[bold blue]Polymath v0.4 (Multi-Mode + New SDK)[/bold blue]\n"
            f"[dim]Servers: {', '.join(mcp_client.sessions.keys())}[/dim]\n"
            f"[dim]Type /help for commands. Current Mode: {CURRENT_MODE}[/dim]",
            border_style="blue"
        ))

        # 5. 聊天主循环
        turn_count = 0
        while True:
            # Show current mode in prompt
            mode_color = "green" if CURRENT_MODE == "default" else ("blue" if CURRENT_MODE == "coder" else "magenta")
            user_input = Prompt.ask(f"\n[bold {mode_color}]You ({CURRENT_MODE})[/bold {mode_color}]")
            
            if user_input.lower() in ["exit", "quit"]:
                break
            
            # 处理 Slash Commands
            if user_input.startswith("/"):
                if await handle_slash_command(user_input, chat_ref, mcp_client, active_tools, client):
                    continue

            chat = chat_ref["session"]

            # 发送消息
            response = None
            try:
                with console.status(f"[bold {mode_color}]Thinking ({CURRENT_MODE})...[/bold {mode_color}]", spinner="dots"):
                    response = chat.send_message(user_input)
            except Exception as e:
                console.print(f"[red]API Error: {e}[/red]")
                continue
            
            # 处理多轮对话 (Tool Loop)
            while True:
                if not response.candidates:
                        console.print("[red]Error: Empty response from model.[/red]")
                        break

                # New SDK: response.candidates[0].content.parts
                parts = response.candidates[0].content.parts
                has_tool_call = False
                tool_results = []
                
                for part in parts:
                    # 1. 处理思考文本 (Text)
                    if part.text:
                        console.print(Panel(Markdown(part.text), title="[bold purple]Thought[/bold purple]", border_style="purple", expand=False))
                        
                    # 2. 处理工具调用 (Function Call)
                    if part.function_call:
                        has_tool_call = True
                        fc = part.function_call
                        tool_name = fc.name
                        args = fc.args # New SDK args is usually a dict or object
                        
                        # Convert args to dict safely
                        if hasattr(args, "items"): 
                            args_dict = {k: v for k, v in args.items()}
                        else:
                            try:
                                args_dict = dict(args)
                            except:
                                args_dict = {}

                        # 注入当前项目上下文
                        if tool_name in ["save_note", "search_notes"]:
                            args_dict["collection_name"] = project

                        # --- Transparency: Diff View ---
                        if tool_name == "write_file" and "content" in args_dict and "path" in args_dict:
                            console.print(f"\n[bold yellow]📝 Proposing changes to:[/bold yellow] {args_dict['path']}")
                            print_diff(args_dict['path'], args_dict['content'])

                        # --- Security: Approval ---
                        tool_result = None
                        if tool_name in SENSITIVE_TOOLS:
                            console.print(f"\n[bold red]⚠️  Sensitive Action:[/bold red] {tool_name}")
                            if tool_name != "write_file":
                                console.print(f"[dim]Args: {json.dumps(args_dict, indent=2, ensure_ascii=False)}[/dim]")
                            
                            confirm = Prompt.ask("Execute?", choices=["y", "n"], default="n")
                            if confirm.lower() != "y":
                                tool_result = "User denied the operation."
                                console.print("[red]Cancelled.[/red]")
                            else:
                                console.print(f"[cyan]Running {tool_name}...[/cyan]")
                                tool_result = await mcp_client.call_tool(tool_name, args_dict)
                        else:
                            console.print(f"[cyan]Running {tool_name}...[/cyan]")
                            tool_result = await mcp_client.call_tool(tool_name, args_dict)
                        
                        tool_results.append({
                            "name": tool_name,
                            "result": {"result": tool_result} 
                        })

                if not has_tool_call:
                    turn_count += 1
                    usage = response.usage_metadata
                    if usage:
                        console.print(f"\n[dim italic]Turn {turn_count} | Input: {usage.prompt_token_count} | Output: {usage.candidates_token_count}[/dim italic]")
                    break
                
                # 发回结果 (New SDK Style)
                # Use types.Part.from_function_response
                response_parts = []
                for tr in tool_results:
                    response_parts.append(types.Part.from_function_response(
                        name=tr["name"],
                        response=tr["result"]
                    ))
                    
                with console.status("[bold cyan]Processing Results...[/bold cyan]", spinner="dots"):
                    response = chat.send_message(response_parts)

    finally:
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
        console.print(f"[red]Error: Textual not installed. Run: pip install textual[/red]")
        console.print(f"[dim]{e}[/dim]")
    except Exception as e:
        console.print(f"[red]TUI Error: {e}[/red]")

@app.command()
def start(project: str = "default"):
    """启动 Polymath CLI（命令行界面）"""
    # 简单的依赖检查
    try:
        import mcp
        import google.genai
    except ImportError:
        console.print("[yellow]警告: 核心依赖似乎未安装。请运行 'pip install google-genai' 或 'pl setup'。[/yellow]")
        if not Prompt.ask("是否继续启动？", choices=["y", "n"], default="n") == "y":
            return
            
    asyncio.run(chat_loop(project=project))

def entry_point():
    app()

if __name__ == "__main__":
    app()