import os
import sys
import asyncio
import json
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

# 加载 .env 环境变量
load_dotenv()

# Google GenAI
import google.generativeai as genai

# 导入通用 MCP 客户端
from src.host.client import MultiServerMCPClient

app = typer.Typer()
console = Console()

def load_config():
    config_path = ".polymath/config.json"
    if not os.path.exists(config_path):
        return {"mcpServers": {}}
    with open(config_path, "r") as f:
        return json.load(f)

async def chat_loop():
    # 1. 环境检查
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        console.print("[red]错误: 未设置 GOOGLE_API_KEY[/red]")
        return
    genai.configure(api_key=api_key)

    # 2. 初始化 MCP 客户端
    config = load_config()
    mcp_client = MultiServerMCPClient()
    await mcp_client.connect_to_config(config)

    try:
        # 3. 获取所有 Server 的工具并转换为 Gemini 格式
        # 注意：目前 Google SDK 对于动态 FunctionDeclaration 的支持比较繁琐
        # 为了演示稳定性，我们这里使用一种 "Tool Wrapping" 技巧
        # 实际上 Gemini SDK 需要你传入可调用的 Python 函数。
        # 所以我们动态生成 Python wrapper 函数。
        
        mcp_genai_tools = []
        
        # 这是一个黑魔法：我们需要动态生成 Python 函数，以便 Gemini SDK 可以 inspect 它们
        # 但既然我们实现了自己的 Tool Call Loop，我们可以只把声明传给模型，自己执行。
        # 遗憾的是，genai.GenerativeModel(tools=...) 强制验证 callable。
        # 让我们换一种方式：使用 chat.send_message 时的 tool_config (如果支持)
        # 或者，我们构建一个包含所有工具声明的 Tool 对象。
        
        # 简化策略：我们创建一个 "Universal Tool" 列表，其中每个函数都是一个 stub
        # 实际上，对于 Python SDK，最简单的方法是定义一个 Map。
        
        active_tools = []
        tool_function_map = {} # name -> callable stub

        for name, session in mcp_client.sessions.items():
            result = await session.list_tools()
            for tool in result.tools:
                # 创建闭包来捕获 tool_name
                async def dynamic_tool_func(**kwargs):
                    # 这个函数实际上不会被 SDK 自动调用，因为它是 async 的
                    # 我们会在 loop 中手动调用 mcp_client.call_tool
                    pass
                
                # 设置元数据以欺骗 SDK (如果需要)
                dynamic_tool_func.__name__ = tool.name
                dynamic_tool_func.__doc__ = tool.description
                
                # Gemini SDK 目前对动态工具支持一般。
                # 在 Phase 4，我们采用 "手动函数调用循环" (Manual Function Calling Loop)
                # 这比让 SDK 自动调用更稳健，尤其是在涉及 async MCP 时。
                
                # 构建给 Gemini 看的声明
                tool_decl = {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
                active_tools.append(tool_decl)

        # 4. 初始化模型 (System Prompt)
        persona = config.get("persona", {})
        sys_prompt = f"""You are {persona.get('name', 'Polymath')}. {persona.get('role', 'Assistant')}.
        You are an intelligent Polymath, capable of handling diverse tasks using your tools.
        You communicate via the Model Context Protocol (MCP).
        Always use tools when you need to read memory, see images, or search data.
        """
        
        # 注意：这里我们使用底层的 model.generate_content (REST 风格) 或者 tool_config
        # 只要我们传入 tools 列表，Gemini 就会返回 FunctionCall part。
        
        model = genai.GenerativeModel("gemini-1.5-pro", tools=[genai.protos.Tool(function_declarations=active_tools)])
        chat = model.start_chat(history=[], enable_automatic_function_calling=False) # 关闭自动调用，我们要手动接管

        console.print("[bold blue]Polymath v0.2 (MCP Architecture)[/bold blue]")
        console.print(f"[dim]Connected Servers: {', '.join(mcp_client.sessions.keys())}[/dim]")

        # 5. 聊天主循环
        while True:
            user_input = Prompt.ask("\n[bold green]You[/bold green]")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            # 发送消息
            with console.status("Thinking...", spinner="dots"):
                response = chat.send_message(user_input)
                
                # 处理可能的工具调用 (多轮)
                while True:
                    part = response.candidates[0].content.parts[0]
                    
                    # 检查是否有函数调用
                    if part.function_call:
                        fc = part.function_call
                        tool_name = fc.name
                        args = dict(fc.args)
                        
                        # 注入当前项目上下文 (如果是 memory 相关工具)
                        if tool_name in ["save_note", "search_notes"]:
                            args["collection_name"] = project

                        # --- 安全审批流 ---
                        if tool_name in SENSITIVE_TOOLS:
                            console.print(f"\n[bold red]⚠️  AI 请求执行敏感操作:[/bold red]")
                            console.print(f"[yellow]Tool:[/yellow] {tool_name}")
                            console.print(f"[yellow]Args:[/yellow] {json.dumps(args, indent=2, ensure_ascii=False)}")
                            
                            confirm = Prompt.ask("允许执行吗？", choices=["y", "n"], default="n")
                            if confirm.lower() != "y":
                                tool_result = "User denied the operation."
                                console.print("[red]已拒绝。[/red]")
                            else:
                                tool_result = await mcp_client.call_tool(tool_name, args)
                        else:
                            # 非敏感工具直接执行
                            console.print(f"[cyan]Calling Tool: {tool_name}...[/cyan]")
                            tool_result = await mcp_client.call_tool(tool_name, args)
                        
                        # --- 结束安全审批流 ---

                        console.print(f"[dim]Result: {str(tool_result)[:100]}...[/dim]")
                            
                            # 把结果喂回给模型
                            response = chat.send_message(
                                genai.protos.Content(
                                    parts=[genai.protos.Part(
                                        function_response=genai.protos.FunctionResponse(
                                            name=tool_name,
                                            response={"result": tool_result}
                                        )
                                    )]
                                )
                            )
                        except Exception as e:
                            console.print(f"[red]Tool Error: {e}[/red]")
                            # 将错误也喂回去，让 AI 知道失败了
                            response = chat.send_message(
                                genai.protos.Content(
                                    parts=[genai.protos.Part(
                                        function_response=genai.protos.FunctionResponse(
                                            name=tool_name,
                                            response={"error": str(e)}
                                        )
                                    )]
                                )
                            )
                    else:
                        # 没有函数调用，说明是纯文本回复，打印并退出内层循环
                        console.print(f"\n[bold purple]Polymath:[/bold purple]\n{response.text}")
                        break

    finally:
        await mcp_client.cleanup()

@app.command()
def start(project: str = "default"):
    """启动 Polymath 并进入特定项目环境"""
    asyncio.run(chat_loop(project=project))

def entry_point():
    app()

if __name__ == "__main__":
    app()