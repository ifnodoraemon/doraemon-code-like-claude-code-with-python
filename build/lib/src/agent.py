import os
import json
import asyncio
from adk import agent, tool
from rich.console import Console

# 导入所有 Server 的逻辑 (In-Process for ADK Web Debugging)
# 这样 ADK Web 可以直接 inspect 到这些函数
from src.servers.memory import save_note, search_notes, update_user_persona, get_user_persona
from src.servers.vision import ocr_image, describe_image
from src.servers.web import search_internet
from src.servers.computer import execute_python
from src.servers.files import read_file, list_directory, write_file

console = Console()

# --------------------------
# 加载配置
# --------------------------
def load_config():
    # 获取当前脚本所在目录 (src/) 的上一级目录 (项目根目录)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, ".polymath", "config.json")
    
    if not os.path.exists(config_path):
        return {"persona": {}}
    with open(config_path, "r") as f:
        return json.load(f)

config = load_config()
persona = config.get("persona", {})

# --------------------------
# Agent 定义
# --------------------------
@agent.define(
    name=persona.get("name", "Polymath"),
    model=os.getenv("MODEL_NAME", "gemini-1.5-pro"),
    instructions=f"""
    You are {persona.get('name', 'Polymath')}, a {persona.get('role', 'Generalist AI Assistant')}.
    
    You have access to a suite of powerful tools:
    1. [Memory]: You can save notes and recall user preferences.
    2. [Vision]: You can see and analyze images (OCR, Description).
    3. [Web]: You can search the internet for real-time info.
    
    Always use the appropriate tool for the task. 
    If asked to write something, check your memory for style guides first.
    """
)
class PolymathAgent:
    # --------------------------
    # 注册所有工具 (Tools)
    # --------------------------
    
    # [Memory Tools]
    save_note = tool(save_note)
    search_notes = tool(search_notes)
    update_user_persona = tool(update_user_persona)
    get_user_persona = tool(get_user_persona)
    
    # [Vision Tools]
    ocr_image = tool(ocr_image)
    describe_image = tool(describe_image)
    
    # [Web Tools]
    search_internet = tool(search_internet)

    # [Computer Tools]
    execute_python = tool(execute_python)

    # [Filesystem Tools]
    read_file = tool(read_file)
    write_file = tool(write_file)
    list_directory = tool(list_directory)

if __name__ == "__main__":
    console.print("[bold green]Polymath Agent Definition Loaded for ADK Web.[/bold green]")