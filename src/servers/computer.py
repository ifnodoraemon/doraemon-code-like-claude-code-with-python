import os
import sys
import subprocess
import tempfile
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("PolymathComputer")

@mcp.tool()
def execute_python(code: str) -> str:
    """
    Execute arbitrary Python code and return stdout/stderr.
    Useful for calculations, data analysis, or generating charts.
    
    The code runs in a temporary environment. 
    Images generated (e.g. via matplotlib) should be saved to 'drafts/'.
    """
    print(f"Executing Code:\n{code}")
    
    # 创建临时文件来存放代码
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        script_path = f.name

    try:
        # 使用当前的 Python 环境执行，这样可以使用已安装的库 (pandas, matplotlib等)
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=30  # 防止死循环
        )
        
        output = result.stdout
        if result.stderr:
            output += f"\n[Stderr]:\n{result.stderr}"
            
        return output if output.strip() else "Code executed successfully (no output)."
        
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out (30s limit)."
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        # 清理临时文件
        if os.path.exists(script_path):
            os.remove(script_path)

if __name__ == "__main__":
    mcp.run()
