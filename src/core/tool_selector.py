"""
Tool Selector - 按模式分配工具

设计原则（参考 OpenCode/Claude Code）：
- 工具精简，不要太细
- 按模式（plan/build）控制权限，而不是动态搜索
- 支持 MCP 扩展外部工具

工具分类：
- 核心工具：所有模式都可用的基础能力
- 写入工具：只有 build 模式可用
- MCP 工具：通过 MCP 协议扩展的外部工具
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ========================================
# 工具定义
# ========================================

# 只读工具 - plan 和 build 模式都可用
READ_TOOLS = [
    "read_file",       # 读文件
    "list_directory",  # 目录结构
    "glob_files",      # 查找文件
    "grep_search",     # 搜索内容
]

# 写入工具 - 只有 build 模式可用
WRITE_TOOLS = [
    "write_file",      # 写文件
    "edit_file",       # 编辑文件
    "shell_execute",   # 执行命令（包含 git、npm、pip 等）
]

# 辅助工具 - 按需使用
AUX_TOOLS = [
    "web_search",      # 网络搜索
    "fetch_url",       # 获取网页
    "save_note",       # 保存笔记
    "search_notes",    # 搜索笔记
    "switch_mode",     # 切换模式 (plan/build)
]

# 高级工具 - 特殊场景
ADVANCED_TOOLS = [
    "read_file_outline",   # 文件大纲（大文件时有用）
    "find_symbol",         # 语义搜索定义
    "execute_python",      # 执行 Python 代码
    "shell_background",    # 后台执行
]


@dataclass
class ToolConfig:
    """工具配置"""
    # 按模式的工具列表
    plan_tools: list[str] = field(default_factory=list)
    build_tools: list[str] = field(default_factory=list)
    # MCP 扩展工具
    mcp_tools: list[str] = field(default_factory=list)


class ToolSelector:
    """
    简化的工具选择器
    
    按模式分配工具，不做动态搜索
    """

    def __init__(self):
        # plan 模式：只读 + 辅助
        self.plan_tools = READ_TOOLS + AUX_TOOLS

        # build 模式：全部工具
        self.build_tools = READ_TOOLS + WRITE_TOOLS + AUX_TOOLS + ADVANCED_TOOLS

        # MCP 扩展工具（运行时加载）
        self.mcp_tools: list[str] = []

    def get_tools_for_mode(self, mode: str) -> list[str]:
        """
        根据模式获取可用工具
        
        Args:
            mode: "plan" 或 "build"
            
        Returns:
            工具名称列表
        """
        if mode == "plan":
            tools = self.plan_tools.copy()
        else:  # build
            tools = self.build_tools.copy()

        # 添加 MCP 扩展工具
        tools.extend(self.mcp_tools)

        return tools

    def register_mcp_tools(self, tool_names: list[str]) -> None:
        """
        注册 MCP 扩展工具
        
        Args:
            tool_names: MCP 工具名称列表
        """
        for name in tool_names:
            if name not in self.mcp_tools:
                self.mcp_tools.append(name)
                logger.info(f"Registered MCP tool: {name}")

    def unregister_mcp_tools(self, tool_names: list[str] | None = None) -> None:
        """
        注销 MCP 扩展工具
        
        Args:
            tool_names: 要注销的工具，None 表示全部注销
        """
        if tool_names is None:
            self.mcp_tools.clear()
        else:
            for name in tool_names:
                if name in self.mcp_tools:
                    self.mcp_tools.remove(name)

    def get_all_builtin_tools(self) -> list[str]:
        """获取所有内置工具"""
        all_tools = set(READ_TOOLS + WRITE_TOOLS + AUX_TOOLS + ADVANCED_TOOLS)
        return list(all_tools)

    def get_tool_categories(self) -> dict[str, list[str]]:
        """获取工具分类（用于显示）"""
        return {
            "read": READ_TOOLS,
            "write": WRITE_TOOLS,
            "aux": AUX_TOOLS,
            "advanced": ADVANCED_TOOLS,
            "mcp": self.mcp_tools,
        }


# ========================================
# 单例
# ========================================

_default_selector: ToolSelector | None = None


def get_default_selector() -> ToolSelector:
    """获取默认的工具选择器"""
    global _default_selector
    if _default_selector is None:
        _default_selector = ToolSelector()
    return _default_selector


def get_tools_for_mode(mode: str) -> list[str]:
    """便捷函数：获取模式对应的工具"""
    return get_default_selector().get_tools_for_mode(mode)
