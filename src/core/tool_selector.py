"""Tool selection by mode."""

# ========================================
# 工具定义
# ========================================

# 只读工具 - plan 和 build 模式都可用
READ_TOOLS = [
    "read",  # 统一读取（文件/目录/大纲/树）
    "search",  # 统一搜索（内容/文件名/符号）
    "ask_user",  # 向用户提问
]

# 写入工具 - 只有 build 模式可用
WRITE_TOOLS = [
    "write",  # 统一写入（创建/编辑/删除/移动/复制）
    "run",  # 统一执行（shell/python/background/install）
]

# 辅助工具 - 按需使用
AUX_TOOLS = [
    "web_search",  # 网络搜索
    "browse_page",  # 网页浏览 (Playwright)
    "take_screenshot",  # 网页截图
    "task",  # 统一任务工具
    "db_read_query",  # 数据库查询
    "db_write_query",  # 数据库修改
    "db_list_tables",  # 列出数据库表
    "db_describe_table",  # 查看表结构
    "fetch_url",  # 获取网页
    "save_note",  # 保存笔记
    "search_notes",  # 搜索笔记
]

class ToolSelector:
    """
    简化的工具选择器

    按模式分配工具，不做动态搜索
    """

    def __init__(self):
        # plan 模式：最小只读工具集
        self.plan_tools = READ_TOOLS.copy()

        # build 模式：最小 coding 工具集
        self.build_tools = READ_TOOLS + WRITE_TOOLS

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

        return tools


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
