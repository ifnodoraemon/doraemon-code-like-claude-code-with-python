"""Tool selection by product mode."""

CAPABILITY_GROUPS: dict[str, list[str]] = {
    "read": [
        "read",
        "search",
        "ask_user",
    ],
    "edit": [
        "write",
        "run",
    ],
    "memory": [
        "memory_get",
        "memory_put",
        "memory_search",
        "memory_list",
    ],
    "research": [
        "web_search",
        "web_fetch",
    ],
    "task": [
        "task",
    ],
}

MODE_CAPABILITY_GROUPS: dict[str, list[str]] = {
    "plan": ["read", "memory", "research", "task"],
    "build": ["read", "edit", "memory", "research", "task"],
}

class ToolSelector:
    """
    Product-facing tool selector.

    Only exposes two modes: `plan` and `build`.
    Each mode expands to a stable set of capability groups.
    """

    def __init__(self):
        self.capability_groups = {name: tools.copy() for name, tools in CAPABILITY_GROUPS.items()}
        self.mode_capability_groups = {
            mode: groups.copy() for mode, groups in MODE_CAPABILITY_GROUPS.items()
        }

    def get_tools_for_mode(self, mode: str) -> list[str]:
        """Expand a product mode into its tool list."""
        groups = self.mode_capability_groups.get(mode, self.mode_capability_groups["build"])
        tools: list[str] = []
        for group in groups:
            for tool_name in self.capability_groups.get(group, []):
                if tool_name not in tools:
                    tools.append(tool_name)
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


def get_capability_groups_for_mode(mode: str) -> list[str]:
    """Get capability groups enabled for a product mode."""
    selector = get_default_selector()
    return selector.mode_capability_groups.get(mode, selector.mode_capability_groups["build"]).copy()
