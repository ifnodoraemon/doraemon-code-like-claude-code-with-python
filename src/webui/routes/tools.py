"""
Tools API Routes

Lists available tools.
"""

from fastapi import APIRouter

from src.core.tool_selector import ToolSelector
from src.host.tools import get_default_registry

router = APIRouter()


@router.get("/")
async def list_tools(mode: str = "build"):
    """List available tools for a mode."""
    selector = ToolSelector()
    tool_names = selector.get_tools_for_mode(mode)
    registry = get_default_registry(tool_names)
    genai_tools = registry.get_genai_tools(tool_names)

    tools = []
    for tool in genai_tools:
        if hasattr(tool, "name"):
            tools.append(
                {
                    "name": tool.name,
                    "description": getattr(tool, "description", ""),
                }
            )
        elif isinstance(tool, dict):
            tools.append(
                {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                }
            )

    return {"tools": tools}
