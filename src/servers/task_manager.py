"""
Task Management MCP Server

Provides tools for AI agents to create, manage, and track tasks.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from src.core.tasks import TaskManager, TaskStatus, TaskPriority


# Initialize task manager
task_manager = TaskManager()

# Create MCP server
app = Server("task-manager")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available task management tools"""
    return [
        Tool(
            name="task_create",
            description="Create a new task in the task list",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Task title (short, descriptive)"
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed task description"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                        "description": "Task priority level",
                        "default": "medium"
                    },
                    "parent_id": {
                        "type": "string",
                        "description": "Parent task ID (for creating subtasks)"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Task tags for organization"
                    }
                },
                "required": ["title"]
            }
        ),
        Tool(
            name="task_list",
            description="List tasks with optional filtering",
            inputSchema={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed", "blocked", "cancelled"],
                        "description": "Filter by task status"
                    },
                    "parent_id": {
                        "type": "string",
                        "description": "Filter by parent task (omit for root tasks)"
                    },
                    "show_tree": {
                        "type": "boolean",
                        "description": "Show hierarchical task tree",
                        "default": False
                    }
                }
            }
        ),
        Tool(
            name="task_update_status",
            description="Update the status of a task",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Task ID to update"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed", "blocked", "cancelled"],
                        "description": "New status for the task"
                    }
                },
                "required": ["task_id", "status"]
            }
        ),
        Tool(
            name="task_get",
            description="Get details of a specific task",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Task ID to retrieve"
                    }
                },
                "required": ["task_id"]
            }
        ),
        Tool(
            name="task_delete",
            description="Delete a task",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Task ID to delete"
                    },
                    "delete_subtasks": {
                        "type": "boolean",
                        "description": "Also delete all subtasks",
                        "default": False
                    }
                },
                "required": ["task_id"]
            }
        ),
        Tool(
            name="task_clear",
            description="Clear all tasks (use with caution!)",
            inputSchema={
                "type": "object",
                "properties": {
                    "confirm": {
                        "type": "boolean",
                        "description": "Must be true to confirm deletion",
                        "default": False
                    }
                },
                "required": ["confirm"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""
    
    if name == "task_create":
        title = arguments["title"]
        description = arguments.get("description", "")
        priority_str = arguments.get("priority", "medium")
        parent_id = arguments.get("parent_id")
        tags = arguments.get("tags", [])
        
        # Convert priority string to enum
        priority = TaskPriority(priority_str)
        
        # Create task
        task = task_manager.create_task(
            title=title,
            description=description,
            priority=priority,
            parent_id=parent_id,
            tags=tags
        )
        
        result = f"✓ Created task: {task.id}\n"
        result += f"  Title: {task.title}\n"
        result += f"  Priority: {task.priority.value}\n"
        if parent_id:
            result += f"  Parent: {parent_id}\n"
        
        return [TextContent(type="text", text=result)]
    
    elif name == "task_list":
        status_str = arguments.get("status")
        parent_id = arguments.get("parent_id")
        show_tree = arguments.get("show_tree", False)
        
        # Convert status string to enum
        status = TaskStatus(status_str) if status_str else None
        
        if show_tree:
            # Show hierarchical tree
            tree = task_manager.get_task_tree()
            result = "Task Tree:\n\n"
            result += _format_task_tree(tree)
        else:
            # Show flat list
            tasks = task_manager.list_tasks(status=status, parent_id=parent_id)
            
            if not tasks:
                result = "No tasks found."
            else:
                result = f"Found {len(tasks)} task(s):\n\n"
                for task in tasks:
                    result += f"[{task.status.value.upper()}] {task.id}\n"
                    result += f"  {task.title}\n"
                    result += f"  Priority: {task.priority.value}"
                    if task.description:
                        result += f"\n  Description: {task.description}"
                    if task.subtasks:
                        result += f"\n  Subtasks: {len(task.subtasks)}"
                    result += "\n\n"
        
        return [TextContent(type="text", text=result)]
    
    elif name == "task_update_status":
        task_id = arguments["task_id"]
        status_str = arguments["status"]
        
        # Convert status string to enum
        status = TaskStatus(status_str)
        
        # Update task
        success = task_manager.update_task_status(task_id, status)
        
        if success:
            result = f"✓ Updated task {task_id} to status: {status.value}"
        else:
            result = f"✗ Task not found: {task_id}"
        
        return [TextContent(type="text", text=result)]
    
    elif name == "task_get":
        task_id = arguments["task_id"]
        task = task_manager.get_task(task_id)
        
        if task:
            result = f"Task: {task.id}\n"
            result += f"Title: {task.title}\n"
            result += f"Status: {task.status.value}\n"
            result += f"Priority: {task.priority.value}\n"
            if task.description:
                result += f"Description: {task.description}\n"
            if task.parent_id:
                result += f"Parent: {task.parent_id}\n"
            if task.subtasks:
                result += f"Subtasks: {', '.join(task.subtasks)}\n"
            if task.tags:
                result += f"Tags: {', '.join(task.tags)}\n"
            result += f"Created: {task.created_at}\n"
            result += f"Updated: {task.updated_at}\n"
        else:
            result = f"✗ Task not found: {task_id}"
        
        return [TextContent(type="text", text=result)]
    
    elif name == "task_delete":
        task_id = arguments["task_id"]
        delete_subtasks = arguments.get("delete_subtasks", False)
        
        success = task_manager.delete_task(task_id, delete_subtasks=delete_subtasks)
        
        if success:
            result = f"✓ Deleted task: {task_id}"
            if delete_subtasks:
                result += " (including subtasks)"
        else:
            result = f"✗ Task not found: {task_id}"
        
        return [TextContent(type="text", text=result)]
    
    elif name == "task_clear":
        confirm = arguments.get("confirm", False)
        
        if confirm:
            task_manager.clear_all_tasks()
            result = "✓ Cleared all tasks"
        else:
            result = "✗ Must set confirm=true to clear all tasks"
        
        return [TextContent(type="text", text=result)]
    
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


def _format_task_tree(tree: list, indent: int = 0) -> str:
    """Format task tree for display"""
    result = ""
    for task_dict in tree:
        prefix = "  " * indent
        status_icon = {
            "pending": "○",
            "in_progress": "◐",
            "completed": "●",
            "blocked": "✗",
            "cancelled": "⊘"
        }.get(task_dict["status"], "○")
        
        result += f"{prefix}{status_icon} {task_dict['title']} ({task_dict['id']})\n"
        
        # Recursively format children
        if task_dict.get("children"):
            result += _format_task_tree(task_dict["children"], indent + 1)
    
    return result


async def main():
    """Run the task management server"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
