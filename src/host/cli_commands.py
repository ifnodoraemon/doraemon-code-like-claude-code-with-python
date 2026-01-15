"""
Enhanced CLI Command Handlers for Polymath

Provides structured command handling with the new command system.
"""

from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from src.core.commands import register_command, CommandCategory, get_registry
from src.core.tasks import TaskManager, TaskStatus


console = Console()


# ========================================
# General Commands
# ========================================

@register_command(
    name="/help",
    description="Show available commands",
    category=CommandCategory.GENERAL,
    aliases=["/?", "/h"]
)
async def cmd_help(args: list, context: dict):
    """Show help information"""
    registry = get_registry()
    
    # Group commands by category
    categories = {}
    for cmd in registry.list_all():
        if cmd.category not in categories:
            categories[cmd.category] = []
        categories[cmd.category].append(cmd)
    
    # Build help text
    help_text = "# Available Commands\n\n"
    
    for category, commands in categories.items():
        help_text += f"## {category.value.title()} Commands\n\n"
        for cmd in commands:
            aliases_str = f" ({', '.join(cmd.aliases)})" if cmd.aliases else ""
            help_text += f"**`{cmd.name}`{aliases_str}**\n"
            help_text += f"  {cmd.description}\n"
            if cmd.args_description:
                help_text += f"  Usage: {cmd.args_description}\n"
            help_text += "\n"
    
    console.print(Panel(Markdown(help_text), title="[bold cyan]Polymath Help[/bold cyan]", border_style="cyan"))
    return True


@register_command(
    name="/clear",
    description="Clear chat history and start fresh",
    category=CommandCategory.GENERAL
)
async def cmd_clear(args: list, context: dict):
    """Clear chat history"""
    console.print("[yellow]Clearing session history...[/yellow]")
    
    # Re-initialize chat with empty history
    from src.host.cli import init_chat_model
    client = context['client']
    mode = context['mode']
    tools = context['tools']
    
    context['chat']['session'] = init_chat_model(client, mode, tools, [])
    return True


@register_command(
    name="/quit",
    description="Exit Polymath",
    category=CommandCategory.GENERAL,
    aliases=["/exit", "/q"]
)
async def cmd_quit(args: list, context: dict):
    """Quit the application"""
    console.print("[yellow]Goodbye![/yellow]")
    return "EXIT"


# ========================================
# Workspace Commands
# ========================================

@register_command(
    name="/init",
    description="Initialize project workspace (create AGENTS.md, etc.)",
    category=CommandCategory.WORKSPACE
)
async def cmd_init(args: list, context: dict):
    """Initialize project workspace"""
    from src.core.rules import create_default_agents_md
    
    console.print("[cyan]Initializing project workspace...[/cyan]")
    
    # Create AGENTS.md if it doesn't exist
    if not Path("AGENTS.md").exists():
        create_default_agents_md(Path.cwd())
        console.print("[green]✓ Created AGENTS.md[/green]")
    else:
        console.print("[yellow]AGENTS.md already exists[/yellow]")
    
    # Create .polymath directory
    config_dir = Path.cwd() / ".polymath"
    config_dir.mkdir(exist_ok=True)
    console.print("[green]✓ Created .polymath/ directory[/green]")
    
    console.print("\n[green]✓ Workspace initialized![/green]")
    console.print("[dim]Tip: Edit AGENTS.md to customize project rules[/dim]")
    
    # Re-init chat to load new rules
    from src.host.cli import init_chat_model
    client = context['client']
    mode = context['mode']
    tools = context['tools']
    
    # Preserve history if possible
    hist = []
    if hasattr(context['chat']['session'], '_history'):
        hist = context['chat']['session']._history
    
    context['chat']['session'] = init_chat_model(client, mode, tools, hist)
    return True


# ========================================
# Mode Commands
# ========================================

@register_command(
    name="/mode",
    description="Switch AI mode (plan/build/coder/architect/default)",
    category=CommandCategory.MODE,
    args_description="/mode <mode_name>",
    examples=["/mode plan", "/mode build", "/mode coder"]
)
async def cmd_mode(args: list, context: dict):
    """Switch between different AI modes"""
    from src.core.prompts import PROMPTS
    from src.host.cli import init_chat_model
    
    if len(args) < 2:
        # Show current mode and available modes
        current_mode = context['mode']
        available_modes = list(PROMPTS.keys())
        
        table = Table(title="AI Modes")
        table.add_column("Mode", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Current", style="green")
        
        mode_descriptions = {
            "default": "General planner & project manager",
            "plan": "Strategic planner (read-only, creates detailed plans)",
            "build": "Implementation engineer (executes tasks, writes code)",
            "coder": "Software engineer (focused coding)",
            "architect": "System architect (high-level design)"
        }
        
        for mode in available_modes:
            desc = mode_descriptions.get(mode, "")
            is_current = "✓" if mode == current_mode else ""
            table.add_row(mode, desc, is_current)
        
        console.print(table)
        console.print(f"\n[dim]Usage: /mode <mode_name>[/dim]")
        return True
    
    new_mode = args[1].lower()
    
    if new_mode not in PROMPTS:
        console.print(f"[red]Unknown mode: {new_mode}[/red]")
        console.print(f"[dim]Available modes: {', '.join(PROMPTS.keys())}[/dim]")
        return True
    
    # Update mode
    old_mode = context['mode']
    context['mode'] = new_mode
    
    # Determine mode color
    mode_colors = {
        "plan": "blue",
        "build": "green",
        "coder": "cyan",
        "architect": "magenta",
        "default": "yellow"
    }
    color = mode_colors.get(new_mode, "yellow")
    
    console.print(f"[{color}]Switched from {old_mode} → {new_mode} mode[/{color}]")
    
    # Re-init chat with new mode but preserve history
    client = context['client']
    tools = context['tools']
    
    hist = []
    if hasattr(context['chat']['session'], '_history'):
        hist = context['chat']['session']._history
    
    context['chat']['session'] = init_chat_model(client, new_mode, tools, hist)
    return True


# ========================================
# Task Commands
# ========================================

@register_command(
    name="/tasks",
    description="View task list",
    category=CommandCategory.TASK,
    aliases=["/task", "/t"],
    args_description="/tasks [status]",
    examples=["/tasks", "/tasks pending", "/tasks completed"]
)
async def cmd_tasks(args: list, context: dict):
    """View task list"""
    task_manager = TaskManager()
    
    # Parse status filter
    status = None
    if len(args) > 1:
        try:
            status = TaskStatus(args[1].lower())
        except ValueError:
            console.print(f"[red]Invalid status: {args[1]}[/red]")
            console.print(f"[dim]Valid statuses: pending, in_progress, completed, blocked, cancelled[/dim]")
            return True
    
    # Get tasks
    tasks = task_manager.list_tasks(status=status)
    
    if not tasks:
        console.print("[yellow]No tasks found.[/yellow]")
        return True
    
    # Display as table
    table = Table(title=f"Tasks ({len(tasks)} total)")
    table.add_column("Status", style="cyan", width=12)
    table.add_column("ID", style="dim", width=25)
    table.add_column("Title", style="white")
    table.add_column("Priority", style="yellow", width=10)
    
    for task in tasks:
        status_icon = {
            TaskStatus.PENDING: "○ Pending",
            TaskStatus.IN_PROGRESS: "◐ Working",
            TaskStatus.COMPLETED: "● Done",
            TaskStatus.BLOCKED: "✗ Blocked",
            TaskStatus.CANCELLED: "⊘ Cancelled"
        }.get(task.status, task.status.value)
        
        table.add_row(
            status_icon,
            task.id,
            task.title,
            task.priority.value
        )
    
    console.print(table)
    return True


@register_command(
    name="/tasks-clear",
    description="Clear all tasks",
    category=CommandCategory.TASK
)
async def cmd_tasks_clear(args: list, context: dict):
    """Clear all tasks"""
    from rich.prompt import Confirm
    
    if not Confirm.ask("[red]Clear all tasks? This cannot be undone.[/red]"):
        console.print("[yellow]Cancelled.[/yellow]")
        return True
    
    task_manager = TaskManager()
    task_manager.clear_all_tasks()
    console.print("[green]✓ Cleared all tasks[/green]")
    return True


# ========================================
# Debug Commands
# ========================================

@register_command(
    name="/debug",
    description="Show debug information",
    category=CommandCategory.DEBUG
)
async def cmd_debug(args: list, context: dict):
    """Show debug information"""
    info = f"""# Debug Information

**Mode**: {context['mode']}
**Active Tools**: {len(context['tools'])}
**MCP Servers**: {', '.join(context.get('mcp_servers', []))}
**Working Directory**: {Path.cwd()}

"""
    console.print(Panel(Markdown(info), title="[bold yellow]Debug Info[/bold yellow]", border_style="yellow"))
    return True


# ========================================
# Command Dispatcher
# ========================================

async def dispatch_command(command: str, context: dict) -> bool:
    """
    Dispatch a slash command to its handler
    
    Args:
        command: Full command string (e.g., "/mode plan")
        context: Execution context with chat, client, mode, tools, etc.
    
    Returns:
        True to continue, "EXIT" to quit, False on error
    """
    parts = command.strip().split()
    cmd_name = parts[0].lower()
    
    # Get command from registry
    registry = get_registry()
    cmd = registry.get(cmd_name)
    
    if not cmd:
        console.print(f"[red]Unknown command: {cmd_name}[/red]")
        console.print("[dim]Type /help for available commands[/dim]")
        return True
    
    # Execute command handler
    try:
        result = await cmd.handler(parts, context)
        return result if result is not None else True
    except Exception as e:
        console.print(f"[red]Command error: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return True
