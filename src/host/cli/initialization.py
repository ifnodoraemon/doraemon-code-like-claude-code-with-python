"""
Session and Manager Initialization

Handles initialization of all managers and components needed for the chat loop.
Extracted from main.py for better maintainability.
"""

import os
from pathlib import Path

from src.core.background_tasks import get_task_manager
from src.core.checkpoint import CheckpointConfig, CheckpointManager
from src.core.command_history import BashModeExecutor, CommandHistory
from src.core.context_manager import ContextConfig, ContextManager
from src.core.cost_tracker import BudgetConfig, CostTracker
from src.core.hooks import HookManager
from src.core.model_client import ModelClient
from src.core.session import SessionManager
from src.core.skills import SkillManager
from src.core.tool_selector import ToolSelector
from src.host.tools import get_default_registry


async def initialize_model_client() -> ModelClient:
    """Initialize the unified model client."""
    return await ModelClient.create()


def initialize_registry():
    """Initialize the tool registry."""
    return get_default_registry()


def initialize_tool_selector() -> ToolSelector:
    """Initialize the tool selector."""
    return ToolSelector()


def initialize_context_manager(project: str) -> ContextManager:
    """Initialize the context manager."""
    ctx_config = ContextConfig(
        max_context_tokens=100_000,
        summarize_threshold=0.7,
        keep_recent_messages=6,
        auto_save=True,
    )
    return ContextManager(project=project, config=ctx_config)


def initialize_skill_manager() -> SkillManager:
    """Initialize the skill manager."""
    return SkillManager(project_dir=Path.cwd(), max_skill_tokens=5000)


def initialize_checkpoint_manager(project: str) -> CheckpointManager:
    """Initialize the checkpoint manager."""
    return CheckpointManager(
        project=project,
        config=CheckpointConfig(enabled=True, retention_days=30),
    )


def initialize_task_manager():
    """Initialize the background task manager."""
    return get_task_manager()


def initialize_hook_manager(ctx: ContextManager) -> HookManager:
    """Initialize the hook manager and load hooks from file."""
    hook_mgr = HookManager(
        project_dir=Path.cwd(),
        session_id=ctx.session_id,
        permission_mode="default",
    )
    hooks_file = Path(".doraemon/hooks.json")
    if hooks_file.exists():
        hook_mgr.load_from_file(hooks_file)
    return hook_mgr


def initialize_cost_tracker(project: str, ctx: ContextManager) -> CostTracker:
    """Initialize the cost tracker."""
    budget_config = BudgetConfig(
        daily_limit_usd=float(os.getenv("DORAEMON_DAILY_BUDGET", "0")) or None,
        session_limit_usd=float(os.getenv("DORAEMON_SESSION_BUDGET", "0")) or None,
    )
    return CostTracker(
        project=project,
        session_id=ctx.session_id,
        budget=budget_config,
    )


def initialize_command_history(project: str) -> CommandHistory:
    """Initialize the command history."""
    cmd_history = CommandHistory(project=project)
    cmd_history.setup_readline()
    return cmd_history


def initialize_bash_executor() -> BashModeExecutor:
    """Initialize the bash mode executor."""
    return BashModeExecutor(cwd=Path.cwd())


def initialize_session_manager() -> SessionManager:
    """Initialize the session manager."""
    return SessionManager()


async def initialize_all_managers(project: str):
    """
    Initialize all managers and components.

    Returns:
        dict with all initialized managers
    """
    model_client = await initialize_model_client()
    registry = initialize_registry()
    tool_selector = initialize_tool_selector()
    ctx = initialize_context_manager(project)
    skill_mgr = initialize_skill_manager()
    checkpoint_mgr = initialize_checkpoint_manager(project)
    task_mgr = initialize_task_manager()
    hook_mgr = initialize_hook_manager(ctx)
    cost_tracker = initialize_cost_tracker(project, ctx)
    cmd_history = initialize_command_history(project)
    bash_executor = initialize_bash_executor()
    session_mgr = initialize_session_manager()

    return {
        "model_client": model_client,
        "registry": registry,
        "tool_selector": tool_selector,
        "ctx": ctx,
        "skill_mgr": skill_mgr,
        "checkpoint_mgr": checkpoint_mgr,
        "task_mgr": task_mgr,
        "hook_mgr": hook_mgr,
        "cost_tracker": cost_tracker,
        "cmd_history": cmd_history,
        "bash_executor": bash_executor,
        "session_mgr": session_mgr,
    }
