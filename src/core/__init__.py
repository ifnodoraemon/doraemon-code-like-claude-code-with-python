"""Core utilities for Polymath."""

from .background_tasks import BackgroundTaskManager, TaskStatus, get_task_manager
from .browser import BrowserManager, get_browser_manager
from .cache import ToolCache, get_tool_cache
from .checkpoint import CheckpointConfig, CheckpointManager
from .command_history import BashModeExecutor, CommandHistory
from .config import load_config
from .cost_tracker import BudgetConfig, CostTracker
from .doctor import Doctor
from .file_watcher import FileWatcher, get_file_watcher
from .git_hooks import GitHooksManager, GitHookType, get_git_hooks_manager
from .hooks import HookEvent, HookManager, HookResult
from .hot_reload import HotReloadManager, get_hot_reload_manager
from .i18n import I18n, get_i18n, t
from .input_mode import InputManager, InputMode
from .log_rotation import LogRotationManager, RotationConfig, setup_rotating_logger
from .logger import setup_logger
from .mcp_client import MCPClient, MCPServerConfig
from .memory_layers import LayeredMemory, MemoryLayer, get_layered_memory
from .model_client import (
    ChatResponse,
    ClientConfig,
    ClientMode,
    Message,
    ModelClient,
    ToolDefinition,
)
from .model_manager import AVAILABLE_MODELS, ModelCapability, ModelManager
from .notifications import NotificationManager, get_notification_manager, notify
from .parallel_executor import ExecutionStrategy, ParallelExecutor, ToolCall
from .permissions import PermissionLevel, PermissionManager, PermissionRule
from .plugins import PluginManager, PluginScope
from .proxy import ProxyConfig, ProxyManager, get_proxy_manager
from .session import SessionData, SessionManager
from .streaming import StreamingChat, StreamManager
from .subagents import BUILTIN_AGENTS, SubagentConfig, SubagentManager
from .task_recovery import TaskRecoveryManager, TaskState, get_recovery_manager
from .themes import BUILTIN_THEMES, Theme, ThemeManager
from .thinking import ThinkingManager, ThinkingMode
from .tool_history import ToolHistoryManager, get_tool_history
from .workspace import WorkspaceManager

__all__ = [
    "load_config",
    "setup_logger",
    # Checkpoint
    "CheckpointManager",
    "CheckpointConfig",
    # Background Tasks
    "BackgroundTaskManager",
    "get_task_manager",
    "TaskStatus",
    # Subagents
    "SubagentManager",
    "SubagentConfig",
    "BUILTIN_AGENTS",
    # Hooks
    "HookManager",
    "HookEvent",
    "HookResult",
    # Session
    "SessionManager",
    "SessionData",
    # Cost Tracking
    "CostTracker",
    "BudgetConfig",
    # Command History
    "CommandHistory",
    "BashModeExecutor",
    # Plugins
    "PluginManager",
    "PluginScope",
    # Workspace
    "WorkspaceManager",
    # Model Manager
    "ModelManager",
    "ModelCapability",
    "AVAILABLE_MODELS",
    # Model Client (Unified API)
    "ModelClient",
    "ClientMode",
    "ClientConfig",
    "ChatResponse",
    "Message",
    "ToolDefinition",
    # Browser
    "BrowserManager",
    "get_browser_manager",
    # Input Mode
    "InputManager",
    "InputMode",
    # Thinking
    "ThinkingManager",
    "ThinkingMode",
    # Doctor
    "Doctor",
    # Themes
    "ThemeManager",
    "Theme",
    "BUILTIN_THEMES",
    # Streaming
    "StreamManager",
    "StreamingChat",
    # Parallel Executor
    "ParallelExecutor",
    "ExecutionStrategy",
    "ToolCall",
    # MCP Client
    "MCPClient",
    "MCPServerConfig",
    # Cache
    "ToolCache",
    "get_tool_cache",
    # File Watcher
    "FileWatcher",
    "get_file_watcher",
    # Hot Reload
    "HotReloadManager",
    "get_hot_reload_manager",
    # Permissions
    "PermissionManager",
    "PermissionLevel",
    "PermissionRule",
    # Tool History
    "ToolHistoryManager",
    "get_tool_history",
    # Task Recovery
    "TaskRecoveryManager",
    "get_recovery_manager",
    "TaskState",
    # Proxy
    "ProxyManager",
    "ProxyConfig",
    "get_proxy_manager",
    # Notifications
    "NotificationManager",
    "get_notification_manager",
    "notify",
    # I18n
    "I18n",
    "get_i18n",
    "t",
    # Git Hooks
    "GitHooksManager",
    "get_git_hooks_manager",
    "GitHookType",
    # Log Rotation
    "LogRotationManager",
    "RotationConfig",
    "setup_rotating_logger",
    # Memory Layers
    "LayeredMemory",
    "MemoryLayer",
    "get_layered_memory",
]
