"""Centralized project-local paths for agent state and instructions."""

from pathlib import Path

STATE_DIRNAME = ".agent"
RULES_FILENAME = "AGENTS.md"
MEMORY_FILENAME = "MEMORY.md"


def project_root(project_dir: Path | None = None) -> Path:
    """Return the active project root."""
    return (project_dir or Path.cwd()).resolve()


def state_dir(project_dir: Path | None = None) -> Path:
    """Return the project-local state directory."""
    return project_root(project_dir) / STATE_DIRNAME


def config_path(project_dir: Path | None = None) -> Path:
    return state_dir(project_dir) / "config.json"


def hooks_path(project_dir: Path | None = None) -> Path:
    return state_dir(project_dir) / "hooks.json"


def permissions_path(project_dir: Path | None = None) -> Path:
    return state_dir(project_dir) / "permissions.json"


def memory_path(project_dir: Path | None = None) -> Path:
    return state_dir(project_dir) / MEMORY_FILENAME


def sessions_dir(project_dir: Path | None = None) -> Path:
    return state_dir(project_dir) / "sessions"


def conversations_dir(project_dir: Path | None = None) -> Path:
    return state_dir(project_dir) / "conversations"


def checkpoints_dir(project_dir: Path | None = None) -> Path:
    return state_dir(project_dir) / "checkpoints"


def usage_dir(project_dir: Path | None = None) -> Path:
    return state_dir(project_dir) / "usage"


def history_dir(project_dir: Path | None = None) -> Path:
    return state_dir(project_dir) / "history"


def recovery_dir(project_dir: Path | None = None) -> Path:
    return state_dir(project_dir) / "recovery"


def tasks_path(project_dir: Path | None = None) -> Path:
    return state_dir(project_dir) / "tasks.json"


def theme_path(project_dir: Path | None = None) -> Path:
    return state_dir(project_dir) / "theme.json"


def logs_dir(project_dir: Path | None = None) -> Path:
    return state_dir(project_dir) / "logs"


def chroma_dir(project_dir: Path | None = None) -> Path:
    return state_dir(project_dir) / "chroma_db"


def persona_path(project_dir: Path | None = None) -> Path:
    return state_dir(project_dir) / "memory.json"


def layered_memory_dir(project_dir: Path | None = None) -> Path:
    return state_dir(project_dir) / "memory"


def skills_dir(project_dir: Path | None = None) -> Path:
    return state_dir(project_dir) / "skills"


def plugins_dir(project_dir: Path | None = None) -> Path:
    return state_dir(project_dir) / "plugins"


def local_plugins_dir(project_dir: Path | None = None) -> Path:
    return state_dir(project_dir) / "plugins.local"
