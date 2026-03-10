"""
Plugin System

Enables extending Doraemon with plugins that bundle:
- Custom slash commands
- Custom tools
- Hooks
- Subagents
- MCP server configurations

Features:
- Plugin discovery and loading
- Plugin marketplace support
- Version pinning (SHA)
- Plugin isolation
"""

import json
import logging
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PluginScope(Enum):
    """Plugin installation scope."""

    USER = "user"  # Legacy alias for project plugins
    PROJECT = "project"  # .agent/plugins
    LOCAL = "local"  # .agent/plugins.local (not committed)


@dataclass
class PluginManifest:
    """Plugin manifest (plugin.json)."""

    name: str
    version: str
    description: str
    author: str = ""
    homepage: str = ""
    repository: str = ""

    # Components
    commands: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    hooks: dict[str, Any] = field(default_factory=dict)
    agents: list[dict[str, Any]] = field(default_factory=list)

    # Requirements
    doraemon_version: str = ">=0.6.0"
    dependencies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "homepage": self.homepage,
            "repository": self.repository,
            "commands": self.commands,
            "tools": self.tools,
            "hooks": self.hooks,
            "agents": self.agents,
            "doraemon_version": self.doraemon_version,
            "dependencies": self.dependencies,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PluginManifest":
        return cls(
            name=data.get("name", "unknown"),
            version=data.get("version", "0.0.0"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            homepage=data.get("homepage", ""),
            repository=data.get("repository", ""),
            commands=data.get("commands", []),
            tools=data.get("tools", []),
            hooks=data.get("hooks", {}),
            agents=data.get("agents", []),
            doraemon_version=data.get("doraemon_version", ">=0.6.0"),
            dependencies=data.get("dependencies", []),
        )


@dataclass
class InstalledPlugin:
    """An installed plugin."""

    manifest: PluginManifest
    path: Path
    scope: PluginScope
    enabled: bool = True
    installed_at: float = field(default_factory=time.time)
    sha: str | None = None  # Git SHA for version pinning

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.manifest.name,
            "version": self.manifest.version,
            "path": str(self.path),
            "scope": self.scope.value,
            "enabled": self.enabled,
            "installed_at": self.installed_at,
            "sha": self.sha,
        }


@dataclass
class PluginCommand:
    """A command provided by a plugin."""

    name: str
    description: str
    handler: Callable
    plugin_name: str


class PluginManager:
    """
    Manages plugin installation, loading, and execution.

    Usage:
        mgr = PluginManager()

        # Install plugin from GitHub
        mgr.install("owner/repo")
        mgr.install("owner/repo@sha256:abc123")

        # List plugins
        plugins = mgr.list_plugins()

        # Enable/disable
        mgr.enable("plugin-name")
        mgr.disable("plugin-name")

        # Get plugin commands
        commands = mgr.get_commands()

        # Uninstall
        mgr.uninstall("plugin-name")
    """

    def __init__(self, project_dir: Path | None = None):
        """
        Initialize plugin manager.

        Args:
            project_dir: Project directory (for project-scope plugins)
        """
        self.project_dir = project_dir or Path.cwd()

        # Plugin directories
        self._user_dir = self.project_dir / ".agent" / "plugins"
        self._project_dir = self.project_dir / ".agent" / "plugins"
        self._local_dir = self.project_dir / ".agent" / "plugins.local"

        # Ensure directories exist
        self._user_dir.mkdir(parents=True, exist_ok=True)

        # Loaded plugins
        self._plugins: dict[str, InstalledPlugin] = {}
        self._commands: dict[str, PluginCommand] = {}

        # Load installed plugins
        self._load_plugins()

    def _get_plugin_dirs(self) -> list[tuple[Path, PluginScope]]:
        """Get plugin directories in order of precedence."""
        dirs: list[tuple[Path, PluginScope]] = []
        seen: set[Path] = set()

        # Local (highest precedence)
        if self._local_dir.exists():
            dirs.append((self._local_dir, PluginScope.LOCAL))
            seen.add(self._local_dir.resolve())

        # Project
        if self._project_dir.exists() and self._project_dir.resolve() not in seen:
            dirs.append((self._project_dir, PluginScope.PROJECT))

        return dirs

    def _load_plugins(self):
        """Load all installed plugins."""
        for plugin_dir, scope in self._get_plugin_dirs():
            if not plugin_dir.exists():
                continue

            for item in plugin_dir.iterdir():
                if item.is_dir():
                    self._load_plugin(item, scope)

    def _load_plugin(self, plugin_path: Path, scope: PluginScope) -> bool:
        """Load a single plugin."""
        manifest_path = plugin_path / "plugin.json"

        if not manifest_path.exists():
            logger.warning(f"No plugin.json found in {plugin_path}")
            return False

        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = PluginManifest.from_dict(data)

            # Check if already loaded (higher precedence wins)
            if manifest.name in self._plugins:
                return False

            plugin = InstalledPlugin(
                manifest=manifest,
                path=plugin_path,
                scope=scope,
            )

            # Load plugin state
            state_path = plugin_path / ".state.json"
            if state_path.exists():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                plugin.enabled = state.get("enabled", True)
                plugin.installed_at = state.get("installed_at", time.time())
                plugin.sha = state.get("sha")

            self._plugins[manifest.name] = plugin

            # Register commands if enabled
            if plugin.enabled:
                self._register_plugin_commands(plugin)

            logger.info(f"Loaded plugin: {manifest.name} v{manifest.version}")
            return True

        except Exception as e:
            logger.error(f"Failed to load plugin from {plugin_path}: {e}")
            return False

    def _register_plugin_commands(self, plugin: InstalledPlugin):
        """Register commands from a plugin."""
        for cmd_def in plugin.manifest.commands:
            name = cmd_def.get("name")
            if not name:
                continue

            # Create command handler
            script_path = plugin.path / cmd_def.get("script", f"{name}.py")

            if script_path.exists():
                # Python script command
                def make_handler(path: Path):
                    def handler(**kwargs):
                        import runpy

                        return runpy.run_path(str(path), run_name="__main__")

                    return handler

                self._commands[name] = PluginCommand(
                    name=name,
                    description=cmd_def.get("description", ""),
                    handler=make_handler(script_path),
                    plugin_name=plugin.manifest.name,
                )

    def install(
        self,
        source: str,
        scope: PluginScope = PluginScope.PROJECT,
        force: bool = False,
    ) -> InstalledPlugin | None:
        """
        Install a plugin.

        Args:
            source: Plugin source (GitHub repo, local path, or URL)
                - "owner/repo" - GitHub repo
                - "owner/repo@sha" - Specific commit
                - "/path/to/plugin" - Local path
            scope: Installation scope
            force: Force reinstall if exists

        Returns:
            Installed plugin or None if failed
        """
        # Determine target directory
        if scope in (PluginScope.USER, PluginScope.PROJECT):
            target_base = self._project_dir
        else:
            target_base = self._local_dir

        target_base.mkdir(parents=True, exist_ok=True)

        # Parse source
        sha = None
        if "@" in source and not source.startswith("/"):
            source, sha = source.rsplit("@", 1)

        # Install based on source type
        if source.startswith("/") or source.startswith("."):
            # Local path
            return self._install_from_local(Path(source), target_base, force)
        elif "/" in source and not source.startswith("http"):
            # GitHub repo (owner/repo)
            return self._install_from_github(source, target_base, sha, force)
        else:
            logger.error(f"Unknown plugin source: {source}")
            return None

    def _install_from_local(
        self, source_path: Path, target_base: Path, force: bool
    ) -> InstalledPlugin | None:
        """Install plugin from local path."""
        source_path = source_path.resolve()

        if not source_path.exists():
            logger.error(f"Plugin path not found: {source_path}")
            return None

        manifest_path = source_path / "plugin.json"
        if not manifest_path.exists():
            logger.error(f"No plugin.json found in {source_path}")
            return None

        try:
            manifest = PluginManifest.from_dict(
                json.loads(manifest_path.read_text(encoding="utf-8"))
            )
        except Exception as e:
            logger.error(f"Invalid plugin.json: {e}")
            return None

        target_path = target_base / manifest.name

        # Check if exists
        if target_path.exists():
            if not force:
                logger.error(f"Plugin already installed: {manifest.name}")
                return None
            shutil.rmtree(target_path)

        # Copy plugin
        shutil.copytree(source_path, target_path)

        # Save state
        state = {
            "enabled": True,
            "installed_at": time.time(),
            "source": str(source_path),
        }
        (target_path / ".state.json").write_text(
            json.dumps(state, indent=2), encoding="utf-8"
        )

        # Load plugin
        scope = self._get_scope_for_path(target_base)
        self._load_plugin(target_path, scope)

        logger.info(f"Installed plugin: {manifest.name}")
        return self._plugins.get(manifest.name)

    def _install_from_github(
        self, repo: str, target_base: Path, sha: str | None, force: bool
    ) -> InstalledPlugin | None:
        """Install plugin from GitHub."""
        # Clone repo
        url = f"https://github.com/{repo}.git"

        # Create temp directory
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / "plugin"

            try:
                # Clone
                cmd = ["git", "clone", "--depth", "1"]
                if sha:
                    cmd = ["git", "clone"]
                cmd.extend([url, str(tmp_path)])

                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.error(f"Git clone failed: {result.stderr}")
                    return None

                # Checkout specific SHA if provided
                if sha:
                    result = subprocess.run(
                        ["git", "checkout", sha],
                        cwd=tmp_path,
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode != 0:
                        logger.error(f"Git checkout failed: {result.stderr}")
                        return None

                # Install from cloned path
                plugin = self._install_from_local(tmp_path, target_base, force)

                if plugin and sha:
                    plugin.sha = sha
                    # Update state
                    state_path = plugin.path / ".state.json"
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                    state["sha"] = sha
                    state["repository"] = repo
                    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

                return plugin

            except Exception as e:
                logger.error(f"Failed to install from GitHub: {e}")
                return None

    def _get_scope_for_path(self, path: Path) -> PluginScope:
        """Get scope for a plugin path."""
        if path == self._local_dir:
            return PluginScope.LOCAL
        return PluginScope.PROJECT

    def uninstall(self, name: str) -> bool:
        """Uninstall a plugin."""
        plugin = self._plugins.get(name)
        if not plugin:
            logger.error(f"Plugin not found: {name}")
            return False

        try:
            # Remove directory
            shutil.rmtree(plugin.path)

            # Remove from loaded plugins
            del self._plugins[name]

            # Remove commands
            self._commands = {
                k: v for k, v in self._commands.items() if v.plugin_name != name
            }

            logger.info(f"Uninstalled plugin: {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to uninstall plugin: {e}")
            return False

    def enable(self, name: str) -> bool:
        """Enable a plugin."""
        plugin = self._plugins.get(name)
        if not plugin:
            return False

        plugin.enabled = True
        self._save_plugin_state(plugin)
        self._register_plugin_commands(plugin)
        return True

    def disable(self, name: str) -> bool:
        """Disable a plugin."""
        plugin = self._plugins.get(name)
        if not plugin:
            return False

        plugin.enabled = False
        self._save_plugin_state(plugin)

        # Remove commands
        self._commands = {
            k: v for k, v in self._commands.items() if v.plugin_name != name
        }

        return True

    def _save_plugin_state(self, plugin: InstalledPlugin):
        """Save plugin state."""
        state_path = plugin.path / ".state.json"
        state = {
            "enabled": plugin.enabled,
            "installed_at": plugin.installed_at,
            "sha": plugin.sha,
        }
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def list_plugins(self, include_disabled: bool = True) -> list[InstalledPlugin]:
        """List installed plugins."""
        plugins = list(self._plugins.values())
        if not include_disabled:
            plugins = [p for p in plugins if p.enabled]
        return plugins

    def get_plugin(self, name: str) -> InstalledPlugin | None:
        """Get a plugin by name."""
        return self._plugins.get(name)

    def get_commands(self) -> dict[str, PluginCommand]:
        """Get all plugin commands."""
        return self._commands.copy()

    def get_hooks(self) -> dict[str, list[dict]]:
        """Get all hooks from enabled plugins."""
        hooks: dict[str, list[dict]] = {}

        for plugin in self._plugins.values():
            if not plugin.enabled:
                continue

            for event, hook_list in plugin.manifest.hooks.items():
                if event not in hooks:
                    hooks[event] = []

                for hook in hook_list if isinstance(hook_list, list) else [hook_list]:
                    hook_copy = hook.copy()
                    hook_copy["_plugin"] = plugin.manifest.name
                    hook_copy["_plugin_path"] = str(plugin.path)
                    hooks[event].append(hook_copy)

        return hooks

    def get_agents(self) -> list[dict]:
        """Get all agents from enabled plugins."""
        agents = []

        for plugin in self._plugins.values():
            if not plugin.enabled:
                continue

            for agent in plugin.manifest.agents:
                agent_copy = agent.copy()
                agent_copy["_plugin"] = plugin.manifest.name
                agents.append(agent_copy)

        return agents

    def search(self, query: str) -> list[dict]:
        """
        Search for plugins (placeholder for marketplace integration).

        In the future, this could search a plugin registry/marketplace.
        """
        # For now, return empty list
        # This could be extended to search GitHub, a registry, etc.
        logger.info(f"Plugin search not yet implemented: {query}")
        return []

    def get_summary(self) -> dict[str, Any]:
        """Get plugin system summary."""
        enabled = [p for p in self._plugins.values() if p.enabled]
        disabled = [p for p in self._plugins.values() if not p.enabled]

        return {
            "total": len(self._plugins),
            "enabled": len(enabled),
            "disabled": len(disabled),
            "commands": len(self._commands),
            "plugins": [
                {
                    "name": p.manifest.name,
                    "version": p.manifest.version,
                    "enabled": p.enabled,
                    "scope": p.scope.value,
                }
                for p in self._plugins.values()
            ],
        }
