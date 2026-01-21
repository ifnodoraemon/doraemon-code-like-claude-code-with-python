"""
Slash Command System for Polymath

Provides a flexible command registration and execution framework.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


class CommandCategory(Enum):
    """Command categories for organization"""

    GENERAL = "general"
    WORKSPACE = "workspace"
    MODE = "mode"
    TASK = "task"
    DEBUG = "debug"


@dataclass
class Command:
    """Represents a slash command"""

    name: str
    description: str
    category: CommandCategory
    handler: Callable
    aliases: list[str] | None = None
    args_description: str = ""
    examples: list[str] | None = None

    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []
        if self.examples is None:
            self.examples = []


class CommandRegistry:
    """Registry for managing slash commands"""

    def __init__(self):
        self.commands: dict[str, Command] = {}
        self._alias_map: dict[str, str] = {}

    def register(self, command: Command):
        """Register a new command"""
        self.commands[command.name] = command

        # Register aliases
        if command.aliases:
            for alias in command.aliases:
                self._alias_map[alias] = command.name

    def get(self, name: str) -> Command | None:
        """Get command by name or alias"""
        # Check if it's an alias first
        if name in self._alias_map:
            name = self._alias_map[name]

        return self.commands.get(name)

    def get_by_category(self, category: CommandCategory) -> list[Command]:
        """Get all commands in a category"""
        return [cmd for cmd in self.commands.values() if cmd.category == category]

    def list_all(self) -> list[Command]:
        """List all registered commands"""
        return list(self.commands.values())


# Global registry instance
_registry = CommandRegistry()


def register_command(
    name: str,
    description: str,
    category: CommandCategory = CommandCategory.GENERAL,
    aliases: list[str] | None = None,
    args_description: str = "",
    examples: list[str] | None = None,
):
    """Decorator to register a command handler"""

    def decorator(handler: Callable):
        command = Command(
            name=name,
            description=description,
            category=category,
            handler=handler,
            aliases=aliases if aliases is not None else [],
            args_description=args_description,
            examples=examples if examples is not None else [],
        )
        _registry.register(command)
        return handler

    return decorator


def get_registry() -> CommandRegistry:
    """Get the global command registry"""
    return _registry
