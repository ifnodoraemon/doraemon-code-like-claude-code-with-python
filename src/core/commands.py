"""
Commands System - Custom Workflow Definitions

Inspired by OpenCode's custom commands, this module provides a way to define
reusable workflows using markdown files with RUN and READ directives.

Command File Format:
    ---
    name: Fetch Issue Context
    description: Fetch context for a GitHub issue
    arguments:
      - ISSUE_NUMBER
      - AUTHOR_NAME (optional)
    ---

    RUN gh issue view $ISSUE_NUMBER --json title,body,comments
    RUN git grep --author="$AUTHOR_NAME" -n .
    READ src/context.md

Usage:
    from src.core.commands import CommandLoader, CommandExecutor

    loader = CommandLoader(commands_dir)
    executor = CommandExecutor()

    # List available commands
    commands = loader.list_commands()

    # Execute a command
    result = await executor.execute("fetch-issue", {"ISSUE_NUMBER": "123"})
"""

import asyncio
import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class CommandArgument:
    """Definition of a command argument."""

    name: str
    required: bool = True
    default: str | None = None
    description: str = ""


@dataclass
class CommandDefinition:
    """Definition of a custom command."""

    name: str
    description: str
    arguments: list[CommandArgument] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "arguments": [
                {"name": arg.name, "required": arg.required, "default": arg.default}
                for arg in self.arguments
            ],
            "steps": self.steps,
        }


@dataclass
class CommandResult:
    """Result of command execution."""

    command_name: str
    success: bool
    outputs: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duration: float = 0.0
    step_results: list[dict[str, Any]] = field(default_factory=list)


class CommandLoader:
    """
    Loads command definitions from markdown files.

    Commands are stored in .agent/commands/ directory as .md files.
    """

    COMMAND_DIR = "commands"

    def __init__(self, project_dir: Path | None = None):
        self.project_dir = project_dir or Path.cwd()
        self._commands: dict[str, CommandDefinition] = {}
        self._loaded = False

    def _get_commands_dir(self) -> Path:
        """Get the commands directory path."""
        return self.project_dir / ".agent" / self.COMMAND_DIR

    def discover_commands(self) -> list[str]:
        """Discover all available command files."""
        commands_dir = self._get_commands_dir()
        if not commands_dir.exists():
            return []

        return [f.stem for f in commands_dir.glob("*.md")]

    def load_command(self, name: str) -> CommandDefinition | None:
        """Load a specific command by name."""
        commands_dir = self._get_commands_dir()
        command_file = commands_dir / f"{name}.md"

        if not command_file.exists():
            logger.warning(f"Command not found: {name}")
            return None

        try:
            content = command_file.read_text(encoding="utf-8")
            return self._parse_command_file(content, command_file)
        except Exception as e:
            logger.error(f"Failed to load command {name}: {e}")
            return None

    def load_all_commands(self) -> dict[str, CommandDefinition]:
        """Load all available commands."""
        if self._loaded:
            return self._commands

        for name in self.discover_commands():
            command = self.load_command(name)
            if command:
                self._commands[name] = command

        self._loaded = True
        return self._commands

    def _parse_command_file(self, content: str, path: Path) -> CommandDefinition:
        """Parse a command file into a CommandDefinition."""
        metadata, body = self._parse_frontmatter(content)

        name = metadata.get("name", path.stem)
        description = metadata.get("description", "")

        arguments = []
        for arg in metadata.get("arguments", []):
            if isinstance(arg, str):
                required = not arg.endswith(" (optional)")
                arg_name = arg.replace(" (optional)", "")
                arguments.append(CommandArgument(name=arg_name, required=required))
            elif isinstance(arg, dict):
                arguments.append(
                    CommandArgument(
                        name=arg.get("name", ""),
                        required=arg.get("required", True),
                        default=arg.get("default"),
                        description=arg.get("description", ""),
                    )
                )

        steps = self._parse_steps(body)

        return CommandDefinition(
            name=name,
            description=description,
            arguments=arguments,
            steps=steps,
            path=path,
        )

    def _parse_frontmatter(self, content: str) -> tuple[dict, str]:
        """Parse YAML frontmatter from content."""
        if not content.startswith("---"):
            return {}, content

        parts = content.split("---", 2)
        if len(parts) < 3:
            return {}, content

        frontmatter = parts[1].strip()
        body = parts[2].strip()

        try:
            metadata = yaml.safe_load(frontmatter) or {}
            return metadata, body
        except yaml.YAMLError as e:
            logger.warning(f"Invalid YAML frontmatter: {e}")
            return {}, body

    def _parse_steps(self, body: str) -> list[dict[str, Any]]:
        """Parse RUN and READ directives from body."""
        steps = []

        for line in body.splitlines():
            line = line.strip()

            if line.startswith("RUN "):
                steps.append(
                    {
                        "type": "run",
                        "command": line[4:].strip(),
                    }
                )
            elif line.startswith("READ "):
                steps.append(
                    {
                        "type": "read",
                        "path": line[5:].strip(),
                    }
                )
            elif line.startswith("ASYNC "):
                steps.append(
                    {
                        "type": "async",
                        "command": line[6:].strip(),
                    }
                )
            elif line.startswith("IF "):
                match = re.match(r"IF\s+\$(\w+)\s+(.+)", line)
                if match:
                    steps.append(
                        {
                            "type": "condition",
                            "variable": match.group(1),
                            "command": match.group(2).strip(),
                        }
                    )

        return steps

    def get_command(self, name: str) -> CommandDefinition | None:
        """Get a loaded command by name."""
        if not self._loaded:
            self.load_all_commands()
        return self._commands.get(name)

    def list_commands(self) -> list[CommandDefinition]:
        """List all loaded commands."""
        if not self._loaded:
            self.load_all_commands()
        return list(self._commands.values())


class CommandExecutor:
    """
    Executes command definitions.

    Handles variable substitution, step execution, and result aggregation.
    """

    def __init__(self, project_dir: Path | None = None, timeout: float = 60.0):
        self.project_dir = project_dir or Path.cwd()
        self.timeout = timeout
        self._cache: dict[str, str] = {}

    async def execute(
        self,
        command: CommandDefinition,
        arguments: dict[str, str],
        use_cache: bool = True,
    ) -> CommandResult:
        """
        Execute a command with given arguments.

        Args:
            command: Command definition to execute
            arguments: Variable values for substitution
            use_cache: Whether to cache step results

        Returns:
            CommandResult with outputs and status
        """
        import time

        start_time = time.time()
        result = CommandResult(command_name=command.name)

        validation_error = self._validate_arguments(command, arguments)
        if validation_error:
            result.errors.append(validation_error)
            result.success = False
            return result

        args = self._prepare_arguments(command, arguments)

        for i, step in enumerate(command.steps):
            try:
                step_result = await self._execute_step(step, args, use_cache)
                result.step_results.append(step_result)

                if step_result.get("output"):
                    result.outputs.append(step_result["output"])
                if step_result.get("error"):
                    result.errors.append(step_result["error"])
                if not step_result.get("success", True):
                    logger.warning(f"Step {i} failed in command {command.name}")

            except Exception as e:
                result.errors.append(f"Step {i} failed: {str(e)}")
                logger.error(f"Error in step {i} of command {command.name}: {e}")

        result.success = len(result.errors) == 0
        result.duration = time.time() - start_time

        return result

    def _validate_arguments(
        self,
        command: CommandDefinition,
        arguments: dict[str, str],
    ) -> str | None:
        """Validate that all required arguments are provided."""
        for arg in command.arguments:
            if arg.required and arg.name not in arguments:
                if arg.default is None:
                    return f"Missing required argument: {arg.name}"
        return None

    def _prepare_arguments(
        self,
        command: CommandDefinition,
        arguments: dict[str, str],
    ) -> dict[str, str]:
        """Prepare arguments with defaults."""
        args = {}

        for arg in command.arguments:
            if arg.name in arguments:
                args[arg.name] = arguments[arg.name]
            elif arg.default is not None:
                args[arg.name] = arg.default
            else:
                args[arg.name] = ""

        args["PROJECT_DIR"] = str(self.project_dir)

        return args

    async def _execute_step(
        self,
        step: dict[str, Any],
        args: dict[str, str],
        use_cache: bool,
    ) -> dict[str, Any]:
        """Execute a single step."""
        step_type = step.get("type", "run")

        if step_type == "run":
            return await self._execute_run(step, args, use_cache)
        elif step_type == "read":
            return await self._execute_read(step, args)
        elif step_type == "async":
            return await self._execute_async(step, args)
        elif step_type == "condition":
            return await self._execute_condition(step, args)

        return {"success": False, "error": f"Unknown step type: {step_type}"}

    def _substitute(self, template: str, args: dict[str, str]) -> str:
        """Substitute variables in a template string."""
        result = template

        for key, value in args.items():
            result = result.replace(f"${key}", str(value))
            result = result.replace(f"${{{key}}}", str(value))

        return result

    async def _execute_run(
        self,
        step: dict[str, Any],
        args: dict[str, str],
        use_cache: bool,
    ) -> dict[str, Any]:
        """Execute a RUN command."""
        command = self._substitute(step["command"], args)

        cache_key = f"run:{command}"
        if use_cache and cache_key in self._cache:
            return {"success": True, "output": self._cache[cache_key], "cached": True}

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_dir,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout,
            )

            output = stdout.decode("utf-8", errors="replace")
            error = stderr.decode("utf-8", errors="replace")

            if proc.returncode == 0:
                if use_cache:
                    self._cache[cache_key] = output
                return {"success": True, "output": output.strip()}
            else:
                return {
                    "success": False,
                    "output": output.strip(),
                    "error": error.strip() or f"Exit code: {proc.returncode}",
                }

        except asyncio.TimeoutError:
            return {"success": False, "error": f"Command timed out after {self.timeout}s"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_read(
        self,
        step: dict[str, Any],
        args: dict[str, str],
    ) -> dict[str, Any]:
        """Execute a READ file directive."""
        path_str = self._substitute(step["path"], args)
        path = self.project_dir / path_str

        if not path.exists():
            return {"success": False, "error": f"File not found: {path}"}

        try:
            content = path.read_text(encoding="utf-8")
            return {"success": True, "output": content}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_async(
        self,
        step: dict[str, Any],
        args: dict[str, str],
    ) -> dict[str, Any]:
        """Execute an ASYNC command (non-blocking)."""
        command = self._substitute(step["command"], args)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                cwd=self.project_dir,
            )

            return {"success": True, "output": "Started async process"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_condition(
        self,
        step: dict[str, Any],
        args: dict[str, str],
    ) -> dict[str, Any]:
        """Execute a conditional IF directive."""
        variable = step["variable"]
        command = step["command"]

        if args.get(variable):
            return await self._execute_run({"command": command}, args, True)

        return {"success": True, "output": "Skipped (condition not met)"}

    def clear_cache(self) -> None:
        """Clear the step result cache."""
        self._cache.clear()


class CommandManager:
    """
    High-level interface for the commands system.

    Integrates with the chat loop to provide /command functionality.
    """

    def __init__(self, project_dir: Path | None = None):
        self.loader = CommandLoader(project_dir)
        self.executor = CommandExecutor(project_dir)

    def list_commands(self) -> list[dict[str, Any]]:
        """List all available commands with their descriptions."""
        commands = self.loader.list_commands()
        return [
            {
                "name": cmd.name,
                "description": cmd.description,
                "arguments": [arg.name for arg in cmd.arguments],
            }
            for cmd in commands
        ]

    async def run_command(
        self,
        name: str,
        arguments: dict[str, str] | None = None,
    ) -> CommandResult:
        """Run a command by name with arguments."""
        command = self.loader.get_command(name)

        if not command:
            return CommandResult(
                command_name=name,
                success=False,
                errors=[f"Command not found: {name}"],
            )

        return await self.executor.execute(command, arguments or {})

    def get_command_help(self, name: str) -> str | None:
        """Get help text for a command."""
        command = self.loader.get_command(name)
        if not command:
            return None

        lines = [f"/{name}", f"  {command.description}"]

        if command.arguments:
            lines.append("  Arguments:")
            for arg in command.arguments:
                required = "required" if arg.required else "optional"
                default = f" (default: {arg.default})" if arg.default else ""
                lines.append(f"    ${arg.name} - {required}{default}")

        return "\n".join(lines)

    def create_command_template(
        self,
        name: str,
        description: str = "",
        arguments: list[str] | None = None,
    ) -> Path:
        """Create a new command template file."""
        commands_dir = self.loader._get_commands_dir()
        commands_dir.mkdir(parents=True, exist_ok=True)

        args_yaml = ""
        if arguments:
            args_yaml = "\narguments:\n" + "\n".join(f"  - {arg}" for arg in arguments)

        template = f"""---
name: {name}
description: {description or "Custom command"}{args_yaml}
---

## Instructions

Add your RUN and READ directives here:

RUN echo "Hello from {name} command"

## Examples

/{name}
"""

        command_file = commands_dir / f"{name}.md"
        command_file.write_text(template, encoding="utf-8")

        logger.info(f"Created command template: {command_file}")
        return command_file
