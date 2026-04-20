"""
Hooks System - Lifecycle Event Hooks

Provides a hook system for intercepting and customizing agent behavior
at various lifecycle points.

Supported Events:
- SessionStart: Session begins or resumes
- SessionEnd: Session terminates
- UserPromptSubmit: User submits a prompt
- PreToolUse: Before tool execution
- PostToolUse: After tool completes
- Stop: Agent finishes responding
- PreCompact: Before context compaction
"""

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class HookEvent(Enum):
    """Hook event types."""

    SESSION_START = "SessionStart"
    SESSION_END = "SessionEnd"
    USER_PROMPT_SUBMIT = "UserPromptSubmit"
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    STOP = "Stop"
    PRE_COMPACT = "PreCompact"
    NOTIFICATION = "Notification"


class HookDecision(Enum):
    """Hook decision for controllable events."""

    ALLOW = "allow"  # Proceed normally
    DENY = "deny"  # Block the action
    ASK = "ask"  # Ask user for permission
    MODIFY = "modify"  # Allow with modifications


@dataclass
class HookResult:
    """Result from a hook execution."""

    success: bool
    decision: HookDecision = HookDecision.ALLOW
    reason: str = ""
    modified_input: dict[str, Any] | None = None
    additional_context: str = ""
    continue_processing: bool = True
    output: str = ""
    duration: float = 0


@dataclass
class HookDefinition:
    """Definition of a hook."""

    event: HookEvent
    matcher: str | None = None  # Tool name pattern for tool hooks (regex supported)
    command: str | None = None  # Shell command to run
    callback: Callable | None = None  # Python callback
    timeout: float = 60  # Timeout in seconds
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event.value,
            "matcher": self.matcher,
            "command": self.command,
            "timeout": self.timeout,
            "enabled": self.enabled,
        }


@dataclass
class HookContext:
    """Context passed to hooks."""

    event: HookEvent
    session_id: str
    project_dir: str
    permission_mode: str
    timestamp: float = field(default_factory=time.time)

    # Event-specific fields
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output: Any | None = None
    user_prompt: str | None = None
    message_count: int = 0
    stop_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "hook_event_name": self.event.value,
            "session_id": self.session_id,
            "cwd": self.project_dir,
            "permission_mode": self.permission_mode,
            "timestamp": self.timestamp,
        }

        if self.tool_name:
            data["tool_name"] = self.tool_name
        if self.tool_input:
            data["tool_input"] = self.tool_input
        if self.tool_output is not None:
            data["tool_output"] = self.tool_output
        if self.user_prompt:
            data["prompt"] = self.user_prompt
        if self.message_count:
            data["message_count"] = self.message_count
        if self.stop_reason:
            data["stop_reason"] = self.stop_reason

        return data


class HookManager:
    """
    Manages lifecycle hooks.

    Usage:
        mgr = HookManager(project_dir="/path/to/project")

        # Register a Python callback
        mgr.register(
            HookEvent.PRE_TOOL_USE,
            callback=my_validator,
            matcher="Bash"
        )

        # Register a shell command hook
        mgr.register(
            HookEvent.POST_TOOL_USE,
            command="./scripts/format.sh",
            matcher="Write|Edit"
        )

        # Trigger hooks
        result = await mgr.trigger(HookEvent.PRE_TOOL_USE, context)

        # Load hooks from config file
        mgr.load_from_file(".agent/hooks.json")
    """

    def __init__(
        self,
        project_dir: str | Path = ".",
        session_id: str = "",
        permission_mode: str = "default",
    ):
        """
        Initialize hook manager.

        Args:
            project_dir: Project directory path
            session_id: Current session ID
            permission_mode: Current permission mode
        """
        self.project_dir = str(Path(project_dir).resolve())
        self.session_id = session_id
        self.permission_mode = permission_mode
        self._hooks: dict[HookEvent, list[HookDefinition]] = {event: [] for event in HookEvent}
        self._env: dict[str, str] = {
            "DORAEMON_PROJECT_DIR": self.project_dir,
        }

    def register(
        self,
        event: HookEvent,
        command: str | None = None,
        callback: Callable | None = None,
        matcher: str | None = None,
        timeout: float = 60,
    ):
        """
        Register a hook.

        Args:
            event: Hook event type
            command: Shell command to execute
            callback: Python callback function
            matcher: Pattern for tool name matching (PreToolUse/PostToolUse only)
            timeout: Execution timeout in seconds
        """
        if not command and not callback:
            raise ValueError("Either command or callback must be provided")

        hook = HookDefinition(
            event=event,
            matcher=matcher,
            command=command,
            callback=callback,
            timeout=timeout,
        )

        self._hooks[event].append(hook)
        logger.debug("Registered hook for %s: %s", event.value, command or callback)

    def unregister(self, event: HookEvent, matcher: str | None = None):
        """Unregister hooks for an event."""
        if matcher:
            self._hooks[event] = [h for h in self._hooks[event] if h.matcher != matcher]
        else:
            self._hooks[event] = []

    async def trigger(
        self,
        event: HookEvent,
        tool_name: str | None = None,
        tool_input: dict[str, Any] | None = None,
        tool_output: Any = None,
        user_prompt: str | None = None,
        message_count: int = 0,
        stop_reason: str | None = None,
    ) -> HookResult:
        """
        Trigger hooks for an event.

        Args:
            event: Event type
            tool_name: Tool name (for tool hooks)
            tool_input: Tool input arguments (for PreToolUse)
            tool_output: Tool output (for PostToolUse)
            user_prompt: User prompt (for UserPromptSubmit)
            message_count: Current message count
            stop_reason: Reason for stopping (for Stop hook)

        Returns:
            Aggregated HookResult
        """
        context = HookContext(
            event=event,
            session_id=self.session_id,
            project_dir=self.project_dir,
            permission_mode=self.permission_mode,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            user_prompt=user_prompt,
            message_count=message_count,
            stop_reason=stop_reason,
        )

        # Get matching hooks
        hooks = self._get_matching_hooks(event, tool_name)

        if not hooks:
            return HookResult(success=True)

        # Run hooks in parallel
        results = await asyncio.gather(
            *[self._run_hook(hook, context) for hook in hooks],
            return_exceptions=True,
        )

        # Aggregate results
        return self._aggregate_results(results)

    def _get_matching_hooks(self, event: HookEvent, tool_name: str | None) -> list[HookDefinition]:
        """Get hooks that match the event and tool name."""
        import re

        matching = []

        for hook in self._hooks[event]:
            if not hook.enabled:
                continue

            # Check matcher for tool events
            if hook.matcher and tool_name:
                # Support regex matching
                try:
                    if not re.match(hook.matcher, tool_name):
                        continue
                except re.error:
                    # Fallback to exact match
                    if hook.matcher != tool_name:
                        continue
            elif hook.matcher and not tool_name:
                # Hook requires tool match but no tool provided
                continue

            matching.append(hook)

        return matching

    async def _run_hook(self, hook: HookDefinition, context: HookContext) -> HookResult:
        """Run a single hook."""
        start_time = time.time()

        try:
            if hook.callback:
                result = await self._run_callback(hook.callback, context, hook.timeout)
            elif hook.command:
                result = await self._run_command(hook.command, context, hook.timeout)
            else:
                return HookResult(success=False, reason="No hook handler defined")

            result.duration = time.time() - start_time
            return result

        except asyncio.TimeoutError:
            return HookResult(
                success=False,
                reason=f"Hook timed out after {hook.timeout}s",
                duration=time.time() - start_time,
            )
        except Exception as e:
            logger.error("Hook execution failed: %s", e)
            return HookResult(
                success=False,
                reason=str(e),
                duration=time.time() - start_time,
            )

    async def _run_callback(
        self,
        callback: Callable,
        context: HookContext,
        timeout: float,
    ) -> HookResult:
        """Run a Python callback hook."""
        result = callback(context)

        # Handle async callbacks
        if asyncio.iscoroutine(result):
            result = await asyncio.wait_for(result, timeout=timeout)

        # Normalize result
        if isinstance(result, HookResult):
            return result
        elif isinstance(result, dict):
            return HookResult(
                success=result.get("success", True),
                decision=HookDecision(result.get("decision", "allow")),
                reason=result.get("reason", ""),
                modified_input=result.get("modified_input"),
                additional_context=result.get("additional_context", ""),
                continue_processing=result.get("continue", True),
            )
        elif isinstance(result, bool):
            return HookResult(
                success=result,
                decision=HookDecision.ALLOW if result else HookDecision.DENY,
            )
        else:
            return HookResult(success=True, output=str(result) if result else "")

    async def _run_command(
        self,
        command: str,
        context: HookContext,
        timeout: float,
    ) -> HookResult:
        """Run a shell command hook."""
        # Prepare environment
        env = {**self._env}
        proc = None

        # Prepare input as JSON
        input_json = json.dumps(context.to_dict(), ensure_ascii=False)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_dir,
                env={**dict(__import__("os").environ), **env},
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=input_json.encode("utf-8")),
                timeout=timeout,
            )

            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

            # Parse result based on exit code
            if proc.returncode == 0:
                # Try to parse JSON output
                try:
                    output_data = json.loads(stdout_str)
                    return HookResult(
                        success=True,
                        decision=HookDecision(output_data.get("decision", "allow")),
                        reason=output_data.get("reason", ""),
                        modified_input=output_data.get("modified_input"),
                        additional_context=output_data.get("additional_context", ""),
                        continue_processing=output_data.get("continue", True),
                        output=stdout_str,
                    )
                except json.JSONDecodeError:
                    # Plain text output
                    return HookResult(
                        success=True,
                        output=stdout_str,
                        additional_context=stdout_str,
                    )

            elif proc.returncode == 2:
                # Blocking error - deny the action
                return HookResult(
                    success=False,
                    decision=HookDecision.DENY,
                    reason=stderr_str or stdout_str,
                    continue_processing=False,
                )

            else:
                # Non-blocking error
                logger.warning("Hook command exited with code %s: %s", proc.returncode, stderr_str)
                return HookResult(
                    success=False,
                    reason=stderr_str,
                    continue_processing=True,
                )

        except asyncio.TimeoutError:
            if proc and proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=1)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
            raise
        except Exception as e:
            if proc and proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=1)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
            return HookResult(success=False, reason=str(e))
        finally:
            transport = getattr(proc, "_transport", None) if proc else None
            if transport is not None:
                try:
                    transport.close()
                except Exception:
                    pass

    def _aggregate_results(self, results: list) -> HookResult:
        """Aggregate multiple hook results."""
        aggregated = HookResult(success=True)
        outputs = []
        additional_contexts = []

        for result in results:
            if isinstance(result, Exception):
                logger.error("Hook raised exception: %s", result)
                continue

            if not isinstance(result, HookResult):
                continue

            # Any failure marks overall as failed
            if not result.success:
                aggregated.success = False
                aggregated.reason = result.reason

            # Most restrictive decision wins
            if result.decision == HookDecision.DENY:
                aggregated.decision = HookDecision.DENY
                aggregated.reason = result.reason
            elif result.decision == HookDecision.ASK and aggregated.decision != HookDecision.DENY:
                aggregated.decision = HookDecision.ASK

            # Any hook can stop processing
            if not result.continue_processing:
                aggregated.continue_processing = False

            # Collect modified inputs (last one wins)
            if result.modified_input:
                aggregated.modified_input = result.modified_input

            # Collect outputs and context
            if result.output:
                outputs.append(result.output)
            if result.additional_context:
                additional_contexts.append(result.additional_context)

            # Sum durations
            aggregated.duration += result.duration

        aggregated.output = "\n".join(outputs)
        aggregated.additional_context = "\n".join(additional_contexts)

        return aggregated

    def load_from_file(self, path: str | Path):
        """
        Load hooks from a JSON configuration file.

        File format:
        {
            "hooks": {
                "PreToolUse": [
                    {
                        "matcher": "Bash",
                        "hooks": [
                            {"type": "command", "command": "./validate.sh", "timeout": 30}
                        ]
                    }
                ],
                "PostToolUse": [...]
            }
        }
        """
        path = Path(path)
        if not path.exists():
            return

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            hooks_config = data.get("hooks", {})

            for event_name, matchers in hooks_config.items():
                try:
                    event = HookEvent(event_name)
                except ValueError:
                    logger.warning("Unknown hook event: %s", event_name)
                    continue

                for matcher_config in matchers:
                    matcher = matcher_config.get("matcher")

                    for hook_def in matcher_config.get("hooks", []):
                        hook_type = hook_def.get("type", "command")

                        if hook_type == "command":
                            self.register(
                                event=event,
                                command=hook_def.get("command"),
                                matcher=matcher,
                                timeout=hook_def.get("timeout", 60),
                            )

            logger.info("Loaded hooks from %s", path)

        except Exception as e:
            logger.error("Failed to load hooks from %s: %s", path, e)

    def save_to_file(self, path: str | Path):
        """Save current hooks to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config = {"hooks": {}}

        for event, hooks in self._hooks.items():
            if not hooks:
                continue

            # Group by matcher
            matchers: dict[str | None, list[dict]] = {}
            for hook in hooks:
                if hook.matcher not in matchers:
                    matchers[hook.matcher] = []

                hook_def = {"type": "command" if hook.command else "callback"}
                if hook.command:
                    hook_def["command"] = hook.command
                if hook.timeout != 60:
                    hook_def["timeout"] = hook.timeout

                matchers[hook.matcher].append(hook_def)

            config["hooks"][event.value] = [
                {"matcher": matcher, "hooks": hook_list} for matcher, hook_list in matchers.items()
            ]

        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        logger.info("Saved hooks to %s", path)

    def get_hooks_summary(self) -> dict[str, list[dict]]:
        """Get a summary of registered hooks."""
        summary = {}

        for event, hooks in self._hooks.items():
            if hooks:
                summary[event.value] = [h.to_dict() for h in hooks]

        return summary
