"""
Shell Security — shared safety logic for command execution.

Extracted from shell.py and run_unified.py to eliminate code duplication.
Both modules should import from here instead of maintaining their own copies.

Enhanced with additional evasion-resistance patterns (C1 fix).
"""

import logging
import os
import re
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ========================================
# Configuration
# ========================================


@dataclass
class ShellConfig:
    """Configuration for shell execution."""

    default_timeout: int = 30  # seconds
    max_timeout: int = 600  # 10 minutes
    max_output_size: int = 100_000  # characters
    shell: str = "/bin/bash"

    # Commands that are blocked for safety
    blocked_commands: list[str] = field(
        default_factory=lambda: [
            "rm -rf /",
            "rm -rf /*",
            "mkfs",
            "dd if=/dev/zero",
            ":(){:|:&};:",  # fork bomb
            "chmod -R 777 /",
            "chown -R",
            "> /dev/sda",
            "mv / ",
            "wget -O- | sh",
            "curl -s | sh",
        ]
    )

    # Commands that require confirmation (handled by HITL in main CLI)
    sensitive_patterns: list[str] = field(
        default_factory=lambda: [
            "rm -rf",
            "rm -r",
            "sudo",
            "chmod 777",
            "curl | bash",
            "wget | bash",
            "pip install",
            "npm install -g",
            "apt install",
            "apt remove",
            "systemctl",
            "service ",
            "kill -9",
            "pkill",
            "docker rm",
            "docker rmi",
        ]
    )

    # Dangerous base commands to detect via shlex parsing
    blocked_base_commands: list[str] = field(
        default_factory=lambda: [
            "mkfs",
            "fdisk",
            "parted",
            "wipefs",
            "shred",
            "halt",
            "poweroff",
            "reboot",
            "shutdown",
            "init",
            "telinit",
        ]
    )


DEFAULT_CONFIG = ShellConfig()


# ========================================
# Background Process Management
# ========================================


@dataclass
class BackgroundProcess:
    """Tracks a background process."""

    pid: int
    command: str
    start_time: float
    working_dir: str
    process: subprocess.Popen
    log_file: str | None = None  # Path to temp log file for cleanup


# Store for background processes
_background_processes: dict[int, BackgroundProcess] = {}
_process_lock = threading.Lock()


def register_background_process(
    proc: subprocess.Popen, command: str, working_dir: str, log_file: str | None = None
) -> int:
    """Register a background process for tracking."""
    with _process_lock:
        bp = BackgroundProcess(
            pid=proc.pid,
            command=command,
            start_time=time.time(),
            working_dir=working_dir,
            process=proc,
            log_file=log_file,
        )
        _background_processes[proc.pid] = bp
        return proc.pid


def cleanup_finished_processes() -> None:
    """Remove finished processes from tracking and clean up temp log files."""
    with _process_lock:
        finished = [
            pid for pid, bp in _background_processes.items() if bp.process.poll() is not None
        ]
        for pid in finished:
            bp = _background_processes[pid]
            if bp.log_file and os.path.exists(bp.log_file):
                try:
                    os.unlink(bp.log_file)
                except OSError:
                    pass
            del _background_processes[pid]


def get_background_processes() -> dict[int, BackgroundProcess]:
    """Return the background process registry (for direct access by tools)."""
    return _background_processes


def get_process_lock() -> threading.Lock:
    """Return the process lock (for synchronized access by tools)."""
    return _process_lock


# ========================================
# Command Validation
# ========================================


def is_command_blocked(command: str, config: ShellConfig = DEFAULT_CONFIG) -> bool:
    """Check if a command is blocked for safety.

    Enhanced with additional evasion-resistance:
    - $() and backtick command substitution detection
    - Prefix command bypass detection (env, command, nice, etc.)
    - Base64/hex decode piping detection
    """
    command_lower = command.lower().strip()
    normalized = command_lower.replace('"', "").replace("'", "").replace("\\", "")

    # Check against blocked command strings
    for blocked in config.blocked_commands:
        bl = blocked.lower()
        if bl in command_lower or bl in normalized:
            return True

    # Parse with shlex for structured analysis
    try:
        tokens = shlex.split(command_lower)
    except ValueError:
        tokens = command_lower.split()

    if tokens:
        base_cmd = os.path.basename(tokens[0])

        # Skip known prefix commands that can wrap dangerous commands
        prefix_commands = {"env", "command", "nice", "nohup", "time", "timeout", "strace"}
        effective_tokens = list(tokens)
        while effective_tokens and os.path.basename(effective_tokens[0]) in prefix_commands:
            effective_tokens.pop(0)
            # Also skip env's VAR=value arguments
            while effective_tokens and "=" in effective_tokens[0]:
                effective_tokens.pop(0)

        if effective_tokens:
            effective_base = os.path.basename(effective_tokens[0])
            if effective_base in config.blocked_base_commands:
                return True
        else:
            # All tokens were prefix commands, check the original base
            if base_cmd in config.blocked_base_commands:
                return True

        # Detect dangerous rm patterns
        if base_cmd == "rm" and any(t in tokens for t in ["-rf", "-fr"]):
            if any(t in ("/", "/*", "/.", "/..") for t in tokens):
                return True

    # Check multi-command chains
    for sep in [";", "&&", "||"]:
        if sep in command:
            return any(
                is_command_blocked(sub.strip(), config) for sub in command.split(sep) if sub.strip()
            )

    # Regex-based dangerous patterns (enhanced)
    dangerous_patterns = [
        r">\s*/dev/sd",  # Write to disk device
        r"\|\s*(bash|sh|zsh)\b",  # Pipe to shell
        r"eval\s*[\s(]",  # eval execution (with space or parens)
        r"exec\s+\d*[<>]",  # exec redirection
        r"\$\([^)]*(?:rm|mkfs|dd|shred)",  # $() command substitution with dangerous cmds
        r"`[^`]*(?:rm|mkfs|dd|shred)",  # backtick substitution with dangerous cmds
        r"base64\s+(?:-d|--decode)\s*\|",  # base64 decode + pipe (obfuscation)
        r"xxd\s+-r\s*\|",  # hex decode + pipe (obfuscation)
        r"python[23]?\s+-c\s+[\"'].*(?:subprocess|os\.system|shutil\.rmtree)",  # Python one-liners
    ]
    return any(re.search(p, command_lower) for p in dangerous_patterns)


def check_git_safety(command: str) -> str | None:
    """Check git commands for dangerous operations.

    Returns error message if blocked, None if safe.
    """
    command_stripped = command.strip()

    # Only check git commands
    if not command_stripped.startswith("git "):
        return None

    # Parse command tokens safely
    try:
        tokens = shlex.split(command_stripped)
    except ValueError:
        tokens = command_stripped.split()

    if len(tokens) < 2:
        return None

    subcommand = tokens[1]

    # --- Force push protection ---
    if subcommand == "push":
        for token in tokens[2:]:
            if token in ("--force", "-f", "--force-with-lease"):
                for t in tokens[2:]:
                    if t in ("main", "master", "origin/main", "origin/master"):
                        return (
                            "Error: Force push to main/master is blocked for safety. "
                            "This could overwrite shared history."
                        )
                return (
                    "Error: Force push (--force) is blocked for safety. "
                    "Use --force-with-lease if you must, or ask the user to confirm."
                )

    # --- Destructive reset protection ---
    if subcommand == "reset":
        if "--hard" in tokens:
            return (
                "Warning: 'git reset --hard' will discard all uncommitted changes. "
                "This is blocked for safety. Use 'git stash' first to preserve changes."
            )

    # --- Checkout discard protection ---
    if subcommand == "checkout":
        if "." in tokens:
            return (
                "Error: 'git checkout .' will discard all uncommitted changes. "
                "Use 'git stash' to preserve them first."
            )

    # --- Clean protection ---
    if subcommand == "clean":
        if "-f" in tokens or "--force" in tokens:
            return (
                "Error: 'git clean -f' will permanently delete untracked files. "
                "This is blocked for safety."
            )

    # --- Branch delete protection ---
    if subcommand == "branch":
        if "-D" in tokens:
            for t in tokens[2:]:
                if t in ("main", "master"):
                    return "Error: Deleting main/master branch is blocked for safety."

    # --- Hooks bypass protection ---
    if "--no-verify" in tokens:
        return (
            "Error: --no-verify bypasses pre-commit hooks. "
            "This is blocked unless explicitly authorized."
        )

    return None


def is_command_sensitive(command: str, config: ShellConfig = DEFAULT_CONFIG) -> bool:
    """Check if a command requires extra confirmation."""
    command_lower = command.lower()
    return any(pattern.lower() in command_lower for pattern in config.sensitive_patterns)


def truncate_output(output: str, max_size: int = DEFAULT_CONFIG.max_output_size) -> str:
    """Truncate output if it exceeds max size, preserving head and tail by lines."""
    if len(output) <= max_size:
        return output

    if max_size <= 0:
        return ""

    lines = output.splitlines(keepends=True)
    total_lines = len(lines)

    # Allocate 40% head, 60% tail (errors/results usually at end)
    head_budget = int(max_size * 0.4)
    tail_budget = max_size - head_budget

    head_lines: list[str] = []
    head_chars = 0
    for line in lines:
        if head_chars + len(line) > head_budget:
            break
        head_lines.append(line)
        head_chars += len(line)

    tail_lines: list[str] = []
    tail_chars = 0
    for line in reversed(lines):
        if tail_chars + len(line) > tail_budget:
            break
        tail_lines.insert(0, line)
        tail_chars += len(line)

    # Guard against overlap when few long lines exist
    if len(head_lines) + len(tail_lines) >= total_lines:
        return output

    omitted = total_lines - len(head_lines) - len(tail_lines)
    if not head_lines and not tail_lines:
        head_chars = max(1, max_size // 3)
        tail_chars = max(1, max_size // 3)
        head = output[:head_chars]
        tail = output[-tail_chars:]
        separator = (
            f"\n\n... [truncated: {omitted} lines omitted, {len(output)} chars total] ...\n\n"
        )
        return head + separator + tail

    separator = f"\n\n... [truncated: {omitted} lines omitted, {len(output)} chars total] ...\n\n"

    return "".join(head_lines) + separator + "".join(tail_lines)
