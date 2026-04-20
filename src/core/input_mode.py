"""
Input Mode System

Vim mode and enhanced input handling.

Features:
- Vim keybindings
- Multiline input
- Input history navigation
- Emacs keybindings (default)
"""

import logging
import readline
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class InputMode(Enum):
    """Input editing mode."""

    EMACS = "emacs"  # Default readline mode
    VI = "vi"  # Vim mode


@dataclass
class InputConfig:
    """Input configuration."""

    mode: InputMode = InputMode.EMACS
    history_size: int = 1000
    enable_multiline: bool = True
    enable_completion: bool = True
    prompt_toolkit: bool = False  # Use prompt_toolkit instead of readline


class InputManager:
    """
    Manages input modes and keybindings.

    Usage:
        input_mgr = InputManager()

        # Switch to vim mode
        input_mgr.set_mode(InputMode.VI)

        # Get input
        text = input_mgr.get_input(">>> ")

        # Multiline input
        text = input_mgr.get_multiline_input()
    """

    def __init__(self, config: InputConfig | None = None):
        """
        Initialize input manager.

        Args:
            config: Input configuration
        """
        self.config = config or InputConfig()
        self._mode = self.config.mode
        self._completers: list[Callable] = []

        # Initialize readline
        self._setup_readline()

    def _setup_readline(self):
        """Setup readline with current mode."""
        try:
            # Set editing mode
            if self._mode == InputMode.VI:
                readline.parse_and_bind("set editing-mode vi")
                readline.parse_and_bind("set show-mode-in-prompt on")
                readline.parse_and_bind("set vi-cmd-mode-string (cmd)")
                readline.parse_and_bind("set vi-ins-mode-string (ins)")
            else:
                readline.parse_and_bind("set editing-mode emacs")

            # Common bindings
            readline.parse_and_bind("tab: complete")
            readline.parse_and_bind(r'"\e[A": history-search-backward')
            readline.parse_and_bind(r'"\e[B": history-search-forward')

            # History size
            readline.set_history_length(self.config.history_size)

            # Setup completion
            if self.config.enable_completion:
                readline.set_completer(self._completer)
                readline.parse_and_bind("tab: complete")

            logger.debug("Input mode set to: %s", self._mode.value)

        except Exception as e:
            logger.warning("Failed to setup readline: %s", e)

    def _completer(self, text: str, state: int) -> str | None:
        """Readline completer function."""
        options = []

        for completer in self._completers:
            try:
                completions = completer(text)
                if completions:
                    options.extend(completions)
            except Exception:
                pass

        if state < len(options):
            return options[state]
        return None

    def add_completer(self, completer: Callable[[str], list[str]]):
        """
        Add a completion function.

        Args:
            completer: Function that takes text and returns completions
        """
        self._completers.append(completer)

    def get_mode(self) -> InputMode:
        """Get current input mode."""
        return self._mode

    def set_mode(self, mode: InputMode) -> bool:
        """
        Set input mode.

        Args:
            mode: Input mode to set

        Returns:
            True if mode was set successfully
        """
        self._mode = mode
        self._setup_readline()
        return True

    def toggle_mode(self) -> InputMode:
        """Toggle between vim and emacs mode."""
        if self._mode == InputMode.VI:
            self.set_mode(InputMode.EMACS)
        else:
            self.set_mode(InputMode.VI)
        return self._mode

    def get_input(self, prompt: str = "") -> str:
        """
        Get single line input.

        Args:
            prompt: Input prompt

        Returns:
            Input text
        """
        try:
            return input(prompt)
        except (EOFError, KeyboardInterrupt):
            return ""

    def get_multiline_input(
        self,
        prompt: str = "",
        continuation_prompt: str = "... ",
        end_marker: str = "",
    ) -> str:
        """
        Get multiline input.

        Supports:
        - Backslash continuation (\\)
        - Empty line to end
        - Ctrl+D to end

        Args:
            prompt: First line prompt
            continuation_prompt: Continuation prompt
            end_marker: Optional end marker (empty = blank line)

        Returns:
            Complete input text
        """
        lines = []
        current_prompt = prompt

        while True:
            try:
                line = input(current_prompt)

                # Check for end marker
                if end_marker and line.strip() == end_marker:
                    break

                # Check for blank line end
                if not end_marker and not line.strip() and lines:
                    break

                # Check for continuation
                if line.endswith("\\"):
                    lines.append(line[:-1])
                    current_prompt = continuation_prompt
                else:
                    lines.append(line)
                    # Only continue if explicitly continued
                    if not self.config.enable_multiline:
                        break
                    current_prompt = continuation_prompt

            except EOFError:
                break
            except KeyboardInterrupt:
                return ""

        return "\n".join(lines)

    def add_history(self, text: str):
        """Add text to history."""
        if text.strip():
            readline.add_history(text)

    def clear_history(self):
        """Clear history."""
        readline.clear_history()

    def get_history(self, limit: int = 50) -> list[str]:
        """Get recent history items."""
        history = []
        length = readline.get_current_history_length()
        start = max(0, length - limit)

        for i in range(start + 1, length + 1):
            item = readline.get_history_item(i)
            if item:
                history.append(item)

        return history

    def get_mode_indicator(self) -> str:
        """Get mode indicator for prompt."""
        if self._mode == InputMode.VI:
            return "[vim]"
        return ""


# Vim-specific keybindings reference
VIM_KEYBINDINGS = """
Vim Mode Keybindings:

NORMAL MODE (press Esc):
  h/j/k/l     - Move left/down/up/right
  w/b         - Move forward/backward word
  0/$         - Move to beginning/end of line
  gg/G        - Move to beginning/end of input
  i/a         - Insert before/after cursor
  I/A         - Insert at beginning/end of line
  x           - Delete character
  dd          - Delete line
  dw          - Delete word
  cw          - Change word
  yy          - Yank (copy) line
  p           - Paste
  u           - Undo
  /           - Search forward
  ?           - Search backward
  n/N         - Next/previous search result

INSERT MODE (press i):
  Type normally
  Esc         - Return to normal mode
  Ctrl+[      - Return to normal mode

COMMAND MODE (press :):
  :q          - Quit
  :w          - Save (submit)
"""


def get_vim_help() -> str:
    """Get vim keybindings help."""
    return VIM_KEYBINDINGS
