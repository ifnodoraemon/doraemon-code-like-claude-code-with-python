"""
Theme System

Color themes for the CLI interface.

Features:
- Multiple built-in themes
- Custom theme support
- Theme persistence
- Syntax highlighting themes
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.paths import theme_path

logger = logging.getLogger(__name__)


@dataclass
class ThemeColors:
    """Theme color definitions."""

    # Primary colors
    primary: str = "blue"
    secondary: str = "cyan"
    accent: str = "magenta"

    # UI elements
    prompt: str = "green"
    prompt_mode: str = "yellow"
    border: str = "blue"
    panel_title: str = "bold blue"

    # Messages
    user_message: str = "green"
    assistant_message: str = "purple"
    system_message: str = "dim"
    error_message: str = "red"
    warning_message: str = "yellow"
    success_message: str = "green"

    # Code
    code_block: str = "cyan"
    code_inline: str = "yellow"

    # Status
    status_ok: str = "green"
    status_warning: str = "yellow"
    status_error: str = "red"

    # Misc
    dim: str = "dim"
    muted: str = "dim white"

    def to_dict(self) -> dict[str, str]:
        return {
            "primary": self.primary,
            "secondary": self.secondary,
            "accent": self.accent,
            "prompt": self.prompt,
            "prompt_mode": self.prompt_mode,
            "border": self.border,
            "panel_title": self.panel_title,
            "user_message": self.user_message,
            "assistant_message": self.assistant_message,
            "system_message": self.system_message,
            "error_message": self.error_message,
            "warning_message": self.warning_message,
            "success_message": self.success_message,
            "code_block": self.code_block,
            "code_inline": self.code_inline,
            "status_ok": self.status_ok,
            "status_warning": self.status_warning,
            "status_error": self.status_error,
            "dim": self.dim,
            "muted": self.muted,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "ThemeColors":
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


@dataclass
class Theme:
    """A complete theme definition."""

    name: str
    description: str
    colors: ThemeColors
    syntax_theme: str = "monokai"  # Rich syntax theme

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "colors": self.colors.to_dict(),
            "syntax_theme": self.syntax_theme,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Theme":
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            colors=ThemeColors.from_dict(data.get("colors", {})),
            syntax_theme=data.get("syntax_theme", "monokai"),
        )


# Built-in themes
BUILTIN_THEMES: dict[str, Theme] = {
    "default": Theme(
        name="default",
        description="Default Doraemon theme",
        colors=ThemeColors(),
        syntax_theme="monokai",
    ),
    "dark": Theme(
        name="dark",
        description="Dark theme with high contrast",
        colors=ThemeColors(
            primary="bright_blue",
            secondary="bright_cyan",
            accent="bright_magenta",
            prompt="bright_green",
            prompt_mode="bright_yellow",
            border="bright_blue",
            panel_title="bold bright_blue",
            user_message="bright_green",
            assistant_message="bright_magenta",
            code_block="bright_cyan",
        ),
        syntax_theme="monokai",
    ),
    "light": Theme(
        name="light",
        description="Light theme for bright terminals",
        colors=ThemeColors(
            primary="blue",
            secondary="dark_cyan",
            accent="dark_magenta",
            prompt="dark_green",
            prompt_mode="dark_orange",
            border="blue",
            panel_title="bold blue",
            user_message="dark_green",
            assistant_message="dark_magenta",
            dim="grey50",
            muted="grey50",
        ),
        syntax_theme="github-dark",
    ),
    "ocean": Theme(
        name="ocean",
        description="Ocean-inspired blue theme",
        colors=ThemeColors(
            primary="deep_sky_blue1",
            secondary="dark_turquoise",
            accent="medium_purple1",
            prompt="spring_green1",
            prompt_mode="gold1",
            border="deep_sky_blue1",
            panel_title="bold deep_sky_blue1",
            user_message="spring_green1",
            assistant_message="medium_purple1",
            code_block="dark_turquoise",
        ),
        syntax_theme="dracula",
    ),
    "forest": Theme(
        name="forest",
        description="Nature-inspired green theme",
        colors=ThemeColors(
            primary="dark_green",
            secondary="green",
            accent="dark_olive_green1",
            prompt="spring_green3",
            prompt_mode="gold3",
            border="dark_green",
            panel_title="bold dark_green",
            user_message="spring_green3",
            assistant_message="dark_olive_green1",
            code_block="green",
        ),
        syntax_theme="monokai",
    ),
    "sunset": Theme(
        name="sunset",
        description="Warm sunset colors",
        colors=ThemeColors(
            primary="orange1",
            secondary="dark_orange",
            accent="hot_pink",
            prompt="gold1",
            prompt_mode="orange_red1",
            border="orange1",
            panel_title="bold orange1",
            user_message="gold1",
            assistant_message="hot_pink",
            code_block="dark_orange",
        ),
        syntax_theme="material",
    ),
    "minimal": Theme(
        name="minimal",
        description="Minimal monochrome theme",
        colors=ThemeColors(
            primary="white",
            secondary="grey70",
            accent="grey50",
            prompt="white",
            prompt_mode="grey70",
            border="grey50",
            panel_title="bold white",
            user_message="white",
            assistant_message="grey70",
            code_block="grey85",
            dim="grey35",
            muted="grey35",
        ),
        syntax_theme="native",
    ),
}


class ThemeManager:
    """
    Manages themes.

    Usage:
        themes = ThemeManager()

        # List themes
        available = themes.list_themes()

        # Get current theme
        current = themes.get_current_theme()

        # Switch theme
        themes.set_theme("ocean")

        # Get color for element
        color = themes.get_color("prompt")
    """

    def __init__(self, config_path: Path | None = None):
        """
        Initialize theme manager.

        Args:
            config_path: Path to theme config file
        """
        self._config_path = config_path or theme_path()
        self._current_theme: str = "default"
        self._custom_themes: dict[str, Theme] = {}

        # Load saved theme
        self._load_config()

    def _load_config(self):
        """Load theme configuration."""
        if self._config_path.exists():
            try:
                data = json.loads(self._config_path.read_text(encoding="utf-8"))
                self._current_theme = data.get("current", "default")

                # Load custom themes
                for name, theme_data in data.get("custom_themes", {}).items():
                    self._custom_themes[name] = Theme.from_dict(theme_data)

            except Exception as e:
                logger.warning(f"Failed to load theme config: {e}")

    def _save_config(self):
        """Save theme configuration."""
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "current": self._current_theme,
                "custom_themes": {
                    name: theme.to_dict() for name, theme in self._custom_themes.items()
                },
            }

            self._config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        except Exception as e:
            logger.warning(f"Failed to save theme config: {e}")

    def list_themes(self) -> list[Theme]:
        """List all available themes."""
        themes = list(BUILTIN_THEMES.values())
        themes.extend(self._custom_themes.values())
        return themes

    def get_theme(self, name: str) -> Theme | None:
        """Get a theme by name."""
        if name in BUILTIN_THEMES:
            return BUILTIN_THEMES[name]
        return self._custom_themes.get(name)

    def get_current_theme(self) -> Theme:
        """Get the current theme."""
        theme = self.get_theme(self._current_theme)
        if not theme:
            theme = BUILTIN_THEMES["default"]
        return theme

    def set_theme(self, name: str) -> bool:
        """
        Set the current theme.

        Args:
            name: Theme name

        Returns:
            True if theme was set
        """
        if name not in BUILTIN_THEMES and name not in self._custom_themes:
            logger.error(f"Theme not found: {name}")
            return False

        self._current_theme = name
        self._save_config()
        logger.info(f"Theme set to: {name}")
        return True

    def get_color(self, element: str) -> str:
        """
        Get color for a UI element.

        Args:
            element: Element name (e.g., "prompt", "error_message")

        Returns:
            Color string
        """
        theme = self.get_current_theme()
        return getattr(theme.colors, element, "white")

    def get_syntax_theme(self) -> str:
        """Get the syntax highlighting theme name."""
        return self.get_current_theme().syntax_theme

    def add_custom_theme(self, theme: Theme) -> bool:
        """
        Add a custom theme.

        Args:
            theme: Theme to add

        Returns:
            True if added
        """
        if theme.name in BUILTIN_THEMES:
            logger.error(f"Cannot override builtin theme: {theme.name}")
            return False

        self._custom_themes[theme.name] = theme
        self._save_config()
        return True

    def remove_custom_theme(self, name: str) -> bool:
        """Remove a custom theme."""
        if name in self._custom_themes:
            del self._custom_themes[name]
            if self._current_theme == name:
                self._current_theme = "default"
            self._save_config()
            return True
        return False

    def format_theme_list(self) -> str:
        """Format theme list for display."""
        lines = ["", "Available Themes:", ""]

        for theme in self.list_themes():
            current = "→ " if theme.name == self._current_theme else "  "
            builtin = "(builtin)" if theme.name in BUILTIN_THEMES else "(custom)"
            lines.append(f"{current}{theme.name}: {theme.description} {builtin}")

        return "\n".join(lines)

    def preview_theme(self, name: str) -> str:
        """
        Generate a preview of a theme.

        Args:
            name: Theme name

        Returns:
            Preview text with colors
        """
        theme = self.get_theme(name)
        if not theme:
            return f"Theme not found: {name}"

        c = theme.colors
        lines = [
            "",
            f"[{c.panel_title}]━━━ {theme.name} ━━━[/{c.panel_title}]",
            f"[{c.dim}]{theme.description}[/{c.dim}]",
            "",
            f"[{c.user_message}]User message example[/{c.user_message}]",
            f"[{c.assistant_message}]Assistant response example[/{c.assistant_message}]",
            f"[{c.error_message}]Error message example[/{c.error_message}]",
            f"[{c.warning_message}]Warning message example[/{c.warning_message}]",
            f"[{c.success_message}]Success message example[/{c.success_message}]",
            f"[{c.code_block}]Code block example[/{c.code_block}]",
            "",
        ]

        return "\n".join(lines)
