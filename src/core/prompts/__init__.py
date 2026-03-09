"""
Doraemon Code — System Prompts Package

Modular, composable prompt system for different agent modes.

Architecture:
    _common.py  → Reusable XML prompt segments (personality, retry, etc.)
    plan.py     → Plan mode prompt (read-only analysis + structured planning)
    build.py    → Build mode prompt (task-driven implementation)
    __init__.py → Public API + PROMPTS dict
"""

from .build import BUILD_PROMPT
from .plan import PLAN_PROMPT

__all__ = [
    "PROMPTS",
    "DEFAULT_MODE",
    "get_system_prompt",
    "BUILD_PROMPT",
    "PLAN_PROMPT",
]

# ─── Registry ────────────────────────────────────────────────────────

PROMPTS: dict[str, str] = {
    "plan": PLAN_PROMPT,
    "build": BUILD_PROMPT,
}

DEFAULT_MODE = "build"


# ─── Public API ──────────────────────────────────────────────────────

def get_system_prompt(
    mode: str = DEFAULT_MODE,
    persona_config: dict | None = None,
) -> str:
    """Get the system prompt for a specific mode.

    Args:
        mode: One of 'plan', 'build'. Falls back to 'build' for unknown modes.
        persona_config: Optional dict with 'name' key to replace 'Doraemon'.

    Returns:
        The assembled system prompt string.
    """
    base = PROMPTS.get(mode, PROMPTS[DEFAULT_MODE])

    if persona_config:
        name = persona_config.get("name", "Doraemon")
        base = base.replace("Doraemon", name)

    return base
