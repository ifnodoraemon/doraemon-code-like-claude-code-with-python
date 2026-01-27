"""
System Tools for Doraemon Code.

These tools allow the agent to interact with the Doraemon system itself,
such as changing modes.
"""

import logging

logger = logging.getLogger(__name__)


def switch_mode(mode: str) -> str:
    """
    Switch the agent's operating mode.

    Available modes:
    - "plan": Read-only mode for research, planning, and design.
    - "build": Execution mode for writing code, running commands, and making changes.

    Args:
        mode: The target mode ("plan" or "build")

    Returns:
        Status message
    """
    if mode not in ["plan", "build"]:
        return f"Error: Invalid mode '{mode}'. Use 'plan' or 'build'."
    
    # This tool is special: the actual mode switching happens in the CLI loop
    # by inspecting the tool call. This function just returns a confirmation.
    logger.info(f"Agent requested switch to mode: {mode}")
    return f"Switched to {mode} mode."
