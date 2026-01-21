"""Rules and instructions loading system for Polymath.

This module handles loading project rules from:
1. AGENTS.md in project directory
2. AGENTS.md in ~/.polymath/
3. Additional instruction files specified in config
"""

from pathlib import Path

from .logger import get_logger

logger = get_logger(__name__)


def load_agents_md(project_dir: Path | None = None) -> str | None:
    """
    Load AGENTS.md file from project directory.

    Args:
        project_dir: Project directory to search in (defaults to cwd)

    Returns:
        Content of AGENTS.md or None if not found
    """
    if project_dir is None:
        project_dir = Path.cwd()

    agents_file = project_dir / "AGENTS.md"

    if agents_file.exists():
        try:
            content = agents_file.read_text(encoding="utf-8")
            logger.info(f"Loaded project AGENTS.md: {agents_file}")
            return content
        except Exception as e:
            logger.error(f"Failed to read AGENTS.md: {e}")
            return None

    logger.debug("No project AGENTS.md found")
    return None


def load_global_agents_md() -> str | None:
    """
    Load AGENTS.md from user's home directory.

    Returns:
        Content of global AGENTS.md or None if not found
    """
    global_agents = Path.home() / ".polymath" / "AGENTS.md"

    if global_agents.exists():
        try:
            content = global_agents.read_text(encoding="utf-8")
            logger.info(f"Loaded global AGENTS.md: {global_agents}")
            return content
        except Exception as e:
            logger.error(f"Failed to read global AGENTS.md: {e}")
            return None

    logger.debug("No global AGENTS.md found")
    return None


def load_instruction_file(file_path: str, base_dir: Path | None = None) -> str | None:
    """
    Load a single instruction file (supports glob patterns).

    Args:
        file_path: Path to instruction file (can be relative or absolute)
        base_dir: Base directory for relative paths (defaults to cwd)

    Returns:
        Content of the file or None if not found
    """
    if base_dir is None:
        base_dir = Path.cwd()

    # Handle absolute paths
    path = Path(file_path)
    if not path.is_absolute():
        path = base_dir / file_path

    # Handle glob patterns
    if "*" in str(path):
        parent = path.parent
        pattern = path.name
        matching_files = list(parent.glob(pattern))

        if not matching_files:
            logger.debug(f"No files matched pattern: {file_path}")
            return None

        contents = []
        for file in matching_files:
            try:
                content = file.read_text(encoding="utf-8")
                contents.append(f"## From {file.name}\n\n{content}")
                logger.info(f"Loaded instruction file: {file}")
            except Exception as e:
                logger.error(f"Failed to read {file}: {e}")

        return "\n\n".join(contents) if contents else None

    # Single file
    if path.exists():
        try:
            content = path.read_text(encoding="utf-8")
            logger.info(f"Loaded instruction file: {path}")
            return content
        except Exception as e:
            logger.error(f"Failed to read {path}: {e}")
            return None

    logger.debug(f"Instruction file not found: {file_path}")
    return None


def load_all_instructions(config: dict, project_dir: Path | None = None) -> str:
    """
    Load all instructions from various sources and combine them.

    Loading order:
    1. Project AGENTS.md
    2. Global AGENTS.md
    3. Additional instruction files from config

    Args:
        config: Configuration dictionary
        project_dir: Project directory (defaults to cwd)

    Returns:
        Combined instructions as a single string
    """
    instructions = []

    # 1. Project AGENTS.md (highest priority)
    project_agents = load_agents_md(project_dir)
    if project_agents:
        instructions.append("# Project Rules (AGENTS.md)\n\n" + project_agents)

    # 2. Global AGENTS.md
    global_agents = load_global_agents_md()
    if global_agents:
        instructions.append("# Global Rules (~/.polymath/AGENTS.md)\n\n" + global_agents)

    # 3. Additional instruction files from config
    instruction_files = config.get("instructions", [])
    if instruction_files:
        logger.info(f"Loading {len(instruction_files)} additional instruction files")

        for file_path in instruction_files:
            content = load_instruction_file(file_path, project_dir)
            if content:
                instructions.append(content)

    if not instructions:
        logger.info("No instructions loaded")
        return ""

    combined = "\n\n---\n\n".join(instructions)
    logger.info(
        f"Loaded total of {len(instructions)} instruction sources, {len(combined)} characters"
    )

    return combined


def create_default_agents_md(project_dir: Path | None = None) -> Path:
    """
    Create a default AGENTS.md file in the project directory.

    Args:
        project_dir: Project directory (defaults to cwd)

    Returns:
        Path to created AGENTS.md file
    """
    if project_dir is None:
        project_dir = Path.cwd()

    agents_file = project_dir / "AGENTS.md"

    default_content = """# Polymath Project Rules

This file contains project-specific rules and conventions that Polymath will follow.

## Project Overview
<!-- Brief description of what this project does -->

## Tech Stack
- Python 3.10+
- [Add your key dependencies here]

## Code Style
- Use 4 spaces for indentation
- Type hints are required for all functions
- Follow PEP 8 naming conventions

## Architecture
<!-- Describe your project's architecture and design patterns -->

## Important Notes
<!-- Any project-specific conventions, gotchas, or important information -->

## Example Usage
<!-- Show how to use your project's main features -->

"""

    agents_file.write_text(default_content, encoding="utf-8")
    logger.info(f"Created default AGENTS.md: {agents_file}")

    return agents_file


def format_instructions_for_prompt(instructions: str) -> str:
    """
    Format instructions for inclusion in system prompt.

    Args:
        instructions: Raw instructions text

    Returns:
        Formatted instructions ready for system prompt
    """
    if not instructions:
        return ""

    return f"""

=== PROJECT RULES AND CONVENTIONS ===

{instructions.strip()}

=== END OF PROJECT RULES ===

Follow these project-specific rules and conventions in all your responses.
"""
