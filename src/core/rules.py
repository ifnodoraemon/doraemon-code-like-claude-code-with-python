"""Rules and instructions loading system for Polymath.

This module handles loading project rules from:
1. POLYMATH.md in project directory (project-specific rules)
2. POLYMATH.md in ~/.polymath/ (global user rules)
3. Additional instruction files specified in config
"""

from pathlib import Path

from .logger import get_logger

logger = get_logger(__name__)

# The rules file name
RULES_FILE = "DORAEMON.md"


def load_project_rules(project_dir: Path | None = None) -> str | None:
    """
    Load POLYMATH.md file from project directory.

    Args:
        project_dir: Project directory to search in (defaults to cwd)

    Returns:
        Content of POLYMATH.md or None if not found
    """
    if project_dir is None:
        project_dir = Path.cwd()

    rules_file = project_dir / RULES_FILE

    if rules_file.exists():
        try:
            content = rules_file.read_text(encoding="utf-8")
            logger.info(f"Loaded project rules: {rules_file}")
            return content
        except Exception as e:
            logger.error(f"Failed to read {RULES_FILE}: {e}")
            return None

    logger.debug(f"No project {RULES_FILE} found")
    return None


def load_global_rules() -> str | None:
    """
    Load POLYMATH.md from user's home directory.

    Returns:
        Content of global POLYMATH.md or None if not found
    """
    global_rules = Path.home() / ".doraemon" / RULES_FILE

    if global_rules.exists():
        try:
            content = global_rules.read_text(encoding="utf-8")
            logger.info(f"Loaded global rules: {global_rules}")
            return content
        except Exception as e:
            logger.error(f"Failed to read global {RULES_FILE}: {e}")
            return None

    logger.debug(f"No global {RULES_FILE} found")
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
    1. Project POLYMATH.md (highest priority)
    2. Global POLYMATH.md
    3. Additional instruction files from config

    Args:
        config: Configuration dictionary
        project_dir: Project directory (defaults to cwd)

    Returns:
        Combined instructions as a single string
    """
    instructions = []

    # 1. Project POLYMATH.md (highest priority)
    project_rules = load_project_rules(project_dir)
    if project_rules:
        instructions.append(f"# Project Rules ({RULES_FILE})\n\n" + project_rules)

    # 2. Global POLYMATH.md
    global_rules = load_global_rules()
    if global_rules:
        instructions.append(f"# Global Rules (~/.doraemon/{RULES_FILE})\n\n" + global_rules)

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


def create_default_rules(project_dir: Path | None = None) -> Path:
    """
    Create a default POLYMATH.md file in the project directory.

    Args:
        project_dir: Project directory (defaults to cwd)

    Returns:
        Path to created POLYMATH.md file
    """
    if project_dir is None:
        project_dir = Path.cwd()

    rules_file = project_dir / RULES_FILE

    default_content = """# Project Rules

This file contains project-specific rules and conventions for Doraemon Code.

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

"""

    rules_file.write_text(default_content, encoding="utf-8")
    logger.info(f"Created default {RULES_FILE}: {rules_file}")

    return rules_file


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
