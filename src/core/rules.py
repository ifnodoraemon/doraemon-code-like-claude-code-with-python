"""Rules and instruction loading.

This module only loads project-local instructions:
1. Hierarchical `AGENTS.md` files from the project root down to cwd
2. Additional instruction files explicitly listed in config
3. Project-local memory from `.agent/MEMORY.md`
"""

from pathlib import Path

from .logger import get_logger
from .paths import MEMORY_FILENAME, RULES_FILENAME, memory_path

logger = get_logger(__name__)


def _read_instruction_file(path: Path) -> str | None:
    """Read a single instruction file."""
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to read {path}: {e}")
        return None


def _find_project_boundary(project_dir: Path) -> Path:
    """Find the repository / project root used for AGENTS.md discovery."""
    for candidate in [project_dir, *project_dir.parents]:
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
            return candidate
    return project_dir


def _combine_instruction_sections(sections: list[tuple[str, str]]) -> str | None:
    """Combine labeled instruction sections into one prompt fragment."""
    if not sections:
        return None
    return "\n\n---\n\n".join(f"# {label}\n\n{content}" for label, content in sections)


def _load_hierarchical_agents(project_dir: Path) -> list[tuple[str, str]]:
    """Load `AGENTS.md` files from project root down to the current directory."""
    boundary = _find_project_boundary(project_dir.resolve())
    search_dirs: list[Path] = []
    current = project_dir.resolve()

    while True:
        search_dirs.append(current)
        if current == boundary or current.parent == current:
            break
        current = current.parent

    sections: list[tuple[str, str]] = []
    for directory in reversed(search_dirs):
        agents_path = directory / RULES_FILENAME
        if not agents_path.exists():
            continue

        content = _read_instruction_file(agents_path)
        if not content:
            continue

        if directory == boundary:
            label = f"Project Instructions ({RULES_FILENAME})"
        else:
            label = f"Nested Instructions ({RULES_FILENAME}: {directory.relative_to(boundary)})"
        sections.append((label, content))

    return sections


def load_project_rules(project_dir: Path | None = None) -> str | None:
    """Load hierarchical project `AGENTS.md` instructions."""
    if project_dir is None:
        project_dir = Path.cwd()

    sections = _load_hierarchical_agents(project_dir)

    combined = _combine_instruction_sections(sections)
    if combined:
        logger.info("Loaded project rules and instructions")
    else:
        logger.debug(f"No project {RULES_FILENAME} found")
    return combined


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
    1. Project AGENTS.md files (top-down)
    2. Additional instruction files from config

    Args:
        config: Configuration dictionary
        project_dir: Project directory (defaults to cwd)

    Returns:
        Combined instructions as a single string
    """
    instructions = []

    # 1. Project AGENTS.md hierarchy
    project_rules = load_project_rules(project_dir)
    if project_rules:
        instructions.append(f"# Project Rules ({RULES_FILENAME})\n\n" + project_rules)

    # 2. Additional instruction files from config
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
    """Create a default `AGENTS.md` file in the project directory."""
    if project_dir is None:
        project_dir = Path.cwd()

    rules_file = project_dir / RULES_FILENAME

    default_content = """# Project Rules

This file contains project-specific rules and conventions for the agent.

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
    logger.info(f"Created default {RULES_FILENAME}: {rules_file}")

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


def load_project_memory(project_dir: Path | None = None) -> str | None:
    """Load project-local `.agent/MEMORY.md`."""
    if project_dir is None:
        project_dir = Path.cwd()

    memory_file = memory_path(project_dir)
    if memory_file.exists():
        try:
            content = memory_file.read_text(encoding="utf-8")
            logger.info(f"Loaded project memory: {memory_file}")
            return content
        except Exception as e:
            logger.error(f"Failed to read {MEMORY_FILENAME}: {e}")
    return None


def format_memory_for_prompt(memory: str) -> str:
    """
    Format memory content for inclusion in system prompt.

    Args:
        memory: Raw memory text

    Returns:
        Formatted memory ready for system prompt
    """
    if not memory:
        return ""

    return f"""

=== PROJECT MEMORY ===

{memory.strip()}

=== END PROJECT MEMORY ===

Use this memory to inform your responses. Update it when you learn important project details.
"""
