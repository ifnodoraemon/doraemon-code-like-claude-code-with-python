"""
Skills System

Inspired by Anthropic's Agent Skills (October 2025):
- Skills are reusable, structured packages of instructions and resources
- They are loaded on-demand based on task relevance
- Uses progressive disclosure to avoid context bloat

Directory Structure:
    .agent/skills/
    ├── python-dev/
    │   ├── SKILL.md          # Required: metadata and main instructions
    │   ├── style-guide.md    # Optional: additional resources
    │   └── templates/        # Optional: code templates
    ├── git-workflow/
    │   └── SKILL.md
    └── ...

SKILL.md Format:
    ---
    name: Python Development
    description: Best practices for Python development
    triggers:
      - python
      - pytest
      - pip
      - .py files
    priority: 10
    ---

    ## Instructions
    ...actual skill content...
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised through fallback tests/runtime
    yaml = None

from .paths import skills_dir

logger = logging.getLogger(__name__)


def _parse_simple_frontmatter(frontmatter: str) -> dict[str, Any]:
    """Parse a minimal YAML subset used by SKILL.md frontmatter."""
    data: dict[str, Any] = {}
    current_key: str | None = None

    for raw_line in frontmatter.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if line.startswith("  - ") and current_key is not None:
            data.setdefault(current_key, []).append(stripped[2:].strip())
            continue

        if ":" not in line:
            current_key = None
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if not value:
            data[key] = []
            current_key = key
            continue

        current_key = None
        if ":" in value:
            raise ValueError(f"Unsupported nested YAML value for key '{key}'")
        lowered = value.lower()
        if lowered in {"true", "false"}:
            data[key] = lowered == "true"
        else:
            try:
                data[key] = int(value)
            except ValueError:
                data[key] = value

    return data


def _safe_load_frontmatter(frontmatter: str) -> dict[str, Any]:
    """Load SKILL.md frontmatter without requiring PyYAML at runtime."""
    if yaml is not None:
        loaded = yaml.safe_load(frontmatter)
        return loaded or {}
    return _parse_simple_frontmatter(frontmatter)


# ========================================
# Data Structures
# ========================================


@dataclass
class SkillMetadata:
    """Metadata from SKILL.md frontmatter."""

    name: str
    description: str
    triggers: list[str] = field(default_factory=list)
    priority: int = 0  # Higher = more important
    requires: list[str] = field(default_factory=list)  # Other skills this depends on
    files: list[str] = field(default_factory=list)  # Additional files to load
    mode: list[str] = field(default_factory=list)  # plan/build applicability
    tools: list[str] = field(default_factory=list)  # preferred tools for this skill
    constraints: list[str] = field(default_factory=list)  # execution constraints

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillMetadata":
        return cls(
            name=data.get("name", "Unnamed Skill"),
            description=data.get("description", ""),
            triggers=data.get("triggers", []),
            priority=data.get("priority", 0),
            requires=data.get("requires", []),
            files=data.get("files", []),
            mode=data.get("mode", []),
            tools=data.get("tools", []),
            constraints=data.get("constraints", []),
        )


@dataclass
class Skill:
    """A loaded skill with content."""

    metadata: SkillMetadata
    content: str  # Main SKILL.md content (after frontmatter)
    path: Path
    additional_content: dict[str, str] = field(default_factory=dict)  # file -> content

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def full_content(self) -> str:
        """Get full skill content including additional files."""
        parts = [self.content]
        for filename, content in self.additional_content.items():
            parts.append(f"\n\n## {filename}\n\n{content}")
        return "\n".join(parts)

    def supports_mode(self, mode: str | None) -> bool:
        """Return whether the skill can be activated in the given product mode."""
        if not self.metadata.mode or mode is None:
            return True
        return mode in self.metadata.mode

    def matches_context(self, context: str, mode: str | None = None) -> float:
        """
        Calculate relevance score for given context.

        Returns a score between 0.0 and 1.0.
        """
        if not self.supports_mode(mode):
            return 0.0

        if not self.metadata.triggers:
            return 0.0

        context_lower = context.lower()
        matches = 0

        for trigger in self.metadata.triggers:
            trigger_lower = trigger.lower()
            if trigger_lower in context_lower:
                matches += 1
            # Also check for file extensions
            if trigger_lower.startswith(".") and trigger_lower in context_lower:
                matches += 1

        if matches == 0:
            return 0.0

        # Normalize by number of triggers, boost by priority
        base_score = min(matches / len(self.metadata.triggers), 1.0)
        priority_boost = self.metadata.priority / 100  # Assuming priority 0-100

        return min(base_score + priority_boost, 1.0)


# ========================================
# Skill Loader
# ========================================


class SkillLoader:
    """
    Loads and manages project-local skills from the filesystem.
    """

    SKILL_FILE = "SKILL.md"

    def __init__(
        self,
        project_dir: Path | None = None,
    ):
        self.project_dir = project_dir or Path.cwd()

        self._skills: dict[str, Skill] = {}
        self._loaded = False

    def discover_skills(self) -> list[SkillMetadata]:
        """
        Discover all available skills (metadata only).

        This is the "progressive disclosure" step - we only load
        metadata until a skill is actually needed.
        """
        skills_metadata = []

        project_skills_dir = skills_dir(self.project_dir)
        if project_skills_dir.exists():
            for skill_dir in project_skills_dir.iterdir():
                if skill_dir.is_dir():
                    metadata = self._load_skill_metadata(skill_dir)
                    if metadata:
                        skills_metadata.append(metadata)

        logger.info("Discovered %s skills", len(skills_metadata))
        return skills_metadata

    def load_skill(self, skill_dir: Path) -> Skill | None:
        """Load a skill fully (content + additional files)."""
        skill_file = skill_dir / self.SKILL_FILE

        if not skill_file.exists():
            logger.warning("No SKILL.md found in %s", skill_dir)
            return None

        try:
            content = skill_file.read_text(encoding="utf-8")
            metadata, body = self._parse_skill_file(content)

            if not metadata:
                return None

            skill = Skill(
                metadata=metadata,
                content=body,
                path=skill_dir,
            )

            # Load additional files if specified
            for filename in metadata.files:
                file_path = skill_dir / filename
                if file_path.exists():
                    try:
                        skill.additional_content[filename] = file_path.read_text(encoding="utf-8")
                    except Exception as e:
                        logger.warning("Failed to load %s: %s", filename, e)

            self._skills[metadata.name] = skill
            logger.info("Loaded skill: %s", metadata.name)
            return skill

        except Exception as e:
            logger.error("Failed to load skill from %s: %s", skill_dir, e)
            return None

    def get_relevant_skills(
        self,
        context: str,
        max_skills: int = 3,
        threshold: float = 0.1,
        mode: str | None = None,
    ) -> list[Skill]:
        """
        Get skills relevant to the given context.

        Args:
            context: The user input or task description
            max_skills: Maximum number of skills to return
            threshold: Minimum relevance score

        Returns:
            List of relevant skills, sorted by relevance
        """
        # First, discover all skills if not done
        if not self._loaded:
            self.discover_skills()
            self._loaded = True

        # Load skills that might be relevant
        project_skills_dir = skills_dir(self.project_dir)

        if project_skills_dir.exists():
            for skill_dir in project_skills_dir.iterdir():
                if skill_dir.is_dir() and skill_dir.name not in self._skills:
                    self.load_skill(skill_dir)

        # Score and filter skills
        scored_skills = []
        for skill in self._skills.values():
            score = skill.matches_context(context, mode=mode)
            if score >= threshold:
                scored_skills.append((score, skill))

        # Sort by score (descending) and return top N
        scored_skills.sort(key=lambda x: x[0], reverse=True)
        return [skill for _, skill in scored_skills[:max_skills]]

    def _load_skill_metadata(self, skill_dir: Path) -> SkillMetadata | None:
        """Load only the metadata from a skill (lightweight)."""
        skill_file = skill_dir / self.SKILL_FILE

        if not skill_file.exists():
            return None

        try:
            content = skill_file.read_text(encoding="utf-8")
            metadata, _ = self._parse_skill_file(content)
            return metadata
        except Exception as e:
            logger.warning("Failed to load metadata from %s: %s", skill_dir, e)
            return None

    def _parse_skill_file(self, content: str) -> tuple[SkillMetadata | None, str]:
        """
        Parse a SKILL.md file into metadata and body.
        """
        # Check for YAML frontmatter
        if not content.startswith("---"):
            return SkillMetadata(name="Unknown", description=""), content

        # Use split to find frontmatter
        # We expect the file to start with --- and have another --- later
        parts = content.split("---", 2)
        if len(parts) < 3:
            return SkillMetadata(name="Unknown", description=""), content

        frontmatter = parts[1].strip()
        body = parts[2].strip()

        try:
            data = _safe_load_frontmatter(frontmatter)
            metadata = SkillMetadata.from_dict(data or {})
            return metadata, body
        except Exception as e:
            logger.warning("Invalid YAML frontmatter: %s", e)
            return SkillMetadata(name="Unknown", description=""), content


# ========================================
# Skill Manager
# ========================================


class SkillManager:
    """
    High-level interface for the skills system.

    Integrates with context management to load relevant skills
    without bloating the context window.
    """

    def __init__(
        self,
        project_dir: Path | None = None,
        max_skill_tokens: int = 5000,  # Max tokens for all skills combined
    ):
        self.loader = SkillLoader(project_dir=project_dir)
        self.max_skill_tokens = max_skill_tokens
        self._active_skills: list[str] = []

    def get_skills_for_context(
        self,
        user_input: str,
        current_context: str = "",
        mode: str | None = None,
    ) -> str:
        """
        Get formatted skill content for the current context.

        Uses progressive disclosure:
        1. Check which skills are relevant based on triggers
        2. Load only those skills
        3. Format them for injection into system prompt
        """
        # Combine user input with current context for better matching
        full_context = f"{current_context}\n{user_input}"

        # Get relevant skills
        skills = self.loader.get_relevant_skills(
            full_context,
            max_skills=3,
            threshold=0.1,
            mode=mode,
        )

        if not skills:
            return ""

        # Track active skills
        self._active_skills = [s.name for s in skills]

        # Format skills for prompt
        parts = ["=== ACTIVE SKILLS ===\n"]

        total_chars = 0
        max_chars = self.max_skill_tokens * 3  # Rough char estimate

        for skill in skills:
            skill_sections = [f"\n### {skill.name}"]
            if skill.metadata.description:
                skill_sections.append(f"{skill.metadata.description}")
            if skill.metadata.tools:
                skill_sections.append(f"Preferred tools: {', '.join(skill.metadata.tools)}")
            if skill.metadata.constraints:
                skill_sections.append(f"Constraints: {', '.join(skill.metadata.constraints)}")
            skill_sections.append(skill.content)

            skill_content = "\n".join(skill_sections) + "\n"

            if total_chars + len(skill_content) > max_chars:
                # Truncate if too long
                remaining = max_chars - total_chars
                if remaining > 200:
                    skill_content = skill_content[:remaining] + "\n... [truncated]"
                else:
                    break

            parts.append(skill_content)
            total_chars += len(skill_content)

        parts.append("\n=== END SKILLS ===")

        return "".join(parts)

    def get_active_skills(self) -> list[str]:
        """Get names of currently active skills."""
        return self._active_skills.copy()


# ========================================
# Utility Functions
# ========================================


def create_skill_template(skill_dir: Path, name: str, description: str) -> Path:
    """
    Create a new skill from template.

    Args:
        skill_dir: Directory to create skill in
        name: Skill name
        description: Brief description

    Returns:
        Path to created SKILL.md
    """
    skill_dir.mkdir(parents=True, exist_ok=True)

    template = f"""---
name: {name}
description: {description}
triggers:
  - # Add trigger keywords here
priority: 0
files: []
---

## Overview

{description}

## Instructions

<!-- Add skill instructions here -->

## Examples

<!-- Add examples of when/how to apply this skill -->
"""

    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(template, encoding="utf-8")

    logger.info("Created skill template: %s", skill_file)
    return skill_file


def format_skills_for_prompt(skills: list[Skill]) -> str:
    """Format a list of skills for inclusion in system prompt."""
    if not skills:
        return ""

    parts = ["\n=== LOADED SKILLS ===\n"]

    for skill in skills:
        parts.append(f"\n### Skill: {skill.name}")
        parts.append(f"*{skill.metadata.description}*\n")
        parts.append(skill.content)

    parts.append("\n=== END SKILLS ===\n")

    return "\n".join(parts)
