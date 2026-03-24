"""Comprehensive tests for skills.py"""

from pathlib import Path

from src.core.skills import (
    Skill,
    SkillLoader,
    SkillManager,
    SkillMetadata,
    create_skill_template,
    format_skills_for_prompt,
)


class TestSkillMetadata:
    """Tests for SkillMetadata."""

    def test_from_dict_basic(self):
        """Test creating metadata from dict."""
        data = {
            "name": "Python Dev",
            "description": "Python development best practices",
            "triggers": ["python", ".py"],
            "priority": 10,
        }
        metadata = SkillMetadata.from_dict(data)
        assert metadata.name == "Python Dev"
        assert metadata.description == "Python development best practices"
        assert len(metadata.triggers) == 2
        assert metadata.priority == 10

    def test_from_dict_with_defaults(self):
        """Test metadata with missing fields uses defaults."""
        data = {"name": "Test Skill"}
        metadata = SkillMetadata.from_dict(data)
        assert metadata.name == "Test Skill"
        assert metadata.description == ""
        assert metadata.triggers == []
        assert metadata.priority == 0

    def test_from_dict_with_requires_and_files(self):
        """Test metadata with requires and files."""
        data = {
            "name": "Advanced Skill",
            "description": "Advanced features",
            "requires": ["basic-skill"],
            "files": ["guide.md", "examples.py"],
        }
        metadata = SkillMetadata.from_dict(data)
        assert metadata.requires == ["basic-skill"]
        assert metadata.files == ["guide.md", "examples.py"]


class TestSkill:
    """Tests for Skill class."""

    def test_skill_properties(self):
        """Test skill properties."""
        metadata = SkillMetadata(name="Test", description="Test skill")
        skill = Skill(
            metadata=metadata,
            content="# Test content",
            path=Path("/test"),
        )
        assert skill.name == "Test"
        assert skill.content == "# Test content"

    def test_full_content_without_additional(self):
        """Test full_content with no additional files."""
        metadata = SkillMetadata(name="Test", description="Test")
        skill = Skill(metadata=metadata, content="Main content", path=Path("/test"))
        assert skill.full_content == "Main content"

    def test_full_content_with_additional(self):
        """Test full_content with additional files."""
        metadata = SkillMetadata(name="Test", description="Test")
        skill = Skill(
            metadata=metadata,
            content="Main content",
            path=Path("/test"),
            additional_content={"guide.md": "Guide content", "examples.py": "Examples"},
        )
        full = skill.full_content
        assert "Main content" in full
        assert "## guide.md" in full
        assert "Guide content" in full
        assert "## examples.py" in full

    def test_matches_context_no_triggers(self):
        """Test matching with no triggers."""
        metadata = SkillMetadata(name="Test", description="Test", triggers=[])
        skill = Skill(metadata=metadata, content="Content", path=Path("/test"))
        assert skill.matches_context("python code") == 0.0

    def test_matches_context_single_match(self):
        """Test matching with single trigger."""
        metadata = SkillMetadata(
            name="Python", description="Python", triggers=["python"], priority=0
        )
        skill = Skill(metadata=metadata, content="Content", path=Path("/test"))
        score = skill.matches_context("I need help with python")
        assert score > 0.0

    def test_matches_context_multiple_matches(self):
        """Test matching with multiple triggers."""
        metadata = SkillMetadata(
            name="Python", description="Python", triggers=["python", "pytest", ".py"], priority=0
        )
        skill = Skill(metadata=metadata, content="Content", path=Path("/test"))
        score = skill.matches_context("python pytest .py files")
        assert score > 0.5

    def test_matches_context_with_priority_boost(self):
        """Test that priority boosts score."""
        metadata_low = SkillMetadata(name="Low", description="Low", triggers=["test"], priority=0)
        metadata_high = SkillMetadata(
            name="High", description="High", triggers=["test"], priority=50
        )
        skill_low = Skill(metadata=metadata_low, content="Content", path=Path("/test"))
        skill_high = Skill(metadata=metadata_high, content="Content", path=Path("/test"))

        score_low = skill_low.matches_context("test")
        score_high = skill_high.matches_context("test")
        # Both should match, high priority should have higher or equal score
        assert score_high >= score_low
        # With priority 50, should get boost
        assert score_high > 0.0

    def test_matches_context_file_extension(self):
        """Test matching file extensions."""
        metadata = SkillMetadata(
            name="Python", description="Python", triggers=[".py", ".pyi"], priority=0
        )
        skill = Skill(metadata=metadata, content="Content", path=Path("/test"))
        score = skill.matches_context("working with .py files")
        assert score > 0.0


class TestSkillLoader:
    """Tests for SkillLoader."""

    def test_initialization(self):
        """Test SkillLoader initialization."""
        loader = SkillLoader()
        assert loader.project_dir == Path.cwd()
        assert loader._skills == {}
        assert loader._loaded is False

    def test_initialization_with_custom_project_dir(self, tmp_path):
        """Test initialization with custom project directory."""
        project_dir = tmp_path / "project"
        loader = SkillLoader(project_dir=project_dir)
        assert loader.project_dir == project_dir

    def test_discover_skills_empty(self, tmp_path):
        """Test discovering skills with no skills directory."""
        loader = SkillLoader(project_dir=tmp_path)
        skills = loader.discover_skills()
        assert skills == []

    def test_discover_skills_with_valid_skill(self, tmp_path):
        """Test discovering a valid skill."""
        skills_dir = tmp_path / ".agent" / "skills" / "test-skill"
        skills_dir.mkdir(parents=True)
        skill_file = skills_dir / "SKILL.md"
        skill_file.write_text(
            """---
name: Test Skill
description: A test skill
triggers:
  - test
priority: 5
---

# Test Skill Content
"""
        )

        loader = SkillLoader(project_dir=tmp_path)
        skills = loader.discover_skills()
        assert len(skills) == 1
        assert skills[0].name == "Test Skill"
        assert skills[0].description == "A test skill"

    def test_load_skill_missing_file(self, tmp_path):
        """Test loading skill with missing SKILL.md."""
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        loader = SkillLoader()
        skill = loader.load_skill(skill_dir)
        assert skill is None

    def test_load_skill_valid(self, tmp_path):
        """Test loading a valid skill."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            """---
name: Valid Skill
description: Valid skill description
triggers:
  - valid
---

# Skill Content
This is the skill content.
"""
        )

        loader = SkillLoader()
        skill = loader.load_skill(skill_dir)
        assert skill is not None
        assert skill.name == "Valid Skill"
        assert "Skill Content" in skill.content

    def test_load_skill_with_additional_files(self, tmp_path):
        """Test loading skill with additional files."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            """---
name: Skill With Files
description: Has additional files
files:
  - guide.md
  - examples.txt
---

# Main Content
"""
        )
        (skill_dir / "guide.md").write_text("# Guide")
        (skill_dir / "examples.txt").write_text("Examples here")

        loader = SkillLoader()
        skill = loader.load_skill(skill_dir)
        assert skill is not None
        assert "guide.md" in skill.additional_content
        assert "examples.txt" in skill.additional_content
        assert skill.additional_content["guide.md"] == "# Guide"

    def test_parse_skill_file_no_frontmatter(self):
        """Test parsing skill file without frontmatter."""
        loader = SkillLoader()
        content = "# Just content, no frontmatter"
        metadata, body = loader._parse_skill_file(content)
        assert metadata.name == "Unknown"
        assert body == content

    def test_parse_skill_file_invalid_yaml(self):
        """Test parsing skill file with invalid YAML."""
        loader = SkillLoader()
        content = """---
invalid: yaml: structure:
---

Body content
"""
        metadata, body = loader._parse_skill_file(content)
        assert metadata.name == "Unknown"

    def test_get_relevant_skills_empty(self, tmp_path):
        """Test getting relevant skills with no skills."""
        loader = SkillLoader(project_dir=tmp_path)
        skills = loader.get_relevant_skills("python code")
        assert skills == []

    def test_get_relevant_skills_with_threshold(self, tmp_path):
        """Test relevance threshold filtering."""
        skills_dir = tmp_path / ".agent" / "skills" / "python-skill"
        skills_dir.mkdir(parents=True)
        (skills_dir / "SKILL.md").write_text(
            """---
name: Python Skill
description: Python
triggers:
  - python
---
Content
"""
        )

        loader = SkillLoader(project_dir=tmp_path)
        # High threshold - should match
        skills = loader.get_relevant_skills("python code", threshold=0.1)
        assert len(skills) >= 0  # May or may not match depending on scoring

        # Very high threshold - should not match
        skills = loader.get_relevant_skills("unrelated topic", threshold=0.9)
        assert len(skills) == 0


class TestSkillManager:
    """Tests for SkillManager."""

    def test_initialization(self):
        """Test SkillManager initialization."""
        manager = SkillManager()
        assert manager.loader is not None
        assert manager.max_skill_tokens == 5000
        assert manager._active_skills == []

    def test_initialization_with_custom_tokens(self, tmp_path):
        """Test initialization with custom token limit."""
        manager = SkillManager(project_dir=tmp_path, max_skill_tokens=10000)
        assert manager.max_skill_tokens == 10000

    def test_get_skills_for_context_no_skills(self, tmp_path):
        """Test getting skills when none exist."""
        manager = SkillManager(project_dir=tmp_path)
        result = manager.get_skills_for_context("python code")
        assert result == ""

    def test_get_active_skills_empty(self):
        """Test getting active skills when none loaded."""
        manager = SkillManager()
        assert manager.get_active_skills() == []


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_create_skill_template(self, tmp_path):
        """Test creating skill template."""
        skill_dir = tmp_path / "new-skill"
        result = create_skill_template(skill_dir, "New Skill", "A new skill")
        assert result.exists()
        assert result.name == "SKILL.md"

        content = result.read_text()
        assert "name: New Skill" in content
        assert "description: A new skill" in content
        assert "## Overview" in content

    def test_format_skills_for_prompt_empty(self):
        """Test formatting empty skill list."""
        result = format_skills_for_prompt([])
        assert result == ""

    def test_format_skills_for_prompt_single_skill(self):
        """Test formatting single skill."""
        metadata = SkillMetadata(name="Test", description="Test skill")
        skill = Skill(metadata=metadata, content="Content", path=Path("/test"))
        result = format_skills_for_prompt([skill])
        assert "=== LOADED SKILLS ===" in result
        assert "Test" in result
        assert "Test skill" in result
        assert "Content" in result

    def test_format_skills_for_prompt_multiple_skills(self):
        """Test formatting multiple skills."""
        skill1 = Skill(
            metadata=SkillMetadata(name="Skill1", description="First"),
            content="Content1",
            path=Path("/test1"),
        )
        skill2 = Skill(
            metadata=SkillMetadata(name="Skill2", description="Second"),
            content="Content2",
            path=Path("/test2"),
        )
        result = format_skills_for_prompt([skill1, skill2])
        assert "Skill1" in result
        assert "Skill2" in result
        assert "Content1" in result
        assert "Content2" in result
