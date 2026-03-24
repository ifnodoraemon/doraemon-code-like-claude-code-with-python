"""Advanced comprehensive tests for skills.py - 20+ tests for improved coverage"""

from pathlib import Path

from src.core.skills import (
    Skill,
    SkillLoader,
    SkillManager,
    SkillMetadata,
    create_skill_template,
    format_skills_for_prompt,
)


class TestSkillMetadataAdvanced:
    """Advanced tests for SkillMetadata."""

    def test_from_dict_minimal(self):
        """Test from_dict with minimal data."""
        data = {"name": "Test Skill"}
        metadata = SkillMetadata.from_dict(data)
        assert metadata.name == "Test Skill"
        assert metadata.description == ""
        assert metadata.triggers == []
        assert metadata.priority == 0
        assert metadata.requires == []
        assert metadata.files == []

    def test_from_dict_complete(self):
        """Test from_dict with all fields."""
        data = {
            "name": "Python Dev",
            "description": "Python development guide",
            "triggers": ["python", "pytest", ".py"],
            "priority": 10,
            "requires": ["git-workflow"],
            "files": ["style-guide.md", "templates/"],
        }
        metadata = SkillMetadata.from_dict(data)
        assert metadata.name == "Python Dev"
        assert metadata.description == "Python development guide"
        assert len(metadata.triggers) == 3
        assert metadata.priority == 10
        assert len(metadata.requires) == 1
        assert len(metadata.files) == 2

    def test_from_dict_with_none_values(self):
        """Test from_dict handles None values."""
        data = {"name": "Test", "description": None, "triggers": None}
        metadata = SkillMetadata.from_dict(data)
        # None values are preserved as-is by from_dict
        assert metadata.description is None or metadata.description == ""
        assert metadata.triggers is None or metadata.triggers == []


class TestSkillAdvanced:
    """Advanced tests for Skill class."""

    def test_skill_name_property(self):
        """Test skill name property."""
        metadata = SkillMetadata(name="Test Skill", description="Test")
        skill = Skill(metadata=metadata, content="Content", path=Path("/test"))
        assert skill.name == "Test Skill"

    def test_skill_full_content_empty_additional(self):
        """Test full_content with no additional files."""
        metadata = SkillMetadata(name="Test", description="Test")
        skill = Skill(metadata=metadata, content="Main content", path=Path("/test"))
        assert skill.full_content == "Main content"

    def test_skill_full_content_with_additional(self):
        """Test full_content includes additional files."""
        metadata = SkillMetadata(name="Test", description="Test")
        skill = Skill(
            metadata=metadata,
            content="Main content",
            path=Path("/test"),
            additional_content={"guide.md": "Guide content", "examples.md": "Examples"},
        )
        full = skill.full_content
        assert "Main content" in full
        assert "Guide content" in full
        assert "Examples" in full
        assert "## guide.md" in full

    def test_skill_matches_context_no_triggers(self):
        """Test matches_context with no triggers."""
        metadata = SkillMetadata(name="Test", description="Test", triggers=[])
        skill = Skill(metadata=metadata, content="Content", path=Path("/test"))
        score = skill.matches_context("python code")
        assert score == 0.0

    def test_skill_matches_context_single_trigger(self):
        """Test matches_context with single trigger."""
        metadata = SkillMetadata(name="Python", description="Python", triggers=["python"])
        skill = Skill(metadata=metadata, content="Content", path=Path("/test"))
        score = skill.matches_context("write python code")
        assert score > 0.0

    def test_skill_matches_context_multiple_triggers(self):
        """Test matches_context with multiple triggers."""
        metadata = SkillMetadata(
            name="Python", description="Python", triggers=["python", "pytest", ".py"]
        )
        skill = Skill(metadata=metadata, content="Content", path=Path("/test"))
        score = skill.matches_context("write python code with pytest")
        assert score > 0.0

    def test_skill_matches_context_file_extension(self):
        """Test matches_context recognizes file extensions."""
        metadata = SkillMetadata(name="Python", description="Python", triggers=[".py"])
        skill = Skill(metadata=metadata, content="Content", path=Path("/test"))
        score = skill.matches_context("edit main.py")
        assert score > 0.0

    def test_skill_matches_context_case_insensitive(self):
        """Test matches_context is case insensitive."""
        metadata = SkillMetadata(name="Python", description="Python", triggers=["PYTHON"])
        skill = Skill(metadata=metadata, content="Content", path=Path("/test"))
        score = skill.matches_context("write python code")
        assert score > 0.0

    def test_skill_matches_context_with_priority(self):
        """Test matches_context includes priority boost."""
        metadata = SkillMetadata(
            name="Python", description="Python", triggers=["python"], priority=50
        )
        skill = Skill(metadata=metadata, content="Content", path=Path("/test"))
        score = skill.matches_context("python")
        # Score should be boosted by priority
        assert score > 0.5


class TestSkillLoaderInitialization:
    """Tests for SkillLoader initialization."""

    def test_initialization_default(self):
        """Test default initialization."""
        loader = SkillLoader()
        assert loader.project_dir == Path.cwd()
        assert loader._skills == {}
        assert loader._loaded is False

    def test_initialization_custom_project_dir(self):
        """Test initialization with custom project directory."""
        project_dir = Path("/custom/project")
        loader = SkillLoader(project_dir=project_dir)
        assert loader.project_dir == project_dir


class TestSkillLoaderDiscovery:
    """Tests for skill discovery."""

    def test_discover_skills_empty(self, tmp_path):
        """Test discovery with no skills."""
        loader = SkillLoader(project_dir=tmp_path)
        skills = loader.discover_skills()
        assert skills == []

    def test_discover_skills_with_valid_skill(self, tmp_path):
        """Test discovery with valid skill."""
        skills_dir = tmp_path / ".agent" / "skills" / "test-skill"
        skills_dir.mkdir(parents=True)
        skill_file = skills_dir / "SKILL.md"
        skill_file.write_text("""---
name: Test Skill
description: A test skill
triggers:
  - test
priority: 5
---

## Content
Test content""")

        loader = SkillLoader(project_dir=tmp_path)
        skills = loader.discover_skills()
        assert len(skills) == 1
        assert skills[0].name == "Test Skill"

    def test_discover_skills_project_only(self, tmp_path):
        """Test discovery only reads project-local skills."""
        project_skills_dir = tmp_path / "project" / ".agent" / "skills" / "test-skill"
        project_skills_dir.mkdir(parents=True)
        (project_skills_dir / "SKILL.md").write_text("""---
name: Test Skill
description: Project version
---
Project""")

        loader = SkillLoader(project_dir=tmp_path / "project")
        skills = loader.discover_skills()
        assert len(skills) == 1


class TestSkillLoaderLoading:
    """Tests for skill loading."""

    def test_load_skill_missing_file(self, tmp_path):
        """Test loading skill with missing SKILL.md."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        loader = SkillLoader()
        skill = loader.load_skill(skill_dir)
        assert skill is None

    def test_load_skill_valid(self, tmp_path):
        """Test loading valid skill."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("""---
name: Test Skill
description: Test description
triggers:
  - test
priority: 5
---

## Content
Skill content here""")

        loader = SkillLoader()
        skill = loader.load_skill(skill_dir)
        assert skill is not None
        assert skill.name == "Test Skill"
        assert "Skill content" in skill.content

    def test_load_skill_with_additional_files(self, tmp_path):
        """Test loading skill with additional files."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()

        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("""---
name: Test Skill
description: Test
files:
  - guide.md
  - examples.md
---

Main content""")

        (skill_dir / "guide.md").write_text("Guide content")
        (skill_dir / "examples.md").write_text("Examples content")

        loader = SkillLoader()
        skill = loader.load_skill(skill_dir)
        assert skill is not None
        assert "guide.md" in skill.additional_content
        assert "examples.md" in skill.additional_content
        assert skill.additional_content["guide.md"] == "Guide content"

    def test_load_skill_missing_additional_file(self, tmp_path):
        """Test loading skill with missing additional file."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()

        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("""---
name: Test Skill
description: Test
files:
  - missing.md
---

Content""")

        loader = SkillLoader()
        skill = loader.load_skill(skill_dir)
        # Should still load, just without the missing file
        assert skill is not None
        assert "missing.md" not in skill.additional_content


class TestSkillLoaderParsing:
    """Tests for SKILL.md parsing."""

    def test_parse_skill_file_no_frontmatter(self):
        """Test parsing file without frontmatter."""
        loader = SkillLoader()
        content = "Just content, no frontmatter"
        metadata, body = loader._parse_skill_file(content)
        assert metadata.name == "Unknown"
        assert body == content

    def test_parse_skill_file_valid_frontmatter(self):
        """Test parsing file with valid frontmatter."""
        loader = SkillLoader()
        content = """---
name: Test Skill
description: Test description
triggers:
  - test
priority: 5
---

Body content here"""
        metadata, body = loader._parse_skill_file(content)
        assert metadata.name == "Test Skill"
        assert metadata.description == "Test description"
        assert "Body content" in body

    def test_parse_skill_file_invalid_yaml(self):
        """Test parsing file with invalid YAML."""
        loader = SkillLoader()
        content = """---
name: Test
invalid: [unclosed
---

Body"""
        metadata, body = loader._parse_skill_file(content)
        # Should return defaults on error
        assert metadata.name == "Unknown"

    def test_parse_skill_file_incomplete_frontmatter(self):
        """Test parsing file with incomplete frontmatter."""
        loader = SkillLoader()
        content = """---
name: Test
"""
        metadata, body = loader._parse_skill_file(content)
        # Should return defaults
        assert metadata.name == "Unknown"


class TestSkillLoaderRelevance:
    """Tests for getting relevant skills."""

    def test_get_relevant_skills_empty(self, tmp_path):
        """Test getting relevant skills with no skills."""
        loader = SkillLoader(project_dir=tmp_path)
        skills = loader.get_relevant_skills("python code")
        assert skills == []

    def test_get_relevant_skills_with_threshold(self, tmp_path):
        """Test getting relevant skills respects threshold."""
        skills_dir = tmp_path / ".agent" / "skills" / "python"
        skills_dir.mkdir(parents=True)
        (skills_dir / "SKILL.md").write_text("""---
name: Python
description: Python
triggers:
  - python
priority: 0
---
Content""")

        loader = SkillLoader(project_dir=tmp_path)
        # High threshold should exclude low-match skills
        skills = loader.get_relevant_skills("javascript", threshold=0.9)
        assert len(skills) == 0

    def test_get_relevant_skills_max_skills(self, tmp_path):
        """Test getting relevant skills respects max_skills."""
        skills_dir = tmp_path / ".agent" / "skills"
        for i in range(5):
            skill_dir = skills_dir / f"skill{i}"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(f"""---
name: Skill {i}
description: Skill {i}
triggers:
  - python
priority: {i}
---
Content""")

        loader = SkillLoader(project_dir=tmp_path)
        skills = loader.get_relevant_skills("python", max_skills=2)
        assert len(skills) <= 2


class TestSkillManagerInitialization:
    """Tests for SkillManager initialization."""

    def test_initialization_default(self):
        """Test default initialization."""
        manager = SkillManager()
        assert manager.max_skill_tokens == 5000
        assert manager._active_skills == []

    def test_initialization_custom_tokens(self):
        """Test initialization with custom token limit."""
        manager = SkillManager(max_skill_tokens=10000)
        assert manager.max_skill_tokens == 10000


class TestSkillManagerContextSkills:
    """Tests for getting skills for context."""

    def test_get_skills_for_context_no_skills(self, tmp_path):
        """Test getting skills when none match."""
        manager = SkillManager(project_dir=tmp_path)
        result = manager.get_skills_for_context("random input")
        assert result == ""

    def test_get_skills_for_context_with_skills(self, tmp_path):
        """Test getting skills for matching context."""
        skills_dir = tmp_path / ".agent" / "skills" / "python"
        skills_dir.mkdir(parents=True)
        (skills_dir / "SKILL.md").write_text("""---
name: Python Dev
description: Python development
triggers:
  - python
priority: 10
---

## Instructions
Write Python code following PEP 8""")

        manager = SkillManager(project_dir=tmp_path)
        result = manager.get_skills_for_context("write python code")
        assert "ACTIVE SKILLS" in result
        assert "Python Dev" in result

    def test_get_skills_for_context_truncation(self, tmp_path):
        """Test skill content is truncated if too long."""
        skills_dir = tmp_path / ".agent" / "skills" / "large"
        skills_dir.mkdir(parents=True)
        large_content = "x" * 20000  # Very large content
        (skills_dir / "SKILL.md").write_text(f"""---
name: Large Skill
description: Large
triggers:
  - test
priority: 0
---

{large_content}""")

        manager = SkillManager(project_dir=tmp_path, max_skill_tokens=100)
        result = manager.get_skills_for_context("test")
        # Should be truncated
        assert "[truncated]" in result or len(result) < len(large_content)

    def test_get_active_skills(self, tmp_path):
        """Test getting active skills."""
        skills_dir = tmp_path / ".agent" / "skills" / "python"
        skills_dir.mkdir(parents=True)
        (skills_dir / "SKILL.md").write_text("""---
name: Python
description: Python
triggers:
  - python
---
Content""")

        manager = SkillManager(project_dir=tmp_path)
        manager.get_skills_for_context("python code")
        active = manager.get_active_skills()
        assert "Python" in active


class TestSkillUtilityFunctions:
    """Tests for utility functions."""

    def test_create_skill_template(self, tmp_path):
        """Test creating skill template."""
        skill_dir = tmp_path / "new-skill"
        result = create_skill_template(skill_dir, "Test Skill", "Test description")
        assert result.exists()
        assert result.name == "SKILL.md"
        content = result.read_text()
        assert "Test Skill" in content
        assert "Test description" in content
        assert "---" in content

    def test_format_skills_for_prompt_empty(self):
        """Test formatting empty skills list."""
        result = format_skills_for_prompt([])
        assert result == ""

    def test_format_skills_for_prompt_single(self):
        """Test formatting single skill."""
        metadata = SkillMetadata(name="Python", description="Python development")
        skill = Skill(metadata=metadata, content="Python guidelines", path=Path("/test"))
        result = format_skills_for_prompt([skill])
        assert "LOADED SKILLS" in result
        assert "Python" in result
        assert "Python guidelines" in result

    def test_format_skills_for_prompt_multiple(self):
        """Test formatting multiple skills."""
        skills = []
        for i in range(3):
            metadata = SkillMetadata(name=f"Skill {i}", description=f"Description {i}")
            skill = Skill(metadata=metadata, content=f"Content {i}", path=Path("/test"))
            skills.append(skill)

        result = format_skills_for_prompt(skills)
        assert "LOADED SKILLS" in result
        assert "Skill 0" in result
        assert "Skill 1" in result
        assert "Skill 2" in result
        assert "END SKILLS" in result


class TestSkillLoaderMetadataOnly:
    """Tests for loading metadata only."""

    def test_load_skill_metadata_missing_file(self, tmp_path):
        """Test loading metadata from missing file."""
        skill_dir = tmp_path / "test"
        skill_dir.mkdir()
        loader = SkillLoader()
        metadata = loader._load_skill_metadata(skill_dir)
        assert metadata is None

    def test_load_skill_metadata_valid(self, tmp_path):
        """Test loading metadata from valid file."""
        skill_dir = tmp_path / "test"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: Test
description: Test skill
triggers:
  - test
priority: 5
---
Content""")

        loader = SkillLoader()
        metadata = loader._load_skill_metadata(skill_dir)
        assert metadata is not None
        assert metadata.name == "Test"
        assert metadata.priority == 5
