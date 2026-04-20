"""Tests for src/core/prompts/__init__.py"""

from src.core.prompts import (
    BUILD_PROMPT,
    DEFAULT_MODE,
    PLAN_PROMPT,
    PROMPTS,
    get_system_prompt,
)


class TestPromptsDict:
    def test_contains_plan(self):
        assert "plan" in PROMPTS

    def test_contains_build(self):
        assert "build" in PROMPTS

    def test_plan_matches(self):
        assert PROMPTS["plan"] == PLAN_PROMPT

    def test_build_matches(self):
        assert PROMPTS["build"] == BUILD_PROMPT

    def test_only_two_modes(self):
        assert len(PROMPTS) == 2


class TestDefaultMode:
    def test_is_build(self):
        assert DEFAULT_MODE == "build"


class TestGetSystemPrompt:
    def test_default_mode(self):
        result = get_system_prompt()
        assert result == BUILD_PROMPT

    def test_plan_mode(self):
        result = get_system_prompt(mode="plan")
        assert result == PLAN_PROMPT

    def test_build_mode(self):
        result = get_system_prompt(mode="build")
        assert result == BUILD_PROMPT

    def test_unknown_mode_falls_back_to_build(self):
        result = get_system_prompt(mode="nonexistent")
        assert result == BUILD_PROMPT

    def test_persona_replaces_code_agent(self):
        result = get_system_prompt(persona_config={"name": "Alfred"})
        assert "Alfred" in result
        assert "Code Agent" not in result

    def test_persona_replaces_doraemon(self):
        result = get_system_prompt(persona_config={"name": "Jarvis"})
        assert "Jarvis" in result

    def test_persona_default_name(self):
        result = get_system_prompt(persona_config={})
        assert "Agent" in result

    def test_persona_none(self):
        result = get_system_prompt(persona_config=None)
        assert "Code Agent" in result

    def test_result_is_string(self):
        assert isinstance(get_system_prompt(), str)
        assert isinstance(get_system_prompt(mode="plan"), str)

    def test_result_non_empty(self):
        assert len(get_system_prompt()) > 0
        assert len(get_system_prompt(mode="plan")) > 0
