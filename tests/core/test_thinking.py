"""Tests for src.core.thinking."""

from src.core.thinking import (
    ThinkingConfig,
    ThinkingManager,
    ThinkingMode,
    ThinkingResult,
)


class TestThinkingMode:
    def test_values(self):
        assert ThinkingMode.NORMAL.value == "normal"
        assert ThinkingMode.EXTENDED.value == "extended"
        assert ThinkingMode.DEEP.value == "deep"


class TestThinkingConfig:
    def test_defaults(self):
        cfg = ThinkingConfig()
        assert cfg.mode == ThinkingMode.NORMAL
        assert cfg.max_thinking_tokens == 8000
        assert cfg.show_thinking is True
        assert cfg.thinking_budget == 0

    def test_custom(self):
        cfg = ThinkingConfig(mode=ThinkingMode.DEEP, max_thinking_tokens=16000)
        assert cfg.mode == ThinkingMode.DEEP
        assert cfg.max_thinking_tokens == 16000


class TestThinkingResult:
    def test_to_dict(self):
        result = ThinkingResult(
            thinking="thought",
            response="answer",
            thinking_tokens=10,
            response_tokens=20,
            duration=1.5,
            mode=ThinkingMode.EXTENDED,
        )
        d = result.to_dict()
        assert d["thinking"] == "thought"
        assert d["response"] == "answer"
        assert d["thinking_tokens"] == 10
        assert d["response_tokens"] == 20
        assert d["duration"] == 1.5
        assert d["mode"] == "extended"


class TestThinkingManager:
    def test_get_mode_default(self):
        mgr = ThinkingManager()
        assert mgr.get_mode() == ThinkingMode.NORMAL

    def test_set_mode(self):
        mgr = ThinkingManager()
        mgr.set_mode(ThinkingMode.EXTENDED)
        assert mgr.get_mode() == ThinkingMode.EXTENDED

    def test_toggle_mode(self):
        mgr = ThinkingManager()
        assert mgr.toggle_mode() == ThinkingMode.EXTENDED
        assert mgr.toggle_mode() == ThinkingMode.NORMAL

    def test_is_extended(self):
        mgr = ThinkingManager()
        assert not mgr.is_extended()
        mgr.set_mode(ThinkingMode.EXTENDED)
        assert mgr.is_extended()
        mgr.set_mode(ThinkingMode.DEEP)
        assert mgr.is_extended()

    def test_build_prompt_normal(self):
        mgr = ThinkingManager()
        assert mgr.build_prompt("hello") == "hello"

    def test_build_prompt_extended(self):
        mgr = ThinkingManager(ThinkingConfig(mode=ThinkingMode.EXTENDED))
        prompt = mgr.build_prompt("hello")
        assert "hello" in prompt
        assert "<thinking>" in prompt

    def test_build_prompt_deep(self):
        mgr = ThinkingManager(ThinkingConfig(mode=ThinkingMode.DEEP))
        prompt = mgr.build_prompt("solve this")
        assert "solve this" in prompt
        assert "## Problem Analysis" in prompt

    def test_parse_response_no_thinking(self):
        mgr = ThinkingManager()
        result = mgr.parse_response("Just a response", duration=1.0)
        assert result.thinking == ""
        assert result.response == "Just a response"
        assert result.thinking_tokens == 0
        assert result.response_tokens > 0
        assert result.duration == 1.0

    def test_parse_response_with_thinking(self):
        mgr = ThinkingManager(ThinkingConfig(mode=ThinkingMode.EXTENDED))
        raw = "<thinking>Step 1: analyze</thinking>\nFinal answer"
        result = mgr.parse_response(raw)
        assert "<thinking>Step 1: analyze</thinking>" in result.thinking
        assert "Final answer" in result.response

    def test_parse_response_multiline_thinking(self):
        mgr = ThinkingManager(ThinkingConfig(mode=ThinkingMode.EXTENDED))
        raw = "<thinking>\nStep 1\nStep 2\n</thinking>\nAnswer"
        result = mgr.parse_response(raw)
        assert "Step 1" in result.thinking
        assert "Answer" in result.response

    def test_get_thinking_usage_unlimited(self):
        mgr = ThinkingManager()
        usage = mgr.get_thinking_usage()
        assert usage["remaining"] == "unlimited"
        assert usage["total_thinking_tokens"] == 0

    def test_get_thinking_usage_with_budget(self):
        mgr = ThinkingManager(ThinkingConfig(thinking_budget=100))
        mgr.parse_response("<thinking>abc</thinking>ans")
        usage = mgr.get_thinking_usage()
        assert usage["remaining"] == 100 - usage["total_thinking_tokens"]

    def test_should_show_thinking(self):
        mgr = ThinkingManager(ThinkingConfig(show_thinking=True))
        assert not mgr.should_show_thinking()
        mgr.set_mode(ThinkingMode.EXTENDED)
        assert mgr.should_show_thinking()

    def test_should_show_thinking_disabled(self):
        mgr = ThinkingManager(ThinkingConfig(show_thinking=False))
        mgr.set_mode(ThinkingMode.EXTENDED)
        assert not mgr.should_show_thinking()

    def test_reset_usage(self):
        mgr = ThinkingManager(ThinkingConfig(mode=ThinkingMode.EXTENDED))
        mgr.parse_response("<thinking>abc</thinking>ans")
        assert mgr._total_thinking_tokens > 0
        mgr.reset_usage()
        assert mgr._total_thinking_tokens == 0

    def test_format_thinking_empty(self):
        mgr = ThinkingManager()
        assert mgr.format_thinking("") == ""

    def test_format_thinking(self):
        mgr = ThinkingManager()
        formatted = mgr.format_thinking("<thinking>Step 1\nStep 2</thinking>")
        assert "Step 1" in formatted
        assert "Step 2" in formatted
        assert "│" in formatted

    def test_get_mode_indicator(self):
        mgr = ThinkingManager()
        assert mgr.get_mode_indicator() == ""
        mgr.set_mode(ThinkingMode.EXTENDED)
        assert mgr.get_mode_indicator() == "🧠"
        mgr.set_mode(ThinkingMode.DEEP)
        assert mgr.get_mode_indicator() == "🧠🧠"

    def test_parse_response_accumulates_tokens(self):
        mgr = ThinkingManager(ThinkingConfig(mode=ThinkingMode.EXTENDED))
        mgr.parse_response("<thinking>abcd</thinking>ef")
        mgr.parse_response("<thinking>ghij</thinking>kl")
        assert mgr._total_thinking_tokens > 0
