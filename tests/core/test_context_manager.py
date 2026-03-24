"""Unit tests for context_manager.py"""

from src.core.context_manager import ContextConfig, ContextManager


class TestContextManagerFixed:
    def test_initialization(self):
        config = ContextConfig(max_context_tokens=1000, auto_save=False)
        manager = ContextManager(project="test", config=config)
        assert manager.project == "test"

    def test_add_messages(self):
        manager = ContextManager(project="test", config=ContextConfig(auto_save=False))
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi")
        assert len(manager.messages) >= 2

    def test_clear_messages(self):
        manager = ContextManager(project="test", config=ContextConfig(auto_save=False))
        manager.add_user_message("Hello")
        manager.clear()
        assert len(manager.messages) == 0


class TestContextConfig:
    def test_default_values(self):
        config = ContextConfig()
        assert config.max_context_tokens > 0
