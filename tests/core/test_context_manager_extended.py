"""Extended tests for context_manager.py"""

from src.core.context_manager import ContextConfig, ContextManager


class TestContextManagerExtended:
    def test_add_user_message(self):
        manager = ContextManager(project="test", config=ContextConfig(auto_save=False))
        manager.add_user_message("Hello")
        assert len(manager.messages) > 0

    def test_add_assistant_message(self):
        manager = ContextManager(project="test", config=ContextConfig(auto_save=False))
        manager.add_assistant_message("Hi there")
        assert len(manager.messages) > 0

    def test_add_system_message(self):
        manager = ContextManager(project="test", config=ContextConfig(auto_save=False))
        manager.add_system_message("System info")
        assert len(manager.messages) > 0

    def test_messages_property(self):
        manager = ContextManager(project="test", config=ContextConfig(auto_save=False))
        manager.add_user_message("Test")
        assert isinstance(manager.messages, list)

    def test_clear_removes_all_messages(self):
        manager = ContextManager(project="test", config=ContextConfig(auto_save=False))
        manager.add_user_message("Test 1")
        manager.add_user_message("Test 2")
        manager.clear()
        assert len(manager.messages) == 0

    def test_project_name_stored(self):
        manager = ContextManager(project="my_project", config=ContextConfig(auto_save=False))
        assert manager.project == "my_project"
