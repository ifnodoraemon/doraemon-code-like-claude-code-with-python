"""
Unit tests for context_manager.py

Tests conversation history management, summarization, and token estimation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.context_manager import ContextConfig, ContextManager
from src.core.model_client import Message


class TestContextManager:
    """Tests for ContextManager."""

    def test_initialization(self):
        """Test that ContextManager initializes correctly."""
        config = ContextConfig(
            max_context_tokens=1000,
            summarize_threshold=0.7,
            keep_recent_messages=6,
            auto_save=False,
        )
        manager = ContextManager(config, project_name="test")

        assert manager.config == config
        assert manager.project_name == "test"
        assert len(manager.messages) == 0

    def test_add_message(self):
        """Test adding messages to history."""
        manager = ContextManager(
            ContextConfig(auto_save=False),
            project_name="test"
        )

        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi there!")

        assert len(manager.messages) == 2
        assert manager.messages[0]["role"] == "user"
        assert manager.messages[0]["content"] == "Hello"
        assert manager.messages[1]["role"] == "assistant"
        assert manager.messages[1]["content"] == "Hi there!"

    def test_add_message_with_thought(self):
        """Test adding message with thought process."""
        manager = ContextManager(
            ContextConfig(auto_save=False),
            project_name="test"
        )

        manager.add_message("assistant", "Response", thought="Thinking...")

        assert len(manager.messages) == 1
        assert manager.messages[0]["content"] == "Response"
        assert manager.messages[0]["thought"] == "Thinking..."

    def test_get_messages_returns_copy(self):
        """Test that get_messages returns a copy, not reference."""
        manager = ContextManager(
            ContextConfig(auto_save=False),
            project_name="test"
        )

        manager.add_message("user", "Hello")
        messages = manager.get_messages()
        messages.append({"role": "user", "content": "Modified"})

        # Original should be unchanged
        assert len(manager.messages) == 1
        assert manager.messages[0]["content"] == "Hello"

    def test_clear_messages(self):
        """Test clearing message history."""
        manager = ContextManager(
            ContextConfig(auto_save=False),
            project_name="test"
        )

        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi")
        assert len(manager.messages) == 2

        manager.clear()
        assert len(manager.messages) == 0

    def test_estimate_tokens_caching(self):
        """Test that token estimation is cached."""
        manager = ContextManager(
            ContextConfig(auto_save=False),
            project_name="test"
        )

        # Add a message
        manager.add_message("user", "Hello world")

        # First call should calculate
        tokens1 = manager.estimate_tokens()
        assert tokens1 > 0

        # Second call should use cache (same result)
        tokens2 = manager.estimate_tokens()
        assert tokens1 == tokens2

        # Add another message - cache should be invalidated
        manager.add_message("assistant", "Hi there")
        tokens3 = manager.estimate_tokens()
        assert tokens3 > tokens1

    @pytest.mark.asyncio
    async def test_summarization_triggered_at_threshold(self):
        """Test that summarization is triggered when threshold is reached."""
        config = ContextConfig(
            max_context_tokens=100,
            summarize_threshold=0.7,  # Trigger at 70 tokens
            keep_recent_messages=2,
            auto_save=False,
        )
        manager = ContextManager(config, project_name="test")

        # Mock the model client
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=MagicMock(
            content="Summary of conversation"
        ))

        # Add messages until we exceed threshold
        for i in range(10):
            manager.add_message("user", f"Message {i} " * 10)  # Long messages

        # Mock estimate_tokens to return value over threshold
        with patch.object(manager, 'estimate_tokens', return_value=80):
            with patch.object(manager, '_get_model_client', return_value=mock_client):
                await manager.maybe_summarize()

        # Should have called summarization
        mock_client.chat.assert_called_once()

        # Should keep recent messages + summary
        assert len(manager.messages) <= config.keep_recent_messages + 1

    @pytest.mark.asyncio
    async def test_no_summarization_below_threshold(self):
        """Test that summarization is NOT triggered below threshold."""
        config = ContextConfig(
            max_context_tokens=1000,
            summarize_threshold=0.7,
            keep_recent_messages=2,
            auto_save=False,
        )
        manager = ContextManager(config, project_name="test")

        mock_client = AsyncMock()

        # Add a few messages
        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi")

        # Mock estimate_tokens to return value below threshold
        with patch.object(manager, 'estimate_tokens', return_value=500):
            with patch.object(manager, '_get_model_client', return_value=mock_client):
                await manager.maybe_summarize()

        # Should NOT have called summarization
        mock_client.chat.assert_not_called()

    def test_get_recent_messages(self):
        """Test getting recent N messages."""
        manager = ContextManager(
            ContextConfig(auto_save=False),
            project_name="test"
        )

        for i in range(10):
            manager.add_message("user", f"Message {i}")

        recent = manager.get_recent_messages(3)
        assert len(recent) == 3
        assert recent[0]["content"] == "Message 7"
        assert recent[1]["content"] == "Message 8"
        assert recent[2]["content"] == "Message 9"

    def test_pop_last_message(self):
        """Test removing the last message."""
        manager = ContextManager(
            ContextConfig(auto_save=False),
            project_name="test"
        )

        manager.add_message("user", "First")
        manager.add_message("assistant", "Second")
        manager.add_message("user", "Third")

        assert len(manager.messages) == 3

        popped = manager.pop_last_message()
        assert popped["content"] == "Third"
        assert len(manager.messages) == 2

    def test_pop_last_message_empty(self):
        """Test popping from empty history returns None."""
        manager = ContextManager(
            ContextConfig(auto_save=False),
            project_name="test"
        )

        popped = manager.pop_last_message()
        assert popped is None

    def test_message_count(self):
        """Test getting message count."""
        manager = ContextManager(
            ContextConfig(auto_save=False),
            project_name="test"
        )

        assert manager.message_count() == 0

        manager.add_message("user", "Hello")
        assert manager.message_count() == 1

        manager.add_message("assistant", "Hi")
        assert manager.message_count() == 2

    def test_has_messages(self):
        """Test checking if history has messages."""
        manager = ContextManager(
            ContextConfig(auto_save=False),
            project_name="test"
        )

        assert not manager.has_messages()

        manager.add_message("user", "Hello")
        assert manager.has_messages()

    @pytest.mark.asyncio
    async def test_save_and_load(self, temp_dir):
        """Test saving and loading conversation history."""
        config = ContextConfig(
            auto_save=False,
            conversation_dir=str(temp_dir)
        )
        manager = ContextManager(config, project_name="test")

        # Add messages
        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi there!")

        # Save
        await manager.save()

        # Create new manager and load
        manager2 = ContextManager(config, project_name="test")
        await manager2.load()

        assert len(manager2.messages) == 2
        assert manager2.messages[0]["content"] == "Hello"
        assert manager2.messages[1]["content"] == "Hi there!"


class TestContextConfig:
    """Tests for ContextConfig."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = ContextConfig()

        assert config.max_context_tokens == 100_000
        assert config.summarize_threshold == 0.7
        assert config.keep_recent_messages == 6
        assert config.auto_save is True

    def test_custom_values(self):
        """Test setting custom values."""
        config = ContextConfig(
            max_context_tokens=50_000,
            summarize_threshold=0.8,
            keep_recent_messages=10,
            auto_save=False,
        )

        assert config.max_context_tokens == 50_000
        assert config.summarize_threshold == 0.8
        assert config.keep_recent_messages == 10
        assert config.auto_save is False

    def test_validation_threshold_range(self):
        """Test that threshold must be between 0 and 1."""
        # Valid values
        ContextConfig(summarize_threshold=0.0)
        ContextConfig(summarize_threshold=0.5)
        ContextConfig(summarize_threshold=1.0)

        # Invalid values should raise
        with pytest.raises(ValueError):
            ContextConfig(summarize_threshold=-0.1)

        with pytest.raises(ValueError):
            ContextConfig(summarize_threshold=1.1)

    def test_validation_positive_values(self):
        """Test that certain values must be positive."""
        # Valid
        ContextConfig(max_context_tokens=1000)
        ContextConfig(keep_recent_messages=1)

        # Invalid
        with pytest.raises(ValueError):
            ContextConfig(max_context_tokens=0)

        with pytest.raises(ValueError):
            ContextConfig(keep_recent_messages=0)
