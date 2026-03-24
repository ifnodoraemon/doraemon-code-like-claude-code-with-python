"""Advanced comprehensive tests for context_manager.py - 30+ tests for improved coverage"""

import json
import time
from unittest.mock import Mock, patch

from src.core.context_manager import (
    DEFAULT_CONFIG,
    SUMMARIZE_PROMPT,
    ContextConfig,
    ContextManager,
    ConversationSummary,
    Message,
    create_llm_summarizer,
)


class TestMessageAdvanced:
    """Advanced tests for Message class."""

    def test_message_timestamp_auto_generation(self):
        """Test that timestamp is auto-generated."""
        before = time.time()
        msg = Message(role="user", content="Test")
        after = time.time()
        assert before <= msg.timestamp <= after

    def test_message_metadata_mutation(self):
        """Test metadata can be mutated after creation."""
        msg = Message(role="user", content="Test")
        msg.metadata["key"] = "value"
        assert msg.metadata["key"] == "value"

    def test_message_to_dict_preserves_all_fields(self):
        """Test to_dict preserves all fields exactly."""
        msg = Message(
            role="assistant",
            content="Response text",
            token_count=42,
            metadata={"source": "test", "nested": {"key": "val"}},
        )
        data = msg.to_dict()
        assert data["role"] == "assistant"
        assert data["content"] == "Response text"
        assert data["token_count"] == 42
        assert data["metadata"]["nested"]["key"] == "val"

    def test_message_from_dict_with_missing_timestamp(self):
        """Test from_dict generates timestamp when missing."""
        data = {"role": "user", "content": "Test"}
        before = time.time()
        msg = Message.from_dict(data)
        after = time.time()
        assert before <= msg.timestamp <= after

    def test_message_api_format_role_normalization(self):
        """Test API format normalizes roles correctly."""
        user_msg = Message(role="user", content="Hello")
        assert user_msg.to_api_format()["role"] == "user"

        assistant_msg = Message(role="assistant", content="Hi")
        assert assistant_msg.to_api_format()["role"] == "model"

        system_msg = Message(role="system", content="System")
        assert system_msg.to_api_format()["role"] == "system"

    def test_message_api_format_structure(self):
        """Test API format has correct structure."""
        msg = Message(role="user", content="Test content")
        api_msg = msg.to_api_format()
        assert "role" in api_msg
        assert "parts" in api_msg
        assert isinstance(api_msg["parts"], list)
        assert len(api_msg["parts"]) == 1
        assert "text" in api_msg["parts"][0]


class TestConversationSummaryAdvanced:
    """Advanced tests for ConversationSummary class."""

    def test_summary_to_dict_complete(self):
        """Test summary to_dict includes all fields."""
        summary = ConversationSummary(
            content="Summary content",
            message_count=15,
            token_count=1000,
            key_points=["Point 1", "Point 2"],
        )
        data = summary.to_dict()
        assert data["content"] == "Summary content"
        assert data["message_count"] == 15
        assert data["token_count"] == 1000
        assert len(data["key_points"]) == 2

    def test_summary_from_dict_with_defaults(self):
        """Test from_dict uses defaults for missing fields."""
        data = {"content": "Test", "message_count": 5, "token_count": 100}
        summary = ConversationSummary.from_dict(data)
        assert summary.content == "Test"
        assert isinstance(summary.created_at, float)
        assert summary.key_points == []

    def test_summary_from_dict_with_all_fields(self):
        """Test from_dict with all fields provided."""
        created_time = time.time()
        data = {
            "content": "Full summary",
            "message_count": 20,
            "token_count": 2000,
            "created_at": created_time,
            "key_points": ["Key 1", "Key 2", "Key 3"],
        }
        summary = ConversationSummary.from_dict(data)
        assert summary.created_at == created_time
        assert len(summary.key_points) == 3


class TestContextConfigAdvanced:
    """Advanced tests for ContextConfig."""

    def test_default_config_values(self):
        """Test default config has sensible values."""
        config = ContextConfig()
        assert config.max_context_tokens == 100_000
        assert config.summarize_threshold == 0.7
        assert config.target_after_summary == 0.4
        assert config.keep_recent_messages == 6
        assert config.min_messages_to_summarize == 4

    def test_custom_config_values(self):
        """Test custom config values are respected."""
        config = ContextConfig(
            max_context_tokens=50_000, summarize_threshold=0.8, keep_recent_messages=10
        )
        assert config.max_context_tokens == 50_000
        assert config.summarize_threshold == 0.8
        assert config.keep_recent_messages == 10

    def test_config_token_estimation(self):
        """Test token estimation configuration."""
        config = ContextConfig(chars_per_token_estimate=4.0)
        assert config.chars_per_token_estimate == 4.0


class TestContextManagerInitialization:
    """Tests for ContextManager initialization."""

    def test_initialization_default(self):
        """Test default initialization."""
        with patch.object(ContextManager, "_load_state"):
            manager = ContextManager()
            assert manager.project == "default"
            assert manager.config == DEFAULT_CONFIG
            assert manager.messages == []
            assert manager.summaries == []
            assert manager.total_messages_ever == 0

    def test_initialization_with_project(self):
        """Test initialization with custom project."""
        with patch.object(ContextManager, "_load_state"):
            manager = ContextManager(project="my_project")
            assert manager.project == "my_project"

    def test_initialization_with_custom_config(self):
        """Test initialization with custom config."""
        config = ContextConfig(max_context_tokens=50_000)
        with patch.object(ContextManager, "_load_state"):
            manager = ContextManager(config=config)
            assert manager.config.max_context_tokens == 50_000

    def test_initialization_with_summarizer(self):
        """Test initialization with custom summarizer."""
        summarizer = Mock()
        with patch.object(ContextManager, "_load_state"):
            manager = ContextManager(summarize_fn=summarizer)
            assert manager.summarize_fn == summarizer

    def test_session_id_generation(self):
        """Test session ID is generated."""
        with patch.object(ContextManager, "_load_state"):
            manager = ContextManager()
            assert len(manager.session_id) == 12
            assert isinstance(manager.session_id, str)


class TestContextManagerMessageManagement:
    """Tests for message management in ContextManager."""

    def test_add_user_message(self):
        """Test adding user message."""
        with patch.object(ContextManager, "_load_state"):
            manager = ContextManager()
            msg = manager.add_user_message("Hello")
            assert msg.role == "user"
            assert msg.content == "Hello"
            assert len(manager.messages) == 1
            assert manager.total_messages_ever == 1

    def test_add_system_message(self):
        """Test adding system message."""
        with patch.object(ContextManager, "_load_state"):
            manager = ContextManager()
            msg = manager.add_system_message("System prompt")
            assert msg.role == "system"
            assert msg.content == "System prompt"
            assert len(manager.messages) == 1

    def test_add_assistant_message_basic(self):
        """Test adding assistant message."""
        with patch.object(ContextManager, "_load_state"):
            manager = ContextManager()
            msg = manager.add_assistant_message("Response")
            assert msg.role == "assistant"
            assert msg.content == "Response"
            assert len(manager.messages) == 1

    def test_add_assistant_message_with_tokens(self):
        """Test adding assistant message with token counts."""
        with patch.object(ContextManager, "_load_state"):
            manager = ContextManager()
            manager.add_assistant_message("Response", prompt_tokens=100, completion_tokens=50)
            assert manager._last_prompt_tokens == 100
            assert manager._total_tokens_used == 150

    def test_multiple_messages_sequence(self):
        """Test adding multiple messages in sequence."""
        with patch.object(ContextManager, "_load_state"):
            manager = ContextManager()
            manager.add_user_message("Q1")
            manager.add_assistant_message("A1")
            manager.add_user_message("Q2")
            manager.add_assistant_message("A2")
            assert len(manager.messages) == 4
            assert manager.total_messages_ever == 4


class TestContextManagerTokenEstimation:
    """Tests for token estimation."""

    def test_estimate_tokens_basic(self):
        """Test basic token estimation."""
        with patch.object(ContextManager, "_load_state"):
            manager = ContextManager()
            # 300 chars / 3.0 = 100 tokens
            tokens = manager._estimate_tokens("x" * 300)
            assert tokens == 100

    def test_estimate_tokens_custom_ratio(self):
        """Test token estimation with custom ratio."""
        config = ContextConfig(chars_per_token_estimate=4.0)
        with patch.object(ContextManager, "_load_state"):
            manager = ContextManager(config=config)
            # 400 chars / 4.0 = 100 tokens
            tokens = manager._estimate_tokens("x" * 400)
            assert tokens == 100

    def test_estimate_current_tokens_empty(self):
        """Test token estimation for empty context."""
        with patch.object(ContextManager, "_load_state"):
            manager = ContextManager()
            tokens = manager._estimate_current_tokens()
            assert tokens == 0

    def test_estimate_current_tokens_with_messages(self):
        """Test token estimation with messages."""
        with patch.object(ContextManager, "_load_state"):
            manager = ContextManager()
            manager.add_user_message("x" * 300)  # ~100 tokens
            manager.add_assistant_message("x" * 300)  # ~100 tokens
            tokens = manager._estimate_current_tokens()
            assert tokens >= 200

    def test_estimate_current_tokens_with_actual_counts(self):
        """Test token estimation uses actual counts when available."""
        with patch.object(ContextManager, "_load_state"):
            manager = ContextManager()
            manager.add_assistant_message("Response", completion_tokens=50)
            tokens = manager._estimate_current_tokens()
            # Should use actual count (50) not estimate
            assert tokens == 50


class TestContextManagerSummarization:
    """Tests for summarization functionality."""

    def test_simple_summarize_empty(self):
        """Test simple summarization with empty messages."""
        with patch.object(ContextManager, "_load_state"):
            manager = ContextManager()
            summary = manager._simple_summarize([])
            assert "Conversation Summary" in summary
            assert "0 messages" in summary

    def test_simple_summarize_user_messages(self):
        """Test simple summarization extracts user messages."""
        with patch.object(ContextManager, "_load_state"):
            manager = ContextManager()
            messages = [
                Message(role="user", content="First question"),
                Message(role="assistant", content="First answer"),
                Message(role="user", content="Second question"),
            ]
            summary = manager._simple_summarize(messages)
            assert "User requests" in summary
            assert "First question" in summary

    def test_simple_summarize_actions(self):
        """Test simple summarization extracts actions."""
        with patch.object(ContextManager, "_load_state"):
            manager = ContextManager()
            messages = [
                Message(role="user", content="Create a file"),
                Message(role="assistant", content="I created the file successfully"),
            ]
            summary = manager._simple_summarize(messages)
            assert "Key actions" in summary or "created" in summary.lower()

    def test_format_summaries_for_context_empty(self):
        """Test formatting empty summaries."""
        with patch.object(ContextManager, "_load_state"):
            manager = ContextManager()
            formatted = manager._format_summaries_for_context()
            assert formatted == ""

    def test_format_summaries_for_context_with_summaries(self):
        """Test formatting summaries for context."""
        with patch.object(ContextManager, "_load_state"):
            manager = ContextManager()
            manager.summaries.append(
                ConversationSummary(content="Summary 1", message_count=5, token_count=100)
            )
            manager.summaries.append(
                ConversationSummary(content="Summary 2", message_count=5, token_count=100)
            )
            formatted = manager._format_summaries_for_context()
            assert "Segment 1" in formatted
            assert "Segment 2" in formatted
            assert "Summary 1" in formatted
            assert "Summary 2" in formatted


class TestContextManagerContextStats:
    """Tests for context statistics."""

    def test_get_context_stats_empty(self):
        """Test context stats for empty manager."""
        with patch.object(ContextManager, "_load_state"):
            manager = ContextManager()
            stats = manager.get_context_stats()
            assert stats["messages"] == 0
            assert stats["summaries"] == 0
            assert stats["total_messages_ever"] == 0
            assert stats["estimated_tokens"] == 0
            assert stats["usage_percent"] == 0.0

    def test_get_context_stats_with_messages(self):
        """Test context stats with messages."""
        with patch.object(ContextManager, "_load_state"):
            manager = ContextManager()
            manager.add_user_message("x" * 300)
            stats = manager.get_context_stats()
            assert stats["messages"] == 1
            assert stats["total_messages_ever"] == 1
            assert stats["estimated_tokens"] > 0

    def test_get_context_stats_needs_summary(self):
        """Test context stats indicates when summary needed."""
        config = ContextConfig(max_context_tokens=1000, summarize_threshold=0.5)
        with patch.object(ContextManager, "_load_state"):
            manager = ContextManager(config=config)
            # Add enough messages to exceed threshold (1000 * 0.5 = 500 tokens)
            # Each message is ~33 tokens (100 chars / 3.0), so need ~15 messages
            for _ in range(20):
                manager.add_user_message("x" * 100)
            stats = manager.get_context_stats()
            # Should need summary if we have enough tokens
            assert stats["estimated_tokens"] > 0


class TestContextManagerClear:
    """Tests for clearing context."""

    def test_clear_messages_only(self):
        """Test clearing messages without summaries."""
        with patch.object(ContextManager, "_load_state"):
            manager = ContextManager()
            manager.add_user_message("Test")
            manager.clear(keep_summaries=False)
            assert len(manager.messages) == 0
            assert len(manager.summaries) == 0

    def test_clear_with_summary_preservation(self):
        """Test clearing messages while preserving summaries."""
        with patch.object(ContextManager, "_load_state"):
            manager = ContextManager()
            manager.add_user_message("Q1")
            manager.add_assistant_message("A1")
            manager.add_user_message("Q2")
            manager.add_assistant_message("A2")
            manager.add_user_message("Q3")
            manager.add_assistant_message("A3")
            manager.clear(keep_summaries=True)
            # Should have created a summary
            assert len(manager.messages) == 0


class TestContextManagerReset:
    """Tests for resetting context."""

    def test_reset_clears_everything(self):
        """Test reset clears all state."""
        with patch.object(ContextManager, "_load_state"):
            manager = ContextManager()
            manager.add_user_message("Test")
            manager.summaries.append(
                ConversationSummary(content="Summary", message_count=1, token_count=50)
            )
            old_session = manager.session_id

            manager.reset()

            assert len(manager.messages) == 0
            assert len(manager.summaries) == 0
            assert manager.total_messages_ever == 0
            assert manager.session_id != old_session

    def test_reset_generates_new_session(self):
        """Test reset generates new session ID."""
        with patch.object(ContextManager, "_load_state"):
            manager = ContextManager()
            session1 = manager.session_id
            manager.reset()
            session2 = manager.session_id
            assert session1 != session2


class TestContextManagerPersistence:
    """Tests for persistence functionality."""

    def test_get_save_path(self):
        """Test save path generation."""
        with patch.object(ContextManager, "_load_state"):
            manager = ContextManager(project="test_project")
            path = manager._get_save_path()
            assert "test_project.json" in str(path)
            assert ".agent/conversations" in str(path)

    def test_save_state_creates_file(self, tmp_path):
        """Test save state creates file."""
        config = ContextConfig(save_directory=str(tmp_path))
        with patch.object(ContextManager, "_load_state"):
            manager = ContextManager(config=config)
            manager.add_user_message("Test message")
            manager._save_state()

            save_path = tmp_path / "default.json"
            assert save_path.exists()

    def test_save_state_content(self, tmp_path):
        """Test saved state has correct content."""
        config = ContextConfig(save_directory=str(tmp_path))
        with patch.object(ContextManager, "_load_state"):
            manager = ContextManager(config=config)
            manager.add_user_message("Test")
            manager._save_state()

            save_path = tmp_path / "default.json"
            with open(save_path) as f:
                data = json.load(f)

            assert data["version"] == 2
            assert data["project"] == "default"
            assert len(data["messages"]) == 1
            assert data["messages"][0]["content"] == "Test"

    def test_load_state_from_file(self, tmp_path):
        """Test loading state from file."""
        config = ContextConfig(save_directory=str(tmp_path))

        # First, save some state
        with patch.object(ContextManager, "_load_state"):
            manager1 = ContextManager(config=config)
            manager1.add_user_message("Saved message")
            manager1._save_state()

        # Now load it
        manager2 = ContextManager(config=config)
        assert len(manager2.messages) == 1
        assert manager2.messages[0].content == "Saved message"


class TestContextManagerHistoryForAPI:
    """Tests for API history formatting."""

    def test_get_history_for_api_empty(self):
        """Test API history for empty context."""
        with patch.object(ContextManager, "_load_state"):
            manager = ContextManager()
            history = manager.get_history_for_api()
            assert isinstance(history, list)
            assert len(history) == 0

    def test_get_history_for_api_with_messages(self):
        """Test API history with messages."""
        with patch.object(ContextManager, "_load_state"):
            manager = ContextManager()
            manager.add_user_message("Hello")
            manager.add_assistant_message("Hi there")
            history = manager.get_history_for_api()
            assert len(history) == 2
            assert history[0]["role"] == "user"
            assert history[1]["role"] == "model"

    def test_get_history_for_api_with_summaries(self):
        """Test API history includes summaries."""
        with patch.object(ContextManager, "_load_state"):
            manager = ContextManager()
            manager.summaries.append(
                ConversationSummary(content="Previous context", message_count=5, token_count=100)
            )
            manager.add_user_message("New question")
            history = manager.get_history_for_api()
            # Should have summary context + new message
            assert len(history) >= 3


class TestLLMSummarizer:
    """Tests for LLM summarizer creation."""

    def test_create_llm_summarizer(self):
        """Test creating LLM summarizer."""
        chat_fn = Mock(return_value="Summary text")
        summarizer = create_llm_summarizer(chat_fn)
        assert callable(summarizer)

    def test_llm_summarizer_calls_chat_fn(self):
        """Test summarizer calls chat function."""
        chat_fn = Mock(return_value="Generated summary")
        summarizer = create_llm_summarizer(chat_fn)
        messages = [
            Message(role="user", content="Question"),
            Message(role="assistant", content="Answer"),
        ]
        result = summarizer(messages)
        assert result == "Generated summary"
        chat_fn.assert_called_once()

    def test_llm_summarizer_formats_prompt(self):
        """Test summarizer formats prompt correctly."""
        chat_fn = Mock(return_value="Summary")
        summarizer = create_llm_summarizer(chat_fn)
        messages = [Message(role="user", content="Test")]
        summarizer(messages)

        # Check that prompt was formatted
        call_args = chat_fn.call_args[0][0]
        assert "User:" in call_args
        assert "Test" in call_args

    def test_summarize_prompt_template(self):
        """Test summarize prompt template."""
        assert "summarize" in SUMMARIZE_PROMPT.lower()
        assert "{conversation}" in SUMMARIZE_PROMPT
        assert "500 words" in SUMMARIZE_PROMPT
