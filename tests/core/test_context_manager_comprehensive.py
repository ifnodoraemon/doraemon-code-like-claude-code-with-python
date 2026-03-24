"""Additional comprehensive tests for context_manager.py"""

from src.core.context_manager import (
    DEFAULT_CONFIG,
    SUMMARIZE_PROMPT,
    ContextConfig,
    ContextManager,
    ConversationSummary,
    Message,
    create_llm_summarizer,
)


class TestMessage:
    """Tests for Message class."""

    def test_creation_basic(self):
        """Test creating a basic message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.token_count is None
        assert isinstance(msg.timestamp, float)

    def test_creation_with_token_count(self):
        """Test creating message with token count."""
        msg = Message(role="assistant", content="Response", token_count=50)
        assert msg.token_count == 50

    def test_creation_with_metadata(self):
        """Test creating message with metadata."""
        metadata = {"source": "api", "model": "gemini"}
        msg = Message(role="user", content="Test", metadata=metadata)
        assert msg.metadata["source"] == "api"
        assert msg.metadata["model"] == "gemini"

    def test_to_dict(self):
        """Test converting message to dictionary."""
        msg = Message(role="user", content="Test", token_count=10)
        data = msg.to_dict()
        assert data["role"] == "user"
        assert data["content"] == "Test"
        assert data["token_count"] == 10
        assert "timestamp" in data

    def test_from_dict(self):
        """Test creating message from dictionary."""
        data = {
            "role": "assistant",
            "content": "Response",
            "timestamp": 1234567890.0,
            "token_count": 25,
            "metadata": {"key": "value"},
        }
        msg = Message.from_dict(data)
        assert msg.role == "assistant"
        assert msg.content == "Response"
        assert msg.token_count == 25
        assert msg.metadata["key"] == "value"

    def test_from_dict_with_defaults(self):
        """Test from_dict with missing optional fields."""
        data = {"role": "user", "content": "Test"}
        msg = Message.from_dict(data)
        assert msg.token_count is None
        assert msg.metadata == {}

    def test_to_api_format_user(self):
        """Test converting user message to API format."""
        msg = Message(role="user", content="Hello")
        api_msg = msg.to_api_format()
        assert api_msg["role"] == "user"
        assert api_msg["parts"][0]["text"] == "Hello"

    def test_to_api_format_assistant(self):
        """Test converting assistant message to API format."""
        msg = Message(role="assistant", content="Response")
        api_msg = msg.to_api_format()
        # Assistant should be converted to "model" for Gemini
        assert api_msg["role"] == "model"
        assert api_msg["parts"][0]["text"] == "Response"


class TestConversationSummary:
    """Tests for ConversationSummary class."""

    def test_creation(self):
        """Test creating a conversation summary."""
        summary = ConversationSummary(content="Summary text", message_count=10, token_count=500)
        assert summary.content == "Summary text"
        assert summary.message_count == 10
        assert summary.token_count == 500

    def test_creation_with_key_points(self):
        """Test creating summary with key points."""
        summary = ConversationSummary(
            content="Summary", message_count=5, token_count=200, key_points=["Point 1", "Point 2"]
        )
        assert len(summary.key_points) == 2

    def test_to_dict(self):
        """Test converting summary to dictionary."""
        summary = ConversationSummary(
            content="Test summary", message_count=8, token_count=400, key_points=["Key point"]
        )
        data = summary.to_dict()
        assert data["content"] == "Test summary"
        assert data["message_count"] == 8
        assert data["token_count"] == 400
        assert len(data["key_points"]) == 1

    def test_from_dict(self):
        """Test creating summary from dictionary."""
        data = {
            "content": "Summary from dict",
            "message_count": 12,
            "token_count": 600,
            "created_at": 1234567890.0,
            "key_points": ["Point A", "Point B"],
        }
        summary = ConversationSummary.from_dict(data)
        assert summary.content == "Summary from dict"
        assert summary.message_count == 12
        assert len(summary.key_points) == 2


class TestContextConfig:
    """Tests for ContextConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ContextConfig()
        assert config.max_context_tokens == 100_000
        assert config.summarize_threshold == 0.7
        assert config.keep_recent_messages == 6
        assert config.auto_save is True

    def test_custom_values(self):
        """Test creating config with custom values."""
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

    def test_default_config_constant(self):
        """Test that DEFAULT_CONFIG is defined."""
        assert isinstance(DEFAULT_CONFIG, ContextConfig)


class TestContextManagerBasic:
    """Basic tests for ContextManager."""

    def test_initialization_defaults(self):
        """Test initialization with defaults."""
        manager = ContextManager(project="test", config=ContextConfig(auto_save=False))
        assert manager.project == "test"
        assert len(manager.messages) == 0
        assert len(manager.summaries) == 0
        assert manager.total_messages_ever == 0

    def test_initialization_with_custom_config(self):
        """Test initialization with custom config."""
        config = ContextConfig(max_context_tokens=50_000, auto_save=False)
        manager = ContextManager(project="custom", config=config)
        assert manager.config.max_context_tokens == 50_000

    def test_add_user_message_increments_count(self):
        """Test that adding user message increments counter."""
        manager = ContextManager(project="test", config=ContextConfig(auto_save=False))
        manager.add_user_message("Hello")
        assert manager.total_messages_ever == 1
        manager.add_user_message("World")
        assert manager.total_messages_ever == 2

    def test_add_system_message_increments_count(self):
        """Test that adding system message increments counter."""
        manager = ContextManager(project="test", config=ContextConfig(auto_save=False))
        manager.add_system_message("System info")
        assert manager.total_messages_ever == 1

    def test_add_assistant_message_with_tokens(self):
        """Test adding assistant message with token counts."""
        manager = ContextManager(project="test", config=ContextConfig(auto_save=False))
        manager.add_assistant_message("Response", prompt_tokens=100, completion_tokens=50)
        assert manager._last_prompt_tokens == 100
        assert manager._total_tokens_used == 150

    def test_get_history_for_api_empty(self):
        """Test getting API history when empty."""
        manager = ContextManager(project="test", config=ContextConfig(auto_save=False))
        history = manager.get_history_for_api()
        assert len(history) == 0

    def test_get_history_for_api_with_messages(self):
        """Test getting API history with messages."""
        manager = ContextManager(project="test", config=ContextConfig(auto_save=False))
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi there")
        history = manager.get_history_for_api()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "model"  # Converted to Gemini format

    def test_get_history_for_api_with_summary(self):
        """Test that summary is prepended to history."""
        manager = ContextManager(project="test", config=ContextConfig(auto_save=False))
        summary = ConversationSummary(
            content="Previous conversation", message_count=5, token_count=200
        )
        manager.summaries.append(summary)
        manager.add_user_message("New message")

        history = manager.get_history_for_api()
        # Should have: summary user, summary model, actual user
        assert len(history) == 3
        assert "[Previous Conversation Summary]" in history[0]["parts"][0]["text"]

    def test_get_context_stats(self):
        """Test getting context statistics."""
        manager = ContextManager(project="test", config=ContextConfig(auto_save=False))
        manager.add_user_message("Test message")
        stats = manager.get_context_stats()
        assert stats["messages"] == 1
        assert stats["summaries"] == 0
        assert stats["total_messages_ever"] == 1
        assert "estimated_tokens" in stats
        assert "session_id" in stats

    def test_clear_without_summaries(self):
        """Test clearing messages without keeping summaries."""
        manager = ContextManager(project="test", config=ContextConfig(auto_save=False))
        manager.add_user_message("Message 1")
        manager.add_user_message("Message 2")
        manager.clear(keep_summaries=False)
        assert len(manager.messages) == 0
        assert len(manager.summaries) == 0

    def test_reset(self):
        """Test complete reset."""
        manager = ContextManager(project="test", config=ContextConfig(auto_save=False))
        manager.add_user_message("Test")
        manager.total_messages_ever = 10
        old_session = manager.session_id

        manager.reset()
        assert len(manager.messages) == 0
        assert manager.total_messages_ever == 0
        assert manager.session_id != old_session


class TestContextManagerSummarization:
    """Tests for summarization functionality."""

    def test_estimate_tokens(self):
        """Test token estimation."""
        manager = ContextManager(project="test", config=ContextConfig(auto_save=False))
        tokens = manager._estimate_tokens("Hello world")
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_estimate_current_tokens_empty(self):
        """Test estimating tokens with no messages."""
        manager = ContextManager(project="test", config=ContextConfig(auto_save=False))
        tokens = manager._estimate_current_tokens()
        assert tokens == 0

    def test_estimate_current_tokens_with_messages(self):
        """Test estimating tokens with messages."""
        manager = ContextManager(project="test", config=ContextConfig(auto_save=False))
        manager.add_user_message("Hello world")
        manager.add_assistant_message("Hi there")
        tokens = manager._estimate_current_tokens()
        assert tokens > 0

    def test_simple_summarize(self):
        """Test simple summarization fallback."""
        manager = ContextManager(project="test", config=ContextConfig(auto_save=False))
        messages = [
            Message(role="user", content="Can you help me?"),
            Message(role="assistant", content="I created a new file."),
            Message(role="user", content="Thanks!"),
        ]
        summary = manager._simple_summarize(messages)
        assert "Conversation Summary" in summary
        assert "Can you help me?" in summary or "help" in summary.lower()

    def test_format_summaries_for_context_empty(self):
        """Test formatting empty summaries."""
        manager = ContextManager(project="test", config=ContextConfig(auto_save=False))
        formatted = manager._format_summaries_for_context()
        assert formatted == ""

    def test_format_summaries_for_context_with_summaries(self):
        """Test formatting summaries for context."""
        manager = ContextManager(project="test", config=ContextConfig(auto_save=False))
        summary = ConversationSummary(content="Summary content", message_count=5, token_count=200)
        manager.summaries.append(summary)
        formatted = manager._format_summaries_for_context()
        assert "Segment 1" in formatted
        assert "Summary content" in formatted


class TestContextManagerPersistence:
    """Tests for persistence functionality."""

    def test_get_save_path(self, tmp_path):
        """Test getting save path."""
        config = ContextConfig(auto_save=False, save_directory=str(tmp_path / "conversations"))
        manager = ContextManager(project="test", config=config)
        path = manager._get_save_path()
        assert path.name == "test.json"
        assert path.parent.exists()

    def test_save_and_load_state(self, tmp_path):
        """Test saving and loading state."""
        config = ContextConfig(auto_save=False, save_directory=str(tmp_path / "conversations"))
        manager = ContextManager(project="test", config=config)
        manager.add_user_message("Test message")
        manager.add_assistant_message("Response")

        # Save
        manager._save_state()

        # Create new manager and load
        manager2 = ContextManager(project="test", config=config)
        manager2._load_state()

        assert len(manager2.messages) == 2
        assert manager2.messages[0].content == "Test message"
        assert manager2.messages[1].content == "Response"


class TestLLMSummarizer:
    """Tests for LLM summarizer helper."""

    def test_create_llm_summarizer(self):
        """Test creating LLM summarizer."""

        def mock_chat(prompt):
            return "This is a summary"

        summarizer = create_llm_summarizer(mock_chat)
        assert callable(summarizer)

    def test_llm_summarizer_execution(self):
        """Test executing LLM summarizer."""

        def mock_chat(prompt):
            assert "summarize" in prompt.lower()
            return "Generated summary"

        summarizer = create_llm_summarizer(mock_chat)
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
        ]
        result = summarizer(messages)
        assert result == "Generated summary"

    def test_summarize_prompt_defined(self):
        """Test that SUMMARIZE_PROMPT is defined."""
        assert isinstance(SUMMARIZE_PROMPT, str)
        assert "summarize" in SUMMARIZE_PROMPT.lower()
        assert "{conversation}" in SUMMARIZE_PROMPT
