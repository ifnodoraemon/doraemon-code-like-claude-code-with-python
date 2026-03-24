"""Tests for errors.py extended functionality"""

import pytest

from src.core.errors import (
    AgentError,
    ConfigurationError,
    ErrorCategory,
    ErrorInfo,
    RateLimitError,
    RetryConfig,
    RetryPolicy,
    TransientError,
    retry,
)


class TestErrorCategories:
    """Tests for error categories."""

    def test_error_categories_exist(self):
        """Test that all error categories exist."""
        assert ErrorCategory.TRANSIENT
        assert ErrorCategory.PERMANENT
        assert ErrorCategory.CONFIGURATION
        assert ErrorCategory.RATE_LIMIT

    def test_error_info_creation(self):
        """Test ErrorInfo creation."""
        exc = Exception("test")
        info = ErrorInfo(
            category=ErrorCategory.TRANSIENT,
            message="Test error",
            original_exception=exc,
            retry_after=5.0,
        )
        assert info.category == ErrorCategory.TRANSIENT
        assert info.retry_after == 5.0


class TestCustomExceptions:
    """Tests for custom exceptions."""

    def test_agent_error(self):
        """Test AgentError."""
        exc = AgentError("Test", category=ErrorCategory.PERMANENT)
        assert exc.message == "Test"
        assert exc.category == ErrorCategory.PERMANENT

    def test_configuration_error(self):
        """Test ConfigurationError."""
        exc = ConfigurationError("Config error")
        assert exc.category == ErrorCategory.CONFIGURATION

    def test_transient_error(self):
        """Test TransientError."""
        exc = TransientError("Transient", retry_after=2.0)
        assert exc.category == ErrorCategory.TRANSIENT
        assert exc.retry_after == 2.0

    def test_rate_limit_error(self):
        """Test RateLimitError."""
        exc = RateLimitError("Rate limited", retry_after=60.0)
        assert exc.category == ErrorCategory.RATE_LIMIT
        assert exc.retry_after == 60.0


class TestRetryPolicy:
    """Tests for RetryPolicy."""

    def test_retry_config_defaults(self):
        """Test RetryConfig default values."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.initial_delay == 1.0
        assert config.jitter is True

    def test_retry_policy_success_first_try(self):
        """Test retry policy succeeds on first try."""
        policy = RetryPolicy()

        def success_func():
            return "success"

        result = policy.execute(success_func)
        assert result == "success"

    def test_retry_policy_retries_on_transient_error(self):
        """Test retry policy retries on transient error."""
        policy = RetryPolicy(RetryConfig(max_attempts=3))

        attempts = []

        def failing_func():
            attempts.append(1)
            if len(attempts) < 2:
                raise TransientError("Fail")
            return "success"

        result = policy.execute(failing_func)
        assert result == "success"
        assert len(attempts) == 2

    @pytest.mark.asyncio
    async def test_retry_policy_async(self):
        """Test async retry policy."""
        policy = RetryPolicy()

        async def async_success():
            return "async_success"

        result = await policy.execute_async(async_success)
        assert result == "async_success"

    def test_retry_decorator(self):
        """Test retry decorator."""
        attempts = []

        @retry(max_attempts=3, initial_delay=0.01)
        def decorated_func():
            attempts.append(1)
            if len(attempts) < 2:
                raise TransientError("Fail")
            return "decorated_success"

        result = decorated_func()
        assert result == "decorated_success"
        assert len(attempts) == 2

    @pytest.mark.asyncio
    async def test_retry_decorator_async(self):
        """Test async retry decorator."""
        attempts = []

        @retry(max_attempts=3, initial_delay=0.01)
        async def async_decorated():
            attempts.append(1)
            if len(attempts) < 2:
                raise TransientError("Fail")
            return "async_decorated_success"

        result = await async_decorated()
        assert result == "async_decorated_success"
        assert len(attempts) == 2
