"""
Unit tests for Error Handling and Retry Mechanisms.

Tests error categorization, retry policies, and circuit breaker pattern.
"""

import time

import pytest

from src.core.errors import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitState,
    ConfigurationError,
    ErrorCategory,
    ErrorHandler,
    PolymathException,
    RateLimitError,
    RetryConfig,
    RetryPolicy,
    TransientError,
    get_error_handler,
    retry,
)


class _CircuitBreakerError(Exception):
    """Custom exception for circuit breaker tests."""

    pass


class TestPolymathExceptions:
    """Tests for custom exception classes"""

    def test_polymath_exception(self):
        """Test base PolymathException"""
        exc = PolymathException("Test error", ErrorCategory.NETWORK, {"key": "value"})

        assert str(exc) == "Test error"
        assert exc.category == ErrorCategory.NETWORK
        assert exc.context == {"key": "value"}

    def test_configuration_error(self):
        """Test ConfigurationError"""
        exc = ConfigurationError("Invalid config", {"file": "config.json"})

        assert exc.category == ErrorCategory.CONFIGURATION
        assert "file" in exc.context

    def test_transient_error(self):
        """Test TransientError with retry_after"""
        exc = TransientError("Temporary failure", retry_after=5.0)

        assert exc.category == ErrorCategory.TRANSIENT
        assert exc.retry_after == 5.0

    def test_rate_limit_error(self):
        """Test RateLimitError"""
        exc = RateLimitError("Too many requests", retry_after=60.0)

        assert exc.category == ErrorCategory.RATE_LIMIT
        assert exc.retry_after == 60.0


class TestRetryPolicy:
    """Tests for RetryPolicy class"""

    def test_successful_execution(self):
        """Test that successful execution returns immediately"""
        policy = RetryPolicy(RetryConfig(max_attempts=3))
        call_count = 0

        def success():
            nonlocal call_count
            call_count += 1
            return "success"

        result = policy.execute(success)

        assert result == "success"
        assert call_count == 1

    def test_retry_on_transient_error(self):
        """Test retry on transient error"""
        config = RetryConfig(
            max_attempts=3,
            initial_delay=0.01,  # Fast for testing
            retryable_exceptions=(TransientError,),
        )
        policy = RetryPolicy(config)
        call_count = 0

        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TransientError("Temporary failure")
            return "success"

        result = policy.execute(fail_then_succeed)

        assert result == "success"
        assert call_count == 3

    def test_max_attempts_exceeded(self):
        """Test that max attempts are enforced"""
        config = RetryConfig(
            max_attempts=3, initial_delay=0.01, retryable_exceptions=(TransientError,)
        )
        policy = RetryPolicy(config)
        call_count = 0

        def always_fail():
            nonlocal call_count
            call_count += 1
            raise TransientError("Always fails")

        with pytest.raises(TransientError):
            policy.execute(always_fail)

        assert call_count == 3

    def test_non_retryable_exception(self):
        """Test that non-retryable exceptions are raised immediately"""
        config = RetryConfig(
            max_attempts=3, initial_delay=0.01, retryable_exceptions=(TransientError,)
        )
        policy = RetryPolicy(config)
        call_count = 0

        def raise_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retryable")

        with pytest.raises(ValueError):
            policy.execute(raise_value_error)

        assert call_count == 1

    def test_exponential_backoff(self):
        """Test exponential backoff delay calculation"""
        config = RetryConfig(
            max_attempts=4, initial_delay=1.0, exponential_base=2.0, jitter=False, max_delay=100.0
        )
        policy = RetryPolicy(config)

        # Manual calculation: delay = initial * (base ^ attempt)
        assert policy._calculate_delay(0, Exception()) == 1.0  # 1 * 2^0
        assert policy._calculate_delay(1, Exception()) == 2.0  # 1 * 2^1
        assert policy._calculate_delay(2, Exception()) == 4.0  # 1 * 2^2

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay"""
        config = RetryConfig(
            initial_delay=10.0, exponential_base=10.0, max_delay=50.0, jitter=False
        )
        policy = RetryPolicy(config)

        delay = policy._calculate_delay(5, Exception())
        assert delay == 50.0

    def test_retry_after_from_exception(self):
        """Test that retry_after from exception is respected"""
        config = RetryConfig(initial_delay=1.0, jitter=False)
        policy = RetryPolicy(config)
        exc = TransientError("Error", retry_after=10.0)

        delay = policy._calculate_delay(0, exc)
        assert delay == 10.0


class TestRetryDecorator:
    """Tests for @retry decorator"""

    def test_retry_decorator_sync(self):
        """Test retry decorator with sync function"""
        call_count = 0

        @retry(max_attempts=3, initial_delay=0.01, exceptions=(TransientError,))
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TransientError("Flaky")
            return "success"

        result = flaky_function()

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_decorator_async(self):
        """Test retry decorator with async function"""
        call_count = 0

        @retry(max_attempts=3, initial_delay=0.01, exceptions=(TransientError,))
        async def flaky_async_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TransientError("Flaky")
            return "async success"

        result = await flaky_async_function()

        assert result == "async success"
        assert call_count == 2


class TestCircuitBreaker:
    """Tests for CircuitBreaker class"""

    def test_initial_state_closed(self):
        """Test that circuit starts in closed state"""
        breaker = CircuitBreaker()
        assert breaker.state == CircuitState.CLOSED

    def test_successful_call(self):
        """Test successful call through circuit breaker"""
        breaker = CircuitBreaker()

        result = breaker.call(lambda: "success")

        assert result == "success"
        assert breaker.state == CircuitState.CLOSED

    def test_opens_after_threshold(self):
        """Test circuit opens after failure threshold"""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker(config)

        def fail():
            raise _CircuitBreakerError("Failure")

        # Cause failures up to threshold
        for _ in range(3):
            with pytest.raises(_CircuitBreakerError):
                breaker.call(fail)

        assert breaker.state == CircuitState.OPEN

    def test_rejects_when_open(self):
        """Test that open circuit rejects calls"""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=100.0)
        breaker = CircuitBreaker(config)

        # Open the circuit
        with pytest.raises(_CircuitBreakerError):
            breaker.call(lambda: (_ for _ in ()).throw(_CircuitBreakerError("Failure")))

        # Should reject immediately
        with pytest.raises(CircuitBreakerOpenError):
            breaker.call(lambda: "should not run")

    def test_half_open_after_timeout(self):
        """Test transition to half-open after timeout"""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            success_threshold=1,  # Close after 1 success in half-open
            timeout=0.01,
        )
        breaker = CircuitBreaker(config)

        # Open the circuit
        with pytest.raises(_CircuitBreakerError):
            breaker.call(lambda: (_ for _ in ()).throw(_CircuitBreakerError("Failure")))

        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(0.02)

        # Next call should be attempted (half-open)
        result = breaker.call(lambda: "success")

        assert result == "success"
        # Should close after success in half-open (with success_threshold=1)
        assert breaker.state == CircuitState.CLOSED

    def test_reset(self):
        """Test manual reset of circuit breaker"""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker(config)

        # Open the circuit
        with pytest.raises(_CircuitBreakerError):
            breaker.call(lambda: (_ for _ in ()).throw(_CircuitBreakerError("Failure")))

        assert breaker.state == CircuitState.OPEN

        breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    def test_protected_decorator(self):
        """Test @breaker.protected decorator"""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CircuitBreaker(config)
        call_count = 0

        @breaker.protected
        def protected_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise _CircuitBreakerError("Failure")
            return "success"

        # First two calls fail
        with pytest.raises(_CircuitBreakerError):
            protected_function()
        with pytest.raises(_CircuitBreakerError):
            protected_function()

        # Circuit should be open
        assert breaker.state == CircuitState.OPEN


class TestErrorHandler:
    """Tests for ErrorHandler class"""

    def test_categorize_polymath_exception(self):
        """Test categorization of PolymathException"""
        handler = ErrorHandler()

        transient = TransientError("temp")
        rate_limit = RateLimitError("limit")
        config_err = ConfigurationError("config")

        assert handler.categorize(transient) == ErrorCategory.TRANSIENT
        assert handler.categorize(rate_limit) == ErrorCategory.RATE_LIMIT
        assert handler.categorize(config_err) == ErrorCategory.CONFIGURATION

    def test_categorize_standard_exceptions(self):
        """Test categorization of standard exceptions"""
        handler = ErrorHandler()

        assert handler.categorize(ConnectionError("conn")) == ErrorCategory.NETWORK
        assert handler.categorize(TimeoutError("timeout")) == ErrorCategory.TRANSIENT

    def test_categorize_by_message(self):
        """Test categorization by exception message"""
        handler = ErrorHandler()

        rate_limit = Exception("rate limit exceeded")
        timeout = Exception("request timed out")
        auth = Exception("unauthorized access")

        assert handler.categorize(rate_limit) == ErrorCategory.RATE_LIMIT
        assert handler.categorize(timeout) == ErrorCategory.TRANSIENT
        assert handler.categorize(auth) == ErrorCategory.AUTHENTICATION

    def test_handle_returns_error_info(self):
        """Test handle method returns ErrorInfo"""
        handler = ErrorHandler()
        exc = TransientError("test", retry_after=5.0)

        info = handler.handle(exc, {"request_id": "123"})

        assert info.category == ErrorCategory.TRANSIENT
        assert info.message == "test"
        assert info.original_exception is exc
        assert info.retry_after == 5.0
        assert info.context == {"request_id": "123"}

    def test_register_custom_mapping(self):
        """Test registering custom exception mapping"""
        handler = ErrorHandler()

        class CustomError(Exception):
            pass

        handler.register_mapping(CustomError, ErrorCategory.AUTHENTICATION)

        assert handler.categorize(CustomError("custom")) == ErrorCategory.AUTHENTICATION

    def test_get_global_error_handler(self):
        """Test global error handler instance"""
        handler = get_error_handler()

        assert isinstance(handler, ErrorHandler)
