"""
Error Handling and Retry Mechanisms

Provides structured error handling, retry logic, and circuit breaker pattern.
"""

import asyncio
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import TypeVar

T = TypeVar("T")


class ErrorCategory(Enum):
    """Error categories for better error handling"""

    TRANSIENT = "transient"  # Temporary, retry likely to succeed
    PERMANENT = "permanent"  # Permanent, retry won't help
    CONFIGURATION = "configuration"  # Configuration error
    AUTHENTICATION = "authentication"  # Auth error
    RATE_LIMIT = "rate_limit"  # Rate limiting
    NETWORK = "network"  # Network error
    UNKNOWN = "unknown"  # Unknown error


@dataclass
class ErrorInfo:
    """Structured error information"""

    category: ErrorCategory
    message: str
    original_exception: Exception
    retry_after: float | None = None  # Seconds to wait before retry
    context: dict = field(default_factory=dict)


class AgentError(Exception):
    """Base exception for agent errors"""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        context: dict | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.context = context or {}


class ConfigurationError(AgentError):
    """Configuration-related errors"""

    def __init__(self, message: str, context: dict | None = None):
        super().__init__(message, ErrorCategory.CONFIGURATION, context)


class TransientError(AgentError):
    """Transient errors that can be retried"""

    def __init__(self, message: str, retry_after: float = 1.0, context: dict | None = None):
        super().__init__(message, ErrorCategory.TRANSIENT, context)
        self.retry_after = retry_after


class RateLimitError(AgentError):
    """Rate limiting errors"""

    def __init__(self, message: str, retry_after: float = 60.0, context: dict | None = None):
        super().__init__(message, ErrorCategory.RATE_LIMIT, context)
        self.retry_after = retry_after


class ToolExecutionError(AgentError):
    """Errors during tool execution"""

    def __init__(self, tool_name: str, message: str, context: dict | None = None):
        super().__init__(f"Tool '{tool_name}' failed: {message}", ErrorCategory.TRANSIENT, context)
        self.tool_name = tool_name


class FileAccessError(AgentError):
    """File access errors"""

    def __init__(self, path: str, message: str, context: dict | None = None):
        super().__init__(f"File access error '{path}': {message}", ErrorCategory.PERMANENT, context)
        self.path = path


class NetworkError(AgentError):
    """Network-related errors"""

    def __init__(self, message: str, context: dict | None = None):
        super().__init__(message, ErrorCategory.NETWORK, context)


class AuthenticationError(AgentError):
    """Authentication errors"""

    def __init__(self, message: str, context: dict | None = None):
        super().__init__(message, ErrorCategory.AUTHENTICATION, context)


class ValidationError(AgentError):
    """Input validation errors"""

    def __init__(self, message: str, context: dict | None = None):
        super().__init__(message, ErrorCategory.PERMANENT, context)


@dataclass
class RetryConfig:
    """Configuration for retry logic"""

    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple[type[Exception], ...] = (TransientError, RateLimitError)


class RetryPolicy:
    """Retry policy with exponential backoff"""

    def __init__(self, config: RetryConfig | None = None):
        self.config = config or RetryConfig()

    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with retry logic.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            Last exception if all retries fail
        """
        last_exception = None

        for attempt in range(self.config.max_attempts):
            try:
                return func(*args, **kwargs)
            except self.config.retryable_exceptions as e:
                last_exception = e

                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt, e)
                    time.sleep(delay)
                else:
                    # Last attempt failed
                    break
            except Exception:
                # Non-retryable exception
                raise

        # All retries failed
        if last_exception is None:
            raise RuntimeError("No attempts were made")
        raise last_exception

    async def execute_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Async version of execute"""
        last_exception = None

        for attempt in range(self.config.max_attempts):
            try:
                return await func(*args, **kwargs)
            except self.config.retryable_exceptions as e:
                last_exception = e

                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt, e)
                    await asyncio.sleep(delay)
                else:
                    break
            except Exception:
                raise

        if last_exception is None:
            raise RuntimeError("No attempts were made")
        raise last_exception

    def _calculate_delay(self, attempt: int, exception: Exception) -> float:
        """Calculate retry delay with exponential backoff"""
        # Check if exception has retry_after
        if hasattr(exception, "retry_after") and exception.retry_after:
            return min(float(exception.retry_after), self.config.max_delay)

        # Exponential backoff
        delay = self.config.initial_delay * (self.config.exponential_base**attempt)
        delay = min(delay, self.config.max_delay)

        # Add jitter to prevent thundering herd
        if self.config.jitter:
            import random

            delay = delay * (0.5 + random.random())

        return delay


def retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple[type[Exception], ...] = (TransientError,),
):
    """
    Decorator for automatic retry with exponential backoff.

    Example:
        @retry(max_attempts=5, initial_delay=2.0)
        def fetch_data():
            # ... code that may fail ...
            pass
    """

    def decorator(func: Callable) -> Callable:
        config = RetryConfig(
            max_attempts=max_attempts,
            initial_delay=initial_delay,
            max_delay=max_delay,
            retryable_exceptions=exceptions,
        )
        policy = RetryPolicy(config)

        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await policy.execute_async(func, *args, **kwargs)

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return policy.execute(func, *args, **kwargs)

            return sync_wrapper

    return decorator


class CircuitState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""

    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes in half-open before closing
    timeout: float = 60.0  # Seconds before trying half-open


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    Prevents cascading failures by failing fast when service is down.

    Example:
        breaker = CircuitBreaker()

        @breaker.protected
        def call_external_service():
            # ... code ...
            pass
    """

    def __init__(self, config: CircuitBreakerConfig | None = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: float | None = None
        self._lock = threading.Lock()

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with circuit breaker protection.

        Raises:
            CircuitBreakerOpenError: If circuit is open
        """
        with self._lock:
            # Check circuit state
            if self.state == CircuitState.OPEN:
                # Check if timeout elapsed
                if (
                    self.last_failure_time
                    and time.time() - self.last_failure_time >= self.config.timeout
                ):
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                else:
                    remaining = (
                        self.config.timeout - (time.time() - self.last_failure_time)
                        if self.last_failure_time
                        else self.config.timeout
                    )
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker is OPEN. Try again in {remaining:.1f}s"
                    )
            current_state = self.state

        # Execute function
        try:
            result = func(*args, **kwargs)
            self._on_success(current_state)
            return result
        except Exception:
            self._on_failure(current_state)
            raise

    def _on_success(self, state_at_call: CircuitState | None = None):
        """Handle successful call"""
        with self._lock:
            state = state_at_call if state_at_call is not None else self.state
            if state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
            else:
                self.failure_count = 0

    def _on_failure(self, state_at_call: CircuitState | None = None):
        """Handle failed call"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
            elif state_at_call == CircuitState.HALF_OPEN or self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN

    async def call_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute async function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitState.OPEN:
                if (
                    self.last_failure_time
                    and time.time() - self.last_failure_time >= self.config.timeout
                ):
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                else:
                    remaining = (
                        self.config.timeout - (time.time() - self.last_failure_time)
                        if self.last_failure_time
                        else self.config.timeout
                    )
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker is OPEN. Try again in {remaining:.1f}s"
                    )
            current_state = self.state

        try:
            result = await func(*args, **kwargs)
            self._on_success(current_state)
            return result
        except Exception:
            self._on_failure(current_state)
            raise

    def protected(self, func: Callable) -> Callable:
        """Decorator for circuit breaker protection"""

        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self.call_async(func, *args, **kwargs)

            return async_wrapper

        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)

        return wrapper

    def reset(self):
        """Reset circuit breaker to closed state"""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""

    pass


class ErrorHandler:
    """
    Central error handler with categorization and recovery.

    Example:
        handler = ErrorHandler()

        try:
            # ... code ...
        except Exception as e:
            error_info = handler.handle(e)
            # ... log or take action ...
    """

    def __init__(self):
        self._error_mappings = {
            TransientError: ErrorCategory.TRANSIENT,
            RateLimitError: ErrorCategory.RATE_LIMIT,
            ConfigurationError: ErrorCategory.CONFIGURATION,
            ConnectionError: ErrorCategory.NETWORK,
            TimeoutError: ErrorCategory.TRANSIENT,
        }

    def categorize(self, exception: Exception) -> ErrorCategory:
        """Categorize an exception"""
        # Check if it's an AgentError with category
        if isinstance(exception, AgentError):
            return exception.category

        # Check error mappings
        for exc_type, category in self._error_mappings.items():
            if isinstance(exception, exc_type):
                return category

        # Check exception message for hints
        message = str(exception).lower()
        if "rate limit" in message or "too many requests" in message:
            return ErrorCategory.RATE_LIMIT
        if "timeout" in message or "timed out" in message:
            return ErrorCategory.TRANSIENT
        if "connection" in message or "network" in message:
            return ErrorCategory.NETWORK
        if "auth" in message or "unauthorized" in message:
            return ErrorCategory.AUTHENTICATION

        return ErrorCategory.UNKNOWN

    def handle(self, exception: Exception, context: dict | None = None) -> ErrorInfo:
        """
        Handle an exception and return structured error information.

        Args:
            exception: The exception to handle
            context: Additional context

        Returns:
            ErrorInfo with categorized error details
        """
        category = self.categorize(exception)

        # Extract retry_after if available
        retry_after = None
        if hasattr(exception, "retry_after"):
            retry_after = exception.retry_after

        return ErrorInfo(
            category=category,
            message=str(exception),
            original_exception=exception,
            retry_after=retry_after,
            context=context or {},
        )

    def register_mapping(self, exception_type: type[Exception], category: ErrorCategory):
        """Register custom exception mapping"""
        self._error_mappings[exception_type] = category


# Global error handler instance
_global_error_handler = ErrorHandler()


def get_error_handler() -> ErrorHandler:
    """Get the global error handler"""
    return _global_error_handler
