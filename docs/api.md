# Doraemon API Documentation

This document provides a comprehensive API reference for Doraemon's core modules and MCP servers.

## Table of Contents

- [Core Modules](#core-modules)
  - [Dependency Injection](#dependency-injection)
  - [Configuration](#configuration)
  - [Events](#events)
  - [Metrics](#metrics)
  - [Error Handling](#error-handling)
  - [Telemetry](#telemetry)
- [MCP Servers](#mcp-servers)
  - [File System Read](#file-system-read)
  - [File System Write](#file-system-write)
  - [File System Edit](#file-system-edit)
  - [Computer (Code Execution)](#computer-code-execution)
  - [Memory (RAG)](#memory-rag)
  - [Web](#web)
  - [Task Manager](#task-manager)

---

## Core Modules

### Dependency Injection

**Module:** `src.core.container`

The DI container manages service lifecycle and automatic dependency resolution.

#### ServiceCollection

```python
from src.core.container import ServiceCollection

services = ServiceCollection()

# Register singleton (one instance for entire app)
services.add_singleton(ILogger, ConsoleLogger)

# Register transient (new instance every time)
services.add_transient(IRepository, UserRepository)

# Register with factory
services.add_singleton(IConfig, lambda: load_config())

# Build provider
provider = services.build_service_provider()
```

#### ServiceProvider

```python
# Get service (returns None if not found)
logger = provider.get_service(ILogger)

# Get required service (raises if not found)
config = provider.get_required_service(IConfig)
```

#### Global Functions

```python
from src.core.container import configure_services, get_service

def setup(services):
    services.add_singleton(ILogger, ConsoleLogger)

configure_services(setup)
logger = get_service(ILogger)
```

---

### Configuration

**Module:** `src.core.configuration`

Hierarchical configuration system with multiple sources.

#### ConfigurationBuilder

```python
from src.core.configuration import ConfigurationBuilder, configure

def setup_config(builder: ConfigurationBuilder):
    # Add default values
    builder.add_defaults({"app": {"name": "MyApp"}})
    
    # Add JSON file (optional)
    builder.add_json_file("config.json", optional=True)
    
    # Add environment variables
    builder.add_environment_variables(prefix="MYAPP_")

config = configure(setup_config)
```

#### Configuration Access

```python
# Basic access
value = config.get("app.name")
value = config.get("app.name", default="Default")

# Typed access
name = config.get_str("app.name")
port = config.get_int("server.port", default=8080)
debug = config.get_bool("debug", default=False)
items = config.get_list("allowed_hosts")

# Check existence
if config.has("database.host"):
    ...

# Get section
db_config = config.get_section("database")
```

---

### Events

**Module:** `src.core.events`

Pub/Sub event system for decoupled communication.

#### Event Types

```python
from src.core.events import SystemEvents, SessionEvents, MessageEvents, TaskEvents

# Available events:
# SystemEvents: STARTUP, SHUTDOWN, CONFIG_CHANGED, ERROR
# SessionEvents: STARTED, ENDED, MODE_CHANGED
# MessageEvents: USER_MESSAGE, AI_RESPONSE, TOOL_CALL, TOOL_RESULT
# TaskEvents: CREATED, UPDATED, COMPLETED, FAILED
```

#### Publishing Events

```python
from src.core.events import publish, publish_async, Event

# Simple publish
publish(MessageEvents.USER_MESSAGE, message="Hello", user_id="123")

# Async publish (waits for handlers)
await publish_async(MessageEvents.TOOL_CALL, tool_name="read_file")

# Publish Event object
event = Event(event_type=MessageEvents.USER_MESSAGE, data={"message": "Hi"})
get_event_bus().publish(event)
```

#### Subscribing to Events

```python
from src.core.events import subscribe, get_event_bus, Event

# Decorator
@subscribe(MessageEvents.USER_MESSAGE)
def on_user_message(event: Event):
    print(f"User said: {event.data['message']}")

# Manual subscription
bus = get_event_bus()
subscription = bus.subscribe(
    MessageEvents.TOOL_CALL,
    handler=my_handler,
    priority=10,  # Higher = called first
    filter_func=lambda e: e.data.get("tool") == "read_file"
)

# Unsubscribe
bus.unsubscribe(subscription)
```

#### Event History

```python
bus = get_event_bus()

# Get recent events
history = bus.get_history(limit=100)
history = bus.get_history(event_type=MessageEvents.TOOL_CALL, limit=50)

# Clear history
bus.clear_history()
```

---

### Metrics

**Module:** `src.core.metrics`

Metrics collection with Counter, Gauge, and Histogram types.

#### Basic Usage

```python
from src.core.metrics import get_metrics

metrics = get_metrics()

# Counters (only increase)
metrics.increment("requests_total")
metrics.increment("requests_total", method="GET", status="200")

# Gauges (can increase/decrease)
metrics.gauge_set("queue_size", 42)
metrics.gauge_inc("active_connections")
metrics.gauge_dec("active_connections")

# Histograms (distributions)
metrics.observe("request_duration_seconds", 0.5)

# Timer context manager
with metrics.timer("operation_duration", operation="fetch"):
    do_something()
```

#### Doraemon Metrics

```python
from src.core.metrics import get_doraemon_metrics

pm = get_doraemon_metrics()

# Session metrics
pm.session_started("my_project")
pm.session_ended("my_project", turns=10, duration_seconds=300.0)

# Tool metrics
pm.tool_called("read_file", "fs_read")
pm.tool_succeeded("read_file", 0.5)
pm.tool_failed("write_file", "permission_denied")

# Cache metrics
pm.cache_hit("read_file")
pm.cache_miss("read_file")

# Token metrics
pm.tokens_used(input_tokens=100, output_tokens=50)
```

#### Export

```python
# Export to dict
data = metrics.to_dict()
# Returns: {"counters": {...}, "gauges": {...}, "histograms": {...}}

# Export to Prometheus format
prom_text = metrics.to_prometheus()
```

---

### Error Handling

**Module:** `src.core.errors`

Structured error handling with retry and circuit breaker patterns.

#### Exception Types

```python
from src.core.errors import (
    DoraemonException,
    ConfigurationError,
    TransientError,
    RateLimitError,
    ErrorCategory,
)

# Base exception
raise DoraemonException("Something went wrong", ErrorCategory.UNKNOWN)

# Configuration error
raise ConfigurationError("Missing required config", {"key": "database.host"})

# Transient error (retryable)
raise TransientError("Temporary failure", retry_after=5.0)

# Rate limit error
raise RateLimitError("Too many requests", retry_after=60.0)
```

#### Retry Policy

```python
from src.core.errors import RetryPolicy, RetryConfig, retry

# Manual retry
policy = RetryPolicy(RetryConfig(
    max_attempts=5,
    initial_delay=1.0,
    max_delay=60.0,
    exponential_base=2.0,
    jitter=True,
))

result = policy.execute(my_function, arg1, arg2)
result = await policy.execute_async(my_async_function, arg1)

# Decorator
@retry(max_attempts=3, initial_delay=1.0)
def fetch_data():
    ...

@retry(max_attempts=5)
async def fetch_data_async():
    ...
```

#### Circuit Breaker

```python
from src.core.errors import CircuitBreaker, CircuitBreakerConfig

breaker = CircuitBreaker(CircuitBreakerConfig(
    failure_threshold=5,
    success_threshold=2,
    timeout=60.0,
))

# Manual use
try:
    result = breaker.call(risky_function)
except CircuitBreakerOpenError:
    # Circuit is open, fail fast
    ...

# Decorator
@breaker.protected
def call_external_service():
    ...

# Check state
print(breaker.state)  # CircuitState.CLOSED/OPEN/HALF_OPEN

# Reset
breaker.reset()
```

#### Error Handler

```python
from src.core.errors import ErrorHandler, get_error_handler

handler = get_error_handler()

try:
    risky_operation()
except Exception as e:
    error_info = handler.handle(e, context={"operation": "fetch"})
    print(f"Category: {error_info.category}")
    print(f"Retry after: {error_info.retry_after}")
```

---

### Telemetry

**Module:** `src.core.telemetry`

Structured logging and distributed tracing.

#### Structured Logger

```python
from src.core.telemetry import StructuredLogger, get_logger, LogLevel

logger = get_logger("my_module", min_level=LogLevel.DEBUG)

# Basic logging
logger.debug("Debug message", key="value")
logger.info("Info message", user_id="123")
logger.warning("Warning message")
logger.error("Error message", exception=e)
logger.critical("Critical message")

# Operation scope
with logger.operation("process_request", request_id="abc"):
    logger.debug("Processing...")
    # Logs will include operation context

# Correlation ID
with logger.correlation("req-12345"):
    logger.info("All logs in this scope share correlation ID")
```

#### Tracer

```python
from src.core.telemetry import Tracer, get_tracer

tracer = get_tracer()

# Create spans
with tracer.start_span("http_request", method="GET", url="/api") as span:
    # Do work
    span.tags["status"] = 200
    span.tags["bytes"] = 1024

# Nested spans
with tracer.start_span("process_order") as parent:
    with tracer.start_span("validate") as child:
        # child.parent_id == parent.span_id
        pass
```

---

## MCP Servers

### File System Read

**Server:** `src.servers.fs_read`

Tools for reading files and navigating code.

#### read_file

Read file content with optional partial reading.

```python
# Full file
content = read_file("path/to/file.txt")

# Partial read (lines 10-20)
content = read_file("file.txt", offset=10, limit=10)

# Supports: .txt, .md, .py, .pdf, .docx, .pptx, .xlsx, .png, .jpg
```

#### read_file_outline

Get structural outline of code files.

```python
outline = read_file_outline("module.py")
# Returns: Classes, functions, and their signatures
```

#### list_directory

List directory contents.

```python
listing = list_directory(".", show_hidden=False, detailed=True)
```

#### glob_files

Find files matching patterns.

```python
files = glob_files("**/*.py", exclude=["**/test_*.py"], max_results=100)
```

#### grep_search

Search for patterns in files.

```python
results = grep_search(r"def \w+\(", include="*.py", path="src/")
```

#### find_symbol

Find symbol definitions.

```python
location = find_symbol("MyClass", path="src/")
```

---

### File System Write

**Server:** `src.servers.fs_write`

Tools for writing files.

#### write_file

Write content to a file.

```python
result = write_file("output.txt", "Hello, World!")
# Automatically creates parent directories
```

---

### File System Edit

**Server:** `src.servers.fs_edit`

Tools for editing files with diff support.

#### edit_file

Apply targeted edits to files.

```python
result = edit_file(
    path="file.py",
    old_content="def old_func():",
    new_content="def new_func():"
)
```

---

### Computer (Code Execution)

**Server:** `src.servers.computer`

Sandboxed Python code execution.

#### execute_python

Execute Python code with resource limits.

```python
result = execute_python("""
import math
print(math.sqrt(2))
""", timeout=30, max_memory_mb=512, sandbox=True)
```

#### install_package

Install Python packages.

```python
result = install_package("numpy")
```

#### list_installed_packages

List installed packages.

```python
packages = list_installed_packages()
```

---

### Memory (RAG)

**Server:** `src.servers.memory`

Vector-based long-term memory using ChromaDB.

#### save_note

Save a note to memory.

```python
result = save_note(
    category="project_notes",
    content="Important architecture decision...",
    collection_name="my_project"
)
```

#### search_notes

Search notes semantically.

```python
results = search_notes(
    query="architecture decisions",
    collection_name="my_project",
    n_results=5
)
```

---

### Web

**Server:** `src.servers.web`

Web content fetching and search.

#### fetch_url

Fetch and extract content from URL.

```python
content = fetch_url("https://example.com")
```

#### web_search

Search the web using DuckDuckGo.

```python
results = web_search("Python async best practices", max_results=5)
```

---

### Task Manager

**Server:** `src.servers.task`

Task management for AI planning.

#### task_create

Create a new task.

```python
task_id = task_create(
    title="Implement feature X",
    description="Details...",
    priority="high",
    parent_id=None
)
```

#### task_list

List tasks.

```python
tasks = task_list(status="pending")
```

#### task_update_status

Update task status.

```python
result = task_update_status(task_id="task_001", status="completed")
```

---

## Error Codes

See `src.core.user_errors` for user-friendly error messages.

| Code | Description |
|------|-------------|
| `TOOL_NOT_FOUND` | Requested tool doesn't exist |
| `SERVER_UNAVAILABLE` | MCP server temporarily unavailable |
| `API_RATE_LIMITED` | API rate limit exceeded |
| `PATH_OUTSIDE_SANDBOX` | Path access denied (outside workspace) |
| `CODE_EXECUTION_TIMEOUT` | Code execution timed out |

---

## Configuration Reference

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google Gemini API key | Required |
| `OPENAI_API_KEY` | OpenAI API key | Optional |
| `DORAEMON_MODEL` | LLM model name | `gemini-3-pro-preview` |
| `DORAEMON_LOG_LEVEL` | Log level (DEBUG/INFO/WARNING/ERROR) | `INFO` |
| `DORAEMON_LOG_FILE` | Enable file logging | `false` |
| `DORAEMON_MAX_MEMORY_MB` | Code execution memory limit | `512` |
| `DORAEMON_MAX_CPU_TIME` | Code execution CPU time limit | `30` |

### Config File

Location: `.doraemon/config.json`

```json
{
  "mcpServers": {
    "fs_read": {
      "command": "python3",
      "args": ["src/servers/fs_read.py"],
      "env": {}
    }
  },
  "persona": {
    "name": "Doraemon",
    "role": "AI Assistant"
  },
  "sensitive_tools": [
    "execute_python",
    "write_file"
  ]
}
```
