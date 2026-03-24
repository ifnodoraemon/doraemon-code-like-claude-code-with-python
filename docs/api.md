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

### Logging and Trace

**Module:** `src.core.logger`

Project-local logging helpers for standard logs and lightweight trace capture.

#### Basic Usage

```python
from src.core.logger import get_logger, setup_logger

logger = get_logger(__name__)
logger.info("Background task started")

custom_logger = setup_logger("my_module", level="DEBUG", log_file=".agent/logs/my_module.log")
custom_logger.debug("Verbose details enabled")
```

#### Trace Logger

```python
from src.core.logger import TraceLogger

trace = TraceLogger()

trace.log("tool_call", "read", {"path": "README.md"})
trace.log("tool_result", "read", {"bytes": 2048}, duration_ms=12.4)

events = trace.export()
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
    AgentError,
    ConfigurationError,
    TransientError,
    RateLimitError,
    ErrorCategory,
)

# Base exception
raise AgentError("Something went wrong", ErrorCategory.UNKNOWN)

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

### Logging

**Module:** `src.core.logger`

Standard logging plus per-operation trace capture.

#### Logger

```python
from src.core.logger import get_logger

logger = get_logger(__name__)
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

#### Trace Capture

```python
from src.core.logger import TraceLogger

trace = TraceLogger()
trace.log("operation_start", "process_request", {"request_id": "abc"})
trace.log("operation_end", "process_request", {"status": 200}, duration_ms=18.7)
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
| `AGENT_LOG_LEVEL` | Log level (DEBUG/INFO/WARNING/ERROR) | `INFO` |
| `AGENT_LOG_FILE` | Enable file logging | `false` |
| `AGENT_MAX_MEMORY_MB` | Code execution memory limit | `512` |
| `AGENT_MAX_CPU_TIME` | Code execution CPU time limit | `30` |

### Config File

Location: `.agent/config.json`

```json
{
  "model": "gemini-3-pro-preview",
  "mcpServers": {
    "filesystem": {
      "command": "python3",
      "args": ["src/servers/filesystem.py"],
      "env": {}
    },
    "memory": {
      "command": "python3",
      "args": ["src/servers/memory.py"],
      "env": {}
    },
    "web": {
      "command": "python3",
      "args": ["src/servers/web.py"],
      "env": {}
    },
    "run": {
      "command": "python3",
      "args": ["src/servers/run.py"],
      "env": {}
    },
    "task": {
      "command": "python3",
      "args": ["src/servers/task.py"],
      "env": {}
    }
  },
  "persona": {
    "name": "Agent",
    "role": "AI Assistant"
  },
  "sensitive_tools": [
    "execute_python",
    "write_file"
  ]
}
```
