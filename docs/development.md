# Doraemon Code Development Guide

This guide covers how to extend and develop for the Doraemon AI agent.

## Table of Contents

- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Adding a New Tool Module](#adding-a-new-tool-module)
- [Extending the Command System](#extending-the-command-system)
- [Adding New Metrics](#adding-new-metrics)
- [Testing Guide](#testing-guide)
- [Code Style](#code-style)

---

## Development Setup

### Prerequisites

- Python 3.10+
- Git

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd doraemon

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install with dev dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/core/test_container.py -v

# Run tests matching pattern
pytest tests/ -k "test_cache" -v
```

### Real Provider Regression

For real provider checks, prefer running the minimal protocol tests first and then the agent evals.

PackyAPI Anthropic-compatible example:

```bash
REAL_API_BASE='https://www.packyapi.com/v1' \
REAL_API_KEY='***' \
REAL_MODEL='claude-sonnet-4-6' \
python3 -m pytest -q tests/integration/test_real_protocols.py -k 'test_real_anthropic_protocol_upstream'

REAL_API_BASE='https://www.packyapi.com/v1' \
REAL_API_KEY='***' \
REAL_MODEL='claude-sonnet-4-6' \
python3 scripts/run_evals.py --category basic
```

Current real-eval baseline as of `2026-03-30`:

- `basic`: `6/6`
- `advanced --limit 3`: `3/3`

### Code Quality

```bash
# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/

# Type check
mypy src/ --ignore-missing-imports
```

---

## Project Structure

```
doraemon/
├── src/
│   ├── core/               # Core infrastructure
│   │   ├── commands.py     # Slash command system
│   │   ├── config.py       # Configuration loading
│   │   ├── configuration.py # Advanced config system
│   │   ├── container.py    # Dependency injection
│   │   ├── errors.py       # Error handling
│   │   ├── events.py       # Event bus
│   │   ├── metrics.py      # Metrics collection
│   │   ├── prompts.py      # Mode prompts
│   │   ├── rules.py        # AGENTS.md system
│   │   ├── schema.py       # Validation
│   │   ├── security.py     # Path sandboxing
│   │   ├── services.py     # DI configuration
│   │   ├── tasks.py        # Task management
│   │   ├── telemetry.py    # Logging & tracing
│   │   └── user_errors.py  # User-friendly messages
│   ├── host/               # CLI host
│   │   ├── cli.py          # Main CLI
│   │   ├── cli_commands.py # Command handlers
│   │   ├── client.py       # In-process tool client
│   │   └── tui.py          # Textual UI
│   ├── servers/            # Built-in tool modules
│   │   ├── computer.py     # Code execution
│   │   ├── fs_edit.py      # File editing
│   │   ├── fs_ops.py       # File operations
│   │   ├── fs_read.py      # File reading
│   │   ├── fs_write.py     # File writing
│   │   ├── memory.py       # File-backed notes
│   │   ├── task.py         # Task management
│   │   └── web.py          # Web fetching
│   └── services/           # Shared services
│       ├── code_nav.py     # Code navigation
│       ├── document.py     # Document parsing
│       ├── outline.py      # AST parsing
│       └── vision.py       # Image processing
├── tests/
│   ├── core/               # Core module tests
│   ├── integration/        # Integration tests
│   └── servers/            # Server tests
└── docs/                   # Documentation
```

---

## Adding a New Tool Module

### Step 1: Create the Tool File

Create `src/servers/my_server.py`:

```python
"""My custom tool module."""

import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def my_tool(param1: str, param2: int = 10) -> str:
    """
    Description of what this tool does.
    
    Args:
        param1: Description of param1
        param2: Description of param2 (default: 10)
    
    Returns:
        Description of return value
    """
    logger.info(f"my_tool called with param1={param1}, param2={param2}")
    
    try:
        # Your implementation here
        result = f"Processed: {param1} x {param2}"
        return result
    except Exception as e:
        logger.error(f"my_tool failed: {e}")
        return f"Error: {str(e)}"

def another_tool(data: dict) -> str:
    """Another tool with dict parameter."""
    return f"Received: {data}"
```

### Step 2: Mark as Sensitive (if needed)

If your tool has side effects, add to sensitive tools:

```json
{
  "sensitive_tools": [
    "my_dangerous_tool"
  ]
}
```

### Step 3: Add Tests

Create `tests/servers/test_my_server.py`:

```python
import pytest
from src.servers.my_server import my_tool, another_tool


class TestMyTool:
    def test_basic_usage(self):
        result = my_tool("test", 5)
        assert "Processed" in result

    def test_default_param(self):
        result = my_tool("test")
        assert "10" in result  # Default param2


class TestAnotherTool:
    def test_dict_input(self):
        result = another_tool({"key": "value"})
        assert "key" in result
```

---

## Extending the Command System

### Adding a New Slash Command

Edit `src/host/cli_commands.py`:

```python
from src.core.commands import register_command, CommandContext

@register_command(
    name="mycommand",
    aliases=["mc", "mycmd"],
    description="Description of my command",
    usage="/mycommand <arg>"
)
async def cmd_mycommand(ctx: CommandContext, args: list[str]) -> bool:
    """
    Handle /mycommand.
    
    Args:
        ctx: Command context with access to chat, client, etc.
        args: Command arguments (split by space)
    
    Returns:
        True to continue, "EXIT" to quit
    """
    if not args:
        ctx.console.print("[yellow]Usage: /mycommand <arg>[/yellow]")
        return True
    
    arg = args[0]
    ctx.console.print(f"[green]Executing with arg: {arg}[/green]")
    
    # Access tool client
    result = await ctx.tool_client.call_tool("some_tool", {"param": arg})
    
    # Access current mode
    current_mode = ctx.get("mode")
    
    return True
```

### Command Context

The `CommandContext` provides:

```python
ctx.console       # Rich Console for output
ctx.tool_client   # Tool client for tool calls
ctx.chat          # Chat session reference
ctx.client        # GenAI client
ctx.get("key")    # Get context value
ctx.set("key", v) # Set context value
```

---

## Adding Trace Logging

### Define Custom Trace Events

```python
from src.core.logger import TraceLogger

trace = TraceLogger()

def feature_used(feature_name: str):
    trace.log("feature_used", feature_name, {"feature": feature_name})

def feature_duration(feature_name: str, duration: float):
    trace.log(
        "feature_duration",
        feature_name,
        {"feature": feature_name, "duration_seconds": duration},
        duration_ms=duration * 1000,
    )
```

### Use in Code

```python
import time

def my_feature():
    start = time.time()

    feature_used("process")
    # ... do work ...

    feature_duration("process", time.time() - start)
```

---

## Testing Guide

### Test Structure

```
tests/
├── core/           # Unit tests for core modules
├── integration/    # Integration tests
├── servers/        # Tool module tests
└── evals/          # Evaluation tests
```

### Writing Unit Tests

```python
import pytest
from src.core.my_module import MyClass

class TestMyClass:
    @pytest.fixture
    def instance(self):
        """Create fresh instance for each test."""
        return MyClass()
    
    def test_basic_operation(self, instance):
        result = instance.do_something()
        assert result == expected
    
    def test_edge_case(self, instance):
        with pytest.raises(ValueError):
            instance.do_something_invalid()
```

### Writing Async Tests

```python
import pytest

@pytest.mark.asyncio
async def test_async_operation():
    result = await async_function()
    assert result is not None
```

### Mocking

```python
from unittest.mock import AsyncMock, MagicMock, patch

def test_with_mock():
    mock_client = MagicMock()
    mock_client.call_tool = AsyncMock(return_value="result")
    
    # Use mock in test
    ...

@patch("src.module.external_function")
def test_with_patch(mock_func):
    mock_func.return_value = "mocked"
    # Test code that uses external_function
```

### Test Fixtures

Common fixtures in `conftest.py`:

```python
import pytest

@pytest.fixture
def temp_config():
    """Temporary configuration for testing."""
    return {
        "persona": {"name": "Test"}
    }
```

---

## Code Style

### Python Style

- Use Python 3.10+ features (type hints, `|` union, etc.)
- Follow PEP 8 naming conventions
- Use 4 spaces for indentation
- Maximum line length: 100 characters

### Type Hints

Always use type hints:

```python
def process_data(
    items: list[str],
    config: dict[str, Any] | None = None,
) -> tuple[list[str], int]:
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def my_function(param1: str, param2: int) -> bool:
    """
    Short description of function.
    
    Longer description if needed, explaining behavior,
    edge cases, etc.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When param1 is empty
    
    Example:
        >>> my_function("test", 5)
        True
    """
```

### Error Handling

```python
# Use specific exceptions
from src.core.errors import AgentError, ConfigurationError, TransientError

# Provide context
raise ConfigurationError(
    "Missing required config",
    context={"key": "database.host", "file": "config.json"}
)

# Log errors appropriately
logger.error("Operation failed", exception=e, operation="fetch")
```

### Logging

```python
from src.core.logger import get_logger

logger = get_logger(__name__)

# Use standard logging
logger.info("User action started")
logger.error("Failed to process item")
```

---

## Debugging Tips

### Enable Debug Logging

```bash
export AGENT_LOG_LEVEL=DEBUG
dora start
```

### Use the Debug Command

```
/debug
```

Shows:
- Current mode
- Connected servers
- Available tools
- Session statistics

### Trace Tool Calls

Tool calls are automatically logged with timing information.
Check `.agent/logs/*.log` for detailed traces.

### Common Issues

1. **Server not connecting**: Check if the server script exists and is executable
2. **Tool not found**: Verify the server registered the tool correctly
3. **Path errors**: Ensure paths are relative to the workspace root
4. **Memory errors**: Reduce `AGENT_MAX_MEMORY_MB` for constrained environments

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Write tests for your changes
4. Ensure all tests pass: `pytest tests/`
5. Run linting: `ruff check src/ tests/`
6. Submit a pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.
