# Doraemon Code Development Guide

This guide covers how to extend and develop for the Doraemon AI agent.

## Table of Contents

- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Adding a New MCP Server](#adding-a-new-mcp-server)
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
│   │   ├── client.py       # MCP client
│   │   └── tui.py          # Textual UI
│   ├── servers/            # MCP servers
│   │   ├── computer.py     # Code execution
│   │   ├── fs_edit.py      # File editing
│   │   ├── fs_ops.py       # File operations
│   │   ├── fs_read.py      # File reading
│   │   ├── fs_write.py     # File writing
│   │   ├── memory.py       # Vector memory
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

## Adding a New MCP Server

### Step 1: Create the Server File

Create `src/servers/my_server.py`:

```python
"""
My Custom MCP Server

Provides custom functionality for Doraemon Code.
"""

import logging
from mcp.server.fastmcp import FastMCP

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create MCP server instance
mcp = FastMCP("DoraemonMyServer")


@mcp.tool()
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


@mcp.tool()
def another_tool(data: dict) -> str:
    """Another tool with dict parameter."""
    return f"Received: {data}"


# Entry point
if __name__ == "__main__":
    mcp.run()
```

### Step 2: Register in Configuration

Add to `.doraemon/config.json`:

```json
{
  "mcpServers": {
    "my_server": {
      "command": "python3",
      "args": ["src/servers/my_server.py"],
      "env": {
        "MY_VAR": "value"
      }
    }
  }
}
```

### Step 3: Mark as Sensitive (if needed)

If your tool has side effects, add to sensitive tools:

```json
{
  "sensitive_tools": [
    "my_dangerous_tool"
  ]
}
```

### Step 4: Add Tests

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
    
    # Access MCP client
    result = await ctx.mcp_client.call_tool("some_tool", {"param": arg})
    
    # Access current mode
    current_mode = ctx.get("mode")
    
    return True
```

### Command Context

The `CommandContext` provides:

```python
ctx.console       # Rich Console for output
ctx.mcp_client    # MCP client for tool calls
ctx.chat          # Chat session reference
ctx.client        # GenAI client
ctx.get("key")    # Get context value
ctx.set("key", v) # Set context value
```

---

## Adding New Metrics

### Define Custom Metrics

```python
from src.core.metrics import get_metrics, DoraemonMetrics

class MyFeatureMetrics:
    def __init__(self):
        self.registry = get_metrics()
    
    def feature_used(self, feature_name: str):
        self.registry.increment(
            "myfeature_usage_total",
            feature=feature_name
        )
    
    def feature_duration(self, feature_name: str, duration: float):
        self.registry.observe(
            "myfeature_duration_seconds",
            duration,
            feature=feature_name
        )
```

### Use in Code

```python
metrics = MyFeatureMetrics()

def my_feature():
    import time
    start = time.time()
    
    metrics.feature_used("process")
    # ... do work ...
    
    metrics.feature_duration("process", time.time() - start)
```

---

## Testing Guide

### Test Structure

```
tests/
├── core/           # Unit tests for core modules
├── integration/    # Integration tests
├── servers/        # MCP server tests
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
        "mcpServers": {},
        "persona": {"name": "Test"}
    }

@pytest.fixture
def mock_mcp_client():
    """Mock MCP client."""
    client = MagicMock()
    client.tool_map = {"read_file": "fs_read"}
    return client
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
from src.core.errors import DoraemonException, TransientError

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
from src.core.telemetry import get_logger

logger = get_logger(__name__)

# Use structured logging
logger.info("User action", user_id="123", action="login")
logger.error("Failed to process", exception=e, item_id="456")

# Use operation scopes
with logger.operation("process_batch", batch_size=100):
    for item in items:
        logger.debug("Processing item", item_id=item.id)
```

---

## Debugging Tips

### Enable Debug Logging

```bash
export DORAEMON_LOG_LEVEL=DEBUG
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
Check `~/.doraemon/logs/doraemon.log` for detailed traces.

### Common Issues

1. **Server not connecting**: Check if the server script exists and is executable
2. **Tool not found**: Verify the server registered the tool correctly
3. **Path errors**: Ensure paths are relative to the workspace root
4. **Memory errors**: Reduce `DORAEMON_MAX_MEMORY_MB` for constrained environments

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Write tests for your changes
4. Ensure all tests pass: `pytest tests/`
5. Run linting: `ruff check src/ tests/`
6. Submit a pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.
