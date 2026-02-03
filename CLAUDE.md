# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Doraemon Code is an AI coding assistant built on the Model Context Protocol (MCP) with a Host-Server architecture. It supports multiple LLM providers through a unified gateway and provides both CLI and Web UI interfaces.

## Development Commands

### Setup
```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Install with all optional features
pip install -e ".[all]"
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_specific.py -v

# Run tests matching pattern
pytest tests/ -k "test_pattern" -v

# Skip slow tests
pytest tests/ -m "not slow"
```

### Linting & Formatting
```bash
# Check code style
ruff check src/ tests/

# Auto-fix issues
ruff check src/ tests/ --fix

# Format code
ruff format src/ tests/

# Type checking
mypy src/
```

### Running the Application
```bash
# Start CLI
doraemon

# Or using alias
dora

# Start with specific project (isolated memory)
doraemon --project "MyProject"

# Start Web UI
python -m src.webui.server

# Start Gateway server
python -m src.gateway.server
```

## Architecture Overview

### Core Design Pattern: Host-Server Architecture

**Host (Brain)**: `src/host/cli/main.py`
- Orchestrates conversation flow
- Manages context and tool execution
- Handles user interaction

**Servers (Limbs)**: `src/servers/`
- Provide specialized capabilities via direct function calls
- No subprocess overhead (FastMCP-based)
- Examples: filesystem, shell, git, browser, database

### Key Components

**1. ModelClient** (`src/core/model_client.py`)
- Unified interface for all LLM providers
- Supports two modes:
  - **Gateway mode**: Routes through unified API server
  - **Direct mode**: Calls provider APIs directly
- Auto-detects mode from environment (DORAEMON_GATEWAY_URL vs API keys)
- Always use this for LLM calls, never call providers directly

**2. ContextManager** (`src/core/context_manager.py`)
- Manages conversation history with automatic summarization
- Summarizes at 70% of context window to prevent overflow
- Persists to `.doraemon/conversations/`
- Keeps recent messages always to maintain context

**3. ToolRegistry** (`src/host/tools.py`)
- Direct function registration (no subprocess)
- Automatic parameter extraction from signatures
- Sensitive tool marking for HITL (Human-in-the-Loop) approval
- Register new tools here with `@register` decorator

**4. ToolSelector** (`src/core/tool_selector.py`)
- Mode-based tool allocation:
  - **Plan mode**: Read-only tools (no modifications)
  - **Build mode**: All tools including write/execute
- Prevents accidental modifications during planning

**5. Gateway System** (`src/gateway/`)
- Unified API for multiple providers (Google, OpenAI, Anthropic, Ollama)
- Provider adapters convert between unified format and provider-specific format
- FastAPI server with CORS for web UI integration

### Data Flow

```
User Input
    ↓
CommandHandler (slash commands) or ContextManager
    ↓
ModelClient (unified interface)
    ↓
[Gateway Mode] → Gateway Server → Provider Adapter → LLM
[Direct Mode] → Provider SDK → LLM
    ↓
Response with tool calls
    ↓
ToolRegistry executes tools (with HITL for sensitive ops)
    ↓
Results back to model
    ↓
Display to user
```

### Critical Abstractions

**Message Format** (unified across providers):
```python
Message(
    role: str,           # "user", "assistant", "system", "tool"
    content: str,        # Main content
    thought: str,        # Reasoning (optional)
    tool_calls: list,    # Function calls (optional)
    tool_call_id: str,   # For tool results
    name: str            # Tool name (for results)
)
```

**Tool Definition** (converted per provider):
- Unified format in `ToolDefinition` class
- Converted to GenAI `FunctionDeclaration` for Google
- Converted to OpenAI function format for OpenAI/Anthropic

## Important Patterns

### Tool Design Principles

**Doraemon Code follows these core principles for tool design:**

1. **Occam's Razor (奥卡姆剃刀)**: Simplicity is paramount. Prefer fewer, well-designed tools over many scattered ones.

2. **Single Responsibility with Functional Cohesion**: Each tool should have one clear purpose, but related operations should be grouped together through parameters rather than creating multiple tools.

3. **Parameterized Design**: Use mode/operation parameters to distinguish behaviors instead of creating separate tools for each variation.

### Unified Filesystem Tools (Recommended)

Doraemon provides **3 unified tools** that replace 15 scattered tools:

```python
# 1. read - Unified reading tool
read(path, mode="file")  # Read file content
read(path, mode="outline")  # Get file structure
read(path, mode="directory")  # List directory
read(path, mode="tree", depth=2)  # Show directory tree

# 2. write - Unified writing tool
write(path, content="...", operation="create")  # Create file
write(path, operation="edit", old_string="...", new_string="...")  # Edit
write(path, operation="delete")  # Delete
write(path, operation="move", destination="...")  # Move/rename
write(path, operation="copy", destination="...")  # Copy

# 3. search - Unified searching tool
search(query, mode="content")  # Search file contents (grep)
search(query, mode="files")  # Search file names (glob)
search(query, mode="symbol")  # Search code symbols
```

**Legacy tools** (read_file, write_file, edit_file, glob_files, grep_search, etc.) are still available for backward compatibility but are deprecated.

### Adding New Tools

Register in `src/host/tools.py`:
```python
@register_tool(
    name="my_tool",
    description="What it does",
    sensitive=True  # Requires HITL approval
)
def my_tool(param: str) -> str:
    """Tool implementation"""
    return result
```

### Adding New MCP Servers

Create in `src/servers/`:
```python
from fastmcp import FastMCP

mcp = FastMCP("server_name")

@mcp.tool()
def my_server_tool(param: str) -> str:
    """Tool implementation"""
    return result
```

### Adding New Model Adapters

Implement `BaseAdapter` in `src/gateway/adapters/`:
```python
class MyAdapter(BaseAdapter):
    async def initialize(self):
        # Setup provider client
        pass

    async def chat(self, request: ChatRequest) -> ChatResponse:
        # Convert and call provider
        pass
```

### Mode-Based Development

**Plan Mode** (`/mode plan`):
- Read-only exploration
- Use for understanding codebase
- No file modifications allowed
- Tools: read_file, grep_search, web_search, etc.

**Build Mode** (`/mode build`):
- Full implementation capabilities
- File modifications, code execution
- Tools: write_file, edit_file, shell_execute, etc.

## Configuration

### Environment Variables

```bash
# Model Selection
DORAEMON_MODEL=gemini-3-pro-preview
DORAEMON_GATEWAY_URL=http://localhost:8000  # Enable gateway mode
DORAEMON_GATEWAY_KEY=...                     # Optional API key

# Direct Mode (at least one required if not using gateway)
GOOGLE_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...

# Budget Control
DORAEMON_DAILY_BUDGET=10.0
DORAEMON_SESSION_BUDGET=5.0

# Logging
DORAEMON_LOG_LEVEL=INFO
```

### Project Configuration

`.doraemon/config.json`:
```json
{
  "mcpServers": {
    "memory": {
      "command": "python3",
      "args": ["src/servers/memory.py"]
    }
  },
  "persona": {
    "name": "Doraemon Code",
    "role": "AI Assistant"
  },
  "sensitive_tools": [
    "write_file",
    "execute_python",
    "shell_execute"
  ]
}
```

## Directory Structure

```
src/
├── core/              # Core infrastructure
│   ├── model_client.py      # Unified LLM interface
│   ├── context_manager.py   # Conversation management
│   ├── tool_selector.py     # Mode-based tool allocation
│   ├── checkpoint.py        # File snapshots & rollback
│   ├── session.py           # Session persistence
│   └── ...
├── host/              # CLI implementation
│   ├── cli/
│   │   ├── main.py          # Main chat loop
│   │   └── commands.py      # Slash command handlers
│   └── tools.py             # Tool registry
├── servers/           # MCP servers (capabilities)
│   ├── filesystem.py        # File operations
│   ├── shell.py             # Command execution
│   ├── git.py               # Version control
│   ├── browser.py           # Web browsing
│   └── ...
├── gateway/           # Model gateway
│   ├── server.py            # FastAPI server
│   ├── router.py            # Model routing
│   └── adapters/            # Provider adapters
└── webui/             # Web interface
    ├── server.py            # FastAPI backend
    └── routes/              # API endpoints
```

## Security & Safety

**HITL (Human-in-the-Loop)**:
- Sensitive tools require user approval
- Preview shown before execution
- Denied in headless mode

**Path Validation**:
- All file operations validated with `validate_path()`
- Prevents directory traversal attacks

**Resource Limits**:
- Code execution has memory/CPU/time limits
- Configurable via environment variables

**Checkpoint System**:
- Automatic file snapshots before modifications
- Rollback capability with `/rewind`
- Stored in `.doraemon/checkpoints/`

## Extension Points

**Easy to extend**:
1. New tools → Register in `ToolRegistry`
2. New servers → Create in `src/servers/`
3. New adapters → Implement `BaseAdapter`
4. New skills → Add SKILL.md in `.doraemon/skills/`
5. New hooks → Define in `hooks.json`
6. New commands → Add handler in `CommandHandler`

## Performance Considerations

- Direct function calls (no subprocess) save ~10ms per tool call
- Context summarization keeps window manageable
- Result caching for read operations
- Streaming responses for real-time feedback
- Lazy loading of skills and plugins

## Tech Stack

- **Language**: Python 3.10+
- **Framework**: FastAPI (gateway & web UI)
- **LLM SDKs**: google-genai, openai, anthropic
- **MCP**: FastMCP for server implementation
- **CLI**: Typer, Rich, Textual
- **Database**: ChromaDB (vector storage)
- **Browser**: Playwright
- **Testing**: pytest, pytest-asyncio, pytest-cov

## Code Style

- 4 space indentation
- Type hints required
- Line length: 100 characters
- Linter: Ruff
- Formatter: Ruff
- Type checker: MyPy
