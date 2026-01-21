# Polymath Project Rules

This file contains project-specific rules and conventions that Polymath will follow.
It replaces the deprecated `POLYMATH.md` file for project memory and conventions.

## Project Overview

Polymath is an extensible AI assistant built on the Model Context Protocol (MCP).
It acts as a "Host" that orchestrates various specialized "Servers" (Memory, Vision, 
Compute, etc.) to assist users with writing, coding, and analysis tasks.

## Tech Stack

- **Language**: Python 3.10+
- **AI SDKs**: Google GenAI, OpenAI, Anthropic (official SDKs only)
- **CLI**: Typer + Rich for terminal UI
- **Storage**: ChromaDB for vector memory, SentenceTransformers for embeddings
- **Protocol**: MCP (Model Context Protocol) for tool execution

## Code Style

- Use 4 spaces for indentation
- Type hints are required for all public functions
- Follow PEP 8 naming conventions
- Use docstrings for all public classes and functions
- Maximum line length: 100 characters (enforced by ruff)

## Architecture

### Core Patterns

1. **Dependency Injection**: Use `src/core/container.py` for service management
2. **Event-Driven**: Use `src/core/events.py` for decoupled communication
3. **Configuration**: Use `src/core/configuration.py` for hierarchical config
4. **Error Handling**: Use `src/core/errors.py` for categorized exceptions

### Directory Structure

```
src/
├── core/           # Core infrastructure (DI, config, events, telemetry)
├── host/           # CLI and MCP client
├── servers/        # MCP server implementations
├── services/       # Shared business logic
└── evals/          # Evaluation harness
```

## Important Notes

### Security

- All file operations must use `validate_path()` from `security.py`
- Sensitive tools require HITL (Human-in-the-Loop) approval
- Code execution has resource limits (memory, CPU, file size)

### Testing

- Run tests with: `pytest tests/`
- Core module tests are in `tests/core/`
- Minimum coverage target: 60%

### Environment Variables

- `GOOGLE_API_KEY`: Required for Gemini API
- `POLYMATH_LOG_LEVEL`: Log level (DEBUG, INFO, WARNING, ERROR)
- `POLYMATH_LOG_FILE`: Enable file logging (true/false)
- `POLYMATH_MODEL`: Override default model (default: gemini-2.0-flash)

## Example Usage

```bash
# Start CLI
pl start

# Start with specific project
pl start --project myproject

# Switch modes
/mode plan    # Planning mode
/mode build   # Build mode
/mode coder   # Coding mode
```

