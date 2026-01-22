# Polymath Project Rules

Project-specific rules and conventions for the Polymath AI coding agent.

## Project Overview

Polymath is an AI coding agent with two modes:
- **plan**: Analyze requirements, investigate code, create plans (read-only)
- **build**: Implement solutions, write code, execute tasks

## Tech Stack

- **Language**: Python 3.10+
- **AI SDK**: Google GenAI (Gemini)
- **CLI**: Typer + Rich for terminal UI
- **Storage**: ChromaDB for vector memory

## Code Style

- Use 4 spaces for indentation
- Type hints are required for all public functions
- Follow PEP 8 naming conventions
- Use docstrings for all public classes and functions
- Maximum line length: 100 characters (enforced by ruff)

## Directory Structure

```
src/
├── core/           # Core infrastructure (config, context, skills, rules)
├── host/           # CLI and tool registry
├── servers/        # Tool implementations
└── services/       # Shared business logic
```

## Context Engineering

### Rules Loading
- `POLYMATH.md` in project root (this file)
- `~/.polymath/POLYMATH.md` for global rules

### Skills System
- `.polymath/skills/<name>/SKILL.md` for domain-specific knowledge
- Skills are loaded automatically based on conversation context
- Use YAML frontmatter with `triggers` for automatic activation

### Context Management
- Automatic summarization when context exceeds 70% of window
- Conversation persistence to `.polymath/conversations/`
- Session restoration on startup

## Important Notes

### Security
- Sensitive tools require HITL (Human-in-the-Loop) approval
- File operations show diff before execution

### Environment Variables
- `GOOGLE_API_KEY`: Required for Gemini API
- `POLYMATH_MODEL`: Override default model (default: gemini-2.0-flash)

## Usage

```bash
# Start CLI
pl start

# Start with specific project
pl start --project myproject

# Switch modes
/mode plan   # Planning mode (read-only)
/mode build  # Build mode (default)
```

