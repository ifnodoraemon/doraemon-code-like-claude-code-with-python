# 🤖 Doraemon Code

**Doraemon Code** is a powerful AI coding assistant built on the Model Context Protocol (MCP), featuring a unified model gateway, comprehensive testing, and rich development tools.

[![Tests](https://img.shields.io/badge/tests-2440%20passing-brightgreen)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-core%20modules%2095%25%2B-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## ✨ Features

### 🌐 Model Gateway
- **Unified API**: Single interface for Google Gemini, OpenAI, Anthropic, and Ollama
- **Auto-detection**: Automatically selects gateway or direct mode based on configuration
- **Provider Adapters**: Seamless conversion between different model formats

### 🧪 Comprehensive Testing
- **2,440 tests** with **95%+ coverage on core modules**
- Comprehensive test suites for all core modules
- Automated testing with pytest

### 🏗️ MCP Architecture
- **Host-Server Model**: Modular design for unlimited extensibility
- **Direct Function Calls**: No subprocess overhead with FastMCP
- **Multiple Servers**: Filesystem, Git, Browser, Database, and more

### 🎯 Core Capabilities
- **Context Management**: Automatic summarization at 70% of context window
- **Checkpoint System**: File snapshots and rollback capability
- **Session Persistence**: Resume conversations across restarts
- **Tool Registry**: Easy tool registration with automatic parameter extraction

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ifnodoraemon/doraemon-code.git
cd doraemon-code

# Install with pip
pip install -e .

# Or with dev dependencies
pip install -e ".[dev]"
```

### Configuration

Create a `.env` file in the project root:

```bash
# For Gateway Mode (recommended)
DORAEMON_GATEWAY_URL=http://localhost:8000
DORAEMON_GATEWAY_KEY=your_api_key  # Optional

# For Direct Mode (at least one required)
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# Model Selection
DORAEMON_MODEL=gemini-2.0-flash-exp  # or gpt-4, claude-3-5-sonnet, etc.
```

### Run

```bash
# Start CLI
doraemon

# Or use the short alias
dora

# With specific project (isolated memory)
doraemon --project "MyProject"
```

## 📂 Project Structure

```
doraemon-code/
├── src/
│   ├── core/              # Core infrastructure
│   │   ├── model_client.py      # Unified LLM interface
│   │   ├── context_manager.py   # Conversation management
│   │   ├── mcp_client.py        # MCP client implementation
│   │   ├── checkpoint.py        # File snapshots & rollback
│   │   └── session.py           # Session persistence
│   ├── host/              # CLI implementation
│   │   ├── cli/                 # Main chat loop
│   │   └── tools.py             # Tool registry
│   ├── servers/           # MCP servers
│   │   ├── filesystem.py        # File operations
│   │   ├── git.py               # Version control
│   │   ├── browser.py           # Web browsing
│   │   └── database.py          # Database operations
│   ├── gateway/           # Model gateway
│   │   ├── server.py            # FastAPI server
│   │   ├── router.py            # Model routing
│   │   └── adapters/            # Provider adapters
│   └── webui/             # Web interface (React + FastAPI)
├── tests/                 # Comprehensive test suite
│   └── core/              # Core module tests
└── docs/                  # Documentation
```

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/core/test_model_client_comprehensive.py -v
```

### Test Coverage

| Module | Tests | Coverage | Status |
|--------|-------|----------|--------|
| `model_client.py` | 117 | 90%+ | ✅ |
| `mcp_client.py` | 103 | 97%+ | ✅ |
| `chat_loop.py` | 72 | 95%+ | ✅ |
| `checkpoint.py` | 70 | 100% | ✅ |
| `session.py` | 121 | 95%+ | ✅ |
| `context_manager.py` | - | 95%+ | ✅ |
| `plugins.py` | - | 95%+ | ✅ |
| `hooks.py` | - | 95%+ | ✅ |

**Total**: 2,440 tests with 95%+ coverage on core modules

## 🛠️ Development

### Setup Development Environment

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run linter
ruff check src/ tests/

# Auto-fix issues
ruff check src/ tests/ --fix

# Format code
ruff format src/ tests/

# Type checking
mypy src/
```

### Code Style

- **Line length**: 100 characters
- **Python version**: 3.10+
- **Linter**: Ruff
- **Formatter**: Ruff
- **Type checker**: MyPy

## 🏗️ Architecture

### Host-Server Model

```
┌─────────────────────────────────────────┐
│              CLI Host                    │
│  ┌──────────────────────────────────┐   │
│  │     Context Manager              │   │
│  │  - Conversation history          │   │
│  │  - Auto summarization            │   │
│  └──────────────────────────────────┘   │
│  ┌──────────────────────────────────┐   │
│  │     Model Client                 │   │
│  │  - Gateway mode / Direct mode    │   │
│  │  - Unified interface             │   │
│  └──────────────────────────────────┘   │
│  ┌──────────────────────────────────┐   │
│  │     Tool Registry                │   │
│  │  - Direct function calls         │   │
│  │  - Auto parameter extraction     │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
   ┌────▼────┐ ┌───▼────┐ ┌───▼────┐
   │Filesystem│ │  Git   │ │Browser │
   │ Server  │ │ Server │ │ Server │
   └─────────┘ └────────┘ └────────┘
```

### Model Gateway

```
┌─────────────────────────────────────────┐
│         Gateway Server                   │
│  ┌──────────────────────────────────┐   │
│  │     Router                       │   │
│  │  - Model selection               │   │
│  │  - Request routing               │   │
│  └──────────────────────────────────┘   │
│  ┌──────────────────────────────────┐   │
│  │     Provider Adapters            │   │
│  │  - Google Gemini                 │   │
│  │  - OpenAI                        │   │
│  │  - Anthropic                     │   │
│  │  - Ollama                        │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DORAEMON_GATEWAY_URL` | Gateway server URL | - |
| `DORAEMON_GATEWAY_KEY` | Gateway API key | - |
| `GOOGLE_API_KEY` | Google Gemini API key | - |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `DORAEMON_MODEL` | Default model | `gemini-2.0-flash-exp` |
| `DORAEMON_LOG_LEVEL` | Logging level | `INFO` |

### Config File

`.doraemon/config.json`:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "python",
      "args": ["src/servers/filesystem.py"]
    },
    "git": {
      "command": "python",
      "args": ["src/servers/git.py"]
    }
  },
  "sensitive_tools": [
    "write_file",
    "execute_python",
    "shell_execute"
  ]
}
```

## 📚 Documentation

- [CLAUDE.md](CLAUDE.md) - Development guide for Claude Code
- [docs/api.md](docs/api.md) - API reference
- [docs/development.md](docs/development.md) - Development guide

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Model Context Protocol](https://modelcontextprotocol.io) - The foundation
- [Google Gemini](https://ai.google.dev) - LLM provider
- [OpenAI](https://openai.com) - LLM provider
- [Anthropic](https://anthropic.com) - LLM provider
- [FastMCP](https://github.com/jlowin/fastmcp) - MCP implementation
- [Rich](https://rich.readthedocs.io) - Beautiful terminal output

## 📊 Project Status

- **Version**: 0.8.0
- **Status**: Active Development
- **Python**: 3.10+
- **Tests**: 2,440 passing
- **Coverage**: 95%+ (core modules)

---

**Built with ❤️ by the Doraemon Code team**
