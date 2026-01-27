# 🤖 Doraemon Code

**Doraemon Code** 是一个功能完整的 AI 编程助手，对标 Claude Code，支持 **Web UI**、**Model Gateway** 多模型接入和丰富的开发功能。

## ✨ 新特性 v0.7.0

- **🌐 Model Gateway**: 统一的模型网关，支持 Google Gemini、OpenAI、Anthropic、Ollama
- **🔄 任务中断恢复**: 自动保存任务状态，支持 Ctrl+C 后恢复
- **🔌 插件系统**: 从 GitHub 安装插件扩展功能
- **📊 成本追踪**: 实时 Token 用量和费用统计
- **🎯 检查点系统**: 代码和对话的回滚能力
- **🤖 子代理系统**: 动态创建专门化的 AI 代理
- **🎨 主题系统**: 多种内置主题可选

- **Polyglot**: Writes and understands Python, JavaScript, Go, Rust, etc.

## 🏗️ Architecture

Doraemon Code follows a modular **Host-Server** architecture:

- **Host (Brain)**: Manages context, LLM interaction, and tool orchestration.
- **Servers (Limbs)**: specialized modules for capabilities:
  - `fs`: Unified filesystem operations
  - `lsp`: Code intelligence (LSP)
  - `git`: Version control & worktrees
  - `webui`: Modern web interface

## 📂 Directory Structure

```
src/
├── core/           # Core logic (context, planning, subagents)
├── host/           # CLI implementation
│   └── cli/        # Modular CLI commands
├── servers/        # Capability servers
├── services/       # Shared services
└── webui/          # Web Interface (React + FastAPI)
```

## 🚀 Getting Started

    
    subgraph Core [Core Infrastructure]
        Config[Configuration]
        Metrics[Metrics]
        Telemetry[Telemetry]
        Errors[Error Handling]
    end
    
    subgraph Servers [MCP Servers]
        FS[Filesystem]
        Computer[computer]
        Memory[memory]
        Web[web]
        Task[task]
    end
    
    subgraph External [External Services]
        Gemini[Google Gemini]
        OpenAI[OpenAI]
        ChromaDB[ChromaDB]
    end
    
    CLI --> Client
    Client --> FS
    Client --> Computer
    Client --> Memory
    Client --> Web
    Client --> Task
    
    CLI --> Gemini
    CLI --> OpenAI
    Memory --> ChromaDB
    
    DI --> Core
    Events --> Core
```

## Features

### v0.4.1 (Enterprise Infrastructure)

- **Parallel Connections**: MCP servers connect in parallel for faster startup
- **Result Caching**: Intelligent caching with TTL for read operations
- **Circuit Breaker**: Automatic fault isolation for failing servers
- **Retry Policy**: Exponential backoff with jitter for transient errors
- **Metrics Collection**: Counter, Gauge, Histogram metrics
- **Distributed Tracing**: Span-based tracing across tool calls
- **User-Friendly Errors**: Localized error messages (EN/ZH)

### v0.4.0 (Multi-Mode Agent)

- **Expert Modes**: Switch between specialized personas
  - **Default**: General assistant
  - **Plan (`/mode plan`)**: Strategic planning and task breakdown
  - **Build (`/mode build`)**: Implementation and execution
  - **Coder (`/mode coder`)**: Code quality and testing focus
  - **Architect (`/mode architect`)**: System design and documentation
- **Code Intelligence**: AST parsing, symbol navigation
- **Task Management**: Built-in todo list with MCP server

### Core Features

- **MCP Architecture**: Client-Server model for unlimited extensibility
- **Multi-modal Vision**: Gemini Vision and GPT-4o support
- **Long-term Memory**: ChromaDB-based vector storage
- **File Processing**: PDF, Word, PPT, Excel, images
- **Code Execution**: Sandboxed Python interpreter
- **Security**: Path jailing, HITL approval, resource limits

## Quick Start

### Installation

```bash
# Clone
git clone https://github.com/doraemon-code/doraemon.git
cd doraemon

# Install
pip install -e .

# Or with dev dependencies
pip install -e ".[dev]"
```

### Configuration

Create `.env` in project root:

```bash
GOOGLE_API_KEY="your_google_api_key"
OPENAI_API_KEY="your_openai_api_key"  # Optional
```

### Run

```bash
# Start CLI
pl start

# With specific project (isolated memory)
pl start --project "MyProject"
```

## Commands

| Command | Description |
|---------|-------------|
| `/help` | 显示所有命令 |
| `/mode <name>` | 切换模式（plan/build）|
| `/model [name]` | 切换/列出可用模型 |
| `/checkpoints` | 列出检查点 |
| `/rewind [id]` | 回滚到检查点 |
| `/cost` | 显示费用统计 |
| `/sessions` | 列出会话 |
| `/resume <id>` | 恢复会话 |
| `/plugins` | 列出插件 |
| `/theme [name]` | 切换主题 |
| `/doctor` | 运行健康检查 |
| `/exit` | 退出 |
| `!<cmd>` | 执行 Shell 命令 |

## MCP Servers

| Server | Tools | Description |
|--------|-------|-------------|
| `filesystem` | `read_file`, `write_file`, `edit_file`, `list_directory`, etc. | Unified filesystem operations |
| `computer` | `execute_python`, `install_package` | Sandboxed code execution |
| `memory` | `save_note`, `search_notes` | Vector-based long-term memory |
| `web` | `fetch_url`, `web_search` | Web content fetching |
| `task` | `task_create`, `task_list`, `task_update_status` | Task management |

## Project Structure

```
doraemon/
├── src/
│   ├── core/           # Core infrastructure
│   │   ├── configuration.py  # Hierarchical config
│   │   ├── events.py         # Event bus (pub/sub)
│   │   ├── metrics.py        # Metrics collection
│   │   ├── errors.py         # Error handling
│   │   └── telemetry.py      # Logging & tracing
│   ├── host/           # CLI host
│   │   ├── cli.py            # Main CLI
│   │   └── client.py         # MCP client
│   ├── servers/        # MCP servers
│   └── services/       # Shared services
├── tests/              # Test suite
└── docs/               # Documentation
```

## Security

- **HITL (Human-in-the-loop)**: Sensitive operations require user approval
- **Path Sandboxing**: File operations restricted to workspace
- **Resource Limits**: Code execution has memory/CPU/time limits
- **Circuit Breaker**: Automatic isolation of failing services

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Gemini API key | Required |
| `DORAEMON_MODEL` | LLM model | `gemini-3-pro-preview` |
| `DORAEMON_LOG_LEVEL` | Log level | `INFO` |
| `DORAEMON_MAX_MEMORY_MB` | Code execution memory | `512` |

### Config File

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
    "execute_python",
    "write_file"
  ]
}
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/
```

See [docs/development.md](docs/development.md) for detailed development guide.

## Documentation

- [API Reference](docs/api.md) - Complete API documentation
- [Development Guide](docs/development.md) - How to extend Doraemon
- [Contributing](CONTRIBUTING.md) - Contribution guidelines

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Model Context Protocol](https://modelcontextprotocol.io) - The foundation
- [Google Gemini](https://ai.google.dev) - LLM provider
- [ChromaDB](https://www.trychroma.com) - Vector database
- [Rich](https://rich.readthedocs.io) - Beautiful terminal output
See [docs/development.md](docs/development.md) for detailed development guide.

## Documentation

- [API Reference](docs/api.md) - Complete API documentation
- [Development Guide](docs/development.md) - How to extend Doraemon
- [Contributing](CONTRIBUTING.md) - Contribution guidelines

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Model Context Protocol](https://modelcontextprotocol.io) - The foundation
- [Google Gemini](https://ai.google.dev) - LLM provider
- [ChromaDB](https://www.trychroma.com) - Vector database
- [Rich](https://rich.readthedocs.io) - Beautiful terminal output
