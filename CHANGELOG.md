# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive project audit and review system
- Automated code quality checks with ruff and mypy
- Enhanced pytest markers (benchmark, security)
- Configuration for Dependabot and security auditing

### Changed
- Updated dependencies to latest versions (cryptography, certifi, Jinja2, bcrypt)
- Improved type annotations in gateway/schema.py
- Fixed code style issues (9 ruff errors fixed)
- Enhanced test collection (excluded fixture directories)

### Fixed
- Missing imports in context_manager.py (hashlib)
- Missing imports in model_client_gateway.py (httpx)
- Test collection errors for fixture test files
- Mock issues in test_model_client.py (AsyncMock for async context managers)
- Type annotation errors in schema.py to_dict methods

## [0.8.0] - 2026-03-07

### Added
- **Unified Model Client**: Single interface for Google, OpenAI, Anthropic, and Ollama
- **Gateway Mode**: Centralized model access through a gateway server
- **Direct Mode**: Direct API access with individual provider SDKs
- **Unified Filesystem Tools**: 3 core tools replacing 15+ scattered tools
  - `read()` - 4 modes (file, directory, outline, tree)
  - `write()` - 5 operations (create, edit, delete, move, copy)
  - `search()` - 3 modes (content, files, symbol)
- **Context Management**: Automatic conversation summarization
- **Safety Features**:
  - Git safety protocol (prevent dangerous operations)
  - Shell command hardening (blacklist and confirmation)
  - Sensitive path protection (.env, .ssh, etc.)
  - Permission rules engine (ALLOW/ASK/DENY/WARN)
- **Testing Infrastructure**:
  - 2,586 test cases
  - Unit, integration, security, and benchmark tests
  - Test coverage reporting
- **Documentation**:
  - Detailed README with architecture diagrams
  - CLAUDE.md for AI assistant guidance
  - Code style guide

### Changed
- Refactored from MCP subprocess to direct function calls (zero overhead)
- Improved error handling with retry logic
- Enhanced streaming support for all providers
- Better async/await patterns throughout

### Technical Details
- **Python Support**: 3.10, 3.11, 3.12
- **Code Size**: 37,085 lines
- **Core Files**: 116 Python modules
- **Dependencies**: 27 core packages

## [0.7.0] - 2026-02-15

### Added
- Initial public release
- Basic chat interface with CLI
- Support for Google Gemini models
- File system operations
- Git integration
- LSP (Language Server Protocol) support

### Known Issues
- Limited provider support (only Google)
- No context summarization
- Basic error handling

## [0.6.0] - 2026-01-20

### Added
- Prototype MCP (Model Context Protocol) implementation
- Basic tool registration system
- Simple context management

## [0.5.0] - 2026-01-10

### Added
- Initial development version
- Basic command-line interface
- Proof of concept for multi-model support

---

## Version History Summary

| Version | Date | Key Features |
|---------|------|--------------|
| 0.8.0 | 2026-03-07 | Unified tools, Gateway mode, Safety features |
| 0.7.0 | 2026-02-15 | Initial release, Gemini support |
| 0.6.0 | 2026-01-20 | MCP prototype |
| 0.5.0 | 2026-01-10 | Development version |

## Upgrade Guide

### From 0.7.0 to 0.8.0

**Breaking Changes**:
- Tool interface unified (old tool names deprecated but still work)
- Model client now requires explicit `await client.connect()`

**Migration Steps**:
1. Update imports: `from src.core.model_client import ModelClient`
2. Update async usage:
   ```python
   # Old
   client = ModelClient()
   
   # New
   client = await ModelClient.create()
   # or
   client = DirectModelClient(config)
   await client.connect()
   ```
3. Update tool calls:
   ```python
   # Old
   read_file(path="main.py")
   
   # New
   read(path="main.py", mode="file")
   ```

**Configuration Changes**:
- `.env` format updated (see `.env.example`)
- Added `DORAEMON_GATEWAY_URL` for gateway mode
- New permission rules in `.doraemon/config.json`

## Future Roadmap

### [0.9.0] - Planned
- Web UI improvements
- More LLM providers (Mistral, Cohere)
- Enhanced code analysis tools
- Performance optimizations

### [1.0.0] - Planned
- Stable API
- Complete documentation
- Production-ready features
- Long-term support

---

For more details on each release, see the [GitHub Releases](https://github.com/ifnodoraemon/doraemon-code/releases) page.
